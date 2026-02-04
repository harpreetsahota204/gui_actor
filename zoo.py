import os
import logging
from PIL import Image
from typing import Dict, Any, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available

from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """"You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>"""

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class GUIActorModel(SamplesMixin, Model):
    """A FiftyOne model for running GUIActorModel vision tasks"""

    def __init__(
        self,
        model_path: str,
        prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self.prompt = prompt

        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "device_map":self.device,
            }

        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        logger.info("Loading processor")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )

        self.tokenizer = self.processor.tokenizer

        self.model.eval()

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"

    def _compute_attention_heatmap(self, pred: Dict[str, Any], image: Image.Image) -> Optional[fo.Heatmap]:
        """Compute attention heatmap as a FiftyOne Heatmap label.
        
        Stores the heatmap at native attention map resolution to avoid MongoDB
        document size limits. FiftyOne handles resizing for visualization.
        
        Args:
            pred: Model prediction dictionary containing attention scores and dimensions
            image: Original PIL Image
            
        Returns:
            fo.Heatmap: Heatmap label with normalized attention scores, or None if data missing
        """
        if not pred.get("attn_scores") or not pred.get("n_height") or not pred.get("n_width"):
            logger.warning("Missing attention data, skipping heatmap generation")
            return None
            
        try:
            W, H = pred["n_width"], pred["n_height"]  # attention map size
            
            # Reshape attention scores to 2D grid (following paper's approach)
            scores = np.array(pred["attn_scores"][0]).reshape(H, W)
            
            # Normalize the attention weights to [0, 1] range
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
            
            # Return as FiftyOne Heatmap label at native resolution
            # FiftyOne will handle resizing for visualization
            return fo.Heatmap(map=scores_norm.astype(np.float32))
            
        except Exception as e:
            logger.error(f"Error computing attention heatmap: {e}")
            return None

    def _to_keypoints(self, pred: Dict[str, Any], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert model predictions to FiftyOne Keypoints.
        
        Args:
            pred: Model prediction dictionary from inference
            image_width: Original image width in pixels  
            image_height: Original image height in pixels
            
        Returns:
            fo.Keypoints: FiftyOne Keypoints object containing the top interaction point
        """
        keypoints = []
        
        # Extract main interaction points and confidence scores
        topk_points = pred.get("topk_points", [])
        topk_values = pred.get("topk_values", [])
        output_text = pred.get("output_text", "")
        
        if not topk_points:
            logger.warning("No topk_points found in prediction")
            return fo.Keypoints(keypoints=[])
        
        # Only process the top (first) prediction
        try:
            point = topk_points[0]  # Get the highest confidence point
            confidence = topk_values[0] if topk_values else None
            
            # Handle tuple format: topk_points contains tuples
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                x, y = point[0], point[1]
            else:
                logger.warning(f"Unexpected point format: {point}")
                return fo.Keypoints(keypoints=[])
            
            # Create the top interaction keypoint
            keypoint = fo.Keypoint(
                label="top_interaction_point",
                points=[[float(x), float(y)]],
                reasoning=output_text,
                confidence=[float(confidence)],  # List with one confidence value
            )
                
            keypoints.append(keypoint)
            logger.info(f"Added top keypoint at ({x:.3f}, {y:.3f}) with confidence {confidence:.3f}")
                            
        except Exception as e:
            logger.error(f"Error processing top interaction point: {e}")
            return fo.Keypoints(keypoints=[])
        
        return fo.Keypoints(keypoints=keypoints)
    
    def _predict(self, image: Image.Image, sample=None) -> fo.Keypoints:
        """Process a single image through the model and return keypoint predictions.
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            fo.Keypoints: Keypoint predictions for GUI interaction points
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        messages = [
            {
                "role": "system", 
                "content": [  
                    {
                        "type": "text",
                        "text": DEFAULT_SYSTEM_PROMPT
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": sample.filepath if sample else image
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ]

        # Run inference
        pred = inference(
            messages, 
            self.model, 
            self.tokenizer, 
            self.processor, 
            use_placeholder=True, 
            topk=3
        )

        # Compute and store heatmap on the sample
        if sample is not None:
            heatmap = self._compute_attention_heatmap(pred, image)
            if heatmap is not None:
                sample["gui_actor_heatmap"] = heatmap
                sample.save()
        
        # Convert predictions to keypoints and return
        return self._to_keypoints(pred, image.width, image.height)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            fo.Keypoints: Keypoint predictions for GUI interaction points
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)