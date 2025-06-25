import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available

from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """"You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."""

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

        if self.torch_dtype:
            self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
                model_path,
                trust_remote_code=True,
                **model_kwargs
            )
        else:
            self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=self.device,
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

    def _save_attention_heatmap(self, pred: Dict[str, Any], image: Image.Image, sample=None):
        """Save attention heatmap as PNG alongside the original image.
        
        Uses the same visualization approach as described in the GUI-Actor paper.
        
        Args:
            pred: Model prediction dictionary containing attention scores and dimensions
            image: Original PIL Image
            sample: FiftyOne sample containing filepath information
        """
        if not pred.get("attn_scores") or not pred.get("n_height") or not pred.get("n_width"):
            logger.warning("Missing attention data, skipping heatmap generation")
            return
            
        try:
            # Extract dimensions
            width, height = image.size
            W, H = pred["n_width"], pred["n_height"]  # attention map size
            
            # Reshape attention scores to 2D grid (following paper's approach)
            scores = np.array(pred["attn_scores"][0]).reshape(H, W)
            
            # Normalize the attention weights for coherent visualization
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
            
            # Resize the attention map to match the image size using PIL BILINEAR
            score_map = Image.fromarray((scores_norm * 255).astype(np.uint8)).resize(
                (width, height), resample=Image.BILINEAR
            )
            
            # Apply jet colormap (following paper's approach)
            colormap = plt.get_cmap('jet')
            colored_score_map = colormap(np.array(score_map) / 255.0)  # returns RGBA
            colored_score_map = (colored_score_map[:, :, :3] * 255).astype(np.uint8)  # Remove alpha, convert to uint8
            colored_overlay = Image.fromarray(colored_score_map)
            
            # Generate filename
            if sample and sample.filepath:
                original_path = Path(sample.filepath)
                heatmap_path = original_path.parent / f"{original_path.stem}_attention.png"
            else:
                # Fallback if no sample filepath
                heatmap_path = Path("attention_heatmap.png")
            
            # Save heatmap
            colored_overlay.save(str(heatmap_path))
            logger.info(f"Saved attention heatmap to: {heatmap_path}")
            
        except Exception as e:
            logger.error(f"Error saving attention heatmap: {e}")



    def _to_keypoints(self, pred: Dict[str, Any], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert model predictions to FiftyOne Keypoints.
        
        Args:
            pred: Model prediction dictionary from inference
            image_width: Original image width in pixels  
            image_height: Original image height in pixels
            
        Returns:
            fo.Keypoints: FiftyOne Keypoints object containing all interaction points
        """
        keypoints = []
        
        # Extract main interaction points and confidence scores
        topk_points = pred.get("topk_points", [])
        topk_values = pred.get("topk_values", [])
        topk_points_all = pred.get("topk_points_all", [])
        output_text = pred.get("output_text", "")
        
        # Process primary interaction points
        for i, point in enumerate(topk_points):
            try:
                # Points are already normalized [x, y] in range [0, 1]
                x, y = point
                confidence = topk_values[i] if i < len(topk_values) else 0.0
                
                # Create primary interaction keypoint
                keypoint = fo.Keypoint(
                    label=f"interaction_point_{i+1}",
                    points=[[float(x), float(y)]],
                    confidence=float(confidence),
                    reasoning=output_text if i == 0 else ""  # Add reasoning to first point only
                )
                keypoints.append(keypoint)
                
                # Add detailed region points if available
                if i < len(topk_points_all):
                    region_points = topk_points_all[i]
                    for j, detail_point in enumerate(region_points):
                        try:
                            detail_x, detail_y = detail_point
                            detail_keypoint = fo.Keypoint(
                                label=f"region_{i+1}_detail_{j+1}",
                                points=[[float(detail_x), float(detail_y)]],
                                confidence=float(confidence * 0.8),  # Slightly lower confidence for detail points
                            )
                            keypoints.append(detail_keypoint)
                        except Exception as e:
                            logger.debug(f"Error processing detail point {detail_point}: {e}")
                            continue
                            
            except Exception as e:
                logger.debug(f"Error processing interaction point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)
    
    def _predict(self, image: Image.Image, sample=None) -> fo.Keypoints:
        """Process a single image through the model and return keypoint predictions.
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            fo.Keypoints: Keypoint predictions for GUI interaction points
        """
        # Get prompt from sample field if available
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self.prompt = str(field_value)

        # Ensure we have a prompt
        if not self.prompt:
            raise ValueError("No prompt provided. Set prompt during initialization or via sample field.")

        # Construct chat messages
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"image": sample.filepath if sample else image}
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

        # Save attention heatmap as side effect
        self._save_attention_heatmap(pred, image, sample)
        
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