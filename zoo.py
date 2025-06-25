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
        print(f"=== INIT DEBUG: Creating GUIActorModel with prompt: '{prompt}' ===")
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
        print(f"=== INIT DEBUG: GUIActorModel created successfully ===")

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        print(f"=== NEEDS_FIELDS DEBUG: Getting fields: {self._fields} ===")
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        print(f"=== NEEDS_FIELDS DEBUG: Setting fields to: {fields} ===")
        self._fields = fields
    
    def _get_field(self):
        print(f"=== GET_FIELD DEBUG: Current needs_fields: {self.needs_fields} ===")
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
            print(f"=== GET_FIELD DEBUG: Found prompt_field: {prompt_field} ===")
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
            print(f"=== GET_FIELD DEBUG: Using first field: {prompt_field} ===")

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
        print(f"=== HEATMAP DEBUG: Saving heatmap for {sample.filepath if sample else 'no sample'} ===")
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
            print(f"=== HEATMAP DEBUG: Saved attention heatmap to: {heatmap_path} ===")
            logger.info(f"Saved attention heatmap to: {heatmap_path}")
            
        except Exception as e:
            print(f"=== HEATMAP DEBUG: Error saving heatmap: {e} ===")
            logger.error(f"Error saving attention heatmap: {e}")

    def _to_keypoints(self, pred: Dict[str, Any], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert model predictions to FiftyOne Keypoints.
        
        Args:
            pred: Model prediction dictionary from inference
            image_width: Original image width in pixels  
            image_height: Original image height in pixels
            
        Returns:
            fo.Keypoints: FiftyOne Keypoints object containing the top interaction point
        """
        print(f"=== KEYPOINTS DEBUG: Converting predictions to keypoints ===")
        keypoints = []
        
        # Extract main interaction points and confidence scores
        topk_points = pred.get("topk_points", [])
        topk_values = pred.get("topk_values", [])
        output_text = pred.get("output_text", "")
        
        print(f"=== KEYPOINTS DEBUG: topk_points: {topk_points} ===")
        print(f"=== KEYPOINTS DEBUG: topk_values: {topk_values} ===")
        print(f"=== KEYPOINTS DEBUG: output_text: '{output_text[:50]}...' ===")
        
        if not topk_points:
            print(f"=== KEYPOINTS DEBUG: No topk_points found! ===")
            logger.warning("No topk_points found in prediction")
            return fo.Keypoints(keypoints=[])
        
        # Only process the top (first) prediction
        try:
            point = topk_points[0]  # Get the highest confidence point
            confidence = topk_values[0] if topk_values else None
            
            print(f"=== KEYPOINTS DEBUG: Processing point: {point}, confidence: {confidence} ===")
            
            # Handle tuple format: topk_points contains tuples
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                x, y = point[0], point[1]
                print(f"=== KEYPOINTS DEBUG: Extracted coordinates: ({x}, {y}) ===")
            else:
                print(f"=== KEYPOINTS DEBUG: Unexpected point format: {point} ===")
                logger.warning(f"Unexpected point format: {point}")
                return fo.Keypoints(keypoints=[])
            
            # Create the top interaction keypoint
            keypoint = fo.Keypoint(
                label="top_interaction_point",
                points=[[float(x), float(y)]],
                reasoning=output_text,
                confidence=[float(confidence)] if confidence is not None else None,  # List with one confidence value
            )
                
            keypoints.append(keypoint)
            print(f"=== KEYPOINTS DEBUG: Created keypoint at ({x:.3f}, {y:.3f}) with confidence {confidence} ===")
            logger.info(f"Added top keypoint at ({x:.3f}, {y:.3f}) with confidence {confidence:.3f if confidence else 'None'}")
                            
        except Exception as e:
            print(f"=== KEYPOINTS DEBUG: Error processing point: {e} ===")
            logger.error(f"Error processing top interaction point: {e}")
            return fo.Keypoints(keypoints=[])
        
        print(f"=== KEYPOINTS DEBUG: Returning {len(keypoints)} keypoints ===")
        return fo.Keypoints(keypoints=keypoints)
    
    def _predict(self, image: Image.Image, sample=None) -> fo.Keypoints:
        """Process a single image through the model and return keypoint predictions.
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            fo.Keypoints: Keypoint predictions for GUI interaction points
        """
        print(f"\n=== PREDICT DEBUG START ===")
        print(f"Sample filepath: {sample.filepath if sample else 'No sample'}")
        print(f"Image size: {image.size}")
        print(f"Instance prompt (self.prompt): '{self.prompt}'")
        
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        print(f"Initial local prompt: '{prompt}'")
        
        if sample is not None and self._get_field() is not None:
            field_name = self._get_field()
            print(f"Looking for field: '{field_name}'")
            field_value = sample.get_field(field_name)
            print(f"Field value from sample: '{field_value}'")
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
                print(f"Updated local prompt to: '{prompt}'")
        
        print(f"Final prompt being used: '{prompt}'")
        print(f"Instance prompt after processing: '{self.prompt}' (should be unchanged)")
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},  # Use local prompt
                    {"image": sample.filepath if sample else image}
                ]
            }
        ]

        print(f"Messages being sent to inference:")
        print(f"  System: {DEFAULT_SYSTEM_PROMPT[:50]}...")
        print(f"  User text: '{prompt}'")
        print(f"  User image: {sample.filepath if sample else 'PIL Image object'}")

        # Run inference
        print("Calling inference...")
        pred = inference(
            messages, 
            self.model, 
            self.tokenizer, 
            self.processor, 
            use_placeholder=True, 
            topk=3
        )

        print(f"Inference completed. Prediction keys: {list(pred.keys())}")
        print(f"Has topk_points: {bool(pred.get('topk_points'))}")
        print(f"topk_points: {pred.get('topk_points')}")
        print(f"topk_values: {pred.get('topk_values')}")
        print(f"Output text: '{pred.get('output_text', '')[:100]}...'")

        # Save attention heatmap as side effect
        self._save_attention_heatmap(pred, image, sample)
        
        # Convert predictions to keypoints and return
        keypoints = self._to_keypoints(pred, image.width, image.height)
        print(f"Generated {len(keypoints.keypoints) if keypoints else 0} keypoints")
        print(f"=== PREDICT DEBUG END ===\n")
        
        return keypoints

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            fo.Keypoints: Keypoint predictions for GUI interaction points
        """
        print(f"=== PREDICT PUBLIC DEBUG: Called with image size {image.shape if hasattr(image, 'shape') else 'unknown'}, sample: {sample.filepath if sample else 'None'} ===")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)