"""
Depth Estimation Module

Uses pre-trained depth estimation models (Depth Anything V2, DPT, MiDaS) 
to estimate depth from a single image. Supports multiple model backends
with automatic fallback.
"""

import numpy as np
import torch
from PIL import Image
import cv2
from typing import Tuple, Optional, Dict, Any
import os
import sys
from termcolor import colored

# Add parent directory to path for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEPTH_MODEL, DEPTH_MODEL_FALLBACK, DEPTH_MAX_SIZE, DEPTH_SCALE


class DepthEstimator:
    """
    Estimates depth maps from single images using transformer-based models.

    Supports multiple depth estimation backends:
    - Depth Anything V2 (recommended): Best quality, modern architecture
    - DPT/MiDaS: Fallback option, widely compatible

    The depth maps are relative (not metric) but provide good geometric structure.
    """

    def __init__(self, model_name: str = DEPTH_MODEL, device: Optional[str] = None):
        """
        Initialize the depth estimator.

        Args:
            model_name: HuggingFace model identifier for depth estimation
            device: 'cuda', 'cpu', or None for auto-detection
        """
        print(
            colored(f"[DepthEstimator] Initializing with model: {model_name}", "cyan")
        )

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(colored(f"[DepthEstimator] Using device: {self.device}", "cyan"))

        self.model_name = model_name
        self.model = None
        self.processor = None
        self.model_type = None  # 'depth_anything_v2', 'dpt', or 'zoedepth'

        # Try to load the primary model, fall back if needed
        try:
            self._load_model(model_name)
        except Exception as e:
            print(
                colored(f"[DepthEstimator] Failed to load {model_name}: {e}", "yellow")
            )
            print(
                colored(
                    f"[DepthEstimator] Falling back to {DEPTH_MODEL_FALLBACK}", "yellow"
                )
            )
            try:
                self._load_model(DEPTH_MODEL_FALLBACK)
            except Exception as e2:
                print(
                    colored(
                        f"[DepthEstimator] ERROR: Could not load fallback model: {e2}",
                        "red",
                    )
                )
                raise RuntimeError(f"Failed to load any depth model: {e2}")

    def _load_model(self, model_name: str):
        """Load the depth estimation model based on model name."""
        print(
            colored(
                "[DepthEstimator] Loading model (this may take a moment on first run)...",
                "cyan",
            )
        )

        # Determine model type and load accordingly
        if "depth-anything" in model_name.lower():
            self._load_depth_anything_v2(model_name)
        elif "zoedepth" in model_name.lower():
            self._load_zoedepth(model_name)
        else:
            self._load_dpt(model_name)

        print(
            colored(
                f"[DepthEstimator] Model loaded successfully! Type: {self.model_type}",
                "green",
            )
        )

    def _load_depth_anything_v2(self, model_name: str):
        """Load Depth Anything V2 model."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "depth_anything_v2"

    def _load_dpt(self, model_name: str):
        """Load DPT/MiDaS model."""
        from transformers import DPTImageProcessor, DPTForDepthEstimation

        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "dpt"

    def _load_zoedepth(self, model_name: str):
        """Load ZoeDepth model for metric depth."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "zoedepth"

    def estimate_depth(
        self,
        image: np.ndarray,
        max_size: int = DEPTH_MAX_SIZE,
        return_raw: bool = False,
    ) -> np.ndarray:
        """
        Estimate depth from a single image.

        Args:
            image: RGB image as numpy array (H, W, 3)
            max_size: Maximum size for processing (for speed)
            return_raw: If True, return raw depth values (not normalized)

        Returns:
            Depth map as numpy array (H, W), normalized 0-1 unless return_raw=True
        """
        # Store original size
        original_h, original_w = image.shape[:2]

        # For Depth Anything V2, use optimal size (multiple of 14)
        if self.model_type == "depth_anything_v2":
            scale = min(max_size / original_h, max_size / original_w, 1.0)
            if scale < 1.0:
                new_h = int(original_h * scale)
                new_w = int(original_w * scale)
                # Round to nearest multiple of 14 for optimal performance
                new_h = max((new_h // 14) * 14, 14)
                new_w = max((new_w // 14) * 14, 14)
                image_resized = cv2.resize(
                    image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            else:
                image_resized = image
        else:
            # Standard resize for other models
            scale = min(max_size / original_h, max_size / original_w, 1.0)
            if scale < 1.0:
                new_h, new_w = int(original_h * scale), int(original_w * scale)
                image_resized = cv2.resize(
                    image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            else:
                image_resized = image

        # Convert to PIL Image
        pil_image = Image.fromarray(image_resized)

        # Prepare inputs for the model
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Post-process depth map
        depth = predicted_depth.squeeze().cpu().numpy()

        # Handle different output formats
        if len(depth.shape) == 3:
            depth = depth[0]  # Take first channel if multi-channel

        # Resize back to original image size
        depth = cv2.resize(
            depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )

        if return_raw:
            return depth

        # Normalize to 0-1 range with robust percentile-based normalization
        depth_min = np.percentile(depth, 2)
        depth_max = np.percentile(depth, 98)
        depth = np.clip(depth, depth_min, depth_max)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        return depth

    def estimate_depth_with_edges(
        self, image: np.ndarray, max_size: int = DEPTH_MAX_SIZE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth and extract depth edges for better wall detection.

        Args:
            image: RGB image as numpy array (H, W, 3)
            max_size: Maximum size for processing

        Returns:
            Tuple of (depth_map, depth_edges)
        """
        depth = self.estimate_depth(image, max_size)

        # Compute depth gradients (edges where depth changes rapidly = walls)
        sobel_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        depth_edges = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize edges
        edge_max = depth_edges.max()
        if edge_max > 0:
            depth_edges = depth_edges / edge_max

        return depth, depth_edges

    def process_multiple_images(self, images: list) -> list:
        """
        Process multiple images and return their depth maps.

        Args:
            images: List of RGB images as numpy arrays

        Returns:
            List of depth maps
        """
        depth_maps = []
        for i, img in enumerate(images):
            print(f"[DepthEstimator] Processing image {i+1}/{len(images)}...")
            depth = self.estimate_depth(img)
            depth_maps.append(depth)
        return depth_maps

    def depth_to_3d_points(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        fx: float = 500,
        fy: float = 500,
        sample_rate: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D point cloud using camera intrinsics.

        Args:
            image: RGB image (for colors)
            depth: Depth map (H, W)
            fx, fy: Focal lengths
            sample_rate: Sample every Nth pixel (for efficiency)

        Returns:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3) normalized 0-1
        """
        h, w = depth.shape
        cx, cy = w / 2, h / 2

        # Create pixel coordinates grid
        u = np.arange(0, w, sample_rate)
        v = np.arange(0, h, sample_rate)
        u, v = np.meshgrid(u, v)

        # Sample depth and colors
        depth_sampled = depth[::sample_rate, ::sample_rate]
        colors_sampled = image[::sample_rate, ::sample_rate] / 255.0

        # Filter out invalid depth (near 0 or 1, which are usually unreliable)
        valid_mask = (depth_sampled > 0.05) & (depth_sampled < 0.95)

        # Convert to 3D points using pinhole camera model
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth (inverted for proper orientation)
        # Inverse depth for better geometric separation, scaled to keep values stable
        z = 1.0 / (depth_sampled + 1e-3)
        z = z * DEPTH_SCALE
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack and filter
        points = np.stack([x, y, z], axis=-1)
        points = points[valid_mask]
        colors = colors_sampled[valid_mask]

        return points, colors


# Quick test
if __name__ == "__main__":
    print("Testing Depth Estimator...")

    # Create a test image (gradient for visualization)
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    test_image[:, :, 0] = np.linspace(0, 255, 256).astype(np.uint8)  # R gradient
    test_image[:, :, 1] = 128
    test_image[:, :, 2] = (
        np.linspace(255, 0, 256).astype(np.uint8).reshape(1, -1).T
    )  # B gradient

    estimator = DepthEstimator()
    depth = estimator.estimate_depth(test_image)
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: {depth.min():.3f} to {depth.max():.3f}")
    print("Test passed!")
