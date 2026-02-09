"""
Metric Depth Estimation Module

Produces depth maps in meters using metric depth models:
- Primary: Apple Depth Pro (also estimates focal length, no intrinsics needed)
- Fallback: Depth Anything V2 Metric Indoor (lighter, great indoor accuracy)

Unlike the relative depth estimator, output values are absolute distances
in meters, eliminating the need for assumed_room_width scaling.
"""

import numpy as np
import torch
from PIL import Image
import cv2
from typing import Tuple, Optional
from termcolor import colored

# Default model identifiers
DEPTH_PRO_MODEL = "apple/DepthPro-hf"
DA_V2_METRIC_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DA_V2_METRIC_SMALL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class MetricDepthEstimator:
    """
    Estimates metric depth (in meters) from single images.

    Primary: Apple Depth Pro — 1B params, estimates focal length + metric depth.
    Fallback: Depth Anything V2 Metric Indoor — 335M params (Large) or 25M (Small).

    Output depth maps have values in meters, not normalized 0-1.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(colored(f"[MetricDepth] Using device: {self.device}", "cyan"))

        self.model = None
        self.processor = None
        self.model_type = None  # "depth_pro" or "da_v2_metric"
        self.estimated_focal_length = None  # Set per-image by Depth Pro

        # Try models in priority order
        for model_id, loader in [
            (DEPTH_PRO_MODEL, self._load_depth_pro),
            (DA_V2_METRIC_MODEL, self._load_da_v2_metric),
            (DA_V2_METRIC_SMALL, self._load_da_v2_metric),
        ]:
            try:
                print(colored(f"[MetricDepth] Loading {model_id}...", "cyan"))
                loader(model_id)
                print(
                    colored(
                        f"[MetricDepth] Loaded {self.model_type} successfully", "green"
                    )
                )
                return
            except Exception as e:
                print(
                    colored(f"[MetricDepth] Failed to load {model_id}: {e}", "yellow")
                )

        raise RuntimeError("Failed to load any metric depth model")

    def _load_depth_pro(self, model_id: str):
        """Load Apple Depth Pro via HuggingFace Transformers."""
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

        self.processor = DepthProImageProcessorFast.from_pretrained(model_id)
        self.model = DepthProForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "depth_pro"

    def _load_da_v2_metric(self, model_id: str):
        """Load Depth Anything V2 Metric Indoor via HuggingFace Transformers."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "da_v2_metric"

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate metric depth from a single image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            Depth map in meters as numpy array (H, W). Values represent
            distance from camera in meters.
        """
        original_h, original_w = image.shape[:2]
        pil_image = Image.fromarray(image)

        if self.model_type == "depth_pro":
            return self._estimate_depth_pro(pil_image, original_h, original_w)
        else:
            return self._estimate_da_v2(pil_image, original_h, original_w)

    def _estimate_depth_pro(self, pil_image: Image.Image, h: int, w: int) -> np.ndarray:
        """Depth Pro inference — returns metric depth + sets focal length."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        post = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)]
        )

        depth = post[0]["predicted_depth"].cpu().numpy()

        # Store focal length estimate for 3D projection
        if "focal_length" in post[0]:
            self.estimated_focal_length = float(post[0]["focal_length"])
            print(
                colored(
                    f"[MetricDepth] Estimated focal length: {self.estimated_focal_length:.1f}px",
                    "cyan",
                )
            )

        return depth

    def _estimate_da_v2(self, pil_image: Image.Image, h: int, w: int) -> np.ndarray:
        """Depth Anything V2 Metric inference — returns metric depth."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

        # Resize to original dimensions
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        self.estimated_focal_length = None  # DA V2 does not estimate focal length
        return depth.cpu().numpy()

    def depth_to_3d_points_metric(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        sample_rate: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert metric depth map to 3D point cloud.

        Uses metric depth values directly as Z (no inverse-depth hack).
        If fx/fy not provided, uses Depth Pro's estimated focal length
        or falls back to a heuristic based on image width.

        Args:
            image: RGB image (H, W, 3), uint8
            depth: Metric depth map (H, W) in meters
            fx: Focal length X in pixels (optional)
            fy: Focal length Y in pixels (optional)
            sample_rate: Sample every Nth pixel

        Returns:
            points: (N, 3) array of 3D points in meters
            colors: (N, 3) array of RGB colors normalized 0-1
        """
        h, w = depth.shape

        # Determine focal length
        if fx is None:
            if self.estimated_focal_length is not None:
                fx = self.estimated_focal_length
            else:
                # Heuristic: assume ~60 degree FOV for typical smartphone
                fx = w / (2.0 * np.tan(np.radians(30)))
        if fy is None:
            fy = fx  # Square pixels

        cx, cy = w / 2.0, h / 2.0

        # Create pixel coordinate grids (subsampled)
        u = np.arange(0, w, sample_rate)
        v = np.arange(0, h, sample_rate)
        u, v = np.meshgrid(u, v)

        # Sample depth and colors
        depth_sampled = depth[::sample_rate, ::sample_rate]
        colors_sampled = image[::sample_rate, ::sample_rate] / 255.0

        # Filter out invalid depths (too close or too far)
        valid_mask = (
            (depth_sampled > 0.1) & (depth_sampled < 20.0) & np.isfinite(depth_sampled)
        )

        # Pinhole back-projection with METRIC depth (no inverse hack)
        z = depth_sampled
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1)
        points = points[valid_mask]
        colors = colors_sampled[valid_mask]

        return points, colors

    def get_focal_length(self, image_width: int) -> float:
        """
        Return the best available focal length estimate.

        Priority:
        1. Depth Pro's per-image estimate (if available)
        2. Heuristic based on ~60 degree FOV
        """
        if self.estimated_focal_length is not None:
            return self.estimated_focal_length
        return image_width / (2.0 * np.tan(np.radians(30)))
