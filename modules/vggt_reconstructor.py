"""
VGGT-based room reconstruction.
Replaces: SfMProcessor + DepthEstimator + ICP/RANSAC registration
in a single forward pass.

VGGT (Visual Geometry Grounded Transformer) — CVPR 2025 Best Paper
Produces: metric depth, camera poses, focal lengths, aligned point cloud.
"""

import numpy as np
from termcolor import colored
from config import VGGT_CONFIDENCE_THRESHOLD, VGGT_MAX_SIZE


class VGGTReconstructor:
    """Single-model 3D reconstruction using VGGT."""

    def __init__(self, device: str = "auto"):
        self.model = None
        self.device = self._select_device(device)

    def _select_device(self, device: str) -> str:
        if device != "auto":
            return device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Lazy-load the VGGT-1B model."""
        if self.model is not None:
            return
        import torch
        from vggt.models.vggt import VGGT

        print(colored("[VGGT] Loading VGGT-1B model...", "cyan"))
        self.model = VGGT.from_pretrained("facebook/VGGT-1B")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = self.model.to(self.device).to(dtype).eval()
        print(colored(f"[VGGT] Model loaded on {self.device} ({dtype})", "green"))

    def _preprocess(self, images: list[np.ndarray]):
        """
        Convert numpy images to VGGT input tensor.
        - Resize so max dim = VGGT_MAX_SIZE (518), dimensions divisible by 14
        - Normalize to [0, 1]
        - Pad to same size across images
        - Return [S, 3, H, W] tensor and list of (orig_h, orig_w) tuples
        """
        import torch
        import cv2

        target_size = VGGT_MAX_SIZE  # 518
        processed = []
        orig_sizes = []

        for img in images:
            h, w = img.shape[:2]
            orig_sizes.append((h, w))

            # Resize so max dim = target_size
            scale = target_size / max(h, w)
            new_h = round(h * scale / 14) * 14  # divisible by 14
            new_w = round(w * scale / 14) * 14
            new_h = max(new_h, 14)
            new_w = max(new_w, 14)

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            tensor = (
                torch.from_numpy(resized).float().div(255.0).permute(2, 0, 1)
            )  # [3, H, W]
            processed.append(tensor)

        # Pad all to same dimensions
        max_h = max(t.shape[1] for t in processed)
        max_w = max(t.shape[2] for t in processed)

        padded = []
        for t in processed:
            pad_h = max_h - t.shape[1]
            pad_w = max_w - t.shape[2]
            if pad_h > 0 or pad_w > 0:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=0.0)
            padded.append(t)

        return torch.stack(padded), orig_sizes  # [S, 3, H, W]

    def reconstruct(self, images: list[np.ndarray]) -> dict:
        """
        Single-pass reconstruction from RGB images.

        Args:
            images: List of RGB numpy arrays (H, W, 3) uint8.

        Returns:
            dict with keys:
                success: bool
                point_cloud: (N, 3) numpy array in world coordinates
                point_colors: (N, 3) numpy array [0, 1]
                depth_maps: list of 2D numpy arrays (metric depth in meters)
                camera_poses: dict mapping image_idx -> {"transform": 4x4 numpy}
                camera_intrinsics: dict mapping image_idx -> {"fx", "fy", "cx", "cy", "width", "height"}
                focal_lengths: list of floats
                num_registered: int
        """
        import torch

        self._load_model()

        n_images = len(images)
        print(colored(f"[VGGT] Processing {n_images} images...", "cyan"))

        input_tensor, orig_sizes = self._preprocess(images)
        _, _, proc_h, proc_w = input_tensor.shape  # processed (VGGT) dims

        dtype = next(self.model.parameters()).dtype
        input_tensor = input_tensor.to(self.device).to(dtype)

        # Forward pass
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    predictions = self.model(input_tensor)
                # Cast all prediction tensors back to float32 for downstream ops
                predictions = {
                    k: v.float() if isinstance(v, torch.Tensor) else v
                    for k, v in predictions.items()
                }
            else:
                predictions = self.model(input_tensor)

        # Extract outputs (all have batch dim [1, S, ...])
        # Decode camera poses from pose encoding
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        pose_enc = predictions["pose_enc"].float()  # [1, S, 9]
        extrinsics, intrinsics_mat = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(proc_h, proc_w)
        )
        # extrinsics: [1, S, 3, 4] (cam-from-world)
        # intrinsics_mat: [1, S, 3, 3]

        extrinsics_np = extrinsics[0].cpu().numpy()  # [S, 3, 4]
        intrinsics_np = intrinsics_mat[0].cpu().numpy()  # [S, 3, 3]

        # Convert extrinsics (3x4 cam-from-world) to 4x4 world-from-camera
        from vggt.utils.geometry import closed_form_inverse_se3

        # Pad to 4x4 for inversion
        ext_4x4 = np.zeros((n_images, 4, 4))
        ext_4x4[:, :3, :] = extrinsics_np
        ext_4x4[:, 3, 3] = 1.0
        world_from_cam = closed_form_inverse_se3(ext_4x4)  # [S, 4, 4]

        # Build camera_poses dict (matching SfM output format)
        camera_poses = {}
        camera_intrinsics = {}
        focal_lengths = []
        for i in range(n_images):
            camera_poses[i] = {"transform": world_from_cam[i]}
            fx = float(intrinsics_np[i, 0, 0])
            fy = float(intrinsics_np[i, 1, 1])
            cx = float(intrinsics_np[i, 0, 2])
            cy = float(intrinsics_np[i, 1, 2])
            camera_intrinsics[i] = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": proc_w,
                "height": proc_h,
                "orig_width": orig_sizes[i][1],
                "orig_height": orig_sizes[i][0],
            }
            focal_lengths.append(fx)

        # Extract depth maps (metric)
        depth_raw = predictions["depth"][0].float().cpu().numpy()  # [S, H, W, 1]
        depth_conf_raw = predictions["depth_conf"][0].float().cpu().numpy()  # [S, H, W]
        depth_maps = []
        for i in range(n_images):
            depth_maps.append(depth_raw[i, :, :, 0])  # [H, W]

        # Extract world points and colors
        world_points = (
            predictions["world_points"][0].float().cpu().numpy()
        )  # [S, H, W, 3]
        world_points_conf = (
            predictions["world_points_conf"][0].float().cpu().numpy()
        )  # [S, H, W]

        all_points = []
        all_colors = []
        conf_threshold = VGGT_CONFIDENCE_THRESHOLD

        for i in range(n_images):
            pts = world_points[i].reshape(-1, 3)  # [H*W, 3]
            conf = world_points_conf[i].reshape(-1)  # [H*W]

            # Get colors from resized images
            import cv2

            h_proc, w_proc = world_points[i].shape[:2]
            colors_img = cv2.resize(
                images[i], (w_proc, h_proc), interpolation=cv2.INTER_AREA
            )
            colors = colors_img.reshape(-1, 3).astype(np.float64) / 255.0

            # Filter by confidence
            mask = conf > conf_threshold
            # Filter by valid coordinates (no NaN/Inf, reasonable range)
            valid = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 100.0).all(axis=1)
            mask = mask & valid

            all_points.append(pts[mask])
            all_colors.append(colors[mask])

        point_cloud = np.concatenate(all_points) if all_points else np.zeros((0, 3))
        point_colors = np.concatenate(all_colors) if all_colors else np.zeros((0, 3))

        print(
            colored(
                f"[VGGT] Reconstruction complete: {len(point_cloud):,} points, "
                f"{n_images} cameras, focal lengths: "
                f"[{', '.join(f'{fl:.0f}' for fl in focal_lengths)}]",
                "green",
            )
        )

        return {
            "success": len(point_cloud) > 0,
            "point_cloud": point_cloud,
            "point_colors": point_colors,
            "depth_maps": depth_maps,
            "camera_poses": camera_poses,
            "camera_intrinsics": camera_intrinsics,
            "focal_lengths": focal_lengths,
            "num_registered": n_images,
        }
