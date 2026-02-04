"""
Structure-from-Motion (SfM) Processor Module

Uses COLMAP (via pycolmap) to perform proper photogrammetric reconstruction:
1. Feature extraction (SIFT/SuperPoint)
2. Feature matching
3. Incremental SfM mapping
4. Camera pose estimation

This is the CORRECT approach for multi-view 3D reconstruction, as opposed to
simply stacking monocular depth maps which don't have geometric consistency.
"""

import numpy as np
import cv2
import os
import sys
import shutil
import tempfile
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COLMAP_WORKSPACE,
    SFM_MIN_IMAGES,
    SFM_FEATURE_TYPE,
    SFM_MATCHER_TYPE,
    SFM_MAX_IMAGE_SIZE,
    ENABLE_SFM,
)

# Check if pycolmap is available
PYCOLMAP_AVAILABLE = False
try:
    import pycolmap

    PYCOLMAP_AVAILABLE = True
except ImportError:
    print(
        colored(
            "[SfMProcessor] Warning: pycolmap not installed. SfM will be disabled.",
            "yellow",
        )
    )
    print(colored("[SfMProcessor] Install with: pip install pycolmap", "yellow"))


class SfMProcessor:
    """
    Structure-from-Motion processor using COLMAP.

    Provides proper photogrammetric reconstruction by:
    1. Extracting features from multiple images
    2. Matching features across image pairs
    3. Solving for camera poses (extrinsics)
    4. Triangulating a sparse 3D point cloud

    The camera poses can then be used to properly align depth maps from
    multiple views into a consistent 3D reconstruction.
    """

    def __init__(self, workspace_path: str = COLMAP_WORKSPACE):
        """
        Initialize the SfM processor.

        Args:
            workspace_path: Path to COLMAP workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        self.database_path = self.workspace_path / "database.db"
        self.image_path = self.workspace_path / "images"
        self.sparse_path = self.workspace_path / "sparse"

        self.reconstruction = None
        self.camera_poses = {}
        self.sparse_points = None
        self.camera_intrinsics = None

        self.enabled = ENABLE_SFM and PYCOLMAP_AVAILABLE

        if self.enabled:
            print(colored("[SfMProcessor] Initialized with COLMAP support", "green"))
        else:
            print(colored("[SfMProcessor] Running in fallback mode (no SfM)", "yellow"))

    def _prepare_workspace(
        self, images: List[np.ndarray], image_paths: Optional[List[str]] = None
    ) -> List[str]:
        """
        Prepare the COLMAP workspace with images.

        Args:
            images: List of RGB images as numpy arrays
            image_paths: Optional original image paths (for metadata)

        Returns:
            List of image paths in the workspace
        """
        # Clean previous workspace
        if self.image_path.exists():
            shutil.rmtree(self.image_path)
        self.image_path.mkdir(parents=True, exist_ok=True)

        if self.sparse_path.exists():
            shutil.rmtree(self.sparse_path)
        self.sparse_path.mkdir(parents=True, exist_ok=True)

        if self.database_path.exists():
            self.database_path.unlink()

        workspace_images = []

        for i, img in enumerate(images):
            # Resize if too large
            h, w = img.shape[:2]
            scale = min(SFM_MAX_IMAGE_SIZE / max(h, w), 1.0)
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Save to workspace
            img_name = f"image_{i:04d}.jpg"
            img_path = self.image_path / img_name

            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            workspace_images.append(str(img_path))

        print(
            colored(
                f"[SfMProcessor] Prepared {len(workspace_images)} images in workspace",
                "cyan",
            )
        )
        return workspace_images

    def run_sfm(
        self, images: List[np.ndarray], progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run the full Structure-from-Motion pipeline.

        Args:
            images: List of RGB images as numpy arrays
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing:
            - success: bool
            - camera_poses: Dict mapping image index to 4x4 transform
            - sparse_points: np.ndarray of 3D points
            - camera_intrinsics: Camera parameters
            - num_registered: Number of successfully registered images
        """
        if not self.enabled:
            return self._fallback_result(len(images))

        if len(images) < SFM_MIN_IMAGES:
            print(
                colored(
                    f"[SfMProcessor] Need at least {SFM_MIN_IMAGES} images for SfM",
                    "yellow",
                )
            )
            return self._fallback_result(len(images))

        try:
            # Prepare workspace
            if progress_callback:
                progress_callback(0.1, "Preparing images...")

            workspace_images = self._prepare_workspace(images)

            # Feature extraction
            if progress_callback:
                progress_callback(0.2, "Extracting features...")

            print(colored("[SfMProcessor] Running feature extraction...", "cyan"))
            self._extract_features()

            # Feature matching
            if progress_callback:
                progress_callback(0.4, "Matching features...")

            print(colored("[SfMProcessor] Running feature matching...", "cyan"))
            self._match_features()

            # Incremental mapping (SfM)
            if progress_callback:
                progress_callback(0.6, "Running Structure-from-Motion...")

            print(colored("[SfMProcessor] Running incremental SfM...", "cyan"))
            self._run_mapping()

            # Extract results
            if progress_callback:
                progress_callback(0.9, "Extracting camera poses...")

            result = self._extract_results()

            if progress_callback:
                progress_callback(1.0, "SfM complete!")

            return result

        except Exception as e:
            print(colored(f"[SfMProcessor] SfM failed: {e}", "red"))
            import traceback

            traceback.print_exc()
            return self._fallback_result(len(images))

    def _extract_features(self):
        """Extract SIFT features from all images."""
        # Configure extraction options (pycolmap 3.13+ API)
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.max_image_size = SFM_MAX_IMAGE_SIZE

        # Configure SIFT options
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = 8192
        sift_options.first_octave = -1  # Include lower octaves for more features
        extraction_options.sift = sift_options

        pycolmap.extract_features(
            database_path=str(self.database_path),
            image_path=str(self.image_path),
            camera_mode=pycolmap.CameraMode.AUTO,
            extraction_options=extraction_options,
        )

    def _match_features(self):
        """Match features between image pairs."""
        if SFM_MATCHER_TYPE == "exhaustive":
            pycolmap.match_exhaustive(
                database_path=str(self.database_path),
            )
        elif SFM_MATCHER_TYPE == "sequential":
            pycolmap.match_sequential(
                database_path=str(self.database_path),
            )
        else:
            # Default to exhaustive for small sets
            pycolmap.match_exhaustive(
                database_path=str(self.database_path),
            )

    def _run_mapping(self):
        """Run incremental SfM mapping."""
        # Configure pipeline options (pycolmap 3.13+ API)
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_num_matches = 15
        mapper_options.init_num_trials = 200
        mapper_options.num_threads = -1  # Use all available threads

        # Run incremental mapping
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(self.database_path),
            image_path=str(self.image_path),
            output_path=str(self.sparse_path),
            options=mapper_options,
        )

        if reconstructions:
            # Use the largest reconstruction
            self.reconstruction = max(
                reconstructions.values(), key=lambda r: r.num_reg_images()
            )
            print(
                colored(
                    f"[SfMProcessor] Registered {self.reconstruction.num_reg_images()} images",
                    "green",
                )
            )
        else:
            self.reconstruction = None
            print(colored("[SfMProcessor] No reconstruction produced", "yellow"))

    def _extract_results(self) -> Dict[str, Any]:
        """Extract camera poses and sparse points from reconstruction."""
        if self.reconstruction is None:
            return self._fallback_result(0)

        camera_poses = {}
        camera_intrinsics = {}

        # Extract camera poses (extrinsics)
        for image_id, image in self.reconstruction.images.items():
            # Get image name to determine index
            img_name = image.name
            try:
                idx = int(img_name.split("_")[1].split(".")[0])
            except:
                idx = image_id - 1

            # Get rotation and translation (cam_from_world is a method in pycolmap)
            cam_from_world = image.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation

            # Create 4x4 transformation matrix (world to camera)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            # Invert to get camera to world (camera pose in world)
            T_inv = np.linalg.inv(T)

            camera_poses[idx] = {
                "transform": T_inv,
                "R": R,
                "t": t,
                "image_id": image_id,
                "image_name": img_name,
            }

            # Get camera intrinsics
            camera = self.reconstruction.cameras[image.camera_id]
            camera_intrinsics[idx] = {
                "fx": camera.focal_length_x,
                "fy": camera.focal_length_y,
                "cx": camera.principal_point_x,
                "cy": camera.principal_point_y,
                "width": camera.width,
                "height": camera.height,
                "model": camera.model_name,
            }

        # Extract sparse 3D points
        sparse_points = []
        sparse_colors = []

        for point_id, point in self.reconstruction.points3D.items():
            sparse_points.append(point.xyz)
            sparse_colors.append(point.color / 255.0)

        sparse_points = np.array(sparse_points) if sparse_points else np.array([])
        sparse_colors = np.array(sparse_colors) if sparse_colors else np.array([])

        self.camera_poses = camera_poses
        self.sparse_points = sparse_points
        self.camera_intrinsics = camera_intrinsics

        print(
            colored(
                f"[SfMProcessor] Extracted {len(camera_poses)} camera poses", "green"
            )
        )
        print(
            colored(
                f"[SfMProcessor] Sparse point cloud: {len(sparse_points)} points",
                "green",
            )
        )

        return {
            "success": True,
            "camera_poses": camera_poses,
            "sparse_points": sparse_points,
            "sparse_colors": sparse_colors,
            "camera_intrinsics": camera_intrinsics,
            "num_registered": len(camera_poses),
        }

    def _fallback_result(self, num_images: int) -> Dict[str, Any]:
        """Return a fallback result when SfM is not available or fails."""
        # Create identity poses for each image (no alignment)
        camera_poses = {}
        for i in range(num_images):
            camera_poses[i] = {
                "transform": np.eye(4),
                "R": np.eye(3),
                "t": np.zeros(3),
                "image_id": i,
                "image_name": f"image_{i:04d}.jpg",
            }

        return {
            "success": False,
            "camera_poses": camera_poses,
            "sparse_points": np.array([]),
            "sparse_colors": np.array([]),
            "camera_intrinsics": {},
            "num_registered": 0,
            "fallback": True,
        }

    def get_camera_pose(self, image_idx: int) -> Optional[np.ndarray]:
        """Get the 4x4 camera pose matrix for a specific image."""
        if image_idx in self.camera_poses:
            return self.camera_poses[image_idx]["transform"]
        return None

    def get_intrinsics(self, image_idx: int) -> Optional[Dict]:
        """Get camera intrinsics for a specific image."""
        if self.camera_intrinsics and image_idx in self.camera_intrinsics:
            return self.camera_intrinsics[image_idx]
        return None

    def transform_points(self, points: np.ndarray, image_idx: int) -> np.ndarray:
        """
        Transform points from camera space to world space using SfM pose.

        Args:
            points: Nx3 array of 3D points in camera space
            image_idx: Index of the image these points came from

        Returns:
            Nx3 array of 3D points in world space
        """
        if len(points) == 0:
            return points

        pose = self.get_camera_pose(image_idx)
        if pose is None:
            return points

        # Convert to homogeneous coordinates
        points_h = np.hstack([points, np.ones((len(points), 1))])

        # Transform
        points_world = (pose @ points_h.T).T

        return points_world[:, :3]


# Quick test
if __name__ == "__main__":
    print("Testing SfM Processor...")

    if not PYCOLMAP_AVAILABLE:
        print("pycolmap not available - skipping test")
    else:
        # Create dummy test images (checkerboard patterns)
        images = []
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some variation
            cv2.rectangle(
                img, (100 + i * 20, 100), (400 + i * 20, 300), (255, 255, 255), -1
            )
            cv2.circle(img, (320, 240), 50 + i * 10, (128, 128, 128), -1)
            images.append(img)

        processor = SfMProcessor()
        result = processor.run_sfm(images)

        print(f"SfM Success: {result['success']}")
        print(f"Registered images: {result['num_registered']}")
        print("Test complete!")
