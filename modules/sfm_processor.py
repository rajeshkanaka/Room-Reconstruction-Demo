"""
Structure-from-Motion (SfM) processor using pycolmap/COLMAP.

This module now returns richer QA metadata (registration ratio, reprojection
errors) and attempts dense MVS reconstruction when available.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COLMAP_DENSE,
    COLMAP_WORKSPACE,
    ENABLE_MVS,
    ENABLE_SFM,
    MVS_MAX_IMAGE_SIZE,
    SFM_MATCHER_TYPE,
    SFM_MATCH_CROSS_CHECK,
    SFM_MATCH_GUIDED,
    SFM_MATCH_MAX_DISTANCE,
    SFM_MATCH_MAX_RATIO,
    SFM_MAX_IMAGE_SIZE,
    SFM_MIN_MODEL_SIZE,
    SFM_MIN_NUM_MATCHES,
    SFM_MIN_IMAGES,
    SFM_ALLOW_TWO_VIEW_TRACKS,
    SFM_ABS_POSE_MIN_NUM_INLIERS,
    SFM_ABS_POSE_MIN_INLIER_RATIO,
    SFM_FILTER_MIN_TRI_ANGLE,
    SFM_INIT_MAX_ERROR,
    SFM_INIT_MAX_FORWARD_MOTION,
    SFM_INIT_MIN_NUM_INLIERS,
    SFM_INIT_MIN_TRI_ANGLE,
    SFM_INIT_NUM_TRIALS,
    SFM_VERIFY_MIN_EF_INLIER_RATIO,
    SFM_VERIFY_MIN_INLIERS,
    SFM_VERIFY_RANSAC_MAX_ERROR,
)

# Check pycolmap availability
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

# Optional dense point cloud loading
OPEN3D_AVAILABLE = False
try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class SfMProcessor:
    """COLMAP SfM wrapper with optional dense MVS output."""

    def __init__(self, workspace_path: str = COLMAP_WORKSPACE):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        self.database_path = self.workspace_path / "database.db"
        self.image_path = self.workspace_path / "images"
        self.sparse_path = self.workspace_path / "sparse"
        self.dense_path = Path(COLMAP_DENSE)

        self.reconstruction = None
        self.camera_poses: Dict[int, Dict] = {}
        self.camera_intrinsics: Dict[int, Dict] = {}
        self.sparse_points = np.array([])
        self.sparse_colors = np.array([])
        self.dense_points = np.array([])
        self.dense_colors = np.array([])

        self.enabled = ENABLE_SFM and PYCOLMAP_AVAILABLE

        if self.enabled:
            print(colored("[SfMProcessor] Initialized with COLMAP support", "green"))
        else:
            print(colored("[SfMProcessor] Running in fallback mode (no SfM)", "yellow"))

    def _prepare_workspace(self, images: List[np.ndarray]) -> List[str]:
        """Prepare clean COLMAP workspace image folder."""
        if self.image_path.exists():
            shutil.rmtree(self.image_path)
        self.image_path.mkdir(parents=True, exist_ok=True)

        if self.sparse_path.exists():
            shutil.rmtree(self.sparse_path)
        self.sparse_path.mkdir(parents=True, exist_ok=True)

        if self.dense_path.exists():
            shutil.rmtree(self.dense_path)
        self.dense_path.mkdir(parents=True, exist_ok=True)

        if self.database_path.exists():
            self.database_path.unlink()

        workspace_images = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            scale = min(SFM_MAX_IMAGE_SIZE / max(h, w), 1.0)
            if scale < 1.0:
                img = cv2.resize(
                    img,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            path = self.image_path / f"image_{i:04d}.jpg"
            cv2.imwrite(
                str(path),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            workspace_images.append(str(path))

        print(
            colored(
                f"[SfMProcessor] Prepared {len(workspace_images)} images in workspace",
                "cyan",
            )
        )
        return workspace_images

    def run_sfm(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Run SfM and return sparse + optional dense reconstruction results."""
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
            if progress_callback:
                progress_callback(0.1, "Preparing SfM workspace...")
            self._prepare_workspace(images)

            if progress_callback:
                progress_callback(0.25, "Extracting features...")
            self._extract_features()

            if progress_callback:
                progress_callback(0.45, "Matching features...")
            self._match_features()

            if progress_callback:
                progress_callback(0.65, "Running incremental SfM...")
            self._run_mapping()

            if progress_callback:
                progress_callback(0.82, "Extracting SfM outputs...")
            result = self._extract_results(total_images=len(images))

            if ENABLE_MVS and result.get("success"):
                if progress_callback:
                    progress_callback(0.92, "Running dense MVS...")
                dense_points, dense_colors = self._run_dense_mvs()
                result["dense_points"] = dense_points
                result["dense_colors"] = dense_colors
            else:
                result["dense_points"] = np.array([])
                result["dense_colors"] = np.array([])

            if progress_callback:
                progress_callback(1.0, "SfM complete")

            return result

        except Exception as exc:
            print(colored(f"[SfMProcessor] SfM failed: {exc}", "red"))
            import traceback

            traceback.print_exc()
            return self._fallback_result(len(images))

    def _extract_features(self) -> None:
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.max_image_size = SFM_MAX_IMAGE_SIZE

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = 8192
        sift_options.first_octave = -1
        extraction_options.sift = sift_options

        pycolmap.extract_features(
            database_path=str(self.database_path),
            image_path=str(self.image_path),
            camera_mode=pycolmap.CameraMode.AUTO,
            extraction_options=extraction_options,
        )

    def _match_features(self) -> None:
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.guided_matching = SFM_MATCH_GUIDED
        matching_options.max_num_matches = max(int(matching_options.max_num_matches), 65536)
        matching_options.sift.cross_check = bool(SFM_MATCH_CROSS_CHECK)
        matching_options.sift.max_ratio = float(SFM_MATCH_MAX_RATIO)
        matching_options.sift.max_distance = float(SFM_MATCH_MAX_DISTANCE)

        verification_options = pycolmap.TwoViewGeometryOptions()
        verification_options.min_num_inliers = int(SFM_VERIFY_MIN_INLIERS)
        verification_options.min_E_F_inlier_ratio = float(SFM_VERIFY_MIN_EF_INLIER_RATIO)
        verification_options.ransac.max_error = float(SFM_VERIFY_RANSAC_MAX_ERROR)
        verification_options.ransac.max_num_trials = max(
            int(verification_options.ransac.max_num_trials),
            20000,
        )

        if SFM_MATCHER_TYPE == "sequential":
            pycolmap.match_sequential(
                database_path=str(self.database_path),
                matching_options=matching_options,
                verification_options=verification_options,
            )
        else:
            pycolmap.match_exhaustive(
                database_path=str(self.database_path),
                matching_options=matching_options,
                verification_options=verification_options,
            )

    def _run_mapping(self) -> None:
        options = pycolmap.IncrementalPipelineOptions()
        options.min_num_matches = int(SFM_MIN_NUM_MATCHES)
        options.init_num_trials = int(SFM_INIT_NUM_TRIALS)
        options.num_threads = -1
        options.min_model_size = int(SFM_MIN_MODEL_SIZE)

        mapper = options.mapper
        mapper.init_min_num_inliers = int(SFM_INIT_MIN_NUM_INLIERS)
        mapper.init_min_tri_angle = float(SFM_INIT_MIN_TRI_ANGLE)
        mapper.init_max_error = float(SFM_INIT_MAX_ERROR)
        mapper.init_max_forward_motion = float(SFM_INIT_MAX_FORWARD_MOTION)
        mapper.abs_pose_min_num_inliers = int(SFM_ABS_POSE_MIN_NUM_INLIERS)
        mapper.abs_pose_min_inlier_ratio = float(SFM_ABS_POSE_MIN_INLIER_RATIO)
        mapper.filter_min_tri_angle = float(SFM_FILTER_MIN_TRI_ANGLE)

        triangulation = options.triangulation
        triangulation.ignore_two_view_tracks = not bool(SFM_ALLOW_TWO_VIEW_TRACKS)
        triangulation.min_angle = float(SFM_FILTER_MIN_TRI_ANGLE)

        reconstructions = pycolmap.incremental_mapping(
            database_path=str(self.database_path),
            image_path=str(self.image_path),
            output_path=str(self.sparse_path),
            options=options,
        )

        if reconstructions:
            self.reconstruction = max(
                reconstructions.values(),
                key=lambda r: r.num_reg_images(),
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

    def _safe_reprojection_error(self, image_obj: Any) -> Optional[float]:
        """Extract image reprojection error across pycolmap versions."""
        for attr in (
            "mean_reprojection_error",
            "mean_observation_error",
            "avg_reprojection_error",
        ):
            value = getattr(image_obj, attr, None)
            try:
                if callable(value):
                    val = float(value())
                    if np.isfinite(val):
                        return val
            except Exception:
                pass
            if isinstance(value, (int, float)):
                val = float(value)
                if np.isfinite(val):
                    return val
        return None

    def _extract_results(self, total_images: int) -> Dict[str, Any]:
        """Extract sparse SfM outputs and QA stats."""
        if self.reconstruction is None:
            return self._fallback_result(total_images)

        camera_poses: Dict[int, Dict] = {}
        camera_intrinsics: Dict[int, Dict] = {}
        reprojection_errors: List[float] = []

        for image_id, image in self.reconstruction.images.items():
            img_name = image.name
            try:
                idx = int(img_name.split("_")[1].split(".")[0])
            except Exception:
                idx = image_id - 1

            cam_from_world = image.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            T_inv = np.linalg.inv(T)
            camera_poses[idx] = {
                "transform": T_inv,
                "R": R,
                "t": t,
                "image_id": image_id,
                "image_name": img_name,
            }

            camera = self.reconstruction.cameras[image.camera_id]
            camera_model = getattr(camera, "model_name", None)
            if camera_model is None:
                raw_model = getattr(camera, "model", None)
                camera_model = getattr(raw_model, "name", str(raw_model))
            camera_intrinsics[idx] = {
                "fx": camera.focal_length_x,
                "fy": camera.focal_length_y,
                "cx": camera.principal_point_x,
                "cy": camera.principal_point_y,
                "width": camera.width,
                "height": camera.height,
                "model": camera_model,
            }

            err = self._safe_reprojection_error(image)
            if err is not None:
                reprojection_errors.append(err)

        sparse_points = []
        sparse_colors = []
        for point in self.reconstruction.points3D.values():
            sparse_points.append(point.xyz)
            sparse_colors.append(point.color / 255.0)

        self.camera_poses = camera_poses
        self.camera_intrinsics = camera_intrinsics
        self.sparse_points = np.array(sparse_points) if sparse_points else np.array([])
        self.sparse_colors = np.array(sparse_colors) if sparse_colors else np.array([])

        num_registered = len(camera_poses)
        registration_ratio = (num_registered / total_images) if total_images else 0.0

        stats = {
            "num_registered": num_registered,
            "num_input_images": total_images,
            "registration_ratio": registration_ratio,
            "mean_reprojection_error_px": float(np.mean(reprojection_errors))
            if reprojection_errors
            else 0.0,
            "median_reprojection_error_px": float(np.median(reprojection_errors))
            if reprojection_errors
            else 0.0,
        }

        print(
            colored(
                f"[SfMProcessor] Sparse point cloud: {len(self.sparse_points)} points",
                "green",
            )
        )

        return {
            "success": True,
            "camera_poses": camera_poses,
            "sparse_points": self.sparse_points,
            "sparse_colors": self.sparse_colors,
            "camera_intrinsics": camera_intrinsics,
            "num_registered": num_registered,
            "reprojection_errors": reprojection_errors,
            "stats": stats,
        }

    def _resolve_sparse_model_path(self) -> Path:
        """Resolve sparse model folder path for pycolmap API variants."""
        if not self.sparse_path.exists():
            return self.sparse_path

        for child in sorted(self.sparse_path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "cameras.bin").exists() or (child / "cameras.txt").exists():
                return child

        return self.sparse_path

    def _run_undistort_images(self, sparse_model_path: Path) -> None:
        """Run image undistortion across pycolmap API versions."""
        if not hasattr(pycolmap, "undistort_images"):
            return

        # Newer pycolmap APIs accept a reconstruction object directly.
        try:
            pycolmap.undistort_images(
                reconstruction=self.reconstruction,
                image_path=str(self.image_path),
                output_path=str(self.dense_path),
                max_image_size=MVS_MAX_IMAGE_SIZE,
            )
            return
        except TypeError:
            pass

        # Older pycolmap APIs accept input sparse model path + undistort options.
        undistort_options = None
        if hasattr(pycolmap, "UndistortCameraOptions"):
            try:
                undistort_options = pycolmap.UndistortCameraOptions()
                if hasattr(undistort_options, "max_image_size"):
                    undistort_options.max_image_size = int(MVS_MAX_IMAGE_SIZE)
            except Exception:
                undistort_options = None

        kwargs = {
            "output_path": str(self.dense_path),
            "input_path": str(sparse_model_path),
            "image_path": str(self.image_path),
        }
        if undistort_options is not None:
            kwargs["undistort_options"] = undistort_options

        try:
            pycolmap.undistort_images(**kwargs)
            return
        except TypeError:
            pass

        # Last-resort positional call.
        pycolmap.undistort_images(
            str(self.dense_path),
            str(sparse_model_path),
            str(self.image_path),
        )

    def _run_patch_match(self) -> None:
        """Run patch-match stereo across pycolmap API versions."""
        if not hasattr(pycolmap, "patch_match_stereo"):
            return
        try:
            pycolmap.patch_match_stereo(workspace_path=str(self.dense_path))
            return
        except TypeError:
            pass
        pycolmap.patch_match_stereo(str(self.dense_path))

    def _run_stereo_fusion(self, fused_path: Path) -> None:
        """Run stereo fusion across pycolmap API versions."""
        if not hasattr(pycolmap, "stereo_fusion"):
            return
        try:
            pycolmap.stereo_fusion(
                workspace_path=str(self.dense_path),
                output_path=str(fused_path),
            )
            return
        except TypeError:
            pass
        pycolmap.stereo_fusion(str(self.dense_path), str(fused_path))

    def _run_dense_mvs(self) -> tuple[np.ndarray, np.ndarray]:
        """Attempt dense MVS and return fused dense points if available."""
        if self.reconstruction is None or not ENABLE_MVS:
            return np.array([]), np.array([])

        try:
            sparse_model_path = self._resolve_sparse_model_path()
            self._run_undistort_images(sparse_model_path)
            self._run_patch_match()

            fused_path = self.dense_path / "fused.ply"
            self._run_stereo_fusion(fused_path)

            if not fused_path.exists():
                return np.array([]), np.array([])

            if not OPEN3D_AVAILABLE:
                return np.array([]), np.array([])

            pcd = o3d.io.read_point_cloud(str(fused_path))
            dense_points = np.asarray(pcd.points)
            dense_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.array([])

            self.dense_points = dense_points
            self.dense_colors = dense_colors

            print(
                colored(
                    f"[SfMProcessor] Dense MVS point cloud: {len(dense_points)} points",
                    "green",
                )
            )

            return dense_points, dense_colors

        except Exception as exc:
            print(colored(f"[SfMProcessor] Dense MVS skipped: {exc}", "yellow"))
            return np.array([]), np.array([])

    def _fallback_result(self, num_images: int) -> Dict[str, Any]:
        """Fallback result when SfM is unavailable or fails."""
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
            "reprojection_errors": [],
            "dense_points": np.array([]),
            "dense_colors": np.array([]),
            "stats": {
                "num_registered": 0,
                "num_input_images": num_images,
                "registration_ratio": 0.0,
                "mean_reprojection_error_px": 0.0,
                "median_reprojection_error_px": 0.0,
            },
        }

    def get_camera_pose(self, image_idx: int) -> Optional[np.ndarray]:
        """Get the 4x4 camera pose matrix for a specific image."""
        if image_idx in self.camera_poses:
            return self.camera_poses[image_idx]["transform"]
        return None

    def get_intrinsics(self, image_idx: int) -> Optional[Dict]:
        """Get camera intrinsics for a specific image."""
        if image_idx in self.camera_intrinsics:
            return self.camera_intrinsics[image_idx]
        return None

    def transform_points(self, points: np.ndarray, image_idx: int) -> np.ndarray:
        """Transform camera-space points to world-space using SfM pose."""
        if len(points) == 0:
            return points

        pose = self.get_camera_pose(image_idx)
        if pose is None:
            return points

        points_h = np.hstack([points, np.ones((len(points), 1))])
        points_world = (pose @ points_h.T).T
        return points_world[:, :3]


if __name__ == "__main__":
    print("Testing SfM Processor...")

    if not PYCOLMAP_AVAILABLE:
        print("pycolmap not available - skipping test")
    else:
        images = []
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(
                img,
                (100 + i * 20, 100),
                (420 + i * 20, 320),
                (255, 255, 255),
                -1,
            )
            cv2.circle(img, (320, 240), 60 + i * 8, (128, 128, 128), -1)
            images.append(img)

        processor = SfMProcessor()
        result = processor.run_sfm(images)
        print(f"SfM success: {result['success']}")
        print(f"Registered images: {result['num_registered']}")
