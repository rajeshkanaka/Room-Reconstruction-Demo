"""
Room Reconstructor Module

Main orchestrator that combines all modules to reconstruct a room
from multiple photographs using proper photogrammetric techniques.

Pipeline:
1. Run SfM (Structure-from-Motion) to get camera poses
2. Estimate depth for each image using AI model
3. Transform depth-based point clouds using SfM camera poses
4. Fuse point clouds into unified reconstruction
5. Generate floor plan and 3D visualization
"""

import numpy as np
import cv2
from PIL import Image
import os
import sys
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OUTPUT_DIR,
    POINT_CLOUD_DENSITY,
    DEPTH_SCALE,
    CAMERA_FX,
    CAMERA_FY,
    ASSUMED_ROOM_WIDTH_METERS,
    ENABLE_REGISTRATION,
    REGISTRATION_VOXEL_SIZE,
    REGISTRATION_RANSAC_ITERATIONS,
    REGISTRATION_ICP_ITERATIONS,
    OUTLIER_NB_NEIGHBORS,
    OUTLIER_STD_RATIO,
    VOXEL_SIZE,
    ENABLE_SFM,
    SFM_MIN_IMAGES,
    ENABLE_METRIC_DEPTH,
    CALIBRATION_METHOD,
    ENABLE_VGGT,
    VGGT_CONFIDENCE_THRESHOLD,
    ENABLE_GEMINI_ANALYSIS,
    GEMINI_TIMEOUT,
)
from modules.depth_estimator import DepthEstimator
from modules.visualizer_3d import Visualizer3D

# Lazy-import SfM/COLMAP modules to avoid pycolmap native crashes on some platforms
SfMProcessor = None
DenseReconstructor = None


def _load_sfm_modules():
    """Load SfM modules on demand (pycolmap may crash on import)."""
    global SfMProcessor, DenseReconstructor
    if SfMProcessor is not None:
        return
    try:
        from modules.sfm_processor import SfMProcessor as _SfM
        from modules.dense_reconstructor import DenseReconstructor as _Dense

        SfMProcessor = _SfM
        DenseReconstructor = _Dense
    except Exception:
        pass


class RoomReconstructor:
    """
    Main class for room reconstruction from photographs.

    Improved pipeline with Structure-from-Motion:
    1. Run SfM to compute camera poses (when available)
    2. Estimate depth for each image
    3. Convert depth maps to 3D point clouds
    4. Transform point clouds using SfM poses for proper alignment
    5. Fuse and clean the combined point cloud
    6. Generate floor plan and 3D visualization
    """

    def __init__(self, assumed_room_width: float = ASSUMED_ROOM_WIDTH_METERS):
        """
        Initialize the room reconstructor.

        Args:
            assumed_room_width: Assumed room width in meters (for scale/calibration hint)
        """
        print(colored("[RoomReconstructor] Initializing components...", "cyan"))

        # Try metric depth first, fall back to legacy relative depth
        self.metric_depth_estimator = None
        self.use_metric_depth = False

        if ENABLE_METRIC_DEPTH:
            try:
                from modules.depth.metric_depth import MetricDepthEstimator

                self.metric_depth_estimator = MetricDepthEstimator()
                self.use_metric_depth = True
                print(colored("[RoomReconstructor] Metric depth enabled", "green"))
            except Exception as e:
                print(
                    colored(
                        f"[RoomReconstructor] Metric depth unavailable ({e}), using legacy",
                        "yellow",
                    )
                )

        # VGGT reconstructor (replaces SfM + depth + registration)
        self.vggt_reconstructor = None
        self.use_vggt = False
        if ENABLE_VGGT:
            try:
                from modules.vggt_reconstructor import VGGTReconstructor

                self.vggt_reconstructor = VGGTReconstructor()
                self.use_vggt = True
                print(
                    colored(
                        "[RoomReconstructor] VGGT enabled as primary backend", "green"
                    )
                )
            except Exception as e:
                print(
                    colored(
                        f"[RoomReconstructor] VGGT unavailable ({e}), using legacy pipeline",
                        "yellow",
                    )
                )

        # Gemini scene analyzer (runs in parallel with reconstruction)
        self.scene_analyzer = None
        if ENABLE_GEMINI_ANALYSIS:
            try:
                from modules.scene_analyzer import SceneAnalyzer

                self.scene_analyzer = SceneAnalyzer()
                print(
                    colored(
                        "[RoomReconstructor] Gemini scene analysis enabled", "green"
                    )
                )
            except Exception as e:
                print(
                    colored(f"[RoomReconstructor] Gemini unavailable ({e})", "yellow")
                )

        # Legacy depth estimator (always loaded as fallback)
        self.depth_estimator = DepthEstimator()
        self.visualizer = Visualizer3D()
        # SfM/COLMAP modules (lazy-loaded to avoid pycolmap native crashes)
        _load_sfm_modules()
        self.sfm_processor = SfMProcessor() if SfMProcessor is not None else None
        self.dense_reconstructor = (
            DenseReconstructor() if DenseReconstructor is not None else None
        )

        # Calibrator for metric depth refinement (needed by both VGGT and metric depth)
        self.depth_calibrator = None
        if self.use_metric_depth or self.use_vggt:
            from modules.depth.depth_calibrator import DepthCalibrator

            self.depth_calibrator = DepthCalibrator()

        # Wall detection + room segmentation pipeline
        from modules.detection.wall_detector import WallDetector
        from modules.detection.room_segmenter import RoomSegmenter
        from modules.geometry.measurement_engine import MeasurementEngine

        self.wall_detector = WallDetector()
        self.room_segmenter = RoomSegmenter()
        self.measurement_engine = MeasurementEngine()

        # Opening detection (door/window)
        self.opening_detector = None
        try:
            from modules.detection.opening_detector import OpeningDetector

            self.opening_detector = OpeningDetector()
        except Exception as e:
            print(
                colored(
                    f"[RoomReconstructor] Opening detector unavailable: {e}",
                    "yellow",
                )
            )

        self.assumed_room_width = assumed_room_width
        self.last_result = None
        self.scale_solution = {
            "scale_factor": 1.0,
            "confidence": "uncalibrated",
            "consistency": 0.0,
            "mode": CALIBRATION_METHOD,
            "cues": [],
        }
        self.use_sfm = (
            ENABLE_SFM and self.sfm_processor is not None and self.sfm_processor.enabled
        )
        self.use_tsdf = True  # Use TSDF fusion when SfM poses are available

        if self.use_sfm:
            print(
                colored(
                    "[RoomReconstructor] SfM enabled for multi-view alignment", "green"
                )
            )
        else:
            print(
                colored(
                    "[RoomReconstructor] SfM disabled - using fallback registration",
                    "yellow",
                )
            )

        print(colored("[RoomReconstructor] Initialization complete!", "green"))

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image.

        Args:
            image_path: Path to the image file

        Returns:
            RGB image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_single_image(
        self,
        image: np.ndarray,
        sample_rate: int = POINT_CLOUD_DENSITY,
        camera_intrinsics: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single image: estimate depth and convert to 3D points.

        Uses metric depth pipeline when available (output in meters).
        Falls back to legacy relative depth pipeline otherwise.

        Args:
            image: RGB image as numpy array
            sample_rate: Sampling rate for point cloud
            camera_intrinsics: Optional camera intrinsics from SfM

        Returns:
            Tuple of (depth_map, points, colors)
        """
        # Use SfM intrinsics if available, otherwise use defaults
        if camera_intrinsics:
            fx = camera_intrinsics.get("fx", CAMERA_FX)
            fy = camera_intrinsics.get("fy", CAMERA_FY)
        else:
            fx, fy = CAMERA_FX, CAMERA_FY

        # Metric depth path: output in meters, no inverse-depth hack
        if self.use_metric_depth and self.metric_depth_estimator is not None:
            depth = self.metric_depth_estimator.estimate_depth(image)

            # Use Depth Pro's estimated focal length if no SfM intrinsics
            metric_fx = fx
            metric_fy = fy
            if not camera_intrinsics:
                estimated_fl = self.metric_depth_estimator.get_focal_length(
                    image.shape[1]
                )
                metric_fx = estimated_fl
                metric_fy = estimated_fl

            points, colors = self.metric_depth_estimator.depth_to_3d_points_metric(
                image, depth, fx=metric_fx, fy=metric_fy, sample_rate=sample_rate
            )
            return depth, points, colors

        # Legacy path: relative depth with inverse-depth hack
        depth = self.depth_estimator.estimate_depth(image)
        points, colors = self.depth_estimator.depth_to_3d_points(
            image, depth, fx=fx, fy=fy, sample_rate=sample_rate
        )
        return depth, points, colors

    def _make_open3d_pcd(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """Create an Open3D point cloud from numpy arrays."""
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    def _apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply a 4x4 transform to Nx3 points."""
        if len(points) == 0:
            return points
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        pts_t = (transform @ pts_h.T).T
        return pts_t[:, :3]

    def _rotation_from_to(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Return a 3x3 rotation that aligns vector src to dst."""
        src = src.astype(np.float64)
        dst = dst.astype(np.float64)
        src /= np.linalg.norm(src) + 1e-12
        dst /= np.linalg.norm(dst) + 1e-12

        c = float(np.dot(src, dst))
        if c > 1.0 - 1e-8:
            return np.eye(3, dtype=np.float64)

        if c < -1.0 + 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(src[0]) > 0.9:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            axis = axis - np.dot(axis, src) * src
            axis /= np.linalg.norm(axis) + 1e-12
            # 180-degree rotation around axis
            return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

        v = np.cross(src, dst)
        s = np.linalg.norm(v)
        vx = np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64
        )
        r = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
        return r

    def _estimate_up_axis_from_poses(self, camera_poses: Optional[Dict]) -> Optional[np.ndarray]:
        """
        Estimate world up-axis from camera poses.

        Assumes camera +Y is image-down (OpenCV convention), so world up is -R[:,1].
        """
        if not camera_poses:
            return None

        ups = []
        ref = None
        for pose_dict in camera_poses.values():
            transform = pose_dict.get("transform")
            if transform is None or transform.shape != (4, 4):
                continue
            up = -transform[:3, 1].astype(np.float64)
            n = np.linalg.norm(up)
            if n < 1e-8:
                continue
            up /= n
            if ref is None:
                ref = up
            elif np.dot(up, ref) < 0:
                up = -up
            ups.append(up)

        if not ups:
            return None

        up_axis = np.mean(np.stack(ups, axis=0), axis=0)
        n = np.linalg.norm(up_axis)
        if n < 1e-8:
            return None
        return up_axis / n

    def _canonicalize_world_frame(
        self, points: np.ndarray, camera_poses: Optional[Dict]
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Rotate world coordinates so vertical aligns with +Y.

        This stabilizes floor-plane detection and 2D projection when upstream
        reconstructions produce arbitrary world orientation.
        """
        up_axis = self._estimate_up_axis_from_poses(camera_poses)
        if up_axis is None:
            return points, camera_poses

        target_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        rotation = self._rotation_from_to(up_axis, target_up)

        points_canonical = points
        if len(points) > 0:
            points_canonical = (rotation @ points.T).T

        poses_canonical = camera_poses
        if camera_poses:
            poses_canonical = {}
            for idx, pose_dict in camera_poses.items():
                transform = pose_dict["transform"]
                transformed = transform.copy()
                transformed[:3, :3] = rotation @ transformed[:3, :3]
                transformed[:3, 3] = rotation @ transformed[:3, 3]
                poses_canonical[idx] = {"transform": transformed}

        print(
            colored(
                f"[RoomReconstructor] Canonicalized world frame (up={up_axis.round(3)})",
                "cyan",
            )
        )
        return points_canonical, poses_canonical

    def _align_with_sfm(
        self,
        views: List[Tuple[np.ndarray, np.ndarray, int]],
        sfm_result: Dict,
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align point clouds using SfM camera poses.

        Args:
            views: List of (points, colors, image_index) tuples
            sfm_result: Result from SfM processor
            progress_callback: Optional progress callback

        Returns:
            Combined (points, colors) aligned in world coordinates
        """
        if not sfm_result.get("success", False):
            print(
                colored(
                    "[RoomReconstructor] SfM failed, using centroid alignment", "yellow"
                )
            )
            return self._fallback_alignment(views)

        camera_poses = sfm_result["camera_poses"]
        combined_points = []
        combined_colors = []

        for i, (points, colors, img_idx) in enumerate(views):
            if progress_callback:
                progress_callback(
                    0.6 + 0.1 * (i / len(views)), f"Aligning view {i+1}..."
                )

            if img_idx in camera_poses:
                # Use SfM camera pose to transform to world coordinates
                pose = camera_poses[img_idx]["transform"]
                aligned_points = self._apply_transform(points, pose)
                print(colored(f"  View {img_idx}: Aligned using SfM pose", "green"))
            else:
                # Fallback: use identity (no alignment)
                aligned_points = points
                print(
                    colored(f"  View {img_idx}: No SfM pose, using identity", "yellow")
                )

            combined_points.append(aligned_points)
            combined_colors.append(colors)

        # Add sparse points from SfM for better structure
        sparse_points = sfm_result.get("sparse_points", np.array([]))
        sparse_colors = sfm_result.get("sparse_colors", np.array([]))

        if len(sparse_points) > 0:
            print(
                colored(
                    f"[RoomReconstructor] Adding {len(sparse_points)} SfM sparse points",
                    "cyan",
                )
            )
            combined_points.append(sparse_points)
            combined_colors.append(sparse_colors)

        points = np.vstack(combined_points) if combined_points else np.array([])
        colors = np.vstack(combined_colors) if combined_colors else np.array([])

        return points, colors

    def _fuse_with_tsdf(
        self,
        images: List[np.ndarray],
        depths: List[np.ndarray],
        sfm_result: Dict,
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform TSDF fusion for dense reconstruction using SfM poses.

        Args:
            images: List of RGB images
            depths: List of depth maps
            sfm_result: Result from SfM processor
            progress_callback: Optional progress callback

        Returns:
            Tuple of (points, colors) from fused volume
        """
        if not sfm_result.get("success", False):
            print(
                colored(
                    "[RoomReconstructor] TSDF requires SfM poses, using direct fusion",
                    "yellow",
                )
            )
            return None, None

        print(colored("[RoomReconstructor] Performing TSDF fusion...", "cyan"))

        camera_poses = sfm_result["camera_poses"]
        camera_intrinsics = sfm_result.get("camera_intrinsics", {})

        try:
            points, colors = self.dense_reconstructor.fuse_tsdf(
                images, depths, camera_poses, camera_intrinsics, progress_callback
            )

            if len(points) > 0:
                print(
                    colored(
                        f"[RoomReconstructor] TSDF fusion produced {len(points):,} points",
                        "green",
                    )
                )
                return points, colors
            else:
                print(
                    colored(
                        "[RoomReconstructor] TSDF fusion produced no points", "yellow"
                    )
                )
                return None, None

        except Exception as e:
            print(colored(f"[RoomReconstructor] TSDF fusion failed: {e}", "red"))
            return None, None

    def _fallback_alignment(
        self, views: List[Tuple[np.ndarray, np.ndarray, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback alignment when SfM is not available."""
        if not views:
            return np.array([]), np.array([])

        # Simple centroid-based alignment
        combined_points = []
        combined_colors = []

        reference_centroid = None

        for points, colors, _ in views:
            if len(points) == 0:
                continue

            centroid = points.mean(axis=0)

            if reference_centroid is None:
                reference_centroid = centroid
                aligned = points
            else:
                # Align centroids
                offset = reference_centroid - centroid
                aligned = points + offset

            combined_points.append(aligned)
            combined_colors.append(colors)

        points = np.vstack(combined_points) if combined_points else np.array([])
        colors = np.vstack(combined_colors) if combined_colors else np.array([])

        return points, colors

    def _prepare_registration(
        self, points: np.ndarray, colors: Optional[np.ndarray], voxel_size: float
    ):
        """Downsample + compute FPFH features for registration."""
        import open3d as o3d

        pcd = self._make_open3d_pcd(points, colors)
        cleaned = pcd.remove_non_finite_points()
        if isinstance(cleaned, tuple):
            pcd = cleaned[0]
        else:
            pcd = cleaned
        pcd_down = pcd.voxel_down_sample(voxel_size)
        if len(pcd_down.points) < 30:
            return pcd_down, None

        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.0, max_nn=30
            )
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
        )
        return pcd_down, fpfh

    def _register_point_clouds_legacy(
        self, views: List[Tuple[np.ndarray, np.ndarray]], progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy registration using FPFH + ICP (fallback when SfM unavailable)."""
        if not views:
            return np.array([]), np.array([])
        if len(views) == 1 or not ENABLE_REGISTRATION:
            points = np.vstack([v[0] for v in views])
            colors = (
                np.vstack([v[1] for v in views]) if views[0][1] is not None else None
            )
            return points, colors

        import open3d as o3d

        voxel_size = REGISTRATION_VOXEL_SIZE
        target_points, target_colors = views[0]
        combined_points = [target_points]
        combined_colors = [target_colors]

        for i in range(1, len(views)):
            if progress_callback:
                progress_callback(
                    0.6 + 0.1 * (i / max(len(views), 2)),
                    f"Registering view {i+1}/{len(views)}...",
                )

            source_points, source_colors = views[i]

            target_stack = np.vstack(combined_points)
            target_colors_stack = (
                np.vstack(combined_colors) if combined_colors[0] is not None else None
            )

            target_down, target_fpfh = self._prepare_registration(
                target_stack, target_colors_stack, voxel_size
            )
            source_down, source_fpfh = self._prepare_registration(
                source_points, source_colors, voxel_size
            )

            if source_fpfh is None or target_fpfh is None:
                # Simple centroid alignment
                transform = np.eye(4)
                if len(source_points) > 0 and len(target_stack) > 0:
                    offset = target_stack.mean(axis=0) - source_points.mean(axis=0)
                    transform[:3, 3] = offset
            else:
                distance_threshold = voxel_size * 1.5
                result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_down,
                    target_down,
                    source_fpfh,
                    target_fpfh,
                    True,
                    distance_threshold,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(
                        False
                    ),
                    4,
                    [
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                            0.9
                        ),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                            distance_threshold
                        ),
                    ],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(
                        REGISTRATION_RANSAC_ITERATIONS, 0.999
                    ),
                )

                if result_ransac.fitness < 0.05:
                    transform = np.eye(4)
                    offset = target_stack.mean(axis=0) - source_points.mean(axis=0)
                    transform[:3, 3] = offset
                else:
                    result_icp = o3d.pipelines.registration.registration_icp(
                        source_down,
                        target_down,
                        distance_threshold * 0.6,
                        result_ransac.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=REGISTRATION_ICP_ITERATIONS
                        ),
                    )
                    transform = result_icp.transformation

            aligned_points = self._apply_transform(source_points, transform)
            combined_points.append(aligned_points)
            combined_colors.append(source_colors)

        points = np.vstack(combined_points)
        colors = np.vstack(combined_colors) if combined_colors[0] is not None else None
        return points, colors

    def _extract_floor_points_2d(
        self,
        points: np.ndarray,
        floor_inliers: np.ndarray,
        floor_height: Optional[float],
        band_height: float = 0.08,
    ) -> np.ndarray:
        """
        Extract robust floor-support points in top-down coordinates (X, Z).

        Prefers a narrow band around floor height; falls back to RANSAC inliers.
        """
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        candidates = np.zeros((0, 3), dtype=np.float64)
        if floor_height is not None:
            band_mask = np.abs(points[:, 1] - floor_height) <= band_height
            if np.any(band_mask):
                candidates = points[band_mask]

        if len(candidates) < 100 and len(floor_inliers) > 0:
            candidates = points[floor_inliers]

        if len(candidates) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        return candidates[:, [0, 2]].astype(np.float64)

    @staticmethod
    def _segment_angle_difference_deg(
        seg_a: Tuple[np.ndarray, np.ndarray],
        seg_b: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Smallest orientation difference in degrees (0..90)."""
        a = seg_a[1] - seg_a[0]
        b = seg_b[1] - seg_b[0]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 90.0
        au = a / na
        bu = b / nb
        dot = float(np.clip(np.dot(au, bu), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot)))
        angle = min(angle, 180.0 - angle)
        return min(angle, 90.0)

    @staticmethod
    def _segment_overlap_along_direction(
        reference: Tuple[np.ndarray, np.ndarray],
        candidate: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Projected overlap between two segments along reference direction."""
        ref_s, ref_e = reference
        c_s, c_e = candidate
        d = ref_e - ref_s
        d_norm = float(np.linalg.norm(d))
        if d_norm < 1e-9:
            return 0.0
        u = d / d_norm
        ref_t0, ref_t1 = 0.0, d_norm
        c_t0 = float(np.dot(c_s - ref_s, u))
        c_t1 = float(np.dot(c_e - ref_s, u))
        lo = max(min(ref_t0, ref_t1), min(c_t0, c_t1))
        hi = min(max(ref_t0, ref_t1), max(c_t0, c_t1))
        return float(max(0.0, hi - lo))

    @staticmethod
    def _point_line_distance(point: np.ndarray, line: Tuple[np.ndarray, np.ndarray]) -> float:
        """Perpendicular distance from point to infinite line."""
        a, b = line
        ab = b - a
        denom = float(np.linalg.norm(ab))
        if denom < 1e-9:
            return float(np.linalg.norm(point - a))
        rel = point - a
        cross_2d = float(ab[0] * rel[1] - ab[1] * rel[0])
        return float(abs(cross_2d) / denom)

    @staticmethod
    def _compute_wall_closure_metrics(walls: List, tolerance: float = 0.2) -> Dict:
        """
        Compute endpoint-closure metrics for a wall graph.

        A well-closed room should have each wall endpoint near at least one
        other endpoint (shared corners). This metric is independent of point
        ordering and robust to small residual snapping noise.
        """
        if not walls:
            return {
                "closure_score": 0.0,
                "closure_mean_gap_m": 0.0,
                "closure_p95_gap_m": 0.0,
                "closure_unpaired_ratio": 1.0,
            }

        endpoints: List[np.ndarray] = []
        for wall in walls:
            start = np.asarray(getattr(wall, "start", None), dtype=np.float64)
            end = np.asarray(getattr(wall, "end", None), dtype=np.float64)
            if start.shape != (2,) or end.shape != (2,):
                continue
            endpoints.append(start)
            endpoints.append(end)

        if len(endpoints) < 4:
            return {
                "closure_score": 0.0,
                "closure_mean_gap_m": 0.0,
                "closure_p95_gap_m": 0.0,
                "closure_unpaired_ratio": 1.0,
            }

        pts = np.vstack(endpoints)
        nearest = []
        for i in range(len(pts)):
            delta = pts - pts[i]
            dists = np.linalg.norm(delta, axis=1)
            dists[i] = np.inf
            nearest.append(float(np.min(dists)))

        nearest_arr = np.asarray(nearest, dtype=np.float64)
        tol = float(max(tolerance, 1e-6))
        paired_ratio = float(np.mean(nearest_arr <= tol))
        mean_gap = float(np.mean(nearest_arr))
        p95_gap = float(np.percentile(nearest_arr, 95))
        gap_penalty = float(np.clip(mean_gap / (tol * 2.0), 0.0, 1.0))
        closure_score = float(np.clip(paired_ratio - 0.25 * gap_penalty, 0.0, 1.0))

        return {
            "closure_score": round(closure_score, 3),
            "closure_mean_gap_m": round(mean_gap, 3),
            "closure_p95_gap_m": round(p95_gap, 3),
            "closure_unpaired_ratio": round(1.0 - paired_ratio, 3),
        }

    def _filter_segments_by_multiview_support(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        depth_segments_by_view: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
        min_support_views: int = 2,
        min_segments_to_keep: int = 3,
        angle_tolerance_deg: float = 18.0,
        distance_tolerance_m: float = 0.30,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
        """
        Keep wall segments that are corroborated by multiple image views.

        Returns:
            (filtered_segments, stats)
        """
        stats = {
            "enabled": False,
            "input_segments": len(segments),
            "retained_segments": len(segments),
            "dropped_segments": 0,
            "min_support_views": int(min_support_views),
            "avg_support_views": 0.0,
            "max_support_views": 0,
            "num_support_views": len(depth_segments_by_view),
        }
        if not segments or not depth_segments_by_view:
            return segments, stats

        support_counts: List[int] = []
        kept: List[Tuple[np.ndarray, np.ndarray]] = []

        for seg in segments:
            s, e = seg
            seg_len = float(np.linalg.norm(e - s))
            if seg_len < 0.2:
                support_counts.append(0)
                continue

            seg_mid = (s + e) / 2.0
            required_overlap = max(0.2, 0.2 * seg_len)
            support_views = 0

            for _, obs_segments in depth_segments_by_view.items():
                has_support = False
                for obs in obs_segments:
                    obs_s, obs_e = obs
                    if float(np.linalg.norm(obs_e - obs_s)) < 0.2:
                        continue
                    angle_diff = self._segment_angle_difference_deg(seg, obs)
                    if angle_diff > angle_tolerance_deg:
                        continue

                    mid_dist = self._point_line_distance(seg_mid, obs)
                    if mid_dist > distance_tolerance_m:
                        continue

                    overlap = self._segment_overlap_along_direction(seg, obs)
                    if overlap < required_overlap:
                        continue

                    has_support = True
                    break

                if has_support:
                    support_views += 1

            support_counts.append(support_views)
            if support_views >= min_support_views:
                kept.append((s.copy(), e.copy()))

        if len(kept) < min_segments_to_keep:
            stats.update(
                {
                    "enabled": True,
                    "retained_segments": len(segments),
                    "dropped_segments": 0,
                    "fallback_to_unfiltered": True,
                    "avg_support_views": float(np.mean(support_counts))
                    if support_counts
                    else 0.0,
                    "max_support_views": int(max(support_counts)) if support_counts else 0,
                }
            )
            return segments, stats

        stats.update(
            {
                "enabled": True,
                "retained_segments": len(kept),
                "dropped_segments": len(segments) - len(kept),
                "fallback_to_unfiltered": False,
                "avg_support_views": float(np.mean(support_counts))
                if support_counts
                else 0.0,
                "max_support_views": int(max(support_counts)) if support_counts else 0,
            }
        )
        return kept, stats

    def _is_room_geometry_usable(
        self,
        rooms: List,
        floor_points_2d: np.ndarray,
        min_area_m2: float = 2.0,
    ) -> bool:
        """
        Validate extracted room geometry quality.

        Rejects tiny/degenerate polygons and polygons that under-cover
        the observed floor footprint.
        """
        if not rooms:
            return False

        primary = max(rooms, key=lambda r: getattr(r, "area", 0.0))
        area = float(getattr(primary, "area", 0.0))
        if area < min_area_m2:
            return False

        boundary = getattr(primary, "boundary", None)
        if boundary is None or len(boundary) < 3:
            return False
        if len(boundary) < 4:
            return False

        edges = np.linalg.norm(np.roll(boundary, -1, axis=0) - boundary, axis=1)
        if np.any(edges < 0.3):
            return False

        if floor_points_2d is not None and len(floor_points_2d) >= 100:
            lo = np.percentile(floor_points_2d, 2, axis=0)
            hi = np.percentile(floor_points_2d, 98, axis=0)
            footprint_area = float(max((hi[0] - lo[0]) * (hi[1] - lo[1]), 0.1))
            if area < 0.5 * footprint_area:
                return False

        return True

    def _regularize_rooms(
        self,
        rooms: List,
        floor_points_2d: np.ndarray,
    ) -> List:
        """
        Regularize noisy single-room polygons into a practical footprint.

        When the extracted room has many vertices (typically from noisy convex
        hull output), replace it with a robust rectangular fit if area remains
        broadly consistent.
        """
        if not rooms:
            return rooms

        primary = max(rooms, key=lambda r: getattr(r, "area", 0.0))
        boundary = getattr(primary, "boundary", None)
        if boundary is None:
            return rooms
        if len(boundary) == 4:
            return rooms

        fit_points = floor_points_2d if floor_points_2d is not None and len(floor_points_2d) > 0 else boundary
        if fit_points is None or len(fit_points) < 20:
            dense = []
            n = len(boundary)
            for i in range(n):
                a = boundary[i].astype(np.float64)
                b = boundary[(i + 1) % n].astype(np.float64)
                seg_len = float(np.linalg.norm(b - a))
                steps = max(3, int(seg_len / 0.15))
                for t in np.linspace(0.0, 1.0, steps, endpoint=False):
                    dense.append(a * (1.0 - t) + b * t)
            fit_points = np.array(dense, dtype=np.float64) if dense else boundary

        rect = self.room_segmenter.points_to_rectangular_room(
            fit_points,
            room_name=getattr(primary, "name", "Room"),
        )
        if rect is None:
            return rooms

        src_area = max(float(getattr(primary, "area", 0.0)), 1e-6)
        ratio = rect.area / src_area
        upper_ratio = 3.0 if len(boundary) < 4 else 1.9
        if ratio < 0.45 or ratio > upper_ratio:
            return rooms

        print(
            colored(
                "[RoomReconstructor] Regularized noisy polygon to rectangular footprint",
                "yellow",
            )
        )
        return [rect]

    def _normalize_model_orientation(self, model):
        """
        Rotate/translate floor plan model into a canonical drafting frame.

        - Longest wall aligns with +X (horizontal)
        - Geometry is shifted so min corner starts near origin
        """
        if model is None or not getattr(model, "walls", None):
            return model

        lengths = []
        for wall in model.walls:
            vec = wall.end - wall.start
            lengths.append(float(np.linalg.norm(vec)))

        if not lengths or max(lengths) < 1e-6:
            return model

        ref_idx = int(np.argmax(lengths))
        ref_wall = model.walls[ref_idx]
        ref_vec = ref_wall.end - ref_wall.start
        ref_angle = float(np.arctan2(ref_vec[1], ref_vec[0]))
        if ref_angle > np.pi / 2:
            ref_angle -= np.pi
        elif ref_angle < -np.pi / 2:
            ref_angle += np.pi

        theta = -ref_angle
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)

        anchors = []
        for wall in model.walls:
            anchors.append(wall.start)
            anchors.append(wall.end)
        center = np.mean(np.array(anchors, dtype=np.float64), axis=0)

        def _xform(p: np.ndarray) -> np.ndarray:
            return (rot @ (p.astype(np.float64) - center)) + center

        for wall in model.walls:
            wall.start = _xform(wall.start)
            wall.end = _xform(wall.end)

        for room in model.rooms:
            if room.boundary is not None and len(room.boundary) > 0:
                room.boundary = np.array([_xform(p) for p in room.boundary])

        for door in model.doors:
            door.position = _xform(door.position)
        for window in model.windows:
            window.position = _xform(window.position)

        all_pts = []
        for wall in model.walls:
            all_pts.append(wall.start)
            all_pts.append(wall.end)
        if all_pts:
            all_pts = np.array(all_pts, dtype=np.float64)
            shift = np.array([max(0.0, -all_pts[:, 0].min()), max(0.0, -all_pts[:, 1].min())])
            if np.linalg.norm(shift) > 0:
                for wall in model.walls:
                    wall.start = wall.start + shift
                    wall.end = wall.end + shift
                for room in model.rooms:
                    if room.boundary is not None and len(room.boundary) > 0:
                        room.boundary = room.boundary + shift
                for door in model.doors:
                    door.position = door.position + shift
                for window in model.windows:
                    window.position = window.position + shift

        model.orientation = 0.0
        return model

    def _detect_walls_and_rooms(
        self,
        points: np.ndarray,
        depth_maps: List[np.ndarray],
        images: List[np.ndarray],
        camera_intrinsics: Optional[Dict] = None,
        camera_poses: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Run the wall detection + room segmentation pipeline.

        Args:
            points: Nx3 point cloud (calibrated)
            depth_maps: Per-image depth maps (calibrated)
            images: Original RGB images
            camera_intrinsics: Optional dict of per-camera intrinsics from
                VGGT or SfM.  Keys are image indices, values have "fx", "fy".
            camera_poses: Optional dict of per-camera 4x4 world-from-camera
                transforms.  Keys are image indices, values have "transform".

        Returns a FloorPlanModel dict, or None if detection fails.
        """
        from modules.geometry.floor_plan_model import FloorPlanModel

        try:
            quality_flags = {
                "used_depth_augmentation": False,
                "used_rectangular_fallback": False,
                "used_sparse_wall_fallback": False,
                "plane_segment_count": 0,
                "aligned_segment_count": 0,
                "wall_multiview_support": {},
                "closure_score": 0.0,
                "closure_mean_gap_m": 0.0,
                "closure_p95_gap_m": 0.0,
                "closure_unpaired_ratio": 1.0,
            }

            # 1. Detect floor plane
            plane, floor_inliers = self.wall_detector.detect_floor_plane(
                points, up_axis=np.array([0.0, 1.0, 0.0], dtype=np.float64)
            )
            floor_height = None
            if len(floor_inliers) > 0:
                floor_height = float(np.median(points[floor_inliers, 1]))
            floor_points_2d = self._extract_floor_points_2d(
                points, floor_inliers, floor_height
            )

            # 2. Detect walls from 3D planes first (primary path).
            all_segments = self.wall_detector.detect_walls_from_point_cloud_planes(
                points=points,
                floor_height=floor_height,
                up_axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )
            quality_flags["plane_segment_count"] = len(all_segments)

            # 3. Optional depth-segment extraction:
            #    - augment geometry when plane extraction is sparse
            #    - estimate cross-view wall support in VGGT/SfM path
            need_depth_augmentation = len(all_segments) < 4
            collect_support = camera_poses is not None and len(depth_maps) > 0
            depth_segments_by_view: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
            if need_depth_augmentation or collect_support:
                depth_segments: List[Tuple[np.ndarray, np.ndarray]] = []
                for i, depth in enumerate(depth_maps):
                    # Use per-camera intrinsics when available (VGGT/SfM)
                    if camera_intrinsics and i in camera_intrinsics:
                        intr = camera_intrinsics[i]
                        fx = float(intr.get("fx", CAMERA_FX))
                        fy = float(intr.get("fy", fx))
                        cx = float(intr.get("cx", depth.shape[1] / 2.0))
                        cy = float(intr.get("cy", depth.shape[0] / 2.0))
                    elif (
                        self.use_metric_depth and self.metric_depth_estimator is not None
                    ):
                        fx = self.metric_depth_estimator.get_focal_length(depth.shape[1])
                        fy = fx
                        cx = depth.shape[1] / 2.0
                        cy = depth.shape[0] / 2.0
                    else:
                        fx = CAMERA_FX
                        fy = CAMERA_FY
                        cx = depth.shape[1] / 2.0
                        cy = depth.shape[0] / 2.0

                    lines = self.wall_detector.detect_wall_lines_from_depth(depth)
                    if not lines:
                        continue

                    pose = (
                        camera_poses[i]["transform"]
                        if camera_poses and i in camera_poses
                        else None
                    )

                    view_segments: List[Tuple[np.ndarray, np.ndarray]] = []
                    for x1, y1, x2, y2 in lines:
                        d1 = self.wall_detector._sample_depth(depth, x1, y1)
                        d2 = self.wall_detector._sample_depth(depth, x2, y2)
                        if d1 <= 0 or d2 <= 0:
                            continue

                        p1_cam = np.array(
                            [(x1 - cx) * d1 / fx, (y1 - cy) * d1 / fy, d1],
                            dtype=np.float64,
                        )
                        p2_cam = np.array(
                            [(x2 - cx) * d2 / fx, (y2 - cy) * d2 / fy, d2],
                            dtype=np.float64,
                        )

                        if pose is not None:
                            p1_world = pose[:3, :3] @ p1_cam + pose[:3, 3]
                            p2_world = pose[:3, :3] @ p2_cam + pose[:3, 3]
                        else:
                            p1_world = p1_cam
                            p2_world = p2_cam

                        if floor_height is not None:
                            # Reject obvious floor/ceiling artifacts.
                            min_h = floor_height - 0.2
                            max_h = floor_height + 3.5
                            if (p1_world[1] < min_h and p2_world[1] < min_h) or (
                                p1_world[1] > max_h and p2_world[1] > max_h
                            ):
                                continue

                        start_2d = np.array([p1_world[0], p1_world[2]], dtype=np.float64)
                        end_2d = np.array([p2_world[0], p2_world[2]], dtype=np.float64)
                        if np.linalg.norm(end_2d - start_2d) < 0.2:
                            continue
                        view_segments.append((start_2d, end_2d))

                    if view_segments:
                        # Keep strongest observations per view (longer segments are usually more stable).
                        view_segments = sorted(
                            view_segments,
                            key=lambda seg: float(np.linalg.norm(seg[1] - seg[0])),
                            reverse=True,
                        )[:24]
                        depth_segments_by_view[i] = view_segments
                        if need_depth_augmentation:
                            depth_segments.extend(view_segments)

                if need_depth_augmentation and depth_segments:
                    all_segments.extend(depth_segments)
                    quality_flags["used_depth_augmentation"] = True
                    print(
                        colored(
                            f"[RoomReconstructor] Augmented walls with {len(depth_segments)} depth segments",
                            "cyan",
                        )
                    )

            if len(all_segments) < 3:
                fallback_room = self.room_segmenter.points_to_rectangular_room(
                    floor_points_2d,
                    room_name="Room",
                )
                if fallback_room is None:
                    print(
                        colored(
                            f"[RoomReconstructor] Too few wall segments ({len(all_segments)}), "
                            "skipping new detection pipeline",
                            "yellow",
                        )
                    )
                    return None

                rooms = [fallback_room]
                wall_segments = self.room_segmenter.rooms_to_wall_segments(rooms)
                all_segments = [
                    (wall.start.copy(), wall.end.copy()) for wall in wall_segments
                ]
                quality_flags["used_rectangular_fallback"] = True
                quality_flags["used_sparse_wall_fallback"] = True
                aligned = all_segments
            else:
                # 4. Manhattan alignment (use wider merge for multi-camera segments)
                merge_dist = 0.2 if camera_poses else 0.15
                aligned = self.wall_detector.align_walls_manhattan(
                    all_segments, merge_distance=merge_dist
                )
                if len(aligned) < 3:
                    print(
                        colored(
                            "[RoomReconstructor] Too few aligned segments, using unaligned",
                            "yellow",
                        )
                    )
                    aligned = all_segments
                quality_flags["aligned_segment_count"] = len(aligned)

                # 5. Multi-view support filtering (drop weak one-off segments).
                min_support_views = 2 if len(depth_segments_by_view) >= 4 else 1
                aligned, support_stats = self._filter_segments_by_multiview_support(
                    aligned,
                    depth_segments_by_view=depth_segments_by_view,
                    min_support_views=min_support_views,
                    min_segments_to_keep=3,
                )
                quality_flags["wall_multiview_support"] = support_stats
                if support_stats.get("enabled"):
                    print(
                        colored(
                            "[RoomReconstructor] Wall multi-view support: "
                            f"{support_stats.get('retained_segments', len(aligned))}/"
                            f"{support_stats.get('input_segments', len(aligned))} kept "
                            f"(avg views {support_stats.get('avg_support_views', 0.0):.2f})",
                            "cyan",
                        )
                    )

                # 6. Optimize wall graph topology (snap + dedupe + closure)
                optimized = self.room_segmenter.optimize_wall_graph(aligned)
                if len(optimized) >= 3:
                    aligned = optimized

                # 7. Extract room polygons (wider snap for multi-camera)
                original_snap = self.room_segmenter.snap_tolerance
                if camera_poses:
                    self.room_segmenter.snap_tolerance = max(original_snap, 0.25)
                rooms = self.room_segmenter.extract_rooms(aligned)
                self.room_segmenter.snap_tolerance = original_snap

                if not self._is_room_geometry_usable(rooms, floor_points_2d):
                    fallback_points = (
                        floor_points_2d
                        if len(floor_points_2d) > 0
                        else np.array(
                            [pt for seg in aligned for pt in seg], dtype=np.float64
                        )
                    )
                    fallback_room = self.room_segmenter.points_to_rectangular_room(
                        fallback_points,
                        room_name="Room",
                    )
                    if fallback_room is not None:
                        rooms = [fallback_room]
                        quality_flags["used_rectangular_fallback"] = True
                        print(
                            colored(
                                "[RoomReconstructor] Replaced fragmented topology with rectangular room fallback",
                                "yellow",
                            )
                        )

                rooms = self._regularize_rooms(rooms, floor_points_2d)

                # 8. Convert to WallSegments
                wall_segments = self.room_segmenter.rooms_to_wall_segments(rooms)
                if not wall_segments:
                    wall_segments = self.room_segmenter.segments_to_wall_segments(
                        aligned
                    )
                if len(wall_segments) < 4:
                    fallback_points = (
                        floor_points_2d
                        if floor_points_2d is not None and len(floor_points_2d) > 0
                        else np.array([pt for seg in aligned for pt in seg], dtype=np.float64)
                    )
                    fallback_room = self.room_segmenter.points_to_rectangular_room(
                        fallback_points,
                        room_name="Room",
                    )
                    if fallback_room is not None:
                        rooms = [fallback_room]
                        wall_segments = self.room_segmenter.rooms_to_wall_segments(rooms)
                        quality_flags["used_rectangular_fallback"] = True
                        print(
                            colored(
                                "[RoomReconstructor] Enforced rectangular fallback due under-constrained wall graph",
                                "yellow",
                            )
                        )

            closure_metrics = self._compute_wall_closure_metrics(
                wall_segments,
                tolerance=max(0.12, self.room_segmenter.snap_tolerance),
            )
            quality_flags.update(closure_metrics)
            print(
                colored(
                    "[RoomReconstructor] Wall closure: "
                    f"score={closure_metrics['closure_score']:.2f}, "
                    f"mean_gap={closure_metrics['closure_mean_gap_m']:.2f}m",
                    "cyan",
                )
            )

            # 9. Detect doors and windows
            doors = []
            windows = []
            if self.opening_detector is not None:
                try:
                    # Fallback focal length (per-view intrinsics are passed when available)
                    if (
                        self.use_metric_depth
                        and self.metric_depth_estimator is not None
                        and images
                    ):
                        fx = self.metric_depth_estimator.get_focal_length(images[0].shape[1])
                    else:
                        fx = CAMERA_FX
                    doors, windows = self.opening_detector.detect_and_project(
                        images,
                        depth_maps,
                        wall_segments,
                        fx=fx,
                        fy=fx,
                        camera_intrinsics=camera_intrinsics,
                        camera_poses=camera_poses,
                    )
                    fusion_stats = getattr(self.opening_detector, "last_fusion_stats", {})
                    if fusion_stats:
                        quality_flags["opening_fusion"] = fusion_stats
                except Exception as e:
                    print(
                        colored(
                            f"[RoomReconstructor] Opening detection failed: {e}",
                            "yellow",
                        )
                    )

            # 9. Build FloorPlanModel
            model = FloorPlanModel(
                walls=wall_segments, rooms=rooms, doors=doors, windows=windows
            )
            model = self._normalize_model_orientation(model)

            # 10. Compute measurements
            measurements = self.measurement_engine.compute_measurements(model)

            print(
                colored(
                    f"[RoomReconstructor] New pipeline: {len(wall_segments)} walls, "
                    f"{len(rooms)} rooms, {len(doors)} doors, {len(windows)} windows",
                    "green",
                )
            )

            return {
                "floor_plan_model": model,
                "measurements": measurements,
                "quality_flags": quality_flags,
            }

        except Exception as e:
            print(
                colored(
                    f"[RoomReconstructor] Wall detection pipeline failed: {e}",
                    "yellow",
                )
            )
            return None

    def _render_floor_plan_model(
        self,
        detection_result: Dict,
        timestamp: str,
    ) -> Dict:
        """
        Render the FloorPlanModel using SVG, DXF, and PNG renderers.

        Returns dict of output paths, or empty dict if rendering fails.
        """
        if detection_result is None:
            return {}

        model = detection_result.get("floor_plan_model")
        if model is None:
            return {}

        outputs = {}
        try:
            from modules.rendering import SVGRenderer, DXFRenderer, PNGRenderer

            title = "Room Floor Plan"

            # SVG
            svg_path = os.path.join(OUTPUT_DIR, f"floor_plan_{timestamp}.svg")
            SVGRenderer().render(model, svg_path, title=title)
            outputs["floor_plan_svg"] = svg_path

            # DXF
            dxf_path = os.path.join(OUTPUT_DIR, f"floor_plan_{timestamp}.dxf")
            DXFRenderer().render(model, dxf_path, title=title)
            outputs["floor_plan_dxf"] = dxf_path

            # PNG (architectural style from FloorPlanModel)
            png_path = os.path.join(OUTPUT_DIR, f"floor_plan_arch_{timestamp}.png")
            PNGRenderer().render_to_image(model, png_path, title=title)
            outputs["floor_plan_arch_png"] = png_path

            print(
                colored(
                    f"[RoomReconstructor] Rendered floor plan: SVG, DXF, PNG",
                    "green",
                )
            )
        except Exception as e:
            print(
                colored(
                    f"[RoomReconstructor] Floor plan rendering failed: {e}",
                    "yellow",
                )
            )

        return outputs

    def _transform_segments_to_world(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        pose: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Transform wall segments from camera-local ground plane to world coordinates.

        Wall segments are in camera frame: (gx, gz) where gx is lateral and
        gz is forward (depth). We lift to 3D camera coords with Y=0, apply
        the world-from-camera pose, then project back to the world X-Z plane.
        """
        if len(segments) == 0:
            return segments

        transformed = []
        for start_2d, end_2d in segments:
            # Camera local 3D: [gx, 0, gz] (Y=0 at camera height)
            cam_start = np.array([start_2d[0], 0.0, start_2d[1], 1.0])
            cam_end = np.array([end_2d[0], 0.0, end_2d[1], 1.0])

            # Transform to world
            world_start = pose @ cam_start
            world_end = pose @ cam_end

            # Project to world ground plane (X, Z)
            transformed.append(
                (
                    np.array([world_start[0], world_start[2]]),
                    np.array([world_end[0], world_end[2]]),
                )
            )

        return transformed

    def _calibrate_metric_points(self, points: np.ndarray) -> np.ndarray:
        """Apply metric depth calibration if available."""
        calibrated, _, _ = self._calibrate_reconstruction_scale(
            points,
            camera_poses=None,
            relative_scale_required=False,
        )
        return calibrated

    def _calibrate_reconstruction_scale(
        self,
        points: np.ndarray,
        camera_poses: Optional[Dict] = None,
        relative_scale_required: bool = False,
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Calibrate global reconstruction scale using multi-cue solver.

        Args:
            points: Nx3 point cloud
            camera_poses: Optional camera poses for baseline cue
            relative_scale_required: Force solving scale (used for relative pipelines)

        Returns:
            (scaled_points, scale_factor_applied, scale_solution)
        """
        default_solution = {
            "scale_factor": 1.0,
            "confidence": "uncalibrated",
            "consistency": 0.0,
            "mode": CALIBRATION_METHOD,
            "cues": [],
        }

        if points is None or len(points) == 0:
            self.scale_solution = default_solution
            return points, 1.0, default_solution

        if self.depth_calibrator is None:
            fallback = {
                **default_solution,
                "confidence": "low",
                "mode": "disabled",
            }
            self.scale_solution = fallback
            return points, 1.0, fallback

        method = CALIBRATION_METHOD
        if method == "none":
            disabled = {
                **default_solution,
                "confidence": "disabled",
                "mode": "none",
                "consistency": 1.0,
            }
            self.scale_solution = disabled
            return points, 1.0, disabled

        plausible = self.depth_calibrator.sanity_check(points)
        should_solve = False

        if method == "user_reference":
            should_solve = self.assumed_room_width > 0
        elif method == "auto":
            should_solve = relative_scale_required or (not plausible)

        if not should_solve:
            passthrough = {
                **default_solution,
                "confidence": "high" if plausible else "low",
                "consistency": 1.0 if plausible else 0.3,
                "mode": method,
                "plausible_before": bool(plausible),
            }
            self.scale_solution = passthrough
            return points, 1.0, passthrough

        assumed_width = self.assumed_room_width if self.assumed_room_width > 0 else None
        solution = self.depth_calibrator.solve_scale_from_cues(
            points,
            assumed_room_width=assumed_width,
            camera_poses=camera_poses,
            mode=method,
        )
        solution["plausible_before"] = bool(plausible)

        scale_factor = float(solution.get("scale_factor", 1.0))
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            scale_factor = 1.0
            solution["scale_factor"] = scale_factor
            solution["confidence"] = "low"

        scaled_points = points * scale_factor
        self.scale_solution = solution

        print(
            colored(
                f"[RoomReconstructor] Scale solved: {scale_factor:.3f}x "
                f"(confidence={solution.get('confidence', 'low')})",
                "green" if solution.get("confidence") == "high" else "yellow",
            )
        )

        return scaled_points, scale_factor, solution

    def _postprocess_point_cloud(
        self, points: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Denoise and downsample the combined cloud for cleaner outputs."""
        if len(points) == 0:
            return points, colors

        import open3d as o3d

        pcd = self._make_open3d_pcd(points, colors)
        cleaned = pcd.remove_non_finite_points()
        if isinstance(cleaned, tuple):
            pcd = cleaned[0]
        else:
            pcd = cleaned

        if OUTLIER_NB_NEIGHBORS > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO
            )

        if VOXEL_SIZE > 0:
            pcd = pcd.voxel_down_sample(VOXEL_SIZE)

        points_clean = np.asarray(pcd.points)
        colors_clean = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points_clean, colors_clean

    def _reconstruct_with_vggt(
        self, images: List[np.ndarray], progress_callback=None
    ) -> Optional[Dict]:
        """
        Run VGGT reconstruction as the primary backend.

        Returns dict with combined_points, combined_colors, all_depths,
        sfm_result, raw_count on success, or None on failure.
        """
        if not self.use_vggt or self.vggt_reconstructor is None:
            return None

        try:
            input_images = images
            selected_indices = list(range(len(images)))
            max_vggt_images = 12
            if len(images) > max_vggt_images:
                idx = np.linspace(0, len(images) - 1, num=max_vggt_images)
                idx = np.round(idx).astype(int)
                selected_indices = sorted(set(int(i) for i in idx.tolist()))
                if len(selected_indices) < max_vggt_images:
                    for i in range(len(images)):
                        if i not in selected_indices:
                            selected_indices.append(i)
                        if len(selected_indices) >= max_vggt_images:
                            break
                    selected_indices = sorted(selected_indices[:max_vggt_images])
                input_images = [images[i] for i in selected_indices]
                print(
                    colored(
                        "[RoomReconstructor] VGGT view selection: "
                        f"using {len(input_images)}/{len(images)} images "
                        f"(indices: {selected_indices})",
                        "yellow",
                    )
                )

            if progress_callback:
                progress_callback(0.15, "Running VGGT reconstruction...")

            print(colored("[RoomReconstructor] Running VGGT reconstruction...", "cyan"))
            vggt_result = self.vggt_reconstructor.reconstruct(input_images)

            if not vggt_result.get("success", False):
                print(
                    colored(
                        "[RoomReconstructor] VGGT reconstruction failed, falling back to legacy",
                        "yellow",
                    )
                )
                return None

            combined_points = vggt_result["point_cloud"]
            combined_colors = vggt_result["point_colors"]

            if len(combined_points) == 0:
                return None

            raw_count = len(combined_points)

            # Post-process (denoise + downsample)
            combined_points, combined_colors = self._postprocess_point_cloud(
                combined_points, combined_colors
            )

            if len(combined_points) == 0:
                return None

            # Calibrate VGGT points using multi-cue solver.
            # VGGT world_points are relative, so scale solving is always required.
            (
                combined_points,
                scale_factor,
                _,
            ) = self._calibrate_reconstruction_scale(
                combined_points,
                camera_poses=vggt_result.get("camera_poses"),
                relative_scale_required=True,
            )

            # Scale depth maps and camera translations by the same factor
            depth_maps_calibrated = []
            for dm in vggt_result["depth_maps"]:
                depth_maps_calibrated.append(dm * scale_factor)

            # Scale camera pose translations so they're in calibrated coords
            calibrated_poses = {}
            for idx, pose_dict in vggt_result["camera_poses"].items():
                pose = pose_dict["transform"].copy()
                pose[:3, 3] *= scale_factor  # Scale translation only
                calibrated_poses[idx] = {"transform": pose}

            # Canonicalize frame so vertical aligns with +Y for downstream geometry.
            combined_points, calibrated_poses = self._canonicalize_world_frame(
                combined_points, calibrated_poses
            )

            print(
                colored(
                    f"[RoomReconstructor] VGGT: {len(combined_points):,} points "
                    f"(raw: {raw_count:,}), {vggt_result['num_registered']} cameras",
                    "green",
                )
            )

            # Build sfm_result-like dict for downstream compatibility
            sfm_result = {
                "success": True,
                "num_registered": vggt_result["num_registered"],
                "camera_poses": calibrated_poses,
                "camera_intrinsics": vggt_result["camera_intrinsics"],
            }

            return {
                "combined_points": combined_points,
                "combined_colors": combined_colors,
                "all_depths": depth_maps_calibrated,
                "sfm_result": sfm_result,
                "raw_count": raw_count,
                "camera_intrinsics": vggt_result["camera_intrinsics"],
                "input_images": input_images,
                "selected_indices": selected_indices,
                "num_vggt_images": len(input_images),
            }

        except Exception as e:
            print(
                colored(
                    f"[RoomReconstructor] VGGT failed: {e}, falling back to legacy",
                    "yellow",
                )
            )
            return None

    def _run_gemini_analysis(self, images: List[np.ndarray]):
        """Run Gemini scene analysis (thread-safe for parallel execution)."""
        if self.scene_analyzer is None:
            return None
        try:
            return self.scene_analyzer.analyze(images)
        except Exception as e:
            print(colored(f"[RoomReconstructor] Gemini analysis failed: {e}", "yellow"))
            return None

    def _merge_gemini_into_model(self, detection_result, gemini_result):
        """Merge Gemini semantic data into the FloorPlanModel from detection."""
        if detection_result is None or gemini_result is None:
            return detection_result
        if not gemini_result.success:
            return detection_result

        from modules.geometry.floor_plan_model import (
            FloorPlanModel,
            DoorOpening,
            WindowOpening,
        )

        model = detection_result.get("floor_plan_model")
        if model is None:
            return detection_result

        # Set room type on room polygons
        if model.rooms and gemini_result.room_type:
            display_name = gemini_result.room_type.replace("_", " ").title()
            for room in model.rooms:
                room.name = display_name
                room.room_type = gemini_result.room_type
                room.room_shape = gemini_result.room_shape

        # Set model-level semantic fields
        model.reconstruction_backend = "vggt" if self.use_vggt else "legacy"
        model.semantic_confidence = gemini_result.confidence

        # Add Gemini-detected doors (with deduplication)
        wall_directions = self._assign_walls_to_directions(model.walls)
        for door_info in gemini_result.doors:
            wall_label = door_info.get("wall", "")
            target_wall = wall_directions.get(wall_label)
            if target_wall is None:
                continue

            # Place door at specified position on wall
            position_label = door_info.get("position", "center")
            wall_vec = target_wall.end - target_wall.start
            wall_len = np.linalg.norm(wall_vec)
            t = {"left": 0.25, "center": 0.5, "right": 0.75}.get(position_label, 0.5)
            pos = target_wall.start + wall_vec * t

            # Check for duplicate (within 0.5m of existing door)
            is_duplicate = any(
                np.linalg.norm(pos - d.position) < 0.5 for d in model.doors
            )
            if is_duplicate:
                continue

            width_map = {"narrow": 0.7, "standard": 0.9, "wide": 1.2, "double": 1.8}
            width = width_map.get(door_info.get("width", "standard"), 0.9)

            door_type = door_info.get("type", "interior")
            swing = "double" if door_type == "sliding" else "left"

            new_door = DoorOpening(position=pos, width=width, swing_direction=swing)
            new_door.door_type = door_type
            new_door.source = "gemini"
            model.doors.append(new_door)

        # Add Gemini-detected windows (with deduplication)
        for win_info in gemini_result.windows:
            wall_label = win_info.get("wall", "")
            target_wall = wall_directions.get(wall_label)
            if target_wall is None:
                continue

            position_label = win_info.get("position", "center")
            wall_vec = target_wall.end - target_wall.start
            t = {"left": 0.25, "center": 0.5, "right": 0.75}.get(position_label, 0.5)
            pos = target_wall.start + wall_vec * t

            is_duplicate = any(
                np.linalg.norm(pos - w.position) < 0.5 for w in model.windows
            )
            if is_duplicate:
                continue

            size_map = {
                "small": 0.6,
                "medium": 1.0,
                "large": 1.5,
                "floor_to_ceiling": 2.0,
            }
            width = size_map.get(win_info.get("size", "medium"), 1.0)

            new_window = WindowOpening(position=pos, width=width)
            new_window.source = "gemini"
            model.windows.append(new_window)

        return detection_result

    def _assign_walls_to_directions(self, walls):
        """Assign cardinal directions (north/south/east/west) to wall segments."""
        if not walls:
            return {}

        # Compute centroid of all wall midpoints
        midpoints = [(w.start + w.end) / 2 for w in walls]
        centroid = np.mean(midpoints, axis=0)

        direction_map = {}
        for wall in walls:
            mid = (wall.start + wall.end) / 2
            offset = mid - centroid
            wall_vec = wall.end - wall.start
            wall_angle = np.arctan2(wall_vec[1], wall_vec[0])

            # Determine if wall is primarily horizontal or vertical
            is_horizontal = abs(np.cos(wall_angle)) > abs(np.sin(wall_angle))

            if is_horizontal:
                label = "north" if offset[1] > 0 else "south"
            else:
                label = "east" if offset[0] > 0 else "west"

            # Keep the wall furthest from centroid for each direction
            if label not in direction_map:
                direction_map[label] = wall
            else:
                existing_mid = (
                    direction_map[label].start + direction_map[label].end
                ) / 2
                if np.linalg.norm(mid - centroid) > np.linalg.norm(
                    existing_mid - centroid
                ):
                    direction_map[label] = wall

        return direction_map

    def _summarize_measurements(
        self,
        detection_result: Optional[Dict],
        points: np.ndarray,
        scale_solution: Optional[Dict] = None,
    ) -> Dict:
        """
        Build top-level measurement summary from FloorPlanModel measurements.

        Falls back to a robust point-cloud bounding box if detection data is absent.
        """
        scale_meta = scale_solution or self.scale_solution or {}
        scale_factor = float(scale_meta.get("scale_factor", 1.0))
        scale_confidence = str(scale_meta.get("confidence", "unknown"))
        scale_method = str(scale_meta.get("mode", CALIBRATION_METHOD))
        scale_consistency = float(scale_meta.get("consistency", 0.0))
        scale_cues = [
            cue.get("name")
            for cue in scale_meta.get("cues", [])
            if isinstance(cue, dict) and cue.get("name")
        ]

        if detection_result and detection_result.get("measurements"):
            det_m = detection_result["measurements"]
            overall = det_m.get("overall", {})
            width_m = float(overall.get("width_m", 0.0))
            depth_m = float(overall.get("depth_m", 0.0))

            area_sqm = 0.0
            for room in det_m.get("rooms", []):
                area_sqm += float(room.get("area_sqm", 0.0))
            if area_sqm <= 0 and width_m > 0 and depth_m > 0:
                area_sqm = width_m * depth_m

            area_sqft = area_sqm * 10.764
            return {
                "width_m": round(width_m, 2),
                "depth_m": round(depth_m, 2),
                "area_sqm": round(area_sqm, 2),
                "area_sqft": round(area_sqft, 1),
                "scale_factor": round(scale_factor, 3),
                "scale_confidence": scale_confidence,
                "scale_method": scale_method,
                "scale_consistency": round(scale_consistency, 2),
                "scale_cues": scale_cues,
            }

        if len(points) == 0:
            return {
                "width_m": 0.0,
                "depth_m": 0.0,
                "area_sqm": 0.0,
                "area_sqft": 0.0,
                "scale_factor": round(scale_factor, 3),
                "scale_confidence": scale_confidence,
                "scale_method": scale_method,
                "scale_consistency": round(scale_consistency, 2),
                "scale_cues": scale_cues,
            }

        lo = np.percentile(points[:, [0, 2]], 2, axis=0)
        hi = np.percentile(points[:, [0, 2]], 98, axis=0)
        width_m = float(max(0.0, hi[0] - lo[0]))
        depth_m = float(max(0.0, hi[1] - lo[1]))
        area_sqm = width_m * depth_m
        return {
            "width_m": round(width_m, 2),
            "depth_m": round(depth_m, 2),
            "area_sqm": round(area_sqm, 2),
            "area_sqft": round(area_sqm * 10.764, 1),
            "scale_factor": round(scale_factor, 3),
            "scale_confidence": scale_confidence,
            "scale_method": scale_method,
            "scale_consistency": round(scale_consistency, 2),
            "scale_cues": scale_cues,
        }

    def _assess_plan_quality(
        self,
        detection_result: Optional[Dict],
        measurements: Dict,
        n_images: int,
        scale_solution: Optional[Dict] = None,
    ) -> Dict:
        """
        Assess floor-plan quality and gate confidence mode.

        Modes:
            - high_confidence
            - approximate
            - needs_more_images
        """
        scale_meta = scale_solution or self.scale_solution or {}
        scale_conf = str(measurements.get("scale_confidence", "low")).lower()
        scale_consistency = float(measurements.get("scale_consistency", 0.0))

        score = 1.0
        warnings = []
        next_steps = []

        if detection_result is None or detection_result.get("floor_plan_model") is None:
            return {
                "mode": "needs_more_images",
                "score": 0.0,
                "warnings": ["No stable room geometry could be reconstructed."],
                "recommended_next_steps": [
                    "Capture at least 6-10 photos with strong overlap and full wall coverage."
                ],
                "quality_flags": {},
                "scale_confidence": scale_meta.get("confidence", scale_conf),
                "industry_ready": False,
                "export_policy": "needs_more_images",
            }

        model = detection_result["floor_plan_model"]
        wall_count = len(model.walls)
        room_count = len(model.rooms)
        area_sqm = float(measurements.get("area_sqm", 0.0))
        detection_warnings = detection_result.get("measurements", {}).get("warnings", [])
        quality_flags = detection_result.get("quality_flags", {})
        opening_fusion = quality_flags.get("opening_fusion", {})
        wall_support = quality_flags.get("wall_multiview_support", {})
        closure_score = float(quality_flags.get("closure_score", 0.0))
        closure_mean_gap_m = float(quality_flags.get("closure_mean_gap_m", 0.0))

        if wall_count < 4:
            score -= 0.2
            warnings.append("Detected fewer than 4 stable walls.")
        if room_count < 1:
            score -= 0.35
            warnings.append("No closed room polygon was extracted.")
        if area_sqm < 2.0:
            score -= 0.15
            warnings.append("Estimated area is very small; geometry may be incomplete.")
        elif area_sqm > 120.0:
            score -= 0.1
            warnings.append("Estimated area is unusually large; scale may be unstable.")

        if scale_conf == "low":
            score -= 0.3
            warnings.append("Scale confidence is low.")
        elif scale_conf == "medium":
            score -= 0.12

        if scale_consistency < 0.55:
            score -= 0.1
            warnings.append("Scale cues are inconsistent.")

        if detection_warnings:
            score -= min(0.2, 0.05 * len(detection_warnings))
            warnings.extend(detection_warnings[:3])

        if quality_flags.get("used_rectangular_fallback"):
            score -= 0.2
            warnings.append("Used rectangular fallback due fragmented wall topology.")

        if quality_flags.get("used_sparse_wall_fallback"):
            score -= 0.1
            warnings.append("Used sparse-wall fallback because wall extraction was weak.")

        if closure_score < 0.65:
            score -= 0.12
            warnings.append("Wall graph closure is weak; corner connectivity is unstable.")
        elif closure_score < 0.82:
            score -= 0.05
            warnings.append("Wall closure is moderate; verify dimensions manually.")
        elif closure_score >= 0.94 and closure_mean_gap_m <= 0.08:
            score += 0.03

        if wall_support.get("enabled"):
            avg_support = float(wall_support.get("avg_support_views", 0.0))
            dropped_segments = int(wall_support.get("dropped_segments", 0))
            retained_segments = int(wall_support.get("retained_segments", wall_count))
            input_segments = int(wall_support.get("input_segments", retained_segments))
            min_support = int(wall_support.get("min_support_views", 1))
            retained_ratio = (
                float(retained_segments) / float(max(input_segments, 1))
                if input_segments > 0
                else 1.0
            )

            if avg_support < max(1.0, float(min_support) - 0.2):
                score -= 0.08
                warnings.append("Wall geometry has limited multi-view support.")
            if dropped_segments > 0 and retained_ratio < 0.7:
                score -= 0.05
                warnings.append("Many wall segments were rejected as cross-view inconsistent.")

        raw_opening_obs = int(
            opening_fusion.get("raw_door_observations", 0)
            + opening_fusion.get("raw_window_observations", 0)
        )
        fused_openings = int(
            opening_fusion.get("fused_doors", 0)
            + opening_fusion.get("fused_windows", 0)
        )
        max_support_views = int(opening_fusion.get("max_support_views", 0))

        if raw_opening_obs > 0 and fused_openings == 0:
            score -= 0.08
            warnings.append("Openings detected but not geometrically stable across views.")
        elif fused_openings > 0 and max_support_views < 2:
            score -= 0.05
            warnings.append("Openings rely on single-view evidence.")
        elif fused_openings > 0 and max_support_views >= 2:
            score += 0.03

        if n_images < 5:
            score -= 0.05
            warnings.append("Using fewer than 5 images reduces geometric stability.")

        score = float(np.clip(score, 0.0, 1.0))

        if score >= 0.75 and scale_conf in ("high", "medium"):
            mode = "high_confidence"
        elif score >= 0.45:
            mode = "approximate"
        else:
            mode = "needs_more_images"

        if mode == "needs_more_images":
            next_steps = [
                "Capture 6-10 images with 30-50% overlap around the room perimeter.",
                "Include full wall-floor intersections in each frame.",
                "Avoid extreme tilt and motion blur; keep lighting consistent.",
            ]
        elif mode == "approximate":
            next_steps = [
                "Add 2-4 additional corner-to-corner views to improve closure.",
                "Provide one known wall/door measurement to tighten scale.",
                "Manually review door/window placement before sharing externally.",
            ]

        industry_ready = (
            mode == "high_confidence"
            and score >= 0.82
            and scale_conf == "high"
            and not quality_flags.get("used_rectangular_fallback")
        )
        if industry_ready:
            export_policy = "normal_export"
        elif mode in ("high_confidence", "approximate"):
            export_policy = "annotate_as_approximate"
        else:
            export_policy = "needs_more_images"

        return {
            "mode": mode,
            "score": round(score, 2),
            "warnings": warnings,
            "recommended_next_steps": next_steps,
            "quality_flags": quality_flags,
            "scale_confidence": scale_meta.get("confidence", scale_conf),
            "industry_ready": bool(industry_ready),
            "export_policy": export_policy,
        }

    @staticmethod
    def _attach_quality_to_model(model, quality: Optional[Dict]) -> None:
        """Attach quality/export metadata to FloorPlanModel for renderers."""
        if model is None or not quality:
            return
        model.quality_mode = str(quality.get("mode", ""))
        model.export_policy = str(quality.get("export_policy", ""))
        model.industry_ready = bool(quality.get("industry_ready", False))
        warnings = quality.get("warnings", [])
        model.quality_warnings = [str(w) for w in warnings[:5]]

    def reconstruct(self, image_paths: List[str], progress_callback=None) -> Dict:
        """
        Perform full room reconstruction from multiple images.

        Args:
            image_paths: List of paths to room images (4-5 recommended)
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary containing all reconstruction results
        """
        if not image_paths:
            raise ValueError("No images provided")

        n_images = len(image_paths)
        print(
            colored(
                f"\n[RoomReconstructor] Starting reconstruction with {n_images} images...",
                "cyan",
            )
        )

        # Load images
        images = []
        for img_path in image_paths:
            try:
                images.append(self.load_image(img_path))
            except Exception as e:
                print(
                    colored(
                        f"[RoomReconstructor] Failed to load {img_path}: {e}", "red"
                    )
                )

        if len(images) < 2:
            raise ValueError("Need at least 2 valid images")

        # Start Gemini analysis in parallel (non-blocking)
        from concurrent.futures import ThreadPoolExecutor

        gemini_future = None
        gemini_executor = None
        if self.scene_analyzer is not None:
            gemini_executor = ThreadPoolExecutor(max_workers=1)
            gemini_future = gemini_executor.submit(self._run_gemini_analysis, images)

        # --- VGGT path (primary) ---
        vggt_data = self._reconstruct_with_vggt(images, progress_callback)

        if vggt_data is not None:
            combined_points = vggt_data["combined_points"]
            combined_colors = vggt_data["combined_colors"]
            all_depths = vggt_data["all_depths"]
            sfm_result = vggt_data["sfm_result"]
            raw_count = vggt_data["raw_count"]
        else:
            # --- Legacy path ---
            sfm_result = None
            if self.use_sfm and len(images) >= SFM_MIN_IMAGES:
                if progress_callback:
                    progress_callback(0.1, "Running Structure-from-Motion...")

                print(colored("[RoomReconstructor] Running SfM pipeline...", "cyan"))
                sfm_result = self.sfm_processor.run_sfm(images, progress_callback=None)

                if sfm_result["success"]:
                    print(
                        colored(
                            f"[RoomReconstructor] SfM registered {sfm_result['num_registered']}/{len(images)} images",
                            "green",
                        )
                    )
                else:
                    print(
                        colored(
                            "[RoomReconstructor] SfM failed, falling back to depth-only reconstruction",
                            "yellow",
                        )
                    )

            all_depths = []
            views = []

            for i, image in enumerate(images):
                if progress_callback:
                    progress_callback(
                        0.3 + 0.3 * (i / len(images)),
                        f"Processing image {i+1}/{len(images)}...",
                    )

                print(
                    colored(
                        f"[RoomReconstructor] Processing image {i+1}/{len(images)}...",
                        "cyan",
                    )
                )

                try:
                    intrinsics = None
                    if sfm_result and sfm_result["success"]:
                        intrinsics = sfm_result["camera_intrinsics"].get(i)

                    depth, points, colors = self.process_single_image(
                        image, camera_intrinsics=intrinsics
                    )

                    all_depths.append(depth)
                    views.append((points, colors, i))

                    print(colored(f"  Generated {len(points):,} 3D points", "green"))

                except Exception as e:
                    print(colored(f"  ERROR: {str(e)}", "red"))
                    continue

            if not views:
                raise ValueError("No valid views generated")

            if progress_callback:
                progress_callback(0.7, "Combining point clouds...")

            combined_points = None
            combined_colors = None

            if sfm_result and sfm_result["success"] and self.use_tsdf:
                print(colored("[RoomReconstructor] Trying TSDF fusion...", "cyan"))
                combined_points, combined_colors = self._fuse_with_tsdf(
                    images, all_depths, sfm_result, progress_callback
                )

            if combined_points is None or len(combined_points) == 0:
                if sfm_result and sfm_result["success"]:
                    print(
                        colored(
                            "[RoomReconstructor] Aligning views using SfM poses...",
                            "cyan",
                        )
                    )
                    combined_points, combined_colors = self._align_with_sfm(
                        views, sfm_result, progress_callback
                    )
                else:
                    print(
                        colored(
                            "[RoomReconstructor] Using legacy registration...",
                            "yellow",
                        )
                    )
                    legacy_views = [(p, c) for p, c, _ in views]
                    (
                        combined_points,
                        combined_colors,
                    ) = self._register_point_clouds_legacy(
                        legacy_views, progress_callback
                    )

            raw_count = len(combined_points)
            combined_points, combined_colors = self._postprocess_point_cloud(
                combined_points, combined_colors
            )

            if len(combined_points) == 0:
                raise ValueError("No valid 3D points generated from images")

            pose_for_scale = (
                sfm_result.get("camera_poses")
                if sfm_result and sfm_result.get("success")
                else None
            )
            combined_points, scale_factor, _ = self._calibrate_reconstruction_scale(
                combined_points,
                camera_poses=pose_for_scale,
                relative_scale_required=False,
            )
            if sfm_result and sfm_result.get("success"):
                if abs(scale_factor - 1.0) > 1e-6:
                    for pose_data in sfm_result["camera_poses"].values():
                        pose = pose_data.get("transform")
                        if pose is not None and pose.shape == (4, 4):
                            pose[:3, 3] *= scale_factor
                combined_points, sfm_result["camera_poses"] = (
                    self._canonicalize_world_frame(
                        combined_points, sfm_result.get("camera_poses")
                    )
                )

        print(
            colored(
                f"[RoomReconstructor] Total points: {len(combined_points):,} (raw: {raw_count:,})",
                "green",
            )
        )

        # Run new wall detection pipeline
        detection_result = None
        vggt_intrinsics = vggt_data.get("camera_intrinsics") if vggt_data else None
        vggt_poses = (
            vggt_data["sfm_result"]["camera_poses"]
            if vggt_data and vggt_data.get("sfm_result")
            else (
                sfm_result.get("camera_poses")
                if sfm_result and sfm_result.get("success")
                else None
            )
        )
        detection_images = (
            vggt_data.get("input_images", images) if vggt_data else images
        )
        if self.use_metric_depth or vggt_data is not None:
            detection_result = self._detect_walls_and_rooms(
                combined_points,
                all_depths,
                detection_images,
                camera_intrinsics=vggt_intrinsics,
                camera_poses=vggt_poses,
            )

        # Collect Gemini result and merge
        gemini_result = None
        if gemini_future is not None:
            try:
                gemini_result = gemini_future.result(timeout=GEMINI_TIMEOUT)
            except Exception as e:
                print(colored(f"[RoomReconstructor] Gemini timed out: {e}", "yellow"))
            finally:
                if gemini_executor:
                    gemini_executor.shutdown(wait=False)

        if gemini_result is not None and detection_result is not None:
            detection_result = self._merge_gemini_into_model(
                detection_result, gemini_result
            )

        model = detection_result.get("floor_plan_model") if detection_result else None
        if model is None or len(model.walls) < 3:
            raise ValueError(
                "Could not extract a usable floor plan geometry from these images."
            )

        if progress_callback:
            progress_callback(0.8, "Rendering floor plan and 3D model...")

        # Create visualizations
        if progress_callback:
            progress_callback(0.9, "Creating visualizations...")

        print(colored("[RoomReconstructor] Creating visualizations...", "cyan"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plotly_fig = self.visualizer.create_plotly_visualization(
            combined_points, combined_colors, title="3D Room Reconstruction"
        )

        html_path = os.path.join(OUTPUT_DIR, f"room_3d_{timestamp}.html")
        self.visualizer.export_html(combined_points, combined_colors, html_path)

        ply_path = os.path.join(OUTPUT_DIR, f"room_pointcloud_{timestamp}.ply")
        self.visualizer.save_point_cloud(combined_points, combined_colors, ply_path)

        measurements = self._summarize_measurements(
            detection_result,
            combined_points,
            scale_solution=self.scale_solution,
        )
        quality = self._assess_plan_quality(
            detection_result=detection_result,
            measurements=measurements,
            n_images=n_images,
            scale_solution=self.scale_solution,
        )
        self._attach_quality_to_model(model, quality)

        # Render FloorPlanModel using new renderers (SVG, DXF, PNG)
        render_outputs = self._render_floor_plan_model(detection_result, timestamp)
        floor_plan_path = render_outputs.get("floor_plan_arch_png")
        if not floor_plan_path or not os.path.exists(floor_plan_path):
            raise ValueError("Architectural floor plan PNG rendering failed.")
        floor_plan_fig = None

        # Try to generate mesh (optional, for better visualization)
        mesh = None
        mesh_path = None
        try:
            if len(combined_points) > 500:
                print(colored("[RoomReconstructor] Generating mesh...", "cyan"))
                mesh = self.visualizer.create_mesh_from_points(
                    combined_points, combined_colors, method="poisson"
                )
                if mesh is not None:
                    mesh_path = os.path.join(OUTPUT_DIR, f"room_mesh_{timestamp}.ply")
                    self.visualizer.save_mesh(mesh, mesh_path)
        except Exception as e:
            print(
                colored(f"[RoomReconstructor] Mesh generation skipped: {e}", "yellow")
            )

        # Compile results
        backend = "vggt" if vggt_data is not None else "legacy"
        result = {
            "success": True,
            "backend": backend,
            "num_images": n_images,
            "num_vggt_images": (
                int(vggt_data.get("num_vggt_images", n_images)) if vggt_data else 0
            ),
            "num_points": len(combined_points),
            "num_points_raw": raw_count,
            "sfm_success": sfm_result["success"] if sfm_result else False,
            "sfm_registered": sfm_result["num_registered"] if sfm_result else 0,
            "is_metric": self.use_metric_depth or backend == "vggt",
            "measurements": measurements,
            "quality": quality,
            "outputs": {
                "floor_plan_image": floor_plan_path,
                "html_3d_model": html_path,
                "point_cloud_ply": ply_path,
                "mesh_ply": mesh_path,
                **render_outputs,
            },
            "data": {
                "points": combined_points,
                "colors": combined_colors,
                "depth_maps": all_depths,
                "mesh": mesh,
                "detection": detection_result,
            },
            "figures": {
                "floor_plan": floor_plan_fig,
                "plotly_3d": plotly_fig,
            },
        }
        if gemini_result is not None and gemini_result.success:
            result["gemini_analysis"] = {
                "room_type": gemini_result.room_type,
                "room_shape": gemini_result.room_shape,
                "confidence": gemini_result.confidence,
                "door_count": len(gemini_result.doors),
                "window_count": len(gemini_result.windows),
                "features": gemini_result.features,
            }

        self.last_result = result

        if progress_callback:
            progress_callback(1.0, "Complete!")

        print(colored("\n" + "=" * 50, "green"))
        print(colored("[RoomReconstructor] RECONSTRUCTION COMPLETE!", "green"))
        print(colored("=" * 50, "green"))
        print(f"\nMeasurements (approximate):")
        print(
            f"  - Width:  {result['measurements']['width_m']:.2f} m ({result['measurements']['width_m']*3.28:.1f} ft)"
        )
        print(
            f"  - Depth:  {result['measurements']['depth_m']:.2f} m ({result['measurements']['depth_m']*3.28:.1f} ft)"
        )
        print(
            f"  - Area:   {result['measurements']['area_sqm']:.1f} m² ({result['measurements']['area_sqft']:.0f} sq ft)"
        )
        print(
            f"  - Quality: {result['quality']['mode']} (score={result['quality']['score']:.2f})"
        )
        if result["quality"]["warnings"]:
            print("  - Warnings:")
            for warning in result["quality"]["warnings"][:3]:
                print(f"    • {warning}")
        if sfm_result and sfm_result["success"]:
            print(f"\nSfM: Registered {sfm_result['num_registered']}/{n_images} images")
        print(f"\nOutputs saved to: {OUTPUT_DIR}")
        print(colored("=" * 50, "green"))

        return result

    def reconstruct_from_arrays(
        self, images: List[np.ndarray], progress_callback=None
    ) -> Dict:
        """
        Reconstruct from image arrays (for Gradio interface).

        Args:
            images: List of RGB images as numpy arrays
            progress_callback: Optional progress callback

        Returns:
            Reconstruction results dictionary
        """
        if not images:
            raise ValueError("No images provided")

        # Filter out None images
        valid_images = []
        for img in images:
            if img is not None:
                if img.max() > 1:
                    img = img.astype(np.uint8)
                else:
                    img = (img * 255).astype(np.uint8)
                valid_images.append(img)

        if len(valid_images) < 2:
            return {
                "success": False,
                "error": "Need at least 2 valid images for reconstruction.",
            }

        n_images = len(valid_images)
        print(colored(f"\n[RoomReconstructor] Processing {n_images} images...", "cyan"))

        # Start Gemini analysis in parallel (non-blocking)
        from concurrent.futures import ThreadPoolExecutor

        gemini_future = None
        gemini_executor = None
        if self.scene_analyzer is not None:
            gemini_executor = ThreadPoolExecutor(max_workers=1)
            gemini_future = gemini_executor.submit(
                self._run_gemini_analysis, valid_images
            )

        # --- VGGT path (primary) ---
        vggt_data = self._reconstruct_with_vggt(valid_images, progress_callback)

        if vggt_data is not None:
            combined_points = vggt_data["combined_points"]
            combined_colors = vggt_data["combined_colors"]
            all_depths = vggt_data["all_depths"]
            sfm_result = vggt_data["sfm_result"]
            raw_count = vggt_data["raw_count"]
        else:
            # --- Legacy path ---
            sfm_result = None
            if self.use_sfm and n_images >= SFM_MIN_IMAGES:
                if progress_callback:
                    progress_callback(0.1, "Running Structure-from-Motion...")

                print(colored("[RoomReconstructor] Running SfM pipeline...", "cyan"))
                sfm_result = self.sfm_processor.run_sfm(
                    valid_images, progress_callback=None
                )

            all_depths = []
            views = []

            for i, image in enumerate(valid_images):
                if progress_callback:
                    progress_callback(
                        0.2 + 0.4 * (i / n_images),
                        f"Processing image {i+1}/{n_images}",
                    )

                print(
                    colored(
                        f"[RoomReconstructor] Processing image {i+1}/{n_images}...",
                        "cyan",
                    )
                )

                try:
                    intrinsics = None
                    if sfm_result and sfm_result["success"]:
                        intrinsics = sfm_result["camera_intrinsics"].get(i)

                    depth, points, colors = self.process_single_image(
                        image, camera_intrinsics=intrinsics
                    )
                    all_depths.append(depth)
                    views.append((points, colors, i))
                    print(colored(f"  Generated {len(points):,} points", "green"))

                except Exception as e:
                    print(colored(f"  ERROR: {str(e)}", "red"))
                    continue

            if not views:
                return {
                    "success": False,
                    "error": "No valid 3D points generated. Please check your images.",
                }

            if progress_callback:
                progress_callback(0.7, "Combining views...")

            combined_points = None
            combined_colors = None

            if sfm_result and sfm_result["success"] and self.use_tsdf:
                combined_points, combined_colors = self._fuse_with_tsdf(
                    valid_images, all_depths, sfm_result, progress_callback
                )

            if combined_points is None or len(combined_points) == 0:
                if sfm_result and sfm_result["success"]:
                    combined_points, combined_colors = self._align_with_sfm(
                        views, sfm_result, progress_callback
                    )
                if combined_points is None or len(combined_points) == 0:
                    legacy_views = [(p, c) for p, c, _ in views]
                    (
                        combined_points,
                        combined_colors,
                    ) = self._register_point_clouds_legacy(
                        legacy_views, progress_callback
                    )

            if combined_points is None:
                return {
                    "success": False,
                    "error": "Point cloud registration failed. Please try different images.",
                }
            raw_count = len(combined_points)
            combined_points, combined_colors = self._postprocess_point_cloud(
                combined_points, combined_colors
            )

            if len(combined_points) == 0:
                return {
                    "success": False,
                    "error": "No valid 3D points generated. Please check your images.",
                }

            pose_for_scale = (
                sfm_result.get("camera_poses")
                if sfm_result and sfm_result.get("success")
                else None
            )
            combined_points, scale_factor, _ = self._calibrate_reconstruction_scale(
                combined_points,
                camera_poses=pose_for_scale,
                relative_scale_required=False,
            )
            if sfm_result and sfm_result.get("success"):
                if abs(scale_factor - 1.0) > 1e-6:
                    for pose_data in sfm_result["camera_poses"].values():
                        pose = pose_data.get("transform")
                        if pose is not None and pose.shape == (4, 4):
                            pose[:3, 3] *= scale_factor
                combined_points, sfm_result["camera_poses"] = (
                    self._canonicalize_world_frame(
                        combined_points, sfm_result.get("camera_poses")
                    )
                )

        # Run new wall detection pipeline
        detection_result = None
        vggt_intrinsics = vggt_data.get("camera_intrinsics") if vggt_data else None
        vggt_poses = (
            vggt_data["sfm_result"]["camera_poses"]
            if vggt_data and vggt_data.get("sfm_result")
            else (
                sfm_result.get("camera_poses")
                if sfm_result and sfm_result.get("success")
                else None
            )
        )
        detection_images = (
            vggt_data.get("input_images", valid_images) if vggt_data else valid_images
        )
        if self.use_metric_depth or vggt_data is not None:
            detection_result = self._detect_walls_and_rooms(
                combined_points,
                all_depths,
                detection_images,
                camera_intrinsics=vggt_intrinsics,
                camera_poses=vggt_poses,
            )

        # Collect Gemini result and merge
        gemini_result = None
        if gemini_future is not None:
            try:
                gemini_result = gemini_future.result(timeout=GEMINI_TIMEOUT)
            except Exception as e:
                print(colored(f"[RoomReconstructor] Gemini timed out: {e}", "yellow"))
            finally:
                if gemini_executor:
                    gemini_executor.shutdown(wait=False)

        if gemini_result is not None and detection_result is not None:
            detection_result = self._merge_gemini_into_model(
                detection_result, gemini_result
            )

        model = detection_result.get("floor_plan_model") if detection_result else None
        if model is None or len(model.walls) < 3:
            return {
                "success": False,
                "error": "Could not extract a usable floor plan geometry from these images.",
            }

        if progress_callback:
            progress_callback(0.8, "Rendering floor plan and 3D model...")

        # Generate outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backend = "vggt" if vggt_data is not None else "legacy"

        plotly_fig = self.visualizer.create_plotly_visualization(
            combined_points, combined_colors
        )

        html_path = os.path.join(OUTPUT_DIR, f"room_3d_{timestamp}.html")
        self.visualizer.export_html(combined_points, combined_colors, html_path)

        measurements = self._summarize_measurements(
            detection_result,
            combined_points,
            scale_solution=self.scale_solution,
        )
        quality = self._assess_plan_quality(
            detection_result=detection_result,
            measurements=measurements,
            n_images=n_images,
            scale_solution=self.scale_solution,
        )
        self._attach_quality_to_model(model, quality)

        # Render FloorPlanModel using new renderers (SVG, DXF, PNG)
        render_outputs = self._render_floor_plan_model(detection_result, timestamp)
        floor_plan_path = render_outputs.get("floor_plan_arch_png")
        if not floor_plan_path or not os.path.exists(floor_plan_path):
            return {
                "success": False,
                "error": "Architectural floor plan PNG rendering failed.",
            }
        floor_plan_fig = None

        if progress_callback:
            progress_callback(1.0, "Complete!")

        result = {
            "success": True,
            "backend": backend,
            "num_images": n_images,
            "num_vggt_images": (
                int(vggt_data.get("num_vggt_images", n_images)) if vggt_data else 0
            ),
            "num_points": len(combined_points),
            "num_points_raw": raw_count,
            "sfm_success": sfm_result["success"] if sfm_result else False,
            "sfm_registered": sfm_result["num_registered"] if sfm_result else 0,
            "is_metric": self.use_metric_depth or backend == "vggt",
            "measurements": measurements,
            "quality": quality,
            "outputs": {
                "floor_plan_image": floor_plan_path,
                "html_3d_model": html_path,
                **render_outputs,
            },
            "data": {
                "points": combined_points,
                "colors": combined_colors,
                "depth_maps": all_depths,
                "detection": detection_result,
            },
            "figures": {
                "floor_plan": floor_plan_fig,
                "plotly_3d": plotly_fig,
            },
        }
        if gemini_result is not None and gemini_result.success:
            result["gemini_analysis"] = {
                "room_type": gemini_result.room_type,
                "room_shape": gemini_result.room_shape,
                "confidence": gemini_result.confidence,
                "door_count": len(gemini_result.doors),
                "window_count": len(gemini_result.windows),
                "features": gemini_result.features,
            }
        return result


# Command-line interface
if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Reconstruct a room from photographs")
    parser.add_argument(
        "images", nargs="+", help="Paths to room images (4-5 recommended)"
    )
    parser.add_argument(
        "--room-width",
        type=float,
        default=4.0,
        help="Assumed room width in meters (default: 4.0)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Open interactive 3D visualization"
    )
    parser.add_argument(
        "--no-sfm", action="store_true", help="Disable Structure-from-Motion"
    )

    args = parser.parse_args()

    # Expand glob patterns
    image_paths = []
    for pattern in args.images:
        image_paths.extend(glob.glob(pattern))

    if not image_paths:
        print("ERROR: No valid image files found!")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    # Run reconstruction
    reconstructor = RoomReconstructor(assumed_room_width=args.room_width)
    if args.no_sfm:
        reconstructor.use_sfm = False

    result = reconstructor.reconstruct(image_paths)

    # Optionally open 3D visualization
    if args.visualize:
        reconstructor.visualizer.visualize_open3d(
            result["data"]["points"], result["data"]["colors"]
        )
