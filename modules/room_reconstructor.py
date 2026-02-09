"""Room reconstruction orchestrator with compliance-aware accurate mode."""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACCURATE_MODE_DEFAULT,
    ASSUMED_ROOM_WIDTH_METERS,
    CAMERA_FX,
    CAMERA_FY,
    DIAGNOSTIC_MIN_REGISTRATION_RATIO,
    DEFAULT_COMPLIANCE_PROFILE,
    ENABLE_REGISTRATION,
    ENABLE_SFM,
    OUTLIER_NB_NEIGHBORS,
    OUTLIER_STD_RATIO,
    OUTPUT_DIR,
    POINT_CLOUD_DENSITY,
    REGISTRATION_ICP_ITERATIONS,
    REGISTRATION_RANSAC_ITERATIONS,
    REGISTRATION_VOXEL_SIZE,
    SFM_MIN_IMAGES,
    VOXEL_SIZE,
)
from modules.calibration import compute_scale_factor, parse_calibration_input, scale_points
from modules.compliance_profile import get_compliance_profile
from modules.dense_reconstructor import DenseReconstructor
from modules.depth_estimator import DepthEstimator
from modules.dxf_exporter import DXFExporter
from modules.floor_plan_generator import FloorPlanGenerator
from modules.qa_report import build_qa_report, save_qa_report
from modules.quality_gate import evaluate_capture_quality
from modules.sfm_processor import SfMProcessor
from modules.visualizer_3d import Visualizer3D


class RoomReconstructor:
    """Main orchestration layer for room reconstruction."""

    def __init__(self, assumed_room_width: float = ASSUMED_ROOM_WIDTH_METERS):
        print(colored("[RoomReconstructor] Initializing components...", "cyan"))

        self.depth_estimator = DepthEstimator()
        self.floor_plan_gen = FloorPlanGenerator(assumed_width=assumed_room_width)
        self.visualizer = Visualizer3D()
        self.sfm_processor = SfMProcessor()
        self.dense_reconstructor = DenseReconstructor()
        self.dxf_exporter = DXFExporter()

        self.assumed_room_width = assumed_room_width
        self.last_result: Optional[Dict] = None
        self.sessions: Dict[str, Dict] = {}

        self.use_sfm = ENABLE_SFM and self.sfm_processor.enabled
        self.use_tsdf = True

        if self.use_sfm:
            print(colored("[RoomReconstructor] SfM enabled", "green"))
        else:
            print(colored("[RoomReconstructor] SfM disabled, fallback mode", "yellow"))

        print(colored("[RoomReconstructor] Initialization complete!", "green"))

    def load_image(self, image_path: str) -> np.ndarray:
        """Load RGB image from path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def process_single_image(
        self,
        image: np.ndarray,
        sample_rate: int = POINT_CLOUD_DENSITY,
        camera_intrinsics: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate depth and convert to a per-view point cloud."""
        depth = self.depth_estimator.estimate_depth(image)

        if camera_intrinsics:
            fx = camera_intrinsics.get("fx", CAMERA_FX)
            fy = camera_intrinsics.get("fy", CAMERA_FY)
        else:
            fx, fy = CAMERA_FX, CAMERA_FY

        points, colors = self.depth_estimator.depth_to_3d_points(
            image,
            depth,
            fx=fx,
            fy=fy,
            sample_rate=sample_rate,
        )

        return depth, points, colors

    def _make_open3d_pcd(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    def _apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        pts_t = (transform @ pts_h.T).T
        return pts_t[:, :3]

    def _align_with_sfm(
        self,
        views: List[Tuple[np.ndarray, np.ndarray, int]],
        sfm_result: Dict,
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align depth-derived views with SfM camera poses."""
        if not sfm_result.get("success", False):
            return self._fallback_alignment(views)

        camera_poses = sfm_result.get("camera_poses", {})
        combined_points = []
        combined_colors = []

        for i, (points, colors, img_idx) in enumerate(views):
            if progress_callback:
                progress_callback(0.62 + 0.1 * (i / max(len(views), 1)), f"Aligning view {i+1}...")

            if img_idx in camera_poses:
                pose = camera_poses[img_idx]["transform"]
                aligned_points = self._apply_transform(points, pose)
            else:
                aligned_points = points

            combined_points.append(aligned_points)
            combined_colors.append(colors)

        sparse_points = sfm_result.get("sparse_points", np.array([]))
        sparse_colors = sfm_result.get("sparse_colors", np.array([]))
        if len(sparse_points) > 0:
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
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """TSDF fusion when SfM poses exist."""
        if not sfm_result.get("success", False):
            return None, None

        camera_poses = sfm_result.get("camera_poses", {})
        camera_intrinsics = sfm_result.get("camera_intrinsics", {})

        try:
            points, colors = self.dense_reconstructor.fuse_tsdf(
                images,
                depths,
                camera_poses,
                camera_intrinsics,
                progress_callback,
            )
            if len(points) > 0:
                return points, colors
        except Exception as exc:
            print(colored(f"[RoomReconstructor] TSDF fusion failed: {exc}", "yellow"))

        return None, None

    def _fallback_alignment(
        self,
        views: List[Tuple[np.ndarray, np.ndarray, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Centroid fallback alignment."""
        if not views:
            return np.array([]), np.array([])

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
                aligned = points + (reference_centroid - centroid)

            combined_points.append(aligned)
            combined_colors.append(colors)

        pts = np.vstack(combined_points) if combined_points else np.array([])
        cols = np.vstack(combined_colors) if combined_colors else np.array([])
        return pts, cols

    def _prepare_registration(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        voxel_size: float,
    ):
        """Downsample and compute features for registration."""
        import open3d as o3d

        pcd = self._make_open3d_pcd(points, colors)
        cleaned = pcd.remove_non_finite_points()
        pcd = cleaned[0] if isinstance(cleaned, tuple) else cleaned

        pcd_down = pcd.voxel_down_sample(voxel_size)
        if len(pcd_down.points) < 30:
            return pcd_down, None

        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.0,
                max_nn=30,
            )
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
        )
        return pcd_down, fpfh

    def _register_point_clouds_legacy(
        self,
        views: List[Tuple[np.ndarray, np.ndarray]],
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FPFH + ICP fallback registration."""
        if not views:
            return np.array([]), np.array([])
        if len(views) == 1 or not ENABLE_REGISTRATION:
            points = np.vstack([v[0] for v in views])
            colors = np.vstack([v[1] for v in views]) if views[0][1] is not None else None
            return points, colors

        import open3d as o3d

        voxel_size = REGISTRATION_VOXEL_SIZE
        target_points, target_colors = views[0]
        combined_points = [target_points]
        combined_colors = [target_colors]

        for i in range(1, len(views)):
            if progress_callback:
                progress_callback(
                    0.62 + 0.1 * (i / max(len(views), 2)),
                    f"Registering view {i+1}/{len(views)}...",
                )

            source_points, source_colors = views[i]

            target_stack = np.vstack(combined_points)
            target_colors_stack = np.vstack(combined_colors) if combined_colors[0] is not None else None

            target_down, target_fpfh = self._prepare_registration(target_stack, target_colors_stack, voxel_size)
            source_down, source_fpfh = self._prepare_registration(source_points, source_colors, voxel_size)

            if source_fpfh is None or target_fpfh is None:
                transform = np.eye(4)
                if len(source_points) > 0 and len(target_stack) > 0:
                    transform[:3, 3] = target_stack.mean(axis=0) - source_points.mean(axis=0)
            else:
                dist_thresh = voxel_size * 1.5
                result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_down,
                    target_down,
                    source_fpfh,
                    target_fpfh,
                    True,
                    dist_thresh,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    4,
                    [
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
                    ],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(
                        REGISTRATION_RANSAC_ITERATIONS,
                        0.999,
                    ),
                )

                if result_ransac.fitness < 0.05:
                    transform = np.eye(4)
                    transform[:3, 3] = target_stack.mean(axis=0) - source_points.mean(axis=0)
                else:
                    result_icp = o3d.pipelines.registration.registration_icp(
                        source_down,
                        target_down,
                        dist_thresh * 0.6,
                        result_ransac.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=REGISTRATION_ICP_ITERATIONS),
                    )
                    transform = result_icp.transformation

            aligned_points = self._apply_transform(source_points, transform)
            combined_points.append(aligned_points)
            combined_colors.append(source_colors)

        points = np.vstack(combined_points)
        colors = np.vstack(combined_colors) if combined_colors[0] is not None else None
        return points, colors

    def _postprocess_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Remove outliers + downsample."""
        if len(points) == 0:
            return points, colors

        import open3d as o3d

        pcd = self._make_open3d_pcd(points, colors)
        cleaned = pcd.remove_non_finite_points()
        pcd = cleaned[0] if isinstance(cleaned, tuple) else cleaned

        if OUTLIER_NB_NEIGHBORS > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=OUTLIER_NB_NEIGHBORS,
                std_ratio=OUTLIER_STD_RATIO,
            )

        if VOXEL_SIZE > 0:
            pcd = pcd.voxel_down_sample(VOXEL_SIZE)

        points_clean = np.asarray(pcd.points)
        colors_clean = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points_clean, colors_clean

    def _run_geometry_pipeline(self, images: List[np.ndarray], progress_callback=None) -> Dict:
        """Run geometry reconstruction pipeline and return point cloud + sfm data."""
        n_images = len(images)
        sfm_result = None
        all_depths: List[np.ndarray] = []
        geometry_source = "none"

        if self.use_sfm and n_images >= SFM_MIN_IMAGES:
            if progress_callback:
                progress_callback(0.1, "Running Structure-from-Motion...")
            sfm_result = self.sfm_processor.run_sfm(images, progress_callback=None)

        if sfm_result and sfm_result.get("success"):
            dense_points = sfm_result.get("dense_points", np.array([]))
            dense_colors = sfm_result.get("dense_colors", np.array([]))
            if len(dense_points) > 0:
                points = dense_points
                colors = dense_colors if len(dense_colors) == len(dense_points) else None
                geometry_source = "sfm_dense"
                points, colors = self._postprocess_point_cloud(points, colors)
                return {
                    "points": points,
                    "colors": colors,
                    "sfm_result": sfm_result,
                    "all_depths": all_depths,
                    "geometry_source": geometry_source,
                }

        views: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(0.2 + 0.35 * (i / max(len(images), 1)), f"Processing image {i+1}/{len(images)}")

            intrinsics = None
            if sfm_result and sfm_result.get("success"):
                intrinsics = sfm_result.get("camera_intrinsics", {}).get(i)

            depth, points, colors = self.process_single_image(image, camera_intrinsics=intrinsics)
            all_depths.append(depth)
            views.append((points, colors, i))

        if not views:
            raise ValueError("No valid views generated")

        combined_points = None
        combined_colors = None

        if sfm_result and sfm_result.get("success") and self.use_tsdf:
            combined_points, combined_colors = self._fuse_with_tsdf(images, all_depths, sfm_result, progress_callback)
            if combined_points is not None and len(combined_points) > 0:
                geometry_source = "depth_tsdf"

        if combined_points is None or len(combined_points) == 0:
            if sfm_result and sfm_result.get("success"):
                combined_points, combined_colors = self._align_with_sfm(views, sfm_result, progress_callback)
                geometry_source = "depth_sfm_aligned"
            else:
                legacy_views = [(p, c) for p, c, _ in views]
                combined_points, combined_colors = self._register_point_clouds_legacy(legacy_views, progress_callback)
                geometry_source = "depth_legacy_registration"

        if combined_points is None or len(combined_points) == 0:
            raise ValueError("No combined point cloud generated")

        combined_points, combined_colors = self._postprocess_point_cloud(combined_points, combined_colors)
        return {
            "points": combined_points,
            "colors": combined_colors,
            "sfm_result": sfm_result,
            "all_depths": all_depths,
            "geometry_source": geometry_source,
        }

    def _create_visual_outputs(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        floor_plan_data: Dict,
        title: str,
    ) -> Dict:
        """Generate floor plan PNG, 3D HTML, and PLY outputs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        floor_plan_path = os.path.join(OUTPUT_DIR, f"floor_plan_{timestamp}.png")
        floor_plan_fig = self.floor_plan_gen.create_floor_plan_image(
            floor_plan_data,
            output_path=floor_plan_path,
            title=title,
        )

        plotly_fig = self.visualizer.create_plotly_visualization(
            points,
            colors,
            title="3D Room Reconstruction",
        )

        html_path = os.path.join(OUTPUT_DIR, f"room_3d_{timestamp}.html")
        self.visualizer.export_html(points, colors, html_path)

        ply_path = os.path.join(OUTPUT_DIR, f"room_pointcloud_{timestamp}.ply")
        self.visualizer.save_point_cloud(points, colors, ply_path)

        return {
            "timestamp": timestamp,
            "outputs": {
                "floor_plan_image": floor_plan_path,
                "html_3d_model": html_path,
                "point_cloud_ply": ply_path,
            },
            "figures": {
                "floor_plan": floor_plan_fig,
                "plotly_3d": plotly_fig,
            },
        }

    def reconstruct(
        self,
        image_paths: List[str],
        progress_callback=None,
        compliance_profile: str = DEFAULT_COMPLIANCE_PROFILE,
        accurate_mode: bool = ACCURATE_MODE_DEFAULT,
        diagnostic_mode: bool = False,
        calibration_input: Optional[Dict] = None,
    ) -> Dict:
        """Reconstruct from image paths."""
        images = [self.load_image(path) for path in image_paths]
        return self.reconstruct_from_arrays(
            images,
            progress_callback=progress_callback,
            compliance_profile=compliance_profile,
            accurate_mode=accurate_mode,
            diagnostic_mode=diagnostic_mode,
            calibration_input=calibration_input,
        )

    def reconstruct_from_arrays(
        self,
        images: List[np.ndarray],
        progress_callback=None,
        compliance_profile: str = DEFAULT_COMPLIANCE_PROFILE,
        accurate_mode: bool = ACCURATE_MODE_DEFAULT,
        diagnostic_mode: bool = False,
        calibration_input: Optional[Dict] = None,
    ) -> Dict:
        """Reconstruct from RGB arrays with accurate-mode two-pass support."""
        if not images:
            return {"success": False, "status": "FAIL", "error": "No images provided."}

        valid_images = []
        for img in images:
            if img is None:
                continue
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            valid_images.append(img)

        if len(valid_images) < 2:
            return {
                "success": False,
                "status": "FAIL",
                "error": "Need at least 2 valid images for reconstruction.",
            }

        profile = get_compliance_profile(compliance_profile)

        if progress_callback:
            progress_callback(0.05, "Starting reconstruction pipeline...")

        pipeline = self._run_geometry_pipeline(valid_images, progress_callback)
        points = pipeline["points"]
        colors = pipeline["colors"]
        sfm_result = pipeline["sfm_result"]
        geometry_source = pipeline["geometry_source"]

        if points is None or len(points) == 0:
            return {
                "success": False,
                "status": "FAIL",
                "error": "No valid 3D points generated.",
            }

        registration_ratio_override = (
            DIAGNOSTIC_MIN_REGISTRATION_RATIO if diagnostic_mode else None
        )
        quality_result = evaluate_capture_quality(
            valid_images,
            profile,
            sfm_result,
            min_registration_ratio_override=registration_ratio_override,
        )

        if accurate_mode and not quality_result.passed:
            return {
                "success": False,
                "status": "INSUFFICIENT_CAPTURE",
                "error": "Capture quality did not meet accurate-mode thresholds.",
                "quality_gate": {
                    "passed": quality_result.passed,
                    "failures": quality_result.failures,
                    "metrics": quality_result.metrics,
                },
                "compliance": {
                    "profile": profile.name,
                    "status": "INSUFFICIENT_CAPTURE",
                },
                "diagnostic_mode": bool(diagnostic_mode),
                "sfm_success": bool((sfm_result or {}).get("success", False)),
                "sfm_registered": int((sfm_result or {}).get("num_registered", 0)),
                "data": {
                    "points": points,
                    "colors": colors,
                },
            }

        if progress_callback:
            progress_callback(0.8, "Generating provisional floor plan...")

        floor_plan_data = self.floor_plan_gen.generate_floor_plan(
            points,
            colors,
            absolute_scale_m_per_unit=None,
            accurate_mode=accurate_mode,
        )

        visuals = self._create_visual_outputs(
            points,
            colors,
            floor_plan_data,
            title="Provisional Floor Plan (Uncalibrated)" if accurate_mode else "Room Floor Plan",
        )

        if accurate_mode:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {
                "points": points,
                "colors": colors,
                "sfm_result": sfm_result,
                "quality_result": quality_result,
                "floor_plan_data": floor_plan_data,
                "geometry_source": geometry_source,
                "profile_name": profile.name,
                "num_images": len(valid_images),
                "diagnostic_mode": bool(diagnostic_mode),
            }

            if calibration_input is not None:
                return self.finalize_with_calibration(
                    session_id,
                    calibration_input,
                    progress_callback=progress_callback,
                )

            result = {
                "success": True,
                "status": "NEEDS_CALIBRATION",
                "session_id": session_id,
                "num_images": len(valid_images),
                "num_points": len(points),
                "measurements": floor_plan_data["measurements"],
                "outputs": visuals["outputs"],
                "figures": visuals["figures"],
                "quality_gate": {
                    "passed": quality_result.passed,
                    "failures": quality_result.failures,
                    "metrics": quality_result.metrics,
                },
                "compliance": {
                    "profile": profile.name,
                    "status": "NEEDS_CALIBRATION",
                },
                "diagnostic_mode": bool(diagnostic_mode),
                "geometry_source": geometry_source,
                "sfm_success": bool((sfm_result or {}).get("success", False)),
                "sfm_registered": int((sfm_result or {}).get("num_registered", 0)),
                "data": {
                    "points": points,
                    "colors": colors,
                    "floor_plan": floor_plan_data,
                },
            }
            self.last_result = result
            return result

        quick_status = "NON_COMPLIANT_QUICK_MODE"
        result = {
            "success": True,
            "status": quick_status,
            "num_images": len(valid_images),
            "num_points": len(points),
            "measurements": floor_plan_data["measurements"],
            "outputs": visuals["outputs"],
            "figures": visuals["figures"],
            "quality_gate": {
                "passed": quality_result.passed,
                "failures": quality_result.failures,
                "metrics": quality_result.metrics,
            },
            "compliance": {
                "profile": profile.name,
                "status": quick_status,
            },
            "diagnostic_mode": bool(diagnostic_mode),
            "geometry_source": geometry_source,
            "sfm_success": bool((sfm_result or {}).get("success", False)),
            "sfm_registered": int((sfm_result or {}).get("num_registered", 0)),
            "data": {
                "points": points,
                "colors": colors,
                "floor_plan": floor_plan_data,
            },
        }
        self.last_result = result
        return result

    def finalize_with_calibration(
        self,
        session_id: str,
        calibration_input: Dict,
        progress_callback=None,
    ) -> Dict:
        """Finalize accurate-mode session with known-distance calibration."""
        if session_id not in self.sessions:
            return {
                "success": False,
                "status": "FAIL",
                "error": "Calibration session not found or expired.",
            }

        session = self.sessions[session_id]
        profile = get_compliance_profile(session["profile_name"])

        if progress_callback:
            progress_callback(0.86, "Applying calibration...")

        try:
            cal_input = parse_calibration_input(calibration_input)
            provisional = session["floor_plan_data"]
            image_shape = provisional["density_map"].shape
            calibration = compute_scale_factor(cal_input, provisional, image_shape)
        except Exception as exc:
            return {
                "success": False,
                "status": "FAIL",
                "error": f"Calibration input invalid: {exc}",
            }

        if calibration["uncertainty_mm"] > profile.tolerance.max_calibration_uncertainty_mm:
            return {
                "success": False,
                "status": "FAIL",
                "error": (
                    "Calibration uncertainty too high "
                    f"({calibration['uncertainty_mm']:.2f} mm > "
                    f"{profile.tolerance.max_calibration_uncertainty_mm:.2f} mm)."
                ),
                "calibration": calibration,
                "compliance": {
                    "profile": profile.name,
                    "status": "FAIL",
                },
            }

        points_scaled = scale_points(session["points"], calibration["scale_factor"])

        if progress_callback:
            progress_callback(0.92, "Generating calibrated outputs...")

        floor_plan_final = self.floor_plan_gen.generate_floor_plan(
            points_scaled,
            session["colors"],
            absolute_scale_m_per_unit=1.0,
            accurate_mode=True,
        )

        visuals = self._create_visual_outputs(
            points_scaled,
            session["colors"],
            floor_plan_final,
            title="Room Floor Plan (Calibrated)",
        )

        timestamp = visuals["timestamp"]
        dxf_path = os.path.join(OUTPUT_DIR, f"floor_plan_{timestamp}.dxf")
        qa_path = os.path.join(OUTPUT_DIR, f"qa_report_{timestamp}.json")

        try:
            self.dxf_exporter.export_floor_plan(
                floor_plan_final,
                dxf_path,
                metadata={
                    "compliance_profile": profile.name,
                    "compliance_status": "PENDING",
                },
            )
        except Exception as exc:
            return {
                "success": False,
                "status": "FAIL",
                "error": f"DXF export failed: {exc}",
                "calibration": calibration,
                "compliance": {
                    "profile": profile.name,
                    "status": "FAIL",
                },
            }

        qa_report = build_qa_report(
            profile=profile,
            compliance_status="PENDING",
            quality_result=session["quality_result"],
            calibration=calibration,
            sfm_result=session["sfm_result"],
            floor_plan_measurements=floor_plan_final.get("measurements", {}),
            notes=[
                "Single-room v1 profile.",
                "Absolute scale anchored by user-provided known distance.",
                "Diagnostic mode used (registration ratio gate relaxed for testing)."
                if session.get("diagnostic_mode", False)
                else "Strict gate mode used.",
            ],
        )

        critical_pass = qa_report["accuracy_metrics"]["critical_pass"]
        overall_pass = qa_report["accuracy_metrics"]["overall_pass"]
        quality_pass = qa_report["pass_flags"]["quality_pass"]

        compliance_status = "PASS" if (quality_pass and critical_pass and overall_pass) else "FAIL"
        if session.get("diagnostic_mode", False):
            compliance_status = "DIAGNOSTIC_ONLY"
        qa_report["compliance_status"] = compliance_status

        save_qa_report(qa_report, qa_path)

        compliance = {
            "profile": profile.name,
            "status": compliance_status,
            "standards": profile.standards,
        }

        result = {
            "success": True,
            "status": compliance_status,
            "session_id": session_id,
            "num_images": session["num_images"],
            "num_points": len(points_scaled),
            "measurements": floor_plan_final["measurements"],
            "calibration": calibration,
            "accuracy_metrics": qa_report["accuracy_metrics"],
            "diagnostic_mode": bool(session.get("diagnostic_mode", False)),
            "quality_gate": {
                "passed": session["quality_result"].passed,
                "failures": session["quality_result"].failures,
                "metrics": session["quality_result"].metrics,
            },
            "compliance": compliance,
            "outputs": {
                **visuals["outputs"],
                "floor_plan_dxf": dxf_path,
                "qa_report_json": qa_path,
            },
            "figures": visuals["figures"],
            "geometry_source": session["geometry_source"],
            "sfm_success": bool((session["sfm_result"] or {}).get("success", False)),
            "sfm_registered": int((session["sfm_result"] or {}).get("num_registered", 0)),
            "data": {
                "points": points_scaled,
                "colors": session["colors"],
                "floor_plan": floor_plan_final,
            },
        }

        self.last_result = result
        return result
