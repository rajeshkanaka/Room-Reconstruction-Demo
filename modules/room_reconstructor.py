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
)
from modules.depth_estimator import DepthEstimator
from modules.floor_plan_generator import FloorPlanGenerator
from modules.visualizer_3d import Visualizer3D
from modules.sfm_processor import SfMProcessor
from modules.dense_reconstructor import DenseReconstructor


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
            assumed_room_width: Assumed room width in meters (for scale)
        """
        print(colored("[RoomReconstructor] Initializing components...", "cyan"))

        self.depth_estimator = DepthEstimator()
        self.floor_plan_gen = FloorPlanGenerator(assumed_width=assumed_room_width)
        self.visualizer = Visualizer3D()
        self.sfm_processor = SfMProcessor()
        self.dense_reconstructor = DenseReconstructor()

        self.assumed_room_width = assumed_room_width
        self.last_result = None
        self.use_sfm = ENABLE_SFM and self.sfm_processor.enabled
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

        Args:
            image: RGB image as numpy array
            sample_rate: Sampling rate for point cloud
            camera_intrinsics: Optional camera intrinsics from SfM

        Returns:
            Tuple of (depth_map, points, colors)
        """
        # Estimate depth
        depth = self.depth_estimator.estimate_depth(image)

        # Use SfM intrinsics if available, otherwise use defaults
        if camera_intrinsics:
            fx = camera_intrinsics.get("fx", CAMERA_FX)
            fy = camera_intrinsics.get("fy", CAMERA_FY)
        else:
            fx, fy = CAMERA_FX, CAMERA_FY

        # Convert to 3D points
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

        # Run SfM if enabled and enough images
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

        # Process each image for depth
        all_depths = []
        views = []  # (points, colors, image_index)

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
                # Get camera intrinsics from SfM if available
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

        # Combine point clouds using appropriate method
        if progress_callback:
            progress_callback(0.7, "Combining point clouds...")

        combined_points = None
        combined_colors = None

        # Try TSDF fusion first if SfM succeeded
        if sfm_result and sfm_result["success"] and self.use_tsdf:
            print(colored("[RoomReconstructor] Trying TSDF fusion...", "cyan"))
            combined_points, combined_colors = self._fuse_with_tsdf(
                images, all_depths, sfm_result, progress_callback
            )

        # Fall back to SfM-based alignment if TSDF didn't work
        if combined_points is None or len(combined_points) == 0:
            if sfm_result and sfm_result["success"]:
                print(
                    colored(
                        "[RoomReconstructor] Aligning views using SfM poses...", "cyan"
                    )
                )
                combined_points, combined_colors = self._align_with_sfm(
                    views, sfm_result, progress_callback
                )
            else:
                print(
                    colored(
                        "[RoomReconstructor] Using legacy registration...", "yellow"
                    )
                )
                legacy_views = [(p, c) for p, c, _ in views]
                combined_points, combined_colors = self._register_point_clouds_legacy(
                    legacy_views, progress_callback
                )

        raw_count = len(combined_points)
        combined_points, combined_colors = self._postprocess_point_cloud(
            combined_points, combined_colors
        )

        if len(combined_points) == 0:
            raise ValueError("No valid 3D points generated from images")

        print(
            colored(
                f"[RoomReconstructor] Total points: {len(combined_points):,} (raw: {raw_count:,})",
                "green",
            )
        )

        # Generate floor plan
        if progress_callback:
            progress_callback(0.8, "Generating floor plan...")

        print(colored("[RoomReconstructor] Generating floor plan...", "cyan"))
        floor_plan_data = self.floor_plan_gen.generate_floor_plan(
            combined_points, combined_colors
        )

        # Create visualizations
        if progress_callback:
            progress_callback(0.9, "Creating visualizations...")

        print(colored("[RoomReconstructor] Creating visualizations...", "cyan"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        floor_plan_path = os.path.join(OUTPUT_DIR, f"floor_plan_{timestamp}.png")
        floor_plan_fig = self.floor_plan_gen.create_floor_plan_image(
            floor_plan_data,
            output_path=floor_plan_path,
            title="Room Floor Plan (Estimated)",
        )

        plotly_fig = self.visualizer.create_plotly_visualization(
            combined_points, combined_colors, title="3D Room Reconstruction"
        )

        html_path = os.path.join(OUTPUT_DIR, f"room_3d_{timestamp}.html")
        self.visualizer.export_html(combined_points, combined_colors, html_path)

        ply_path = os.path.join(OUTPUT_DIR, f"room_pointcloud_{timestamp}.ply")
        self.visualizer.save_point_cloud(combined_points, combined_colors, ply_path)

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
        result = {
            "success": True,
            "num_images": n_images,
            "num_points": len(combined_points),
            "num_points_raw": raw_count,
            "sfm_success": sfm_result["success"] if sfm_result else False,
            "sfm_registered": sfm_result["num_registered"] if sfm_result else 0,
            "measurements": floor_plan_data["measurements"],
            "outputs": {
                "floor_plan_image": floor_plan_path,
                "html_3d_model": html_path,
                "point_cloud_ply": ply_path,
                "mesh_ply": mesh_path,
            },
            "data": {
                "points": combined_points,
                "colors": combined_colors,
                "floor_plan": floor_plan_data,
                "depth_maps": all_depths,
                "mesh": mesh,
            },
            "figures": {
                "floor_plan": floor_plan_fig,
                "plotly_3d": plotly_fig,
            },
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

        # Run SfM if enabled
        sfm_result = None
        if self.use_sfm and n_images >= SFM_MIN_IMAGES:
            if progress_callback:
                progress_callback(0.1, "Running Structure-from-Motion...")

            print(colored("[RoomReconstructor] Running SfM pipeline...", "cyan"))
            sfm_result = self.sfm_processor.run_sfm(
                valid_images, progress_callback=None
            )

        # Process images for depth
        all_depths = []
        views = []

        for i, image in enumerate(valid_images):
            if progress_callback:
                progress_callback(
                    0.2 + 0.4 * (i / n_images), f"Processing image {i+1}/{n_images}"
                )

            print(
                colored(
                    f"[RoomReconstructor] Processing image {i+1}/{n_images}...", "cyan"
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

        # Combine point clouds
        if progress_callback:
            progress_callback(0.7, "Combining views...")

        combined_points = None
        combined_colors = None

        # Try TSDF fusion first if SfM succeeded
        if sfm_result and sfm_result["success"] and self.use_tsdf:
            combined_points, combined_colors = self._fuse_with_tsdf(
                valid_images, all_depths, sfm_result, progress_callback
            )

        # Fall back to SfM-based alignment, then legacy registration
        if combined_points is None or len(combined_points) == 0:
            if sfm_result and sfm_result["success"]:
                combined_points, combined_colors = self._align_with_sfm(
                    views, sfm_result, progress_callback
                )
            if combined_points is None or len(combined_points) == 0:
                legacy_views = [(p, c) for p, c, _ in views]
                combined_points, combined_colors = self._register_point_clouds_legacy(
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

        if progress_callback:
            progress_callback(0.8, "Generating floor plan and 3D model...")

        # Generate outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        floor_plan_data = self.floor_plan_gen.generate_floor_plan(
            combined_points, combined_colors
        )
        floor_plan_path = os.path.join(OUTPUT_DIR, f"floor_plan_{timestamp}.png")
        floor_plan_fig = self.floor_plan_gen.create_floor_plan_image(
            floor_plan_data, output_path=floor_plan_path
        )

        plotly_fig = self.visualizer.create_plotly_visualization(
            combined_points, combined_colors
        )

        html_path = os.path.join(OUTPUT_DIR, f"room_3d_{timestamp}.html")
        self.visualizer.export_html(combined_points, combined_colors, html_path)

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return {
            "success": True,
            "num_images": n_images,
            "num_points": len(combined_points),
            "num_points_raw": raw_count,
            "sfm_success": sfm_result["success"] if sfm_result else False,
            "sfm_registered": sfm_result["num_registered"] if sfm_result else 0,
            "measurements": floor_plan_data["measurements"],
            "outputs": {
                "floor_plan_image": floor_plan_path,
                "html_3d_model": html_path,
            },
            "figures": {
                "floor_plan": floor_plan_fig,
                "plotly_3d": plotly_fig,
            },
        }


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
