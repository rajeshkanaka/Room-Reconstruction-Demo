"""
Dense Reconstruction Module

Implements dense 3D reconstruction using:
1. TSDF (Truncated Signed Distance Function) fusion
2. Multi-view depth fusion with SfM poses
3. Optional Poisson surface reconstruction

This produces cleaner, more coherent geometry compared to 
simply concatenating point clouds from individual depth maps.
"""

import numpy as np
import cv2
import os
import sys
from typing import List, Dict, Optional, Tuple, Any
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEPTH_SCALE,
    CAMERA_FX,
    CAMERA_FY,
    VOXEL_SIZE,
    MESH_DEPTH,
    MESH_SCALE,
    MESH_SIMPLIFY_TARGET,
    DEPTH_FUSION_METHOD,
)

# Check for Open3D
try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print(colored("[DenseReconstructor] Warning: Open3D not available", "yellow"))


class DenseReconstructor:
    """
    Dense 3D reconstruction using multi-view depth fusion.

    Supports two main methods:
    1. TSDF Fusion: Integrates multiple depth maps into a volumetric
       representation, then extracts mesh using Marching Cubes
    2. Direct Fusion: Combines point clouds with proper weighting

    Both methods use SfM camera poses for proper alignment.
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        sdf_trunc: float = 0.04,
        depth_scale: float = 1.0,
        depth_max: float = 3.0,
    ):
        """
        Initialize the dense reconstructor.

        Args:
            voxel_size: Voxel size for TSDF (smaller = more detail)
            sdf_trunc: Truncation distance for SDF (typically 3-5x voxel_size)
            depth_scale: Scale factor for depth values
            depth_max: Maximum depth to consider
        """
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.depth_scale = depth_scale
        self.depth_max = depth_max

        self.volume = None
        self.mesh = None
        self.point_cloud = None

        if OPEN3D_AVAILABLE:
            print(
                colored(
                    f"[DenseReconstructor] Initialized (voxel_size={voxel_size})",
                    "green",
                )
            )
        else:
            print(
                colored(
                    "[DenseReconstructor] Running in limited mode (no Open3D)", "yellow"
                )
            )

    def create_intrinsic_matrix(
        self,
        width: int,
        height: int,
        fx: float = None,
        fy: float = None,
        cx: float = None,
        cy: float = None,
    ) -> Any:
        """
        Create Open3D camera intrinsic matrix.

        Args:
            width, height: Image dimensions
            fx, fy: Focal lengths (defaults to CAMERA_FX/FY)
            cx, cy: Principal point (defaults to image center)

        Returns:
            Open3D PinholeCameraIntrinsic
        """
        if not OPEN3D_AVAILABLE:
            return None

        fx = fx or CAMERA_FX
        fy = fy or CAMERA_FY
        cx = cx or width / 2
        cy = cy or height / 2

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(width), height=int(height), fx=fx, fy=fy, cx=cx, cy=cy
        )

        return intrinsic

    def depth_to_rgbd(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        depth_scale: float = 1.0,
        depth_trunc: float = 3.0,
        convert_rgb_to_intensity: bool = False,
    ) -> Any:
        """
        Convert color and depth images to Open3D RGBD image.

        Args:
            color: RGB image (H, W, 3), uint8
            depth: Depth image (H, W), float32
            depth_scale: Scale factor for depth
            depth_trunc: Maximum depth
            convert_rgb_to_intensity: Convert to grayscale

        Returns:
            Open3D RGBDImage
        """
        if not OPEN3D_AVAILABLE:
            return None

        # Ensure correct types
        if color.dtype != np.uint8:
            if color.max() <= 1.0:
                color = (color * 255).astype(np.uint8)
            else:
                color = color.astype(np.uint8)

        # Convert depth to appropriate format
        # For TSDF, depth should be in meters
        if depth.max() <= 1.0:
            # Normalized depth - convert to metric
            # Assuming inverse depth representation
            depth_metric = 1.0 / (depth + 1e-6) * DEPTH_SCALE
        else:
            depth_metric = depth * depth_scale

        # Clip depth
        depth_metric = np.clip(depth_metric, 0, depth_trunc)

        # Convert to uint16 for Open3D (millimeters)
        depth_mm = (depth_metric * 1000).astype(np.uint16)

        # Create Open3D images
        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth_mm)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,  # millimeters to meters
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=convert_rgb_to_intensity,
        )

        return rgbd

    def fuse_tsdf(
        self,
        images: List[np.ndarray],
        depths: List[np.ndarray],
        camera_poses: Dict[int, Dict],
        camera_intrinsics: Optional[Dict[int, Dict]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform TSDF fusion of multiple depth maps.

        Args:
            images: List of RGB images
            depths: List of depth maps
            camera_poses: Dict mapping image index to pose info
            camera_intrinsics: Optional camera intrinsics per image
            progress_callback: Progress callback

        Returns:
            Tuple of (points, colors) from fused volume
        """
        if not OPEN3D_AVAILABLE:
            print(colored("[DenseReconstructor] TSDF fusion requires Open3D", "red"))
            return self._fallback_fusion(images, depths, camera_poses)

        print(colored("[DenseReconstructor] Starting TSDF fusion...", "cyan"))

        # Initialize TSDF volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        n_images = len(images)
        integrated_count = 0

        for i, (image, depth) in enumerate(zip(images, depths)):
            if progress_callback:
                progress_callback(
                    0.1 + 0.6 * (i / n_images), f"Fusing view {i+1}/{n_images}..."
                )

            # Get camera pose
            if i not in camera_poses:
                print(colored(f"  View {i}: No pose available, skipping", "yellow"))
                continue

            pose_info = camera_poses[i]

            # Get extrinsic matrix (camera pose)
            if "transform" in pose_info:
                extrinsic = pose_info["transform"]
            else:
                R = pose_info.get("R", np.eye(3))
                t = pose_info.get("t", np.zeros(3))
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t

            # Get intrinsics
            h, w = image.shape[:2]
            if camera_intrinsics and i in camera_intrinsics:
                intr = camera_intrinsics[i]
                intrinsic = self.create_intrinsic_matrix(
                    w,
                    h,
                    fx=intr.get("fx", CAMERA_FX),
                    fy=intr.get("fy", CAMERA_FY),
                    cx=intr.get("cx", w / 2),
                    cy=intr.get("cy", h / 2),
                )
            else:
                intrinsic = self.create_intrinsic_matrix(w, h)

            # Create RGBD image
            rgbd = self.depth_to_rgbd(
                image, depth, depth_scale=self.depth_scale, depth_trunc=self.depth_max
            )

            if rgbd is None:
                continue

            # Integrate into volume
            try:
                self.volume.integrate(
                    rgbd, intrinsic, np.linalg.inv(extrinsic)  # World to camera
                )
                integrated_count += 1
                print(colored(f"  View {i}: Integrated successfully", "green"))
            except Exception as e:
                print(colored(f"  View {i}: Integration failed: {e}", "red"))

        if integrated_count == 0:
            print(
                colored(
                    "[DenseReconstructor] No views integrated, using fallback", "yellow"
                )
            )
            return self._fallback_fusion(images, depths, camera_poses)

        print(
            colored(
                f"[DenseReconstructor] Integrated {integrated_count}/{n_images} views",
                "green",
            )
        )

        # Extract point cloud from volume
        if progress_callback:
            progress_callback(0.8, "Extracting point cloud...")

        try:
            pcd = self.volume.extract_point_cloud()
            self.point_cloud = pcd

            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            print(
                colored(
                    f"[DenseReconstructor] Extracted {len(points):,} points", "green"
                )
            )

            return points, colors

        except Exception as e:
            print(colored(f"[DenseReconstructor] Extraction failed: {e}", "red"))
            return self._fallback_fusion(images, depths, camera_poses)

    def extract_mesh(
        self, progress_callback: Optional[callable] = None
    ) -> Optional[Any]:
        """
        Extract mesh from TSDF volume using Marching Cubes.

        Returns:
            Open3D TriangleMesh or None
        """
        if not OPEN3D_AVAILABLE or self.volume is None:
            return None

        if progress_callback:
            progress_callback(0.9, "Extracting mesh...")

        print(
            colored("[DenseReconstructor] Extracting mesh from TSDF volume...", "cyan")
        )

        try:
            mesh = self.volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()

            # Simplify if too many triangles
            if len(mesh.triangles) > MESH_SIMPLIFY_TARGET:
                target_triangles = MESH_SIMPLIFY_TARGET
                mesh = mesh.simplify_quadric_decimation(target_triangles)
                mesh.compute_vertex_normals()
                print(
                    colored(
                        f"[DenseReconstructor] Simplified to {len(mesh.triangles):,} triangles",
                        "cyan",
                    )
                )

            self.mesh = mesh
            print(
                colored(
                    f"[DenseReconstructor] Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles",
                    "green",
                )
            )

            return mesh

        except Exception as e:
            print(colored(f"[DenseReconstructor] Mesh extraction failed: {e}", "red"))
            return None

    def _fallback_fusion(
        self,
        images: List[np.ndarray],
        depths: List[np.ndarray],
        camera_poses: Dict[int, Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback point cloud fusion without TSDF."""
        print(colored("[DenseReconstructor] Using simple point cloud fusion", "yellow"))

        all_points = []
        all_colors = []

        for i, (image, depth) in enumerate(zip(images, depths)):
            # Convert depth to points
            h, w = depth.shape
            fx, fy = CAMERA_FX, CAMERA_FY
            cx, cy = w / 2, h / 2

            # Sample pixels
            sample_rate = 4
            u = np.arange(0, w, sample_rate)
            v = np.arange(0, h, sample_rate)
            u, v = np.meshgrid(u, v)

            depth_sampled = depth[::sample_rate, ::sample_rate]
            colors_sampled = image[::sample_rate, ::sample_rate] / 255.0

            # Valid depth mask
            valid = (depth_sampled > 0.05) & (depth_sampled < 0.95)

            # Convert to 3D
            z = 1.0 / (depth_sampled + 1e-3) * DEPTH_SCALE
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points = np.stack([x, y, z], axis=-1)
            points = points[valid]
            colors = colors_sampled[valid]

            # Apply camera pose if available
            if i in camera_poses:
                pose = camera_poses[i]
                if "transform" in pose:
                    T = pose["transform"]
                    pts_h = np.hstack([points, np.ones((len(points), 1))])
                    points = (T @ pts_h.T).T[:, :3]

            all_points.append(points)
            all_colors.append(colors)

        points = np.vstack(all_points) if all_points else np.array([])
        colors = np.vstack(all_colors) if all_colors else np.array([])

        return points, colors

    def create_poisson_mesh(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        depth: int = MESH_DEPTH,
    ) -> Optional[Any]:
        """
        Create mesh using Poisson surface reconstruction.

        Args:
            points: Nx3 point cloud
            colors: Optional Nx3 colors
            depth: Poisson reconstruction depth

        Returns:
            Open3D TriangleMesh or None
        """
        if not OPEN3D_AVAILABLE or len(points) < 100:
            return None

        print(colored("[DenseReconstructor] Running Poisson reconstruction...", "cyan"))

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(100)

        try:
            # Run Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=MESH_SCALE
            )

            # Remove low-density vertices (noise)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            mesh.compute_vertex_normals()

            # Transfer colors if available
            if colors is not None:
                # Use nearest neighbor to assign colors to mesh vertices
                mesh_vertices = np.asarray(mesh.vertices)
                from scipy.spatial import cKDTree

                tree = cKDTree(points)
                _, indices = tree.query(mesh_vertices, k=1)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors[indices])

            self.mesh = mesh
            print(
                colored(
                    f"[DenseReconstructor] Poisson mesh: {len(mesh.vertices):,} vertices",
                    "green",
                )
            )

            return mesh

        except Exception as e:
            print(
                colored(
                    f"[DenseReconstructor] Poisson reconstruction failed: {e}", "red"
                )
            )
            return None

    def save_mesh(self, mesh: Any, filepath: str) -> bool:
        """Save mesh to file (PLY, OBJ, etc.)."""
        if mesh is None or not OPEN3D_AVAILABLE:
            return False

        try:
            o3d.io.write_triangle_mesh(filepath, mesh)
            print(colored(f"[DenseReconstructor] Saved mesh to: {filepath}", "green"))
            return True
        except Exception as e:
            print(colored(f"[DenseReconstructor] Failed to save mesh: {e}", "red"))
            return False


# Quick test
if __name__ == "__main__":
    print("Testing Dense Reconstructor...")

    if not OPEN3D_AVAILABLE:
        print("Open3D not available - skipping test")
    else:
        # Create dummy data
        reconstructor = DenseReconstructor()

        # Create synthetic depth and image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.uniform(0.2, 0.8, (480, 640)).astype(np.float32)

        # Dummy camera pose
        camera_poses = {0: {"transform": np.eye(4)}}

        points, colors = reconstructor.fuse_tsdf([image], [depth], camera_poses)

        print(f"Result: {len(points)} points")
        print("Test complete!")
