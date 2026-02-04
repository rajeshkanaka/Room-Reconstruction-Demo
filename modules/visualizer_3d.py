"""
3D Visualization Module

Creates interactive 3D visualizations of the reconstructed room
using both Open3D (for desktop) and Plotly (for web/Gradio).

Features:
- Point cloud visualization
- Mesh visualization and export
- Scale estimation utilities
- Multiple export formats (PLY, OBJ, HTML)
"""

import numpy as np
import os
import sys
from typing import Optional, Tuple, Dict, Any
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISUALIZATION_POINT_SIZE, OUTPUT_DIR, MESH_DEPTH


class ScaleEstimator:
    """
    Utilities for estimating real-world scale from reconstructed geometry.

    Methods:
    1. Reference object: Scale using a known-size object in the scene
    2. SfM scale: Use sparse points from SfM to estimate scale
    3. Assumed dimension: Scale based on typical room dimensions
    """

    @staticmethod
    def estimate_from_reference(
        points: np.ndarray,
        ref_point1: np.ndarray,
        ref_point2: np.ndarray,
        known_distance_m: float,
    ) -> float:
        """
        Estimate scale factor from known distance between two points.

        Args:
            points: Point cloud (for context, not used directly)
            ref_point1: First reference point (3D)
            ref_point2: Second reference point (3D)
            known_distance_m: Known real-world distance in meters

        Returns:
            Scale factor (multiply model units by this to get meters)
        """
        model_distance = np.linalg.norm(ref_point2 - ref_point1)
        if model_distance < 1e-6:
            return 1.0
        return known_distance_m / model_distance

    @staticmethod
    def estimate_from_bounding_box(
        points: np.ndarray, assumed_width_m: float = 4.0, axis: int = 0
    ) -> Tuple[float, Dict]:
        """
        Estimate scale based on assumed room dimension.

        Args:
            points: Point cloud (N, 3)
            assumed_width_m: Assumed width in meters
            axis: Which axis represents width (0=X, 1=Y, 2=Z)

        Returns:
            Tuple of (scale_factor, dimensions_dict)
        """
        if len(points) == 0:
            return 1.0, {"width": 0, "height": 0, "depth": 0}

        # Get bounding box
        min_pt = np.percentile(points, 2, axis=0)
        max_pt = np.percentile(points, 98, axis=0)
        extents = max_pt - min_pt

        # Calculate scale
        model_width = extents[axis]
        if model_width < 1e-6:
            scale = 1.0
        else:
            scale = assumed_width_m / model_width

        # Calculate scaled dimensions
        dimensions = {
            "width_m": extents[0] * scale,
            "height_m": extents[1] * scale,
            "depth_m": extents[2] * scale,
            "scale_factor": scale,
        }

        return scale, dimensions

    @staticmethod
    def estimate_from_sfm_sparse(
        sfm_sparse_points: np.ndarray, sfm_camera_poses: Dict
    ) -> Tuple[float, str]:
        """
        Estimate scale from SfM camera baseline and sparse points.

        Note: SfM reconstruction is up to scale. This method tries to
        estimate reasonable scale from camera positions.

        Args:
            sfm_sparse_points: Sparse point cloud from SfM
            sfm_camera_poses: Camera poses from SfM

        Returns:
            Tuple of (estimated_scale, confidence_level)
        """
        if not sfm_camera_poses or len(sfm_sparse_points) == 0:
            return 1.0, "low"

        # Extract camera positions
        camera_positions = []
        for idx, pose_info in sfm_camera_poses.items():
            if "transform" in pose_info:
                T = pose_info["transform"]
                position = T[:3, 3]
                camera_positions.append(position)

        if len(camera_positions) < 2:
            return 1.0, "low"

        camera_positions = np.array(camera_positions)

        # Calculate average camera baseline
        baselines = []
        for i in range(len(camera_positions)):
            for j in range(i + 1, len(camera_positions)):
                dist = np.linalg.norm(camera_positions[i] - camera_positions[j])
                baselines.append(dist)

        avg_baseline = np.mean(baselines)

        # Assume typical photographer distances
        # Indoor room photography: typically 1-3 meters between positions
        assumed_avg_baseline_m = 1.5  # meters

        if avg_baseline < 1e-6:
            return 1.0, "low"

        scale = assumed_avg_baseline_m / avg_baseline
        confidence = "medium" if len(camera_positions) >= 3 else "low"

        return scale, confidence


class Visualizer3D:
    """
    Creates 3D visualizations from point cloud data.

    Supports multiple output formats:
    - Interactive Plotly plots (for web/Gradio)
    - Open3D visualization (for desktop)
    - HTML export (for sharing)
    - PLY/OBJ file export (for 3D software)
    """

    def __init__(self):
        """Initialize the 3D visualizer."""
        self.point_size = VISUALIZATION_POINT_SIZE

    def create_plotly_visualization(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        title: str = "3D Room Reconstruction",
        max_points: int = 50000,
    ):
        """
        Create an interactive 3D visualization using Plotly.

        This is ideal for web interfaces like Gradio.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3), values 0-1
            title: Plot title
            max_points: Maximum points to display (for performance)

        Returns:
            Plotly figure object
        """
        import plotly.graph_objects as go

        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]

        # Prepare colors
        if colors is not None:
            # Convert to RGB strings for Plotly
            color_strings = [
                f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors
            ]
        else:
            # Default blue gradient based on depth
            depths = points[:, 2]
            depths_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
            color_strings = [
                f"rgb({int(50 + d*100)},{int(100 + d*100)},{int(200)})"
                for d in depths_norm
            ]

        # Create 3D scatter plot
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(size=self.point_size, color=color_strings, opacity=0.8),
                    hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            scene=dict(
                xaxis_title="Width (X)",
                yaxis_title="Height (Y)",
                zaxis_title="Depth (Z)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=600,
        )

        return fig

    def create_open3d_point_cloud(self, points: np.ndarray, colors: np.ndarray = None):
        """
        Create an Open3D point cloud object.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3), values 0-1

        Returns:
            Open3D PointCloud object
        """
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Default coloring based on height (Y coordinate)
            heights = points[:, 1]
            heights_norm = (heights - heights.min()) / (
                heights.max() - heights.min() + 1e-8
            )
            default_colors = np.zeros((len(points), 3))
            default_colors[:, 0] = 0.3 + 0.4 * heights_norm  # R
            default_colors[:, 1] = 0.5 + 0.3 * heights_norm  # G
            default_colors[:, 2] = 0.8 - 0.3 * heights_norm  # B
            pcd.colors = o3d.utility.Vector3dVector(default_colors)

        return pcd

    def visualize_open3d(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        window_name: str = "3D Room Reconstruction",
    ):
        """
        Open an interactive Open3D visualization window.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3), values 0-1
            window_name: Window title
        """
        import open3d as o3d

        pcd = self.create_open3d_point_cloud(points, colors)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1200, height=800)
        vis.add_geometry(pcd)

        # Set render options
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array([0.1, 0.1, 0.15])  # Dark background

        vis.run()
        vis.destroy_window()

    def save_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        output_path: str = None,
        format: str = "ply",
    ):
        """
        Save point cloud to file.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3), values 0-1
            output_path: Path to save (auto-generated if None)
            format: "ply" or "pcd"

        Returns:
            Path to saved file
        """
        import open3d as o3d

        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, f"room_model.{format}")

        pcd = self.create_open3d_point_cloud(points, colors)
        o3d.io.write_point_cloud(output_path, pcd)

        print(f"[Visualizer3D] Saved point cloud to: {output_path}")
        return output_path

    def export_html(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        output_path: str = None,
        title: str = "3D Room Model",
    ):
        """
        Export interactive 3D visualization as standalone HTML file.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3), values 0-1
            output_path: Path to save HTML
            title: Page title

        Returns:
            Path to saved HTML file
        """
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "room_model_3d.html")

        fig = self.create_plotly_visualization(points, colors, title)
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)

        print(f"[Visualizer3D] Saved HTML visualization to: {output_path}")
        return output_path

    def create_mesh_from_points(
        self, points: np.ndarray, colors: np.ndarray = None, method: str = "poisson"
    ) -> Optional[Any]:
        """
        Create a mesh from point cloud using surface reconstruction.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3)
            method: "poisson" or "ball_pivot"

        Returns:
            Open3D TriangleMesh or None if failed
        """
        import open3d as o3d

        if len(points) < 100:
            print(colored("[Visualizer3D] Too few points for mesh creation", "yellow"))
            return None

        try:
            pcd = self.create_open3d_point_cloud(points, colors)

            # Estimate normals (required for mesh reconstruction)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=30)

            if method == "poisson":
                # Poisson surface reconstruction (better quality)
                print(
                    colored("[Visualizer3D] Running Poisson reconstruction...", "cyan")
                )
                (
                    mesh,
                    densities,
                ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=MESH_DEPTH
                )

                # Remove low-density vertices (noise)
                densities = np.asarray(densities)
                density_threshold = np.quantile(densities, 0.01)
                vertices_to_remove = densities < density_threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)

            else:
                # Ball pivoting (faster but lower quality)
                radii = [0.05, 0.1, 0.2, 0.4]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

            mesh.compute_vertex_normals()

            # Transfer colors to mesh vertices
            if colors is not None:
                mesh_vertices = np.asarray(mesh.vertices)
                from scipy.spatial import cKDTree

                tree = cKDTree(points)
                _, indices = tree.query(mesh_vertices, k=1)
                valid_indices = indices[indices < len(colors)]
                mesh_colors = np.zeros((len(mesh_vertices), 3))
                mesh_colors[: len(valid_indices)] = colors[valid_indices]
                mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

            print(
                colored(
                    f"[Visualizer3D] Created mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles",
                    "green",
                )
            )
            return mesh

        except Exception as e:
            print(colored(f"[Visualizer3D] Mesh creation failed: {e}", "red"))
            return None

    def save_mesh(self, mesh: Any, output_path: str, format: str = "ply") -> bool:
        """
        Save mesh to file.

        Args:
            mesh: Open3D TriangleMesh
            output_path: Path to save
            format: "ply", "obj", "stl", or "gltf"

        Returns:
            True if successful
        """
        import open3d as o3d

        if mesh is None:
            return False

        try:
            if not output_path.endswith(f".{format}"):
                output_path = f"{output_path}.{format}"

            o3d.io.write_triangle_mesh(output_path, mesh)
            print(colored(f"[Visualizer3D] Saved mesh to: {output_path}", "green"))
            return True
        except Exception as e:
            print(colored(f"[Visualizer3D] Failed to save mesh: {e}", "red"))
            return False

    def visualize_mesh_plotly(self, mesh: Any, title: str = "3D Mesh") -> Any:
        """
        Create Plotly visualization of a mesh.

        Args:
            mesh: Open3D TriangleMesh
            title: Plot title

        Returns:
            Plotly figure
        """
        import plotly.graph_objects as go
        import open3d as o3d

        if mesh is None:
            return None

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Get colors if available
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
            colors = [
                f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
                for c in vertex_colors
            ]
        else:
            colors = "lightblue"

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    vertexcolor=colors if isinstance(colors, list) else None,
                    color=colors if isinstance(colors, str) else None,
                    opacity=0.9,
                    flatshading=True,
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            scene=dict(
                xaxis_title="Width (X)",
                yaxis_title="Height (Y)",
                zaxis_title="Depth (Z)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=600,
        )

        return fig

    def create_measurement_overlay(
        self, points: np.ndarray, scale_factor: float = 1.0
    ) -> Dict:
        """
        Create measurement data for the reconstruction.

        Args:
            points: Point cloud (N, 3)
            scale_factor: Scale to convert to meters

        Returns:
            Dictionary with measurements
        """
        if len(points) == 0:
            return {"error": "No points"}

        # Calculate bounding box
        min_pt = np.percentile(points, 2, axis=0)
        max_pt = np.percentile(points, 98, axis=0)
        extents = max_pt - min_pt

        # Apply scale
        measurements = {
            "width_m": float(extents[0] * scale_factor),
            "height_m": float(extents[1] * scale_factor),
            "depth_m": float(extents[2] * scale_factor),
            "width_ft": float(extents[0] * scale_factor * 3.28084),
            "height_ft": float(extents[1] * scale_factor * 3.28084),
            "depth_ft": float(extents[2] * scale_factor * 3.28084),
            "floor_area_sqm": float(extents[0] * extents[2] * scale_factor**2),
            "floor_area_sqft": float(
                extents[0] * extents[2] * scale_factor**2 * 10.764
            ),
            "volume_m3": float(np.prod(extents) * scale_factor**3),
            "scale_factor": scale_factor,
            "bounding_box": {
                "min": min_pt.tolist(),
                "max": max_pt.tolist(),
            },
        }

        return measurements


class CombinedPointCloud:
    """
    Combines multiple point clouds from different viewpoints
    into a single unified point cloud.
    """

    def __init__(self):
        self.all_points = []
        self.all_colors = []

    def add_view(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        view_index: int = 0,
        offset: float = 2.0,
    ):
        """
        Add a point cloud from one view.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3)
            view_index: Index of this view (for positioning)
            offset: Offset between views
        """
        # Simple approach: offset each view along X axis
        # A more sophisticated approach would use ICP registration
        points_offset = points.copy()

        # Apply rotation based on view index to simulate different angles
        angle = view_index * np.pi / 4  # 45 degrees per view
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Rotate around Y axis
        x_rot = points_offset[:, 0] * cos_a - points_offset[:, 2] * sin_a
        z_rot = points_offset[:, 0] * sin_a + points_offset[:, 2] * cos_a
        points_offset[:, 0] = x_rot
        points_offset[:, 2] = z_rot

        self.all_points.append(points_offset)
        self.all_colors.append(colors)

    def get_combined(self):
        """
        Get the combined point cloud.

        Returns:
            Combined points and colors
        """
        if not self.all_points:
            return np.array([]), np.array([])

        combined_points = np.vstack(self.all_points)
        combined_colors = np.vstack(self.all_colors)

        return combined_points, combined_colors


# Quick test
if __name__ == "__main__":
    print("Testing 3D Visualizer...")

    # Create dummy point cloud
    n_points = 5000
    # Create points resembling a room corner
    x = np.random.uniform(-2, 2, n_points)
    y = np.random.uniform(0, 2.5, n_points)
    z = np.random.uniform(0, 3, n_points)
    points = np.stack([x, y, z], axis=1)

    # Random colors
    colors = np.random.rand(n_points, 3)

    visualizer = Visualizer3D()

    # Test Plotly visualization
    fig = visualizer.create_plotly_visualization(points, colors, "Test 3D View")
    print("Plotly figure created successfully!")

    # Save HTML
    html_path = visualizer.export_html(points, colors)
    print(f"HTML exported to: {html_path}")

    print("Test passed!")
