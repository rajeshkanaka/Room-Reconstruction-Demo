"""
Floor Plan Generator Module

Generates 2D floor plans from 3D point clouds by analyzing the
room structure and creating a top-down view with measurements.

Improvements:
- Hough transform for line/wall detection
- Convex hull for room boundary estimation
- Better noise filtering with morphological operations
- Improved wall extraction using depth edges
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure, morphology
from skimage.transform import probabilistic_hough_line
import cv2
from typing import Tuple, Optional, Dict, List
import os
import sys
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FLOOR_PLAN_HEIGHT_MIN,
    FLOOR_PLAN_HEIGHT_MAX,
    FLOOR_PLAN_RESOLUTION,
    ASSUMED_ROOM_WIDTH_METERS,
    FIGURE_SIZE,
)


class FloorPlanGenerator:
    """
    Generates 2D floor plans from 3D point cloud data.

    Improved pipeline:
    1. Take horizontal slice of point cloud (floor level)
    2. Create top-down density map
    3. Apply advanced wall detection (Hough lines + edge detection)
    4. Extract room boundary using convex hull
    5. Estimate dimensions with proper scaling
    """

    def __init__(self, assumed_width: float = ASSUMED_ROOM_WIDTH_METERS):
        """
        Initialize the floor plan generator.

        Args:
            assumed_width: Assumed room width in meters (for scale estimation)
        """
        self.assumed_width = assumed_width
        self.scale_factor = 1.0  # meters per unit

    def generate_floor_plan(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        resolution: int = FLOOR_PLAN_RESOLUTION,
    ) -> Dict:
        """
        Generate a floor plan from 3D points.

        Args:
            points: 3D points (N, 3) with X, Y, Z coordinates
            colors: Optional RGB colors (N, 3)
            resolution: Grid resolution for the floor plan

        Returns:
            Dictionary with floor plan data and measurements
        """
        if len(points) == 0:
            return self._create_empty_result()

        print(colored("[FloorPlanGenerator] Generating floor plan...", "cyan"))

        # 1. Remove invalid points
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        if len(points) == 0:
            return self._create_empty_result()

        # 2. Slice a height band for floor-level view
        points_floor = self._extract_floor_slice(points)
        if len(points_floor) < 50:
            print(
                colored(
                    "[FloorPlanGenerator] Warning: Few points in floor slice, using all points",
                    "yellow",
                )
            )
            points_floor = points

        # 3. Center the point cloud
        centroid = points_floor.mean(axis=0)
        points_centered = points_floor - centroid

        # 4. Extract X and Z coordinates (top-down view)
        x_coords = points_centered[:, 0]
        z_coords = points_centered[:, 2]

        # 5. Calculate robust bounds using percentiles
        if len(x_coords) > 50:
            x_min, x_max = np.percentile(x_coords, [1, 99])
            z_min, z_max = np.percentile(z_coords, [1, 99])
        else:
            x_min, x_max = x_coords.min(), x_coords.max()
            z_min, z_max = z_coords.min(), z_coords.max()

        # Add padding
        padding = 0.05 * max(x_max - x_min, z_max - z_min, 0.1)
        x_min -= padding
        x_max += padding
        z_min -= padding
        z_max += padding

        # 6. Create 2D histogram (density map)
        x_bins = np.linspace(x_min, x_max, resolution + 1)
        z_bins = np.linspace(z_min, z_max, resolution + 1)

        density, _, _ = np.histogram2d(x_coords, z_coords, bins=[x_bins, z_bins])
        density = density.T  # Transpose for proper orientation

        # 7. Apply improved occupancy and wall detection
        occupancy = self._create_improved_occupancy(density)
        walls = self._detect_walls_hough(occupancy)
        boundary = self._extract_room_boundary(occupancy)

        # 8. Calculate scale factor
        room_width_units = x_max - x_min
        self.scale_factor = (
            self.assumed_width / room_width_units if room_width_units > 0 else 1.0
        )

        # 9. Calculate approximate dimensions
        width_meters = (x_max - x_min) * self.scale_factor
        depth_meters = (z_max - z_min) * self.scale_factor
        area_sqm = width_meters * depth_meters
        area_sqft = area_sqm * 10.764  # Convert to square feet

        # 10. Try to get more accurate area from boundary
        if boundary is not None and len(boundary) > 2:
            try:
                boundary_area_pixels = self._polygon_area(boundary)
                pixel_area = ((x_max - x_min) / resolution) * (
                    (z_max - z_min) / resolution
                )
                boundary_area_units = boundary_area_pixels * pixel_area
                area_sqm = boundary_area_units * (self.scale_factor**2)
                area_sqft = area_sqm * 10.764
            except Exception:
                pass  # Fall back to bounding box area

        print(
            colored(
                f"[FloorPlanGenerator] Estimated: {width_meters:.2f}m x {depth_meters:.2f}m ({area_sqft:.0f} sq ft)",
                "green",
            )
        )

        result = {
            "density_map": density,
            "occupancy": occupancy,
            "walls": walls,
            "boundary": boundary,
            "x_range": (x_min, x_max),
            "z_range": (z_min, z_max),
            "measurements": {
                "width_m": round(width_meters, 2),
                "depth_m": round(depth_meters, 2),
                "area_sqm": round(area_sqm, 2),
                "area_sqft": round(area_sqft, 1),
                "scale_factor": self.scale_factor,
            },
            "num_points": len(points_floor),
        }

        return result

    def _extract_floor_slice(self, points: np.ndarray) -> np.ndarray:
        """Extract points at floor level for better top-down view."""
        y_coords = points[:, 1]

        if len(y_coords) > 100:
            y_min, y_max = np.percentile(y_coords, [5, 95])
        else:
            y_min, y_max = y_coords.min(), y_coords.max()

        height_range = max(y_max - y_min, 1e-6)

        # Take a slice at lower portion (floor area)
        floor_min = y_min + FLOOR_PLAN_HEIGHT_MIN * height_range
        floor_max = y_min + FLOOR_PLAN_HEIGHT_MAX * height_range

        floor_mask = (y_coords >= floor_min) & (y_coords <= floor_max)
        points_floor = points[floor_mask]

        # If too few points, expand the slice
        if len(points_floor) < max(100, int(0.1 * len(points))):
            floor_max = y_min + 0.5 * height_range
            floor_mask = (y_coords >= floor_min) & (y_coords <= floor_max)
            points_floor = points[floor_mask]

        return points_floor

    def _create_improved_occupancy(self, density: np.ndarray) -> np.ndarray:
        """Create a cleaned occupancy map with improved morphological operations."""
        if density.max() == 0:
            return np.zeros_like(density)

        # Normalize density
        density_norm = density / density.max()

        # Adaptive thresholding based on density distribution
        nonzero = density_norm[density_norm > 0]
        if nonzero.size > 0:
            # Use Otsu-like threshold selection
            threshold = np.percentile(nonzero, 30)
            threshold = max(threshold, 0.05)  # Minimum threshold
        else:
            threshold = 0.1

        binary = (density_norm > threshold).astype(np.uint8)

        # Apply morphological operations for cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Close small gaps
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Fill holes in the occupancy map
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

        return binary.astype(float)

    def _detect_walls_hough(self, occupancy: np.ndarray) -> np.ndarray:
        """
        Detect walls using Hough line transform for better line detection.

        Args:
            occupancy: Binary occupancy map

        Returns:
            Wall detection map
        """
        if occupancy.max() == 0:
            return np.zeros_like(occupancy)

        binary = (occupancy > 0.5).astype(np.uint8) * 255

        # Edge detection using Canny
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Apply Hough line transform
        walls = np.zeros_like(occupancy, dtype=np.float32)

        try:
            # Probabilistic Hough Line Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=20,
                minLineLength=int(occupancy.shape[0] * 0.1),
                maxLineGap=int(occupancy.shape[0] * 0.05),
            )

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(walls, (x1, y1), (x2, y2), 1.0, thickness=2)
        except Exception as e:
            print(
                colored(f"[FloorPlanGenerator] Hough transform warning: {e}", "yellow")
            )

        # Combine with edge detection as fallback
        sobel_x = ndimage.sobel(occupancy, axis=0)
        sobel_y = ndimage.sobel(occupancy, axis=1)
        edge_magnitude = np.hypot(sobel_x, sobel_y)

        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()

        # Combine Hough lines with edge detection
        walls = np.maximum(walls, edge_magnitude * 0.5)
        walls = np.clip(walls, 0, 1)

        return walls

    def _extract_room_boundary(self, occupancy: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract room boundary using contour detection and convex hull.

        Args:
            occupancy: Binary occupancy map

        Returns:
            Array of boundary points or None
        """
        if occupancy.max() == 0:
            return None

        binary = (occupancy > 0.5).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 100:  # Too small
            return None

        # Simplify the contour using Douglas-Peucker algorithm
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Convert to numpy array
        boundary = simplified.squeeze()

        # If too few points, use convex hull instead
        if len(boundary) < 3:
            try:
                hull = ConvexHull(largest_contour.squeeze())
                boundary = largest_contour.squeeze()[hull.vertices]
            except Exception:
                return None

        return boundary

    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using Shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]

        return abs(area) / 2.0

    def _create_empty_result(self) -> Dict:
        """Create empty result when no points available."""
        return {
            "density_map": np.zeros((100, 100)),
            "occupancy": np.zeros((100, 100)),
            "walls": np.zeros((100, 100)),
            "boundary": None,
            "x_range": (0, 1),
            "z_range": (0, 1),
            "measurements": {
                "width_m": 0,
                "depth_m": 0,
                "area_sqm": 0,
                "area_sqft": 0,
                "scale_factor": 1.0,
            },
            "num_points": 0,
        }

    def create_floor_plan_image(
        self,
        floor_plan_data: Dict,
        output_path: Optional[str] = None,
        title: str = "Room Floor Plan",
    ) -> plt.Figure:
        """
        Create a visual floor plan image with measurements.

        Args:
            floor_plan_data: Output from generate_floor_plan()
            output_path: Optional path to save the image
            title: Title for the floor plan

        Returns:
            Matplotlib figure
        """
        measurements = floor_plan_data["measurements"]
        density = floor_plan_data["density_map"]
        occupancy = floor_plan_data.get("occupancy")
        walls = floor_plan_data["walls"]
        boundary = floor_plan_data.get("boundary")

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # --- Left: Density/Layout Map ---
        ax1 = axes[0]

        # Create custom colormap (white to blue)
        colors_map = ["white", "#e6f2ff", "#99ccff", "#3399ff", "#0066cc"]
        cmap = LinearSegmentedColormap.from_list("room", colors_map)

        # Display density with log scaling for better visibility
        density_display = np.log1p(density)
        ax1.imshow(density_display, cmap=cmap, origin="lower", aspect="equal")

        # Overlay occupancy
        if occupancy is not None and occupancy.max() > 0:
            ax1.imshow(occupancy, cmap="Blues", alpha=0.3, origin="lower")

        # Overlay walls
        if walls is not None and walls.max() > 0:
            ax1.imshow(walls, cmap="Reds", alpha=0.4, origin="lower")

        # Draw boundary if available
        if boundary is not None and len(boundary) > 2:
            boundary_closed = np.vstack([boundary, boundary[0]])
            ax1.plot(
                boundary_closed[:, 0],
                boundary_closed[:, 1],
                "g-",
                linewidth=2,
                label="Room Boundary",
            )

        ax1.set_title("Room Layout (Top-Down View)", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Width", fontsize=11)
        ax1.set_ylabel("Depth", fontsize=11)

        # Add measurement annotations
        width_label = (
            f"{measurements['width_m']:.1f}m ({measurements['width_m']*3.28:.1f}ft)"
        )
        depth_label = (
            f"{measurements['depth_m']:.1f}m ({measurements['depth_m']*3.28:.1f}ft)"
        )

        # Width annotation (top)
        ax1.annotate(
            "",
            xy=(density.shape[1], density.shape[0] * 1.05),
            xytext=(0, density.shape[0] * 1.05),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2),
        )
        ax1.text(
            density.shape[1] / 2,
            density.shape[0] * 1.1,
            width_label,
            ha="center",
            va="bottom",
            fontsize=11,
            color="red",
            fontweight="bold",
        )

        # Depth annotation (right)
        ax1.annotate(
            "",
            xy=(density.shape[1] * 1.05, density.shape[0]),
            xytext=(density.shape[1] * 1.05, 0),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2),
        )
        ax1.text(
            density.shape[1] * 1.15,
            density.shape[0] / 2,
            depth_label,
            ha="left",
            va="center",
            fontsize=11,
            color="red",
            fontweight="bold",
            rotation=90,
        )

        ax1.set_xlim(-density.shape[1] * 0.1, density.shape[1] * 1.3)
        ax1.set_ylim(-density.shape[0] * 0.1, density.shape[0] * 1.2)

        # --- Right: Schematic Floor Plan ---
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)

        # Draw room outline from boundary or occupancy contour
        room_patch = None
        if boundary is not None and len(boundary) > 2:
            h, w = occupancy.shape if occupancy is not None else (100, 100)
            xs = boundary[:, 0]
            ys = boundary[:, 1]
            x_plot = 1 + (xs / max(w - 1, 1)) * 8
            y_plot = 1 + ((h - 1 - ys) / max(h - 1, 1)) * 8
            poly = np.stack([x_plot, y_plot], axis=1)
            room_patch = patches.Polygon(
                poly,
                closed=True,
                linewidth=2.5,
                edgecolor="#333333",
                facecolor="#f0f8ff",
            )
        elif occupancy is not None and occupancy.max() > 0:
            mask = (occupancy > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 10:
                    epsilon = 0.02 * cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, epsilon, True)
                    h, w = occupancy.shape
                    xs = approx[:, 0, 0]
                    ys = approx[:, 0, 1]
                    x_plot = 1 + (xs / max(w - 1, 1)) * 8
                    y_plot = 1 + ((h - 1 - ys) / max(h - 1, 1)) * 8
                    poly = np.stack([x_plot, y_plot], axis=1)
                    room_patch = patches.Polygon(
                        poly,
                        closed=True,
                        linewidth=2.5,
                        edgecolor="#333333",
                        facecolor="#f0f8ff",
                    )

        if room_patch is None:
            room_patch = patches.Rectangle(
                (1, 1), 8, 8, linewidth=3, edgecolor="#333333", facecolor="#f0f8ff"
            )

        ax2.add_patch(room_patch)

        # Add room label in center
        ax2.text(
            5,
            5,
            f"ROOM\n{measurements['area_sqft']:.0f} sq ft\n({measurements['area_sqm']:.1f} m²)",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#333",
        )

        # Add dimension lines
        # Width (bottom)
        ax2.annotate(
            "",
            xy=(9, 0.5),
            xytext=(1, 0.5),
            arrowprops=dict(arrowstyle="<->", color="#cc0000", lw=2),
        )
        ax2.text(
            5,
            0.2,
            f"{measurements['width_m']:.2f} m / {measurements['width_m']*3.28:.1f} ft",
            ha="center",
            va="top",
            fontsize=11,
            color="#cc0000",
            fontweight="bold",
        )

        # Depth (left)
        ax2.annotate(
            "",
            xy=(0.5, 9),
            xytext=(0.5, 1),
            arrowprops=dict(arrowstyle="<->", color="#cc0000", lw=2),
        )
        ax2.text(
            0.2,
            5,
            f"{measurements['depth_m']:.2f} m\n{measurements['depth_m']*3.28:.1f} ft",
            ha="right",
            va="center",
            fontsize=10,
            color="#cc0000",
            fontweight="bold",
            rotation=90,
        )

        ax2.set_title("Schematic Floor Plan", fontsize=14, fontweight="bold")
        ax2.set_aspect("equal")
        ax2.axis("off")

        # Add note about approximate measurements
        fig.text(
            0.5,
            0.02,
            f"Note: Measurements are approximate estimates based on depth analysis. "
            f"Scale assumes room width = {self.assumed_width}m.",
            ha="center",
            fontsize=9,
            color="#666",
            style="italic",
        )

        plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save if path provided
        if output_path:
            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(
                colored(
                    f"[FloorPlanGenerator] Saved floor plan to: {output_path}", "green"
                )
            )

        return fig


# Quick test
if __name__ == "__main__":
    print("Testing Floor Plan Generator...")

    # Create dummy point cloud (simple room shape)
    n_points = 2000

    # Create points along walls (more realistic)
    wall_points = []
    # Left wall
    wall_points.append(
        np.column_stack(
            [
                np.full(200, -2),
                np.random.uniform(0, 2.5, 200),
                np.random.uniform(0, 4, 200),
            ]
        )
    )
    # Right wall
    wall_points.append(
        np.column_stack(
            [
                np.full(200, 2),
                np.random.uniform(0, 2.5, 200),
                np.random.uniform(0, 4, 200),
            ]
        )
    )
    # Back wall
    wall_points.append(
        np.column_stack(
            [
                np.random.uniform(-2, 2, 200),
                np.random.uniform(0, 2.5, 200),
                np.full(200, 4),
            ]
        )
    )
    # Floor
    wall_points.append(
        np.column_stack(
            [
                np.random.uniform(-2, 2, 400),
                np.random.uniform(0, 0.3, 400),
                np.random.uniform(0, 4, 400),
            ]
        )
    )

    points = np.vstack(wall_points)

    generator = FloorPlanGenerator()
    result = generator.generate_floor_plan(points)

    print(f"Measurements: {result['measurements']}")

    fig = generator.create_floor_plan_image(result, title="Test Floor Plan")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "test_floor_plan.png"))
    print("Test passed!")
