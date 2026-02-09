"""
Floor plan generation from 3D point cloud data.

This version adds:
- Floor-plane normalization (RANSAC + PCA fallback)
- Vertical wall extraction
- Manhattan-first polygonization for single-room v1
- Optional calibrated absolute scale for accurate mode
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FIGURE_SIZE,
    FLOOR_PLAN_HEIGHT_MAX,
    FLOOR_PLAN_HEIGHT_MIN,
    FLOOR_PLAN_RESOLUTION,
    MANHATTAN_SNAP_DEFAULT,
    ASSUMED_ROOM_WIDTH_METERS,
)


class FloorPlanGenerator:
    """Generate 2D floor plan and measurements from point cloud."""

    def __init__(
        self,
        assumed_width: float = ASSUMED_ROOM_WIDTH_METERS,
        manhattan_snap: bool = MANHATTAN_SNAP_DEFAULT,
    ):
        self.assumed_width = assumed_width
        self.scale_factor = 1.0
        self.manhattan_snap = manhattan_snap

    def generate_floor_plan(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        resolution: int = FLOOR_PLAN_RESOLUTION,
        absolute_scale_m_per_unit: Optional[float] = None,
        accurate_mode: bool = False,
    ) -> Dict:
        """Generate floor plan with optional calibrated scale."""
        if len(points) == 0:
            return self._create_empty_result()

        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        if len(points) == 0:
            return self._create_empty_result()

        print(colored("[FloorPlanGenerator] Generating floor plan...", "cyan"))

        points_norm, floor_meta = self._normalize_floor_plane(points)
        floor_slice, floor_y = self._extract_floor_slice(points_norm)
        wall_points = self._extract_vertical_wall_points(points_norm, floor_y)

        if len(wall_points) >= 80:
            plan_points = wall_points
        elif len(floor_slice) >= 80:
            plan_points = floor_slice
        else:
            plan_points = points_norm

        centroid = plan_points.mean(axis=0)
        plan_points_centered = plan_points - centroid

        x_coords = plan_points_centered[:, 0]
        z_coords = plan_points_centered[:, 2]

        x_min, x_max = np.percentile(x_coords, [1, 99])
        z_min, z_max = np.percentile(z_coords, [1, 99])

        pad = 0.05 * max(x_max - x_min, z_max - z_min, 0.1)
        x_min -= pad
        x_max += pad
        z_min -= pad
        z_max += pad

        x_bins = np.linspace(x_min, x_max, resolution + 1)
        z_bins = np.linspace(z_min, z_max, resolution + 1)
        density, _, _ = np.histogram2d(x_coords, z_coords, bins=[x_bins, z_bins])
        density = density.T

        occupancy = self._create_occupancy(density)
        boundary = self._extract_boundary(
            occupancy,
            plan_points_centered,
            x_range=(x_min, x_max),
            z_range=(z_min, z_max),
            resolution=resolution,
            manhattan=self.manhattan_snap,
        )

        if absolute_scale_m_per_unit is not None:
            self.scale_factor = float(absolute_scale_m_per_unit)
            scale_source = "calibrated"
        else:
            width_units = max(x_max - x_min, 1e-6)
            self.scale_factor = float(self.assumed_width / width_units)
            scale_source = "assumed_width"

        width_m = float((x_max - x_min) * self.scale_factor)
        depth_m = float((z_max - z_min) * self.scale_factor)

        boundary_world_units = self._boundary_pixels_to_world(
            boundary,
            x_range=(x_min, x_max),
            z_range=(z_min, z_max),
            resolution=resolution,
        )
        boundary_world_m = boundary_world_units * self.scale_factor

        if len(boundary_world_m) >= 3:
            area_sqm = self._polygon_area(boundary_world_m)
            perimeter_m = self._polygon_perimeter(boundary_world_m)
        else:
            area_sqm = width_m * depth_m
            perimeter_m = 2.0 * (width_m + depth_m)

        area_sqft = area_sqm * 10.7639104
        width_ft = width_m * 3.28084
        depth_ft = depth_m * 3.28084

        width_ft_round = round(width_ft / 0.1) * 0.1
        depth_ft_round = round(depth_ft / 0.1) * 0.1

        measurements = {
            "width_m": round(width_m, 3),
            "depth_m": round(depth_m, 3),
            "width_ft": round(width_ft, 3),
            "depth_ft": round(depth_ft, 3),
            "width_ft_rounded_0_1": round(width_ft_round, 1),
            "depth_ft_rounded_0_1": round(depth_ft_round, 1),
            "area_sqm": round(area_sqm, 3),
            "area_sqft": round(area_sqft, 3),
            "perimeter_m": round(perimeter_m, 3),
            "scale_factor": float(self.scale_factor),
            "scale_source": scale_source,
        }

        result = {
            "density_map": density,
            "occupancy": occupancy,
            "walls": self._compute_wall_map(occupancy),
            "boundary": boundary,
            "x_range": (float(x_min), float(x_max)),
            "z_range": (float(z_min), float(z_max)),
            "measurements": measurements,
            "num_points": int(len(plan_points)),
            "floor_plane": floor_meta,
            "boundary_world_units": boundary_world_units.tolist(),
            "boundary_world_m": boundary_world_m.tolist(),
            "accurate_mode": bool(accurate_mode),
        }

        print(
            colored(
                "[FloorPlanGenerator] "
                f"Estimated {measurements['width_m']:.3f}m x {measurements['depth_m']:.3f}m"
                f" ({measurements['area_sqft']:.1f} sqft)",
                "green",
            )
        )

        return result

    def _normalize_floor_plane(self, points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize floor plane to be horizontal (Y-up)."""
        if len(points) < 100:
            return points.copy(), {
                "method": "none",
                "normal": [0.0, 1.0, 0.0],
                "inlier_ratio": 0.0,
            }

        pts = points.astype(np.float64)
        scene_scale = np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0))
        threshold = max(scene_scale * 0.01, 0.01)

        rng = np.random.default_rng(42)
        sample_count = min(len(pts), 2500)
        sample_idx = rng.choice(len(pts), size=sample_count, replace=False)
        sample_pts = pts[sample_idx]

        best_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        best_inliers = 0

        for _ in range(160):
            i0, i1, i2 = rng.choice(sample_count, size=3, replace=False)
            p0, p1, p2 = sample_pts[i0], sample_pts[i1], sample_pts[i2]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm
            if normal[1] < 0:
                normal = -normal

            d = -np.dot(normal, p0)
            dist = np.abs(sample_pts @ normal + d)
            inliers = int(np.sum(dist < threshold))
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal

        if best_normal[1] < 0:
            best_normal = -best_normal

        target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(best_normal, target)
        s = np.linalg.norm(v)
        c = float(np.dot(best_normal, target))

        if s < 1e-8:
            R = np.eye(3)
        else:
            vx = np.array(
                [
                    [0.0, -v[2], v[1]],
                    [v[2], 0.0, -v[0]],
                    [-v[1], v[0], 0.0],
                ]
            )
            R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s + 1e-8))

        points_rot = (R @ pts.T).T
        inlier_ratio = best_inliers / max(sample_count, 1)

        return points_rot, {
            "method": "ransac",
            "normal": best_normal.tolist(),
            "inlier_ratio": float(inlier_ratio),
            "threshold": float(threshold),
        }

    def _extract_floor_slice(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Extract floor-level slice from normalized points."""
        y = points[:, 1]
        y_min = float(np.percentile(y, 5))
        y_max = float(np.percentile(y, 95))
        height = max(y_max - y_min, 1e-6)

        floor_min = y_min + FLOOR_PLAN_HEIGHT_MIN * height
        floor_max = y_min + FLOOR_PLAN_HEIGHT_MAX * height

        mask = (y >= floor_min) & (y <= floor_max)
        floor_points = points[mask]

        if len(floor_points) < max(80, int(len(points) * 0.08)):
            mask = (y >= y_min) & (y <= y_min + 0.45 * height)
            floor_points = points[mask]

        return floor_points, y_min

    def _extract_vertical_wall_points(self, points: np.ndarray, floor_y: float) -> np.ndarray:
        """Extract likely wall points using height bands in normalized frame."""
        y = points[:, 1]
        y_top = float(np.percentile(y, 95))

        low = floor_y + 0.12
        high = min(floor_y + 2.9, y_top)

        mask = (y >= low) & (y <= high)
        candidates = points[mask]

        if len(candidates) < 100:
            alt_mask = (y >= floor_y + 0.05) & (y <= floor_y + 2.5)
            candidates = points[alt_mask]

        return candidates

    def _create_occupancy(self, density: np.ndarray) -> np.ndarray:
        """Create cleaned occupancy map from density map."""
        if density.size == 0 or density.max() <= 0:
            return np.zeros_like(density)

        d = density / density.max()
        nz = d[d > 0]
        threshold = max(float(np.percentile(nz, 30)) if nz.size else 0.08, 0.05)
        binary = (d > threshold).astype(np.uint8)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

        return binary.astype(np.float32)

    def _compute_wall_map(self, occupancy: np.ndarray) -> np.ndarray:
        """Simple wall map from occupancy edges."""
        if occupancy.max() <= 0:
            return np.zeros_like(occupancy)
        img = (occupancy > 0.5).astype(np.uint8) * 255
        edges = cv2.Canny(img, 60, 180)
        return (edges / 255.0).astype(np.float32)

    def _extract_boundary(
        self,
        occupancy: np.ndarray,
        plan_points_centered: np.ndarray,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        resolution: int,
        manhattan: bool,
    ) -> np.ndarray:
        """Extract room boundary in occupancy pixel coordinates."""
        if manhattan:
            if plan_points_centered is None or len(plan_points_centered) == 0:
                return np.zeros((0, 2), dtype=np.float32)
            xs = plan_points_centered[:, 0]
            zs = plan_points_centered[:, 2]
            x0, x1 = np.percentile(xs, [2, 98])
            z0, z1 = np.percentile(zs, [2, 98])

            poly_world = np.array(
                [[x0, z0], [x1, z0], [x1, z1], [x0, z1]],
                dtype=np.float32,
            )
            poly_px = self._world_to_pixel(poly_world, x_range, z_range, resolution)
            return poly_px

        if occupancy.max() <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        binary = (occupancy > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((0, 2), dtype=np.float32)

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.015 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return approx[:, 0, :].astype(np.float32)

    @staticmethod
    def _world_to_pixel(
        points_world: np.ndarray,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        resolution: int,
    ) -> np.ndarray:
        """Map world-unit x/z points to occupancy pixel coordinates."""
        x0, x1 = x_range
        z0, z1 = z_range

        x = (points_world[:, 0] - x0) / max((x1 - x0), 1e-8) * (resolution - 1)
        y = (z1 - points_world[:, 1]) / max((z1 - z0), 1e-8) * (resolution - 1)
        return np.stack([x, y], axis=1).astype(np.float32)

    @staticmethod
    def _boundary_pixels_to_world(
        boundary: np.ndarray,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        resolution: int,
    ) -> np.ndarray:
        """Convert occupancy pixel boundary to world-unit x/z boundary."""
        if boundary is None or len(boundary) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        x0, x1 = x_range
        z0, z1 = z_range

        bx = boundary[:, 0].astype(np.float64)
        by = boundary[:, 1].astype(np.float64)

        x = x0 + (bx / max(resolution - 1, 1)) * (x1 - x0)
        z = z1 - (by / max(resolution - 1, 1)) * (z1 - z0)

        return np.stack([x, z], axis=1)

    @staticmethod
    def _polygon_area(vertices: np.ndarray) -> float:
        """Polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _polygon_perimeter(vertices: np.ndarray) -> float:
        """Polygon perimeter."""
        if len(vertices) < 2:
            return 0.0
        closed = np.vstack([vertices, vertices[0]])
        seg = np.diff(closed, axis=0)
        return float(np.sum(np.linalg.norm(seg, axis=1)))

    def _create_empty_result(self) -> Dict:
        """Empty floor-plan response."""
        return {
            "density_map": np.zeros((100, 100), dtype=np.float32),
            "occupancy": np.zeros((100, 100), dtype=np.float32),
            "walls": np.zeros((100, 100), dtype=np.float32),
            "boundary": np.zeros((0, 2), dtype=np.float32),
            "x_range": (0.0, 1.0),
            "z_range": (0.0, 1.0),
            "measurements": {
                "width_m": 0.0,
                "depth_m": 0.0,
                "area_sqm": 0.0,
                "area_sqft": 0.0,
                "perimeter_m": 0.0,
                "scale_factor": 1.0,
                "scale_source": "none",
            },
            "num_points": 0,
            "floor_plane": {"method": "none", "normal": [0.0, 1.0, 0.0]},
            "boundary_world_units": [],
            "boundary_world_m": [],
            "accurate_mode": False,
        }

    def create_floor_plan_image(
        self,
        floor_plan_data: Dict,
        output_path: Optional[str] = None,
        title: str = "Room Floor Plan",
    ) -> plt.Figure:
        """Create visual floor plan summary image."""
        measurements = floor_plan_data["measurements"]
        density = floor_plan_data["density_map"]
        occupancy = floor_plan_data.get("occupancy")
        walls = floor_plan_data.get("walls")
        boundary = floor_plan_data.get("boundary")

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        ax1, ax2 = axes

        cmap = LinearSegmentedColormap.from_list(
            "room_density", ["white", "#e8f4ff", "#a6d2ff", "#2e86de"]
        )

        ax1.imshow(np.log1p(density), cmap=cmap, origin="lower", aspect="equal")
        if occupancy is not None and occupancy.max() > 0:
            ax1.imshow(occupancy, cmap="Blues", alpha=0.30, origin="lower")
        if walls is not None and walls.max() > 0:
            ax1.imshow(walls, cmap="Reds", alpha=0.45, origin="lower")
        if boundary is not None and len(boundary) > 2:
            closed = np.vstack([boundary, boundary[0]])
            ax1.plot(closed[:, 0], closed[:, 1], "g-", linewidth=2)

        ax1.set_title("Extracted Layout")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Z")

        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        room_patch = patches.Rectangle(
            (1.0, 1.0),
            8.0,
            8.0,
            linewidth=2.5,
            edgecolor="#333333",
            facecolor="#f4fbff",
        )
        ax2.add_patch(room_patch)

        ax2.text(
            5.0,
            6.0,
            "ROOM",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        ax2.text(
            5.0,
            4.6,
            f"{measurements.get('area_sqft', 0):.1f} sq ft",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax2.text(
            5.0,
            3.8,
            f"{measurements.get('area_sqm', 0):.2f} m²",
            ha="center",
            va="center",
            fontsize=11,
        )

        ax2.annotate(
            "",
            xy=(9, 0.8),
            xytext=(1, 0.8),
            arrowprops=dict(arrowstyle="<->", color="#cc0000", lw=2),
        )
        ax2.text(
            5,
            0.4,
            f"{measurements.get('width_ft_rounded_0_1', measurements.get('width_ft', 0)):.1f} ft",
            ha="center",
            va="top",
            color="#cc0000",
            fontsize=11,
            fontweight="bold",
        )

        ax2.annotate(
            "",
            xy=(0.8, 9),
            xytext=(0.8, 1),
            arrowprops=dict(arrowstyle="<->", color="#cc0000", lw=2),
        )
        ax2.text(
            0.4,
            5,
            f"{measurements.get('depth_ft_rounded_0_1', measurements.get('depth_ft', 0)):.1f} ft",
            ha="right",
            va="center",
            color="#cc0000",
            fontsize=11,
            fontweight="bold",
            rotation=90,
        )

        scale_source = measurements.get("scale_source", "unknown")
        fig.text(
            0.5,
            0.02,
            (
                "Scale source: calibrated known-distance reference"
                if scale_source == "calibrated"
                else f"Scale source: assumed width ({self.assumed_width}m)"
            ),
            ha="center",
            fontsize=9,
            color="#555",
        )

        ax2.set_title("Schematic Plan")
        ax2.axis("off")

        plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(colored(f"[FloorPlanGenerator] Saved floor plan to: {output_path}", "green"))

        return fig


if __name__ == "__main__":
    print("Testing Floor Plan Generator...")
    rng = np.random.default_rng(1)
    wall_left = np.column_stack([np.full(300, -2.0), rng.uniform(0, 2.6, 300), rng.uniform(0, 4.0, 300)])
    wall_right = np.column_stack([np.full(300, 2.0), rng.uniform(0, 2.6, 300), rng.uniform(0, 4.0, 300)])
    wall_back = np.column_stack([rng.uniform(-2, 2, 300), rng.uniform(0, 2.6, 300), np.full(300, 4.0)])
    floor = np.column_stack([rng.uniform(-2, 2, 800), rng.uniform(0, 0.2, 800), rng.uniform(0, 4, 800)])

    points = np.vstack([wall_left, wall_right, wall_back, floor])
    gen = FloorPlanGenerator()
    res = gen.generate_floor_plan(points)
    print(res["measurements"])
    gen.create_floor_plan_image(res, title="Test Plan")
