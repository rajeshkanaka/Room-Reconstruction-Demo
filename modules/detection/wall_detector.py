"""
Wall Detection Module

Detects walls from point clouds and depth maps using:
- RANSAC floor plane detection (replaces fixed height slice)
- LSD line segment detection from depth discontinuities
- Manhattan World alignment (snaps walls to 90-degree grid)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from termcolor import colored


class WallDetector:
    """
    Detects walls from 3D point clouds and depth maps.

    Pipeline:
    1. detect_floor_plane() — RANSAC plane fitting for the floor
    2. detect_walls_from_depth() — depth gradient → line segments
    3. align_walls_manhattan() — snap to perpendicular grid
    """

    def __init__(
        self,
        floor_distance_threshold: float = 0.05,
        floor_ransac_n: int = 3,
        floor_num_iterations: int = 1000,
    ):
        self.floor_distance_threshold = floor_distance_threshold
        self.floor_ransac_n = floor_ransac_n
        self.floor_num_iterations = floor_num_iterations

    def detect_floor_plane(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the dominant horizontal plane (floor) using RANSAC.

        Args:
            points: Nx3 point cloud

        Returns:
            plane: 4-element array [a, b, c, d] for plane equation ax+by+cz+d=0
            inlier_indices: indices of points belonging to the floor plane
        """
        if len(points) < 10:
            return np.array([0, 1, 0, 0], dtype=np.float64), np.array([], dtype=int)

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

            plane, inliers = pcd.segment_plane(
                distance_threshold=self.floor_distance_threshold,
                ransac_n=self.floor_ransac_n,
                num_iterations=self.floor_num_iterations,
            )

            plane = np.array(plane)
            inlier_indices = np.array(inliers)

            # Ensure normal points "up" (positive Y component)
            if plane[1] < 0:
                plane = -plane

            print(
                colored(
                    f"[WallDetector] Floor plane normal: [{plane[0]:.3f}, {plane[1]:.3f}, {plane[2]:.3f}], "
                    f"inliers: {len(inlier_indices)}",
                    "green",
                )
            )

            return plane, inlier_indices

        except ImportError:
            print(
                colored(
                    "[WallDetector] Open3D not available, using fallback floor detection",
                    "yellow",
                )
            )
            return self._fallback_floor_detection(points)

    def _fallback_floor_detection(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback floor detection using Y-coordinate histogram."""
        y_coords = points[:, 1]
        y_min = np.percentile(y_coords, 5)
        y_max = np.percentile(y_coords, 95)
        height_range = max(y_max - y_min, 1e-6)

        # Floor is the bottom 20% of the point cloud
        floor_threshold = y_min + 0.2 * height_range
        inlier_mask = y_coords <= floor_threshold
        inlier_indices = np.where(inlier_mask)[0]

        # Approximate floor plane as horizontal at mean floor Y
        floor_y = y_coords[inlier_mask].mean() if inlier_mask.any() else y_min
        plane = np.array([0, 1, 0, -floor_y])

        return plane, inlier_indices

    def detect_walls_from_depth(
        self,
        depth: np.ndarray,
        fx: float = 500.0,
        fy: float = 500.0,
        gradient_threshold: float = 0.3,
        min_line_length_ratio: float = 0.08,
        max_line_gap_ratio: float = 0.03,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect wall line segments from a depth map using depth gradients + LSD.

        Walls appear as depth discontinuities (wall-floor, wall-wall edges).
        Extracts line segments and projects them to ground-plane coordinates.

        Args:
            depth: Depth map (H, W) in meters
            fx: Focal length X in pixels
            fy: Focal length Y in pixels
            gradient_threshold: Minimum gradient magnitude for edge detection
            min_line_length_ratio: Minimum line length as fraction of image width
            max_line_gap_ratio: Maximum gap between line segments to merge

        Returns:
            List of (start_2d, end_2d) tuples in metric ground-plane coordinates
        """
        h, w = depth.shape

        # Compute depth gradient magnitude
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize gradient
        if grad_mag.max() > 0:
            grad_norm = grad_mag / grad_mag.max()
        else:
            return []

        # Threshold to get edges
        edges = (grad_norm > gradient_threshold).astype(np.uint8) * 255

        # Apply morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Detect line segments using Hough
        min_line_length = int(w * min_line_length_ratio)
        max_line_gap = int(w * max_line_gap_ratio)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=max(min_line_length, 10),
            maxLineGap=max(max_line_gap, 5),
        )

        if lines is None:
            return []

        # Project pixel line segments to ground-plane coordinates
        cx, cy = w / 2.0, h / 2.0
        segments = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Get depth at line endpoints (use local median for robustness)
            d1 = self._sample_depth(depth, x1, y1)
            d2 = self._sample_depth(depth, x2, y2)

            if d1 <= 0 or d2 <= 0:
                continue

            # Back-project to 3D ground plane (X, Z)
            gx1 = (x1 - cx) * d1 / fx
            gz1 = d1
            gx2 = (x2 - cx) * d2 / fx
            gz2 = d2

            segments.append((np.array([gx1, gz1]), np.array([gx2, gz2])))

        print(
            colored(
                f"[WallDetector] Detected {len(segments)} wall segments from depth",
                "green" if segments else "yellow",
            )
        )

        return segments

    def _sample_depth(
        self, depth: np.ndarray, x: int, y: int, radius: int = 3
    ) -> float:
        """Sample depth at a point using local median for robustness."""
        h, w = depth.shape
        y0 = max(0, int(y) - radius)
        y1 = min(h, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(w, int(x) + radius + 1)

        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > 0.1) & (patch < 20.0) & np.isfinite(patch)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def align_walls_manhattan(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        angle_tolerance_deg: float = 15.0,
        merge_distance: float = 0.2,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Snap wall segments to Manhattan World (two perpendicular directions).

        Finds the two dominant perpendicular directions from segment angles,
        then snaps each segment to the nearest axis. Merges colinear segments.

        Args:
            segments: List of (start, end) 2D point pairs
            angle_tolerance_deg: Max deviation from grid axis (degrees)
            merge_distance: Max perpendicular distance to merge colinear segments

        Returns:
            Aligned and merged wall segments
        """
        if len(segments) < 2:
            return segments

        # Compute angles of each segment
        angles = []
        for start, end in segments:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            angle = np.degrees(np.arctan2(dy, dx)) % 180  # 0-180
            angles.append(angle)

        angles = np.array(angles)

        # Find dominant direction using histogram
        hist, bin_edges = np.histogram(angles, bins=36, range=(0, 180))

        # Smooth histogram to find peaks
        from scipy.ndimage import gaussian_filter1d

        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1.5)
        peak_bin = np.argmax(hist_smooth)
        dominant_angle = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2

        # Second dominant direction is ~90 degrees from first
        perp_angle = (dominant_angle + 90) % 180

        print(
            colored(
                f"[WallDetector] Manhattan directions: {dominant_angle:.1f}° and {perp_angle:.1f}°",
                "cyan",
            )
        )

        # Snap each segment to the nearest axis
        aligned = []
        for (start, end), angle in zip(segments, angles):
            # Determine which axis this segment is closest to
            diff_dom = min(
                abs(angle - dominant_angle), 180 - abs(angle - dominant_angle)
            )
            diff_perp = min(abs(angle - perp_angle), 180 - abs(angle - perp_angle))

            if diff_dom > angle_tolerance_deg and diff_perp > angle_tolerance_deg:
                continue  # Skip segments that don't align with either axis

            if diff_dom <= diff_perp:
                target_angle = dominant_angle
            else:
                target_angle = perp_angle

            snapped = self._snap_segment_to_angle(start, end, target_angle)
            aligned.append(snapped)

        # Merge colinear segments
        aligned = self._merge_colinear(aligned, merge_distance)

        print(
            colored(
                f"[WallDetector] Aligned {len(aligned)} wall segments",
                "green",
            )
        )

        return aligned

    def _snap_segment_to_angle(
        self, start: np.ndarray, end: np.ndarray, target_angle_deg: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Snap a segment to a target angle while preserving its midpoint and length."""
        mid = (start + end) / 2
        length = np.linalg.norm(end - start)
        rad = np.radians(target_angle_deg)
        direction = np.array([np.cos(rad), np.sin(rad)])

        new_start = mid - direction * length / 2
        new_end = mid + direction * length / 2

        return (new_start, new_end)

    def _merge_colinear(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        distance_threshold: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Merge approximately colinear wall segments."""
        if len(segments) <= 1:
            return segments

        merged = list(segments)
        changed = True

        while changed:
            changed = False
            new_merged = []
            used = set()

            for i in range(len(merged)):
                if i in used:
                    continue
                seg_i = merged[i]
                best_j = -1
                best_dist = float("inf")

                for j in range(i + 1, len(merged)):
                    if j in used:
                        continue
                    seg_j = merged[j]

                    # Check if segments are approximately colinear
                    dist = self._segment_perpendicular_distance(seg_i, seg_j)
                    if dist < distance_threshold and dist < best_dist:
                        best_j = j
                        best_dist = dist

                if best_j >= 0:
                    # Merge: take the union of endpoints along the line direction
                    combined = self._merge_two_segments(merged[i], merged[best_j])
                    new_merged.append(combined)
                    used.add(i)
                    used.add(best_j)
                    changed = True
                else:
                    new_merged.append(seg_i)
                    used.add(i)

            merged = new_merged

        return merged

    def _segment_perpendicular_distance(
        self,
        seg1: Tuple[np.ndarray, np.ndarray],
        seg2: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Average perpendicular distance between two segments."""
        s1_start, s1_end = seg1
        s2_start, s2_end = seg2

        # Direction of seg1
        d = s1_end - s1_start
        length = np.linalg.norm(d)
        if length < 1e-6:
            return float("inf")
        d_norm = d / length

        # Perpendicular component: distance from seg2 endpoints to seg1 line
        perp = np.array([-d_norm[1], d_norm[0]])
        d1 = abs(np.dot(s2_start - s1_start, perp))
        d2 = abs(np.dot(s2_end - s1_start, perp))

        return (d1 + d2) / 2

    def _merge_two_segments(
        self,
        seg1: Tuple[np.ndarray, np.ndarray],
        seg2: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge two colinear segments by projecting to a common line."""
        all_points = np.array([seg1[0], seg1[1], seg2[0], seg2[1]])
        direction = seg1[1] - seg1[0]
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return seg1

        d_norm = direction / length

        # Project all 4 endpoints onto the line direction
        origin = seg1[0]
        projections = np.array([np.dot(p - origin, d_norm) for p in all_points])

        # Take min and max projections as new endpoints
        min_proj = projections.min()
        max_proj = projections.max()

        # Compute perpendicular average for the merged line position
        perp = np.array([-d_norm[1], d_norm[0]])
        perp_offsets = np.array([np.dot(p - origin, perp) for p in all_points])
        avg_perp = perp_offsets.mean()

        new_origin = origin + avg_perp * perp
        new_start = new_origin + min_proj * d_norm
        new_end = new_origin + max_proj * d_norm

        return (new_start, new_end)
