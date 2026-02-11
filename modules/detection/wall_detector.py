"""
Wall Detection Module

Detects walls from point clouds and depth maps using:
- RANSAC floor plane detection (replaces fixed height slice)
- Vertical plane fitting in 3D point cloud (primary wall extraction)
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
    2. detect_walls_from_point_cloud_planes() — 3D planes → 2D segments
    3. detect_walls_from_depth() — depth gradient → line segments (fallback)
    4. align_walls_manhattan() — snap to perpendicular grid
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

    def detect_floor_plane(
        self,
        points: np.ndarray,
        up_axis: Optional[np.ndarray] = None,
        min_up_dot: float = 0.75,
        max_candidates: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        if up_axis is None:
            up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-12)

        try:
            import open3d as o3d

            remaining = np.arange(len(points))
            candidates = []

            for _ in range(max_candidates):
                if len(remaining) < max(self.floor_ransac_n * 3, 30):
                    break

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[remaining].astype(np.float64))

                plane, inliers_local = pcd.segment_plane(
                    distance_threshold=self.floor_distance_threshold,
                    ransac_n=self.floor_ransac_n,
                    num_iterations=self.floor_num_iterations,
                )

                if len(inliers_local) < 20:
                    break

                plane = np.array(plane, dtype=np.float64)
                normal = plane[:3]
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-9:
                    break
                normal = normal / normal_norm
                up_dot = float(abs(np.dot(normal, up_axis)))

                inlier_indices = remaining[np.array(inliers_local, dtype=int)]
                heights = points[inlier_indices] @ up_axis
                median_height = float(np.median(heights))

                if np.dot(plane[:3], up_axis) < 0:
                    plane = -plane

                candidates.append(
                    {
                        "plane": plane,
                        "inliers": inlier_indices,
                        "up_dot": up_dot,
                        "height": median_height,
                    }
                )

                keep_mask = np.ones(len(remaining), dtype=bool)
                keep_mask[np.array(inliers_local, dtype=int)] = False
                remaining = remaining[keep_mask]

            if not candidates:
                return self._fallback_floor_detection(points)

            horizontal = [c for c in candidates if c["up_dot"] >= min_up_dot]
            if horizontal:
                # Prefer the lowest plausible horizontal plane as the floor.
                min_inliers = max(20, int(0.02 * len(points)))
                plausible = [c for c in horizontal if len(c["inliers"]) >= min_inliers]
                floor = min(plausible or horizontal, key=lambda c: c["height"])
            else:
                # Fallback to the most horizontal + well-supported plane.
                floor = max(
                    candidates, key=lambda c: c["up_dot"] * len(c["inliers"])
                )

            print(
                colored(
                    f"[WallDetector] Floor plane normal: "
                    f"[{floor['plane'][0]:.3f}, {floor['plane'][1]:.3f}, {floor['plane'][2]:.3f}], "
                    f"inliers: {len(floor['inliers'])}, up_dot={floor['up_dot']:.3f}",
                    "green",
                )
            )

            return floor["plane"], floor["inliers"]

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

    def detect_walls_from_point_cloud_planes(
        self,
        points: np.ndarray,
        floor_height: Optional[float],
        up_axis: Optional[np.ndarray] = None,
        distance_threshold: float = 0.05,
        min_height_span: float = 1.2,
        min_length: float = 0.8,
        max_up_dot: float = 0.25,
        max_planes: int = 8,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect wall segments directly from 3D vertical planes.

        This is a structure-first alternative to depth-edge extraction.
        Each detected vertical plane is projected into a top-down line segment.

        Args:
            points: Nx3 world-space point cloud.
            floor_height: Estimated floor Y in world-space.
            up_axis: World up direction (default +Y).
            distance_threshold: RANSAC inlier threshold in meters.
            min_height_span: Minimum vertical span for a plane to count as wall.
            min_length: Minimum 2D wall segment length in meters.
            max_up_dot: Maximum |dot(normal, up)| allowed for vertical walls.
            max_planes: Maximum number of plane candidates to evaluate.

        Returns:
            List of (start_2d, end_2d) wall segments in world XZ coordinates.
        """
        if points is None or len(points) < 120:
            return []

        if up_axis is None:
            up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-12)

        finite_mask = np.isfinite(points).all(axis=1)
        candidates = points[finite_mask]
        if len(candidates) < 120:
            return []

        if floor_height is not None:
            # Keep likely wall points above floor, below likely ceiling.
            h = candidates @ up_axis
            mask_h = (h > floor_height + 0.12) & (h < floor_height + 3.6)
            candidates = candidates[mask_h]

        if len(candidates) < 120:
            return []

        try:
            import open3d as o3d
        except ImportError:
            return []

        remaining = np.arange(len(candidates))
        segments: List[Tuple[np.ndarray, np.ndarray]] = []

        for _ in range(max_planes):
            if len(remaining) < 80:
                break

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                candidates[remaining].astype(np.float64)
            )
            plane, inliers_local = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=max(self.floor_num_iterations, 800),
            )

            if len(inliers_local) < 60:
                break

            inliers_local = np.array(inliers_local, dtype=int)
            inlier_idx = remaining[inliers_local]
            plane = np.array(plane, dtype=np.float64)
            normal = plane[:3]
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-9:
                remaining = np.delete(remaining, inliers_local)
                continue
            normal = normal / normal_norm
            up_dot = float(abs(np.dot(normal, up_axis)))

            # Vertical wall planes should have normals nearly perpendicular to up.
            if up_dot > max_up_dot:
                remaining = np.delete(remaining, inliers_local)
                continue

            inlier_points = candidates[inlier_idx]
            heights = inlier_points @ up_axis
            height_span = float(np.percentile(heights, 95) - np.percentile(heights, 5))
            if height_span < min_height_span:
                remaining = np.delete(remaining, inliers_local)
                continue

            segment = self._plane_inliers_to_topdown_segment(
                inlier_points, min_length=min_length
            )
            if segment is not None:
                segments.append(segment)

            remaining = np.delete(remaining, inliers_local)

        if len(segments) > 1:
            segments = self._merge_colinear(segments, distance_threshold=0.15)

        print(
            colored(
                f"[WallDetector] Detected {len(segments)} wall segments from 3D planes",
                "green" if segments else "yellow",
            )
        )
        return segments

    def _plane_inliers_to_topdown_segment(
        self,
        inlier_points: np.ndarray,
        min_length: float = 0.8,
        trim_percentile: float = 5.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Project plane inliers to XZ and fit a robust dominant line segment."""
        if inlier_points is None or len(inlier_points) < 20:
            return None

        pts2d = inlier_points[:, [0, 2]].astype(np.float64)
        if not np.isfinite(pts2d).all():
            pts2d = pts2d[np.isfinite(pts2d).all(axis=1)]
        if len(pts2d) < 20:
            return None

        centered = pts2d - pts2d.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, int(np.argmax(eigvals))]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        orth = np.array([-direction[1], direction[0]], dtype=np.float64)

        t = pts2d @ direction
        o = pts2d @ orth
        t0, t1 = np.percentile(t, [trim_percentile, 100.0 - trim_percentile])
        o_mid = float(np.median(o))

        length = float(t1 - t0)
        if length < min_length:
            return None

        start = direction * float(t0) + orth * o_mid
        end = direction * float(t1) + orth * o_mid
        return start.astype(np.float64), end.astype(np.float64)

    def detect_wall_lines_from_depth(
        self,
        depth: np.ndarray,
        gradient_threshold: float = 0.3,
        min_line_length_ratio: float = 0.08,
        max_line_gap_ratio: float = 0.03,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect wall line segments (pixel coordinates) from depth gradients.

        Walls appear as depth discontinuities (wall-floor, wall-wall edges).

        Args:
            depth: Depth map (H, W) in meters
            gradient_threshold: Minimum gradient magnitude for edge detection
            min_line_length_ratio: Minimum line length as fraction of image width
            max_line_gap_ratio: Maximum gap between line segments to merge

        Returns:
            List of (x1, y1, x2, y2) line segments in depth-map pixel space
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

        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append((int(x1), int(y1), int(x2), int(y2)))

        print(
            colored(
                f"[WallDetector] Detected {len(segments)} wall segments from depth",
                "green" if segments else "yellow",
            )
        )

        return segments

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
        Backward-compatible API that returns approximate camera-local XZ segments.
        """
        lines = self.detect_wall_lines_from_depth(
            depth,
            gradient_threshold=gradient_threshold,
            min_line_length_ratio=min_line_length_ratio,
            max_line_gap_ratio=max_line_gap_ratio,
        )
        if not lines:
            return []

        h, w = depth.shape
        cx, cy = w / 2.0, h / 2.0
        segments = []
        for x1, y1, x2, y2 in lines:
            d1 = self._sample_depth(depth, x1, y1)
            d2 = self._sample_depth(depth, x2, y2)
            if d1 <= 0 or d2 <= 0:
                continue

            x1_cam = (x1 - cx) * d1 / fx
            y1_cam = (y1 - cy) * d1 / fy
            z1_cam = d1
            x2_cam = (x2 - cx) * d2 / fx
            y2_cam = (y2 - cy) * d2 / fy
            z2_cam = d2

            # Approximate top-down by dropping Y.
            _ = y1_cam, y2_cam
            segments.append((np.array([x1_cam, z1_cam]), np.array([x2_cam, z2_cam])))
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
