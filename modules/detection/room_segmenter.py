"""
Room Segmenter Module

Builds room polygons from wall line segments using Shapely.
Handles non-convex rooms (L-shaped, U-shaped, T-shaped).
"""

import numpy as np
from typing import List, Tuple, Optional
from termcolor import colored

from modules.geometry.floor_plan_model import RoomPolygon, WallSegment


class RoomSegmenter:
    """
    Extracts room polygons from wall line segments.

    Strategy:
    1. Build a polygon from wall segment endpoints
    2. Use Shapely to create a valid polygon (handles self-intersections)
    3. Supports non-convex room shapes
    """

    def __init__(self, snap_tolerance: float = 0.15):
        """
        Args:
            snap_tolerance: Max distance (meters) to snap nearby endpoints together
        """
        self.snap_tolerance = snap_tolerance

    def extract_rooms(
        self,
        wall_segments: List[Tuple[np.ndarray, np.ndarray]],
        room_name: str = "Room",
    ) -> List[RoomPolygon]:
        """
        Extract room polygons from wall line segments.

        Builds a closed polygon by ordering wall segments into a connected
        boundary. Uses Shapely for robust polygon operations.

        Args:
            wall_segments: List of (start_2d, end_2d) tuples in meters
            room_name: Name for the room

        Returns:
            List of RoomPolygon instances
        """
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import polygonize, unary_union

        if len(wall_segments) < 3:
            print(
                colored(
                    f"[RoomSegmenter] Too few segments ({len(wall_segments)}) for polygon",
                    "yellow",
                )
            )
            return self._fallback_convex_hull(wall_segments, room_name)

        # Try ordered boundary approach first
        ordered = self._order_segments_into_boundary(wall_segments)
        if ordered is not None and len(ordered) >= 3:
            try:
                poly = Polygon(ordered)
                if not poly.is_valid:
                    poly = poly.buffer(0)  # Fix self-intersections

                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)

                if poly.is_valid and poly.area > 0.5:
                    boundary = np.array(
                        poly.exterior.coords[:-1]
                    )  # Remove closing point
                    room = RoomPolygon(boundary=boundary, name=room_name)
                    print(
                        colored(
                            f"[RoomSegmenter] Extracted room: {room.area:.1f}m^2",
                            "green",
                        )
                    )
                    return [room]
            except Exception as e:
                print(
                    colored(
                        f"[RoomSegmenter] Ordered boundary failed: {e}",
                        "yellow",
                    )
                )

        # Fallback: try polygonize from line segments
        try:
            from shapely.geometry import LineString

            lines = []
            for start, end in wall_segments:
                lines.append(LineString([start, end]))

            merged = unary_union(lines)
            polygons = list(polygonize(merged))

            if polygons:
                # Take the largest polygon
                largest = max(polygons, key=lambda p: p.area)
                boundary = np.array(largest.exterior.coords[:-1])
                room = RoomPolygon(boundary=boundary, name=room_name)
                print(
                    colored(
                        f"[RoomSegmenter] Extracted room via polygonize: {room.area:.1f}m^2",
                        "green",
                    )
                )
                return [room]
        except Exception as e:
            print(
                colored(
                    f"[RoomSegmenter] Polygonize failed: {e}",
                    "yellow",
                )
            )

        # Final fallback: convex hull
        return self._fallback_convex_hull(wall_segments, room_name)

    def _order_segments_into_boundary(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """
        Order wall segments into a connected boundary polygon.

        Greedily connects segments by finding the nearest endpoint.
        """
        if not segments:
            return None

        segments = self._snap_segments(segments)

        # Build adjacency from snapped segments
        remaining = list(range(len(segments)))
        ordered = [segments[0][0].copy()]
        current = segments[0][1].copy()
        remaining.remove(0)

        for _ in range(len(segments)):
            ordered.append(current.copy())

            if not remaining:
                break

            # Find the segment whose start or end is closest to current
            best_idx = -1
            best_dist = float("inf")
            use_reverse = False

            for idx in remaining:
                start, end = segments[idx]
                d_start = np.linalg.norm(current - start)
                d_end = np.linalg.norm(current - end)

                if d_start < best_dist:
                    best_dist = d_start
                    best_idx = idx
                    use_reverse = False
                if d_end < best_dist:
                    best_dist = d_end
                    best_idx = idx
                    use_reverse = True

            # Avoid long jumps that create unrealistic polygon shortcuts.
            if best_idx < 0 or best_dist > self.snap_tolerance * 3.5:
                break

            remaining.remove(best_idx)
            start, end = segments[best_idx]
            if use_reverse:
                current = start.copy()
            else:
                current = end.copy()

        if len(ordered) >= 3:
            return np.array(ordered)
        return None

    def _snap_segments(
        self, segments: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Snap nearby segment endpoints and rebuild snapped segments."""
        if not segments:
            return []

        points = []
        for start, end in segments:
            points.append(start.copy())
            points.append(end.copy())

        snapped = self._snap_endpoints(points)
        snapped_segments = []
        for i in range(0, len(snapped), 2):
            snapped_segments.append((snapped[i], snapped[i + 1]))

        return snapped_segments

    def _snap_endpoints(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Snap nearby endpoints to cluster centroids.

        Uses union-find clustering to avoid order-dependent pairwise averaging.
        """
        if not points:
            return []

        pts = np.array(points, dtype=np.float64)
        n = len(pts)
        parent = list(range(n))

        def _find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i: int, j: int) -> None:
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[rj] = ri

        tol = float(max(self.snap_tolerance, 1e-4))
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts[i] - pts[j]) <= tol:
                    _union(i, j)

        clusters = {}
        for i in range(n):
            root = _find(i)
            clusters.setdefault(root, []).append(i)

        snapped = [p.copy() for p in pts]
        for members in clusters.values():
            center = np.mean(pts[members], axis=0)
            for idx in members:
                snapped[idx] = center.copy()

        return snapped

    def optimize_wall_graph(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        min_length: float = 0.3,
        min_area: float = 0.5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Optimize wall segments into a cleaner, closed graph.

        Steps:
        1) snap endpoints
        2) remove short/noisy segments
        3) deduplicate near-identical segments
        4) try polygon closure via Shapely polygonize
        5) fallback to ordered boundary closure if polygonize fails
        """
        if not segments:
            return []

        snapped = self._snap_segments(segments)
        snapped = [
            (s.copy(), e.copy())
            for s, e in snapped
            if np.linalg.norm(e - s) >= min_length
        ]
        if not snapped:
            return []

        deduped = self._deduplicate_segments(
            snapped, tol=max(0.05, self.snap_tolerance * 0.7)
        )
        if not deduped:
            return []

        # Snap near-perpendicular junctions to their line intersection to close gaps.
        closed = self._snap_perpendicular_junctions(
            deduped,
            junction_tolerance=max(0.12, self.snap_tolerance * 1.2),
            min_length=min_length,
        )
        deduped = self._deduplicate_segments(
            closed, tol=max(0.05, self.snap_tolerance * 0.7)
        )
        deduped = [
            (s.copy(), e.copy())
            for s, e in deduped
            if np.linalg.norm(e - s) >= min_length
        ]
        if not deduped:
            return []

        polygonized = self._polygonize_to_boundary_segments(deduped, min_area=min_area)
        if polygonized is not None and len(polygonized) >= 3:
            print(
                colored(
                    f"[RoomSegmenter] Optimized wall graph to {len(polygonized)} closed segments",
                    "green",
                )
            )
            return polygonized

        ordered = self._order_segments_into_boundary(deduped)
        if ordered is not None and len(ordered) >= 3:
            boundary_segments = self._boundary_to_segments(ordered)
            if boundary_segments:
                print(
                    colored(
                        "[RoomSegmenter] Optimized wall graph via ordered boundary",
                        "yellow",
                    )
                )
                return boundary_segments

        return deduped

    def _snap_perpendicular_junctions(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        junction_tolerance: float,
        min_length: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Close small endpoint gaps at near-perpendicular wall junctions.

        For each segment pair, if the infinite lines intersect and that
        intersection is close to one endpoint of each segment, snap those
        endpoints to the shared corner.
        """
        if len(segments) < 2:
            return segments

        snapped = [(s.copy(), e.copy()) for s, e in segments]
        tol = float(max(junction_tolerance, 1e-3))

        for i in range(len(snapped)):
            for j in range(i + 1, len(snapped)):
                s1, e1 = snapped[i]
                s2, e2 = snapped[j]
                len1 = float(np.linalg.norm(e1 - s1))
                len2 = float(np.linalg.norm(e2 - s2))
                if len1 < max(0.15, 0.6 * min_length) or len2 < max(
                    0.15, 0.6 * min_length
                ):
                    continue

                angle_diff = self._segment_angle_difference_deg((s1, e1), (s2, e2))
                if angle_diff < 35.0 or angle_diff > 145.0:
                    continue

                corner = self._line_intersection_2d(s1, e1, s2, e2)
                if corner is None:
                    continue

                d1s = float(np.linalg.norm(corner - s1))
                d1e = float(np.linalg.norm(corner - e1))
                d2s = float(np.linalg.norm(corner - s2))
                d2e = float(np.linalg.norm(corner - e2))
                if min(d1s, d1e) > tol or min(d2s, d2e) > tol:
                    continue

                if d1s <= d1e:
                    s1 = corner.copy()
                else:
                    e1 = corner.copy()

                if d2s <= d2e:
                    s2 = corner.copy()
                else:
                    e2 = corner.copy()

                snapped[i] = (s1, e1)
                snapped[j] = (s2, e2)

        return snapped

    @staticmethod
    def _line_intersection_2d(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return intersection of two infinite 2D lines, or None if parallel."""
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        x3, y3 = float(p3[0]), float(p3[1])
        x4, y4 = float(p4[0]), float(p4[1])

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            return None

        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / den
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / den
        return np.array([px, py], dtype=np.float64)

    @staticmethod
    def _segment_angle_difference_deg(
        seg_a: Tuple[np.ndarray, np.ndarray],
        seg_b: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Smallest orientation difference in degrees (0..180)."""
        a = seg_a[1] - seg_a[0]
        b = seg_b[1] - seg_b[0]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        au = a / na
        bu = b / nb
        dot = float(np.clip(np.dot(au, bu), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot)))
        return min(angle, 180.0 - angle)

    def _polygonize_to_boundary_segments(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        min_area: float = 0.5,
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """Try to convert line segments into a closed boundary using polygonize."""
        if len(segments) < 3:
            return None

        try:
            from shapely.geometry import LineString
            from shapely.ops import polygonize, unary_union
        except Exception:
            return None

        try:
            lines = [LineString([start, end]) for start, end in segments]
            merged = unary_union(lines)
            polygons = list(polygonize(merged))
            if not polygons:
                return None

            poly = max(polygons, key=lambda p: p.area)
            if poly.area < min_area:
                return None

            boundary = np.array(poly.exterior.coords[:-1], dtype=np.float64)
            return self._boundary_to_segments(boundary)
        except Exception:
            return None

    def _boundary_to_segments(
        self,
        boundary: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert boundary points to closed segment list."""
        if boundary is None or len(boundary) < 3:
            return []

        segments = []
        n = len(boundary)
        for i in range(n):
            start = boundary[i].astype(np.float64)
            end = boundary[(i + 1) % n].astype(np.float64)
            if np.linalg.norm(end - start) < 1e-4:
                continue
            segments.append((start, end))
        return segments

    def _deduplicate_segments(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        tol: float = 0.1,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Remove near-duplicate segments (including reversed duplicates)."""
        unique: List[Tuple[np.ndarray, np.ndarray]] = []

        for start, end in segments:
            replaced = False
            length = float(np.linalg.norm(end - start))
            for idx, (u_start, u_end) in enumerate(unique):
                same_dir = (
                    np.linalg.norm(start - u_start) <= tol
                    and np.linalg.norm(end - u_end) <= tol
                )
                rev_dir = (
                    np.linalg.norm(start - u_end) <= tol
                    and np.linalg.norm(end - u_start) <= tol
                )
                if same_dir or rev_dir:
                    u_len = float(np.linalg.norm(u_end - u_start))
                    if length > u_len:
                        unique[idx] = (start.copy(), end.copy())
                    replaced = True
                    break

            if not replaced:
                unique.append((start.copy(), end.copy()))

        return unique

    def _fallback_convex_hull(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        room_name: str,
    ) -> List[RoomPolygon]:
        """Fallback: build convex hull from all segment endpoints."""
        if not segments:
            return []

        all_points = []
        for start, end in segments:
            all_points.append(start)
            all_points.append(end)

        all_points = np.array(all_points)

        if len(all_points) < 3:
            return []

        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(all_points)
            boundary = all_points[hull.vertices]
            room = RoomPolygon(boundary=boundary, name=room_name)

            print(
                colored(
                    f"[RoomSegmenter] Fallback convex hull: {room.area:.1f}m^2",
                    "yellow",
                )
            )
            return [room]
        except Exception:
            return []

    def segments_to_wall_segments(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        thickness: float = 0.15,
    ) -> List[WallSegment]:
        """Convert raw (start, end) tuples to WallSegment dataclasses."""
        return [
            WallSegment(start=start, end=end, thickness=thickness)
            for start, end in segments
        ]

    def rooms_to_wall_segments(
        self,
        rooms: List[RoomPolygon],
        thickness: float = 0.15,
    ) -> List[WallSegment]:
        """
        Convert room polygon boundaries into closed wall segments.

        Using boundary-derived segments produces a connected floor plan
        suitable for architectural rendering and dimensioning.
        """
        wall_segments: List[WallSegment] = []
        for room in rooms:
            boundary = room.boundary
            if boundary is None or len(boundary) < 3:
                continue

            for i in range(len(boundary)):
                start = boundary[i]
                end = boundary[(i + 1) % len(boundary)]
                if np.linalg.norm(end - start) < 1e-4:
                    continue
                wall_segments.append(
                    WallSegment(start=start.copy(), end=end.copy(), thickness=thickness)
                )

        return wall_segments

    def points_to_rectangular_room(
        self,
        points_2d: np.ndarray,
        room_name: str = "Room",
        lower_percentile: float = 3.0,
        upper_percentile: float = 97.0,
        min_span_m: float = 1.0,
    ) -> Optional[RoomPolygon]:
        """
        Build a robust rectangular room from noisy 2D points.

        Useful as a fallback when wall-segment topology is fragmented.
        The rectangle is computed in a PCA-aligned frame with percentile
        trimming to reduce outlier impact.
        """
        if points_2d is None:
            return None

        pts = np.asarray(points_2d, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None

        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if len(pts) < 20:
            return None

        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axis_u = eigvecs[:, order[0]]
        axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
        axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float64)

        proj_u = pts @ axis_u
        proj_v = pts @ axis_v
        u_min, u_max = np.percentile(proj_u, [lower_percentile, upper_percentile])
        v_min, v_max = np.percentile(proj_v, [lower_percentile, upper_percentile])

        span_u = float(u_max - u_min)
        span_v = float(v_max - v_min)
        if span_u < min_span_m or span_v < min_span_m:
            return None

        corners_uv = np.array(
            [
                [u_min, v_min],
                [u_max, v_min],
                [u_max, v_max],
                [u_min, v_max],
            ],
            dtype=np.float64,
        )

        corners_xy = np.array(
            [axis_u * uv[0] + axis_v * uv[1] for uv in corners_uv], dtype=np.float64
        )

        if self._signed_area(corners_xy) < 0:
            corners_xy = corners_xy[::-1]

        room = RoomPolygon(boundary=corners_xy, name=room_name)
        if room.area < 2.0:
            return None

        print(
            colored(
                f"[RoomSegmenter] Rectangular fallback room: {room.area:.1f}m^2",
                "yellow",
            )
        )
        return room

    @staticmethod
    def _signed_area(poly: np.ndarray) -> float:
        """Signed polygon area (>0 for counter-clockwise ordering)."""
        if poly is None or len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
