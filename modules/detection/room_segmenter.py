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

        # Collect all endpoints
        points = []
        for start, end in segments:
            points.append(start)
            points.append(end)

        # Snap nearby points together
        points = self._snap_endpoints(points)

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

            if best_idx < 0 or best_dist > self.snap_tolerance * 5:
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

    def _snap_endpoints(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Snap nearby endpoints to the same location."""
        snapped = [p.copy() for p in points]

        for i in range(len(snapped)):
            for j in range(i + 1, len(snapped)):
                if np.linalg.norm(snapped[i] - snapped[j]) < self.snap_tolerance:
                    avg = (snapped[i] + snapped[j]) / 2
                    snapped[i] = avg
                    snapped[j] = avg

        return snapped

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
