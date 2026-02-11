"""
Measurement Engine

Computes per-wall lengths, room dimensions, areas, and generates
DimensionLine objects for the FloorPlanModel.
"""

import numpy as np
from typing import List, Dict
from termcolor import colored

from modules.geometry.floor_plan_model import (
    FloorPlanModel,
    WallSegment,
    RoomPolygon,
    DimensionLine,
    DoorOpening,
)


class MeasurementEngine:
    """
    Computes measurements from a FloorPlanModel.

    Produces:
    - Per-wall lengths
    - Room width x depth (bounding box of room polygon)
    - Total area from polygon (Shoelace), not bounding box
    - Metric and imperial values
    - Cross-validation warnings (opposite walls should match in rectangular rooms)
    - DimensionLine objects for renderers
    """

    METERS_TO_FEET = 3.28084
    SQM_TO_SQFT = 10.7639

    def compute_measurements(self, model: FloorPlanModel) -> Dict:
        """
        Compute all measurements for a FloorPlanModel.

        Args:
            model: FloorPlanModel with walls and rooms populated

        Returns:
            Dict with wall_lengths, rooms, overall dimensions, warnings
        """
        wall_lengths = [self._wall_length(w) for w in model.walls]

        rooms_data = []
        for room in model.rooms:
            room_info = self._measure_room(room)
            rooms_data.append(room_info)

        # Overall bounding box
        all_points = []
        for w in model.walls:
            all_points.append(w.start)
            all_points.append(w.end)
        for room in model.rooms:
            for pt in room.boundary:
                all_points.append(pt)

        if all_points:
            all_pts = np.array(all_points)
            overall_min = all_pts.min(axis=0)
            overall_max = all_pts.max(axis=0)
            overall_width = overall_max[0] - overall_min[0]
            overall_depth = overall_max[1] - overall_min[1]
        else:
            overall_width = 0
            overall_depth = 0

        # Cross-validation
        warnings = self._cross_validate(model)

        # Generate DimensionLine objects
        dimension_lines = self._generate_dimension_lines(model)
        model.dimensions = dimension_lines

        result = {
            "wall_lengths": [round(l, 2) for l in wall_lengths],
            "wall_lengths_ft": [
                round(l * self.METERS_TO_FEET, 1) for l in wall_lengths
            ],
            "rooms": rooms_data,
            "overall": {
                "width_m": round(overall_width, 2),
                "depth_m": round(overall_depth, 2),
                "width_ft": round(overall_width * self.METERS_TO_FEET, 1),
                "depth_ft": round(overall_depth * self.METERS_TO_FEET, 1),
            },
            "warnings": warnings,
        }

        return result

    def _wall_length(self, wall: WallSegment) -> float:
        """Compute the length of a wall segment."""
        return float(np.linalg.norm(wall.end - wall.start))

    def _measure_room(self, room: RoomPolygon) -> Dict:
        """Compute measurements for a single room."""
        boundary = room.boundary
        if len(boundary) < 3:
            return {
                "name": room.name,
                "area_sqm": 0,
                "area_sqft": 0,
                "width_m": 0,
                "depth_m": 0,
            }

        area = room.area
        bb_min = boundary.min(axis=0)
        bb_max = boundary.max(axis=0)
        width = bb_max[0] - bb_min[0]
        depth = bb_max[1] - bb_min[1]

        return {
            "name": room.name,
            "area_sqm": round(area, 2),
            "area_sqft": round(area * self.SQM_TO_SQFT, 1),
            "width_m": round(width, 2),
            "depth_m": round(depth, 2),
            "width_ft": round(width * self.METERS_TO_FEET, 1),
            "depth_ft": round(depth * self.METERS_TO_FEET, 1),
        }

    def _cross_validate(self, model: FloorPlanModel) -> List[str]:
        """Check for measurement inconsistencies."""
        warnings = []

        if len(model.walls) == 4:
            lengths = [self._wall_length(w) for w in model.walls]
            # In a rectangle, opposite walls should be similar
            pair1_diff = abs(lengths[0] - lengths[2])
            pair2_diff = abs(lengths[1] - lengths[3])

            if pair1_diff > 0.3:
                warnings.append(
                    f"Opposite walls differ: {lengths[0]:.2f}m vs {lengths[2]:.2f}m"
                )
            if pair2_diff > 0.3:
                warnings.append(
                    f"Opposite walls differ: {lengths[1]:.2f}m vs {lengths[3]:.2f}m"
                )

        return warnings

    def _generate_dimension_lines(self, model: FloorPlanModel) -> List[DimensionLine]:
        """Generate DimensionLine objects for each wall (prefer exterior offsets)."""
        if not model.walls:
            return []

        all_pts = []
        for wall in model.walls:
            all_pts.append(wall.start)
            all_pts.append(wall.end)
        centroid = np.mean(np.array(all_pts), axis=0)

        dims = []
        for wall in model.walls:
            length = self._wall_length(wall)
            if length < 0.5:
                continue

            wall_vec = wall.end - wall.start
            wall_unit = wall_vec / max(length, 1e-12)
            perp = np.array([-wall_unit[1], wall_unit[0]])
            mid = (wall.start + wall.end) / 2
            sign = 1.0 if np.dot(mid - centroid, perp) >= 0 else -1.0
            offset = 0.35 * sign

            dim = DimensionLine(
                start=wall.start.copy(),
                end=wall.end.copy(),
                value_m=length,
                offset=offset,
            )
            dims.append(dim)
        return dims

    def compute_chain_dimensions(
        self,
        wall: WallSegment,
        openings: List[DoorOpening],
    ) -> List[Dict]:
        """
        Compute chain dimensions for a wall with door/window openings.

        Splits the wall into segments separated by openings.
        Example: |--1.2m--| [D 0.9m] |--2.4m--|

        Args:
            wall: The parent wall segment
            openings: Doors/windows along this wall

        Returns:
            List of dicts with type ("wall" or "opening"), length, position
        """
        wall_dir = wall.end - wall.start
        wall_length = np.linalg.norm(wall_dir)

        if wall_length < 1e-6:
            return []

        wall_unit = wall_dir / wall_length

        # Project each opening onto the wall line
        opening_intervals = []
        for opening in openings:
            # Project opening center onto wall
            t = np.dot(opening.position - wall.start, wall_unit)
            half_w = opening.width / 2
            interval_start = max(0, t - half_w)
            interval_end = min(wall_length, t + half_w)
            opening_intervals.append(
                {
                    "start": interval_start,
                    "end": interval_end,
                    "width": opening.width,
                    "type": "door",
                }
            )

        # Sort by position along wall
        opening_intervals.sort(key=lambda x: x["start"])

        # Build chain
        chain = []
        current_pos = 0.0

        for interval in opening_intervals:
            # Wall segment before the opening
            gap = interval["start"] - current_pos
            if gap > 0.01:
                chain.append({"type": "wall", "length": round(gap, 3)})

            # The opening itself
            chain.append(
                {
                    "type": interval["type"],
                    "length": round(interval["width"], 3),
                }
            )

            current_pos = interval["end"]

        # Remaining wall after last opening
        remaining = wall_length - current_pos
        if remaining > 0.01:
            chain.append({"type": "wall", "length": round(remaining, 3)})

        return chain
