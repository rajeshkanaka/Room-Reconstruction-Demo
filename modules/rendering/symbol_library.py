"""
Architectural Symbol Library

Reusable architectural drawing symbols for floor plan renderers.
Returns coordinate data (renderer-agnostic) that SVG/DXF/PNG renderers consume.
"""

import numpy as np
from typing import List, Dict, Tuple


class SymbolLibrary:
    """
    Standard architectural symbols for floor plans.

    Each method returns a dict of drawing primitives (lines, arcs, fills)
    that renderers can convert to their format (SVG paths, DXF entities, etc.).
    """

    def door_symbol(
        self,
        position: np.ndarray,
        width: float,
        wall_direction: np.ndarray,
        swing: str = "left",
        thickness: float = 0.15,
    ) -> Dict:
        """
        Door symbol: gap in wall + 90-degree swing arc.

        Args:
            position: Center point of door on wall line (2D)
            width: Door width in meters
            wall_direction: Unit vector along the wall
            swing: "left", "right", or "double"
            thickness: Wall thickness for gap

        Returns:
            Dict with 'gap', 'arc', 'lines' primitives
        """
        wall_dir = wall_direction / max(np.linalg.norm(wall_direction), 1e-6)
        perp = np.array([-wall_dir[1], wall_dir[0]])

        half_w = width / 2
        gap_start = position - wall_dir * half_w
        gap_end = position + wall_dir * half_w

        # Arc center is at one edge of the door gap
        if swing == "left":
            arc_center = gap_start
            arc_start_angle = np.degrees(np.arctan2(wall_dir[1], wall_dir[0]))
            arc_end_angle = arc_start_angle + 90
        elif swing == "right":
            arc_center = gap_end
            arc_start_angle = np.degrees(np.arctan2(-wall_dir[1], -wall_dir[0]))
            arc_end_angle = arc_start_angle - 90
        else:  # double
            arc_center = position
            arc_start_angle = 0
            arc_end_angle = 180

        return {
            "type": "door",
            "gap": {"start": gap_start, "end": gap_end},
            "arc": {
                "center": arc_center,
                "radius": width,
                "start_angle": arc_start_angle,
                "end_angle": arc_end_angle,
            },
            "swing": swing,
            "width": width,
        }

    def window_symbol(
        self,
        position: np.ndarray,
        width: float,
        wall_direction: np.ndarray,
        thickness: float = 0.15,
    ) -> Dict:
        """
        Window symbol: triple parallel lines in wall gap.

        Args:
            position: Center point of window on wall line (2D)
            width: Window width in meters
            wall_direction: Unit vector along the wall
            thickness: Wall thickness

        Returns:
            Dict with 'gap', 'lines' primitives
        """
        wall_dir = wall_direction / max(np.linalg.norm(wall_direction), 1e-6)
        perp = np.array([-wall_dir[1], wall_dir[0]])

        half_w = width / 2
        gap_start = position - wall_dir * half_w
        gap_end = position + wall_dir * half_w

        # Three parallel lines across the gap (glass panes)
        offsets = [-thickness / 3, 0, thickness / 3]
        lines = []
        for offset in offsets:
            p1 = gap_start + perp * offset
            p2 = gap_end + perp * offset
            lines.append((p1, p2))

        return {
            "type": "window",
            "gap": {"start": gap_start, "end": gap_end},
            "lines": lines,
            "width": width,
        }

    def dimension_tick(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        size: float = 0.1,
    ) -> Dict:
        """
        Dimension tick mark: 45-degree architectural slash.

        Args:
            position: Tick position (2D)
            direction: Direction perpendicular to dimension line
            size: Tick size in meters

        Returns:
            Dict with 'line' primitive (start, end)
        """
        # 45-degree tick relative to the dimension direction
        d = direction / max(np.linalg.norm(direction), 1e-6)
        perp = np.array([-d[1], d[0]])

        # Rotated 45 degrees
        tick_dir = (d + perp) / np.sqrt(2)
        start = position - tick_dir * size / 2
        end = position + tick_dir * size / 2

        return {
            "type": "tick",
            "line": {"start": start, "end": end},
        }

    def scale_bar(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        total_length: float = 2.0,
        divisions: int = 4,
    ) -> Dict:
        """
        Graphical scale bar with metric markings.

        Args:
            origin: Start point of scale bar (2D)
            direction: Direction unit vector
            total_length: Total length in meters
            divisions: Number of equal divisions

        Returns:
            Dict with 'lines', 'fills', 'labels' primitives
        """
        d = direction / max(np.linalg.norm(direction), 1e-6)
        perp = np.array([-d[1], d[0]])
        bar_height = 0.08  # Height of scale bar

        segment_len = total_length / divisions
        lines = []
        fills = []
        labels = []

        # Outer border
        p0 = origin
        p1 = origin + d * total_length
        p2 = p1 + perp * bar_height
        p3 = origin + perp * bar_height
        lines.append({"points": [p0, p1, p2, p3, p0], "closed": True})

        # Alternating fills and division lines
        for i in range(divisions):
            seg_start = origin + d * (i * segment_len)
            seg_end = origin + d * ((i + 1) * segment_len)

            if i % 2 == 0:
                fills.append(
                    {
                        "corners": [
                            seg_start,
                            seg_end,
                            seg_end + perp * bar_height,
                            seg_start + perp * bar_height,
                        ],
                        "fill": "black",
                    }
                )

            # Division tick
            lines.append(
                {
                    "points": [seg_end, seg_end + perp * bar_height],
                    "closed": False,
                }
            )

            # Label at division
            label_pos = seg_end - perp * 0.05
            labels.append(
                {
                    "position": label_pos,
                    "text": f"{(i + 1) * segment_len:.1f}m",
                }
            )

        # Zero label
        labels.insert(
            0,
            {
                "position": origin - perp * 0.05,
                "text": "0",
            },
        )

        return {
            "type": "scale_bar",
            "lines": lines,
            "fills": fills,
            "labels": labels,
            "total_length": total_length,
        }

    def north_arrow(
        self,
        center: np.ndarray,
        size: float = 0.5,
        angle_deg: float = 0.0,
    ) -> Dict:
        """
        North arrow symbol.

        Args:
            center: Center point (2D)
            size: Arrow size in meters
            angle_deg: North angle in degrees (0 = up)

        Returns:
            Dict with 'polygon', 'label' primitives
        """
        rad = np.radians(angle_deg)
        half = size / 2

        # Arrow points (pointing up by default, rotated by angle)
        tip = center + np.array([np.sin(rad), np.cos(rad)]) * half
        left = (
            center
            + np.array(
                [
                    np.sin(rad - np.radians(150)),
                    np.cos(rad - np.radians(150)),
                ]
            )
            * half
            * 0.6
        )
        right = (
            center
            + np.array(
                [
                    np.sin(rad + np.radians(150)),
                    np.cos(rad + np.radians(150)),
                ]
            )
            * half
            * 0.6
        )

        return {
            "type": "north_arrow",
            "polygon": {
                "points": [tip, left, center, right],
                "fill_left": "black",
                "fill_right": "white",
            },
            "label": {
                "position": tip + np.array([np.sin(rad), np.cos(rad)]) * 0.15,
                "text": "N",
            },
        }
