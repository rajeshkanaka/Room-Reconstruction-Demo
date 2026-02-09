"""
SVG Floor Plan Renderer

Renders a FloorPlanModel to SVG with architectural line weight hierarchy:
- Heavy (0.6-1.0mm): walls
- Medium (0.3-0.5mm): openings (doors, windows)
- Light (0.1-0.2mm): dimensions, annotations
"""

import numpy as np
import svgwrite
from typing import Optional
from termcolor import colored

from modules.geometry.floor_plan_model import FloorPlanModel
from modules.rendering.symbol_library import SymbolLibrary


class SVGRenderer:
    """
    Renders FloorPlanModel to SVG with proper architectural conventions.
    """

    # Drawing scale: 1 meter = this many SVG units (pixels at 96dpi)
    SCALE = 100  # 1m = 100px

    # Line weights in SVG units
    WALL_WEIGHT = 3.0  # Heavy
    OPENING_WEIGHT = 1.5  # Medium
    DIM_WEIGHT = 0.5  # Light
    ANNO_WEIGHT = 0.3  # Very light

    # Colors
    WALL_COLOR = "#333333"
    WALL_FILL = "#e8e8e8"
    DIM_COLOR = "#cc0000"
    ROOM_FILL = "#f5f9ff"
    GRID_COLOR = "#e0e0e0"

    def __init__(self):
        self.symbols = SymbolLibrary()

    def render(
        self,
        model: FloorPlanModel,
        output_path: str,
        title: str = "Floor Plan",
    ) -> str:
        """
        Render FloorPlanModel to an SVG file.

        Args:
            model: FloorPlanModel to render
            output_path: Path to save SVG file
            title: Drawing title

        Returns:
            Path to saved SVG file
        """
        svg_string = self.render_to_string(model, title=title)

        with open(output_path, "w") as f:
            f.write(svg_string)

        print(colored(f"[SVGRenderer] Saved SVG to: {output_path}", "green"))
        return output_path

    def render_to_string(
        self,
        model: FloorPlanModel,
        title: str = "Floor Plan",
    ) -> str:
        """
        Render FloorPlanModel to an SVG string.

        Args:
            model: FloorPlanModel to render
            title: Drawing title

        Returns:
            SVG content as string
        """
        # Compute bounding box
        bbox = self._compute_bbox(model)
        if bbox is None:
            return "<svg></svg>"

        x_min, y_min, x_max, y_max = [float(v) for v in bbox]
        margin = 1.5  # meters margin for dimensions and annotations

        # SVG canvas dimensions
        width = (x_max - x_min + 2 * margin) * self.SCALE
        height = (y_max - y_min + 2 * margin) * self.SCALE

        dwg = svgwrite.Drawing(
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}",
        )

        # Add a white background
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

        # Transform: flip Y axis (SVG Y goes down, architecture Y goes up)
        # and apply margin offset
        group = dwg.g(
            transform=f"translate({margin * self.SCALE}, {(y_max + margin) * self.SCALE}) scale(1, -1)"
        )

        # Draw room fills
        for room in model.rooms:
            self._draw_room_fill(dwg, group, room, x_min, y_min)

        # Draw walls (heavy line weight)
        for wall in model.walls:
            self._draw_wall(dwg, group, wall)

        # Draw doors
        for door in model.doors:
            self._draw_door(dwg, group, door, model)

        # Draw windows
        for window in model.windows:
            self._draw_window(dwg, group, window, model)

        dwg.add(group)

        # Dimensions (drawn in non-flipped space for text readability)
        dim_group = dwg.g(
            transform=f"translate({margin * self.SCALE}, {margin * self.SCALE})"
        )
        for dim in model.dimensions:
            self._draw_dimension(dwg, dim_group, dim, y_max - y_min)

        dwg.add(dim_group)

        # Room labels (non-flipped for readability)
        label_group = dwg.g(
            transform=f"translate({margin * self.SCALE}, {margin * self.SCALE})"
        )
        for room in model.rooms:
            self._draw_room_label(dwg, label_group, room, y_max - y_min)

        dwg.add(label_group)

        # Title block
        self._draw_title_block(dwg, title, width, height, model.scale)

        # Scale bar
        self._draw_scale_indicator(dwg, width, height)

        # North arrow
        self._draw_north_arrow(dwg, width, model.orientation)

        return dwg.tostring()

    def _compute_bbox(self, model: FloorPlanModel):
        """Compute bounding box of all elements."""
        all_points = []
        for w in model.walls:
            all_points.extend([w.start, w.end])
        for r in model.rooms:
            all_points.extend(r.boundary)

        if not all_points:
            return None

        pts = np.array(all_points)
        return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    def _to_svg(self, x: float, y: float) -> tuple:
        """Convert model coordinates to SVG coordinates."""
        return (float(x * self.SCALE), float(y * self.SCALE))

    def _draw_room_fill(self, dwg, group, room, x_min, y_min):
        """Draw a light fill for room area."""
        points = [
            (float(p[0] * self.SCALE), float(p[1] * self.SCALE)) for p in room.boundary
        ]
        group.add(
            dwg.polygon(
                points=points,
                fill=self.ROOM_FILL,
                stroke="none",
                opacity=0.5,
            )
        )

    def _draw_wall(self, dwg, group, wall):
        """Draw a wall as a double line with fill (representing thickness)."""
        start = wall.start
        end = wall.end
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return

        d = direction / length
        perp = np.array([-d[1], d[0]])
        half_t = wall.thickness / 2

        # Wall outline (rectangle)
        corners = [
            start + perp * half_t,
            end + perp * half_t,
            end - perp * half_t,
            start - perp * half_t,
        ]
        points = [(float(c[0] * self.SCALE), float(c[1] * self.SCALE)) for c in corners]

        group.add(
            dwg.polygon(
                points=points,
                fill=self.WALL_FILL,
                stroke=self.WALL_COLOR,
                stroke_width=self.WALL_WEIGHT,
            )
        )

    def _draw_door(self, dwg, group, door, model):
        """Draw door symbol (gap + arc)."""
        # Find parent wall
        parent_wall = self._find_parent_wall(door.position, model.walls)
        if parent_wall is None:
            return

        wall_dir = parent_wall.end - parent_wall.start
        wall_length = np.linalg.norm(wall_dir)
        if wall_length < 1e-6:
            return
        wall_unit = wall_dir / wall_length

        symbol = self.symbols.door_symbol(
            door.position, door.width, wall_unit, door.swing_direction
        )

        # Draw arc
        arc = symbol["arc"]
        cx, cy = arc["center"][0] * self.SCALE, arc["center"][1] * self.SCALE
        r = arc["radius"] * self.SCALE

        # SVG arc path
        start_rad = np.radians(arc["start_angle"])
        end_rad = np.radians(arc["end_angle"])
        x1 = cx + r * np.cos(start_rad)
        y1 = cy + r * np.sin(start_rad)
        x2 = cx + r * np.cos(end_rad)
        y2 = cy + r * np.sin(end_rad)

        large_arc = 1 if abs(arc["end_angle"] - arc["start_angle"]) > 180 else 0
        sweep = 1 if arc["end_angle"] > arc["start_angle"] else 0

        path_data = f"M {x1},{y1} A {r},{r} 0 {large_arc},{sweep} {x2},{y2}"
        group.add(
            dwg.path(
                d=path_data,
                fill="none",
                stroke=self.WALL_COLOR,
                stroke_width=self.OPENING_WEIGHT,
                stroke_dasharray="4,2",
            )
        )

    def _draw_window(self, dwg, group, window, model):
        """Draw window symbol (triple lines)."""
        parent_wall = self._find_parent_wall(window.position, model.walls)
        if parent_wall is None:
            return

        wall_dir = parent_wall.end - parent_wall.start
        wall_length = np.linalg.norm(wall_dir)
        if wall_length < 1e-6:
            return
        wall_unit = wall_dir / wall_length

        symbol = self.symbols.window_symbol(window.position, window.width, wall_unit)

        for line_start, line_end in symbol["lines"]:
            s = self._to_svg(line_start[0], line_start[1])
            e = self._to_svg(line_end[0], line_end[1])
            group.add(
                dwg.line(
                    start=s,
                    end=e,
                    stroke=self.WALL_COLOR,
                    stroke_width=self.OPENING_WEIGHT,
                )
            )

    def _draw_dimension(self, dwg, group, dim, total_height):
        """Draw a dimension line with extension lines and text."""
        # Flip Y for dimension drawing (non-flipped coordinate space)
        # Cast all to Python float for svgwrite compatibility
        sx = float(dim.start[0] * self.SCALE)
        sy = float((total_height - dim.start[1]) * self.SCALE)
        ex = float(dim.end[0] * self.SCALE)
        ey = float((total_height - dim.end[1]) * self.SCALE)

        # Direction and perpendicular
        dx = ex - sx
        dy = ey - sy
        length = float(np.sqrt(dx**2 + dy**2))
        if length < 1:
            return

        # Perpendicular offset direction
        px = -dy / length
        py = dx / length
        offset = float(dim.offset) * self.SCALE

        # Offset dimension line
        d_sx = sx + px * offset
        d_sy = sy + py * offset
        d_ex = ex + px * offset
        d_ey = ey + py * offset

        # Extension lines
        group.add(
            dwg.line(
                start=(sx, sy),
                end=(d_sx, d_sy),
                stroke=self.DIM_COLOR,
                stroke_width=self.DIM_WEIGHT,
            )
        )
        group.add(
            dwg.line(
                start=(ex, ey),
                end=(d_ex, d_ey),
                stroke=self.DIM_COLOR,
                stroke_width=self.DIM_WEIGHT,
            )
        )

        # Dimension line with arrows
        group.add(
            dwg.line(
                start=(d_sx, d_sy),
                end=(d_ex, d_ey),
                stroke=self.DIM_COLOR,
                stroke_width=self.DIM_WEIGHT,
            )
        )

        # Tick marks at ends (45-degree architectural style)
        tick_size = 4
        for tx, ty in [(d_sx, d_sy), (d_ex, d_ey)]:
            group.add(
                dwg.line(
                    start=(tx - tick_size, ty - tick_size),
                    end=(tx + tick_size, ty + tick_size),
                    stroke=self.DIM_COLOR,
                    stroke_width=self.DIM_WEIGHT * 2,
                )
            )

        # Dimension text
        mid_x = (d_sx + d_ex) / 2
        mid_y = (d_sy + d_ey) / 2
        value_ft = dim.value_m * 3.28084
        text = f"{dim.value_m:.2f}m ({value_ft:.1f}ft)"

        group.add(
            dwg.text(
                text,
                insert=(mid_x, mid_y - 5),
                text_anchor="middle",
                font_size="10px",
                font_family="Arial, sans-serif",
                fill=self.DIM_COLOR,
            )
        )

    def _draw_room_label(self, dwg, group, room, total_height):
        """Draw room name and area label."""
        if len(room.boundary) < 3:
            return

        centroid = room.boundary.mean(axis=0)
        cx = float(centroid[0]) * self.SCALE
        cy = float(total_height - centroid[1]) * self.SCALE

        group.add(
            dwg.text(
                room.name,
                insert=(cx, cy - 8),
                text_anchor="middle",
                font_size="14px",
                font_weight="bold",
                font_family="Arial, sans-serif",
                fill="#333",
            )
        )

        area_text = f"{room.area:.1f} m\u00B2"
        group.add(
            dwg.text(
                area_text,
                insert=(cx, cy + 10),
                text_anchor="middle",
                font_size="11px",
                font_family="Arial, sans-serif",
                fill="#666",
            )
        )

    def _draw_title_block(self, dwg, title, width, height, scale):
        """Draw title block at bottom of drawing."""
        y = height - 30
        dwg.add(
            dwg.line(
                start=(10, y - 5),
                end=(width - 10, y - 5),
                stroke="#999",
                stroke_width=0.5,
            )
        )
        dwg.add(
            dwg.text(
                title,
                insert=(15, y + 10),
                font_size="12px",
                font_weight="bold",
                font_family="Arial, sans-serif",
                fill="#333",
            )
        )
        dwg.add(
            dwg.text(
                f"Scale 1:{int(scale)}",
                insert=(width - 100, y + 10),
                font_size="10px",
                font_family="Arial, sans-serif",
                fill="#666",
            )
        )

    def _draw_scale_indicator(self, dwg, width, height):
        """Draw a simple scale indicator."""
        # 1-meter scale bar at bottom-left
        bar_y = height - 50
        bar_x = 20
        bar_len = self.SCALE  # 1m in SVG units

        dwg.add(
            dwg.line(
                start=(bar_x, bar_y),
                end=(bar_x + bar_len, bar_y),
                stroke="#333",
                stroke_width=2,
            )
        )
        # End ticks
        for x in [bar_x, bar_x + bar_len]:
            dwg.add(
                dwg.line(
                    start=(x, bar_y - 4),
                    end=(x, bar_y + 4),
                    stroke="#333",
                    stroke_width=2,
                )
            )
        dwg.add(
            dwg.text(
                "1m",
                insert=(bar_x + bar_len / 2, bar_y - 6),
                text_anchor="middle",
                font_size="9px",
                font_family="Arial, sans-serif",
                fill="#333",
            )
        )

    def _draw_north_arrow(self, dwg, width, orientation):
        """Draw north arrow at top-right."""
        cx = width - 40
        cy = 40
        size = 20

        rad = np.radians(orientation)
        tip_x = float(cx + size * np.sin(rad))
        tip_y = float(cy - size * np.cos(rad))

        # Arrow
        dwg.add(
            dwg.line(
                start=(cx, cy),
                end=(tip_x, tip_y),
                stroke="#333",
                stroke_width=2,
            )
        )
        # N label
        dwg.add(
            dwg.text(
                "N",
                insert=(tip_x, tip_y - 5),
                text_anchor="middle",
                font_size="12px",
                font_weight="bold",
                font_family="Arial, sans-serif",
                fill="#333",
            )
        )

    def _find_parent_wall(self, position, walls):
        """Find the wall closest to a given position."""
        if not walls:
            return None

        best_wall = None
        best_dist = float("inf")

        for wall in walls:
            # Distance from point to line segment
            d = self._point_to_segment_distance(position, wall.start, wall.end)
            if d < best_dist:
                best_dist = d
                best_wall = wall

        return best_wall if best_dist < 1.0 else None

    def _point_to_segment_distance(self, p, a, b):
        """Compute distance from point p to line segment a-b."""
        ab = b - a
        ap = p - a
        t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
