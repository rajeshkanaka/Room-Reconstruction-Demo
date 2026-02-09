"""
DXF Floor Plan Renderer

Renders a FloorPlanModel to DXF with standard CAD layer structure:
- A-WALL: Wall outlines (heavy line weight)
- A-DOOR: Door symbols
- A-GLAZ: Window/glazing symbols
- A-DIMS: Dimension annotations
- A-AREA: Room fills and labels
- A-ANNO: Title block, scale bar, north arrow
"""

import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
from typing import Optional

from modules.geometry.floor_plan_model import FloorPlanModel
from modules.rendering.symbol_library import SymbolLibrary


class DXFRenderer:
    """
    Renders FloorPlanModel to DXF with proper CAD layer structure.
    """

    # Layer definitions: (name, color_index, lineweight in 1/100mm)
    LAYERS = {
        "A-WALL": (7, 50),  # White, 0.50mm
        "A-WALL-FILL": (9, 25),  # Light gray, 0.25mm
        "A-DOOR": (3, 25),  # Green, 0.25mm
        "A-GLAZ": (4, 25),  # Cyan, 0.25mm
        "A-DIMS": (1, 13),  # Red, 0.13mm
        "A-AREA": (5, 13),  # Blue, 0.13mm
        "A-ANNO": (7, 13),  # White, 0.13mm
    }

    def __init__(self):
        self.symbols = SymbolLibrary()

    def render(
        self,
        model: FloorPlanModel,
        output_path: str,
        title: str = "Floor Plan",
    ) -> str:
        """
        Render FloorPlanModel to a DXF file.

        Args:
            model: FloorPlanModel to render
            output_path: Path to save DXF file
            title: Drawing title

        Returns:
            Path to saved DXF file
        """
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Create layers
        for layer_name, (color, lineweight) in self.LAYERS.items():
            doc.layers.add(
                layer_name,
                color=color,
                lineweight=lineweight,
            )

        # Draw room fills
        for room in model.rooms:
            self._draw_room_fill(msp, room)

        # Draw walls
        for wall in model.walls:
            self._draw_wall(msp, wall)

        # Draw doors
        for door in model.doors:
            self._draw_door(msp, door, model)

        # Draw windows
        for window in model.windows:
            self._draw_window(msp, window, model)

        # Draw dimensions
        for dim in model.dimensions:
            self._draw_dimension(msp, dim)

        # Draw room labels
        for room in model.rooms:
            self._draw_room_label(msp, room)

        # Title block
        self._draw_title_block(msp, title, model)

        doc.saveas(output_path)
        return output_path

    def _draw_room_fill(self, msp, room):
        """Draw room boundary as a lightweight polyline on A-AREA layer."""
        if len(room.boundary) < 3:
            return

        points = [(float(p[0]), float(p[1])) for p in room.boundary]
        msp.add_lwpolyline(
            points,
            close=True,
            dxfattribs={"layer": "A-AREA", "color": 254},  # Very light gray
        )

    def _draw_wall(self, msp, wall):
        """Draw wall as a filled rectangle (polyline outline) on A-WALL layer."""
        start = wall.start
        end = wall.end
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return

        d = direction / length
        perp = np.array([-d[1], d[0]])
        half_t = wall.thickness / 2

        # Wall outline corners
        corners = [
            start + perp * half_t,
            end + perp * half_t,
            end - perp * half_t,
            start - perp * half_t,
        ]
        points = [(float(c[0]), float(c[1])) for c in corners]

        # Filled wall polygon
        hatch = msp.add_hatch(
            color=9,  # Light gray
            dxfattribs={"layer": "A-WALL-FILL"},
        )
        hatch.paths.add_polyline_path(points, is_closed=True)

        # Wall outline
        msp.add_lwpolyline(
            points,
            close=True,
            dxfattribs={"layer": "A-WALL"},
        )

    def _draw_door(self, msp, door, model):
        """Draw door symbol (arc) on A-DOOR layer."""
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

        arc = symbol["arc"]
        center = (float(arc["center"][0]), float(arc["center"][1]))
        radius = float(arc["radius"])
        start_angle = float(arc["start_angle"])
        end_angle = float(arc["end_angle"])

        msp.add_arc(
            center=center,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            dxfattribs={"layer": "A-DOOR"},
        )

    def _draw_window(self, msp, window, model):
        """Draw window symbol (triple lines) on A-GLAZ layer."""
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
            msp.add_line(
                start=(float(line_start[0]), float(line_start[1])),
                end=(float(line_end[0]), float(line_end[1])),
                dxfattribs={"layer": "A-GLAZ"},
            )

    def _draw_dimension(self, msp, dim):
        """Draw dimension with extension lines and text on A-DIMS layer."""
        sx = float(dim.start[0])
        sy = float(dim.start[1])
        ex = float(dim.end[0])
        ey = float(dim.end[1])

        # Direction and perpendicular
        dx = ex - sx
        dy = ey - sy
        length = float(np.sqrt(dx**2 + dy**2))
        if length < 0.01:
            return

        px = -dy / length
        py = dx / length
        offset = float(dim.offset)

        # Offset dimension line positions
        d_sx = sx + px * offset
        d_sy = sy + py * offset
        d_ex = ex + px * offset
        d_ey = ey + py * offset

        # Extension lines
        msp.add_line(
            start=(sx, sy),
            end=(d_sx, d_sy),
            dxfattribs={"layer": "A-DIMS"},
        )
        msp.add_line(
            start=(ex, ey),
            end=(d_ex, d_ey),
            dxfattribs={"layer": "A-DIMS"},
        )

        # Dimension line
        msp.add_line(
            start=(d_sx, d_sy),
            end=(d_ex, d_ey),
            dxfattribs={"layer": "A-DIMS"},
        )

        # Dimension text
        mid_x = (d_sx + d_ex) / 2
        mid_y = (d_sy + d_ey) / 2
        value_ft = float(dim.value_m) * 3.28084
        text = f"{float(dim.value_m):.2f}m ({value_ft:.1f}ft)"

        msp.add_text(
            text,
            height=0.08,
            dxfattribs={
                "layer": "A-DIMS",
                "insert": (mid_x, mid_y + 0.05),
                "halign": 1,  # center
            },
        )

    def _draw_room_label(self, msp, room):
        """Draw room name and area label on A-AREA layer."""
        if len(room.boundary) < 3:
            return

        centroid = room.boundary.mean(axis=0)
        cx = float(centroid[0])
        cy = float(centroid[1])

        # Room name
        msp.add_text(
            room.name,
            height=0.15,
            dxfattribs={
                "layer": "A-AREA",
                "insert": (cx, cy + 0.1),
                "halign": 1,  # center
            },
        )

        # Area text
        area_text = f"{room.area:.1f} m\u00B2"
        msp.add_text(
            area_text,
            height=0.10,
            dxfattribs={
                "layer": "A-AREA",
                "insert": (cx, cy - 0.15),
                "halign": 1,  # center
            },
        )

    def _draw_title_block(self, msp, title, model):
        """Draw title block on A-ANNO layer."""
        # Compute bounding box for positioning
        all_points = []
        for w in model.walls:
            all_points.extend([w.start, w.end])
        for r in model.rooms:
            all_points.extend(r.boundary)

        if not all_points:
            return

        pts = np.array(all_points)
        x_min = float(pts[:, 0].min())
        y_min = float(pts[:, 1].min())
        x_max = float(pts[:, 0].max())

        # Title text below floor plan
        msp.add_text(
            title,
            height=0.2,
            dxfattribs={
                "layer": "A-ANNO",
                "insert": (x_min, y_min - 0.8),
            },
        )

        # Scale text
        msp.add_text(
            f"Scale 1:{int(model.scale)}",
            height=0.12,
            dxfattribs={
                "layer": "A-ANNO",
                "insert": (x_max - 1.5, y_min - 0.8),
            },
        )

    def _find_parent_wall(self, position, walls):
        """Find the wall closest to a given position."""
        if not walls:
            return None

        best_wall = None
        best_dist = float("inf")

        for wall in walls:
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
