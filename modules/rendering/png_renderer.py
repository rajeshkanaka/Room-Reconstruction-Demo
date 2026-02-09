"""
PNG Floor Plan Renderer

Renders a FloorPlanModel to a clean architectural-style PNG using matplotlib.
Replaces the old heatmap-based floor plan image for Gradio preview.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, FancyArrowPatch
from typing import Optional

from modules.geometry.floor_plan_model import FloorPlanModel
from modules.rendering.symbol_library import SymbolLibrary


class PNGRenderer:
    """
    Renders FloorPlanModel to matplotlib Figure for PNG export.
    """

    # Colors
    WALL_COLOR = "#333333"
    WALL_FILL = "#e0e0e0"
    DIM_COLOR = "#cc0000"
    ROOM_FILL = "#f0f4ff"
    DOOR_COLOR = "#333333"
    WINDOW_COLOR = "#333333"

    def __init__(self, dpi: int = 150):
        self.dpi = dpi
        self.symbols = SymbolLibrary()

    def render(
        self,
        model: FloorPlanModel,
        output_path: Optional[str] = None,
        title: str = "Floor Plan",
    ):
        """
        Render FloorPlanModel to a matplotlib Figure.

        Args:
            model: FloorPlanModel to render
            output_path: Optional path to save PNG file
            title: Drawing title

        Returns:
            matplotlib Figure (also saved to output_path if provided)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Compute bounding box
        bbox = self._compute_bbox(model)
        if bbox is None:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            return fig

        x_min, y_min, x_max, y_max = bbox
        margin = 1.0

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect("equal")
        ax.set_facecolor("white")

        # Draw room fills
        for room in model.rooms:
            self._draw_room_fill(ax, room)

        # Draw walls
        for wall in model.walls:
            self._draw_wall(ax, wall)

        # Draw doors
        for door in model.doors:
            self._draw_door(ax, door, model)

        # Draw windows
        for window in model.windows:
            self._draw_window(ax, window, model)

        # Draw dimensions
        for dim in model.dimensions:
            self._draw_dimension(ax, dim)

        # Draw room labels
        for room in model.rooms:
            self._draw_room_label(ax, room)

        # Title and scale
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.text(
            0.98,
            0.02,
            f"Scale 1:{int(model.scale)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666",
        )

        # Clean up axes
        ax.grid(False)
        ax.set_xlabel("meters")
        ax.set_ylabel("meters")
        ax.tick_params(labelsize=7, colors="#999")
        for spine in ax.spines.values():
            spine.set_color("#ccc")

        fig.tight_layout()

        if output_path:
            fig.savefig(
                output_path, dpi=self.dpi, bbox_inches="tight", facecolor="white"
            )

        return fig

    def render_to_image(
        self, model: FloorPlanModel, output_path: str, title: str = "Floor Plan"
    ) -> str:
        """Render and save to PNG, close figure, return path."""
        fig = self.render(model, output_path=output_path, title=title)
        plt.close(fig)
        return output_path

    def _compute_bbox(self, model):
        """Compute bounding box of all elements."""
        all_points = []
        for w in model.walls:
            all_points.extend([w.start, w.end])
        for r in model.rooms:
            all_points.extend(r.boundary)

        if not all_points:
            return None

        pts = np.array(all_points)
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    def _draw_room_fill(self, ax, room):
        """Draw room polygon fill."""
        if len(room.boundary) < 3:
            return

        poly = plt.Polygon(
            room.boundary,
            closed=True,
            facecolor=self.ROOM_FILL,
            edgecolor="none",
            alpha=0.5,
        )
        ax.add_patch(poly)

    def _draw_wall(self, ax, wall):
        """Draw wall as filled rectangle with outline."""
        start = wall.start
        end = wall.end
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return

        d = direction / length
        perp = np.array([-d[1], d[0]])
        half_t = wall.thickness / 2

        corners = np.array(
            [
                start + perp * half_t,
                end + perp * half_t,
                end - perp * half_t,
                start - perp * half_t,
            ]
        )

        poly = plt.Polygon(
            corners,
            closed=True,
            facecolor=self.WALL_FILL,
            edgecolor=self.WALL_COLOR,
            linewidth=1.5,
        )
        ax.add_patch(poly)

    def _draw_door(self, ax, door, model):
        """Draw door symbol (arc)."""
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

        arc_data = symbol["arc"]
        center = arc_data["center"]
        radius = float(arc_data["radius"])
        start_angle = float(arc_data["start_angle"])
        end_angle = float(arc_data["end_angle"])

        arc = Arc(
            (float(center[0]), float(center[1])),
            width=radius * 2,
            height=radius * 2,
            angle=0,
            theta1=min(start_angle, end_angle),
            theta2=max(start_angle, end_angle),
            color=self.DOOR_COLOR,
            linewidth=1.0,
            linestyle="--",
        )
        ax.add_patch(arc)

    def _draw_window(self, ax, window, model):
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
            ax.plot(
                [float(line_start[0]), float(line_end[0])],
                [float(line_start[1]), float(line_end[1])],
                color=self.WINDOW_COLOR,
                linewidth=1.0,
            )

    def _draw_dimension(self, ax, dim):
        """Draw dimension line with extension lines and text."""
        sx = float(dim.start[0])
        sy = float(dim.start[1])
        ex = float(dim.end[0])
        ey = float(dim.end[1])

        dx = ex - sx
        dy = ey - sy
        length = float(np.sqrt(dx**2 + dy**2))
        if length < 0.01:
            return

        px = -dy / length
        py = dx / length
        offset = float(dim.offset)

        d_sx = sx + px * offset
        d_sy = sy + py * offset
        d_ex = ex + px * offset
        d_ey = ey + py * offset

        # Extension lines
        ax.plot([sx, d_sx], [sy, d_sy], color=self.DIM_COLOR, linewidth=0.5)
        ax.plot([ex, d_ex], [ey, d_ey], color=self.DIM_COLOR, linewidth=0.5)

        # Dimension line
        ax.annotate(
            "",
            xy=(d_ex, d_ey),
            xytext=(d_sx, d_sy),
            arrowprops=dict(arrowstyle="<->", color=self.DIM_COLOR, lw=0.8),
        )

        # Text
        mid_x = (d_sx + d_ex) / 2
        mid_y = (d_sy + d_ey) / 2
        value_ft = float(dim.value_m) * 3.28084
        text = f"{float(dim.value_m):.2f}m ({value_ft:.1f}ft)"

        angle = np.degrees(np.arctan2(dy, dx))
        ax.text(
            mid_x,
            mid_y,
            text,
            ha="center",
            va="bottom",
            fontsize=7,
            color=self.DIM_COLOR,
            rotation=angle,
            rotation_mode="anchor",
            bbox=dict(
                boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8
            ),
        )

    def _draw_room_label(self, ax, room):
        """Draw room name and area."""
        if len(room.boundary) < 3:
            return

        centroid = room.boundary.mean(axis=0)
        cx = float(centroid[0])
        cy = float(centroid[1])

        ax.text(
            cx,
            cy + 0.15,
            room.name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#333",
        )
        ax.text(
            cx,
            cy - 0.15,
            f"{room.area:.1f} m\u00B2",
            ha="center",
            va="center",
            fontsize=8,
            color="#666",
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
