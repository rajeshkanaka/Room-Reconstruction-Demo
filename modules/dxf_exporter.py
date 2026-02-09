"""DXF export utilities for reconstruction-ready floor plans."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import ezdxf

    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


DEFAULT_LAYERS = [
    "A-WALL-EXT",
    "A-WALL-INT",
    "A-OPENING",
    "A-DIMS",
    "A-ANNO",
]


class DXFExporter:
    """Export calibrated floor plan geometry to DXF."""

    def __init__(self, unit: str = "m"):
        self.unit = unit

    @staticmethod
    def _ensure_polygon(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        return points

    @staticmethod
    def _manual_dxf(path: str, boundary: np.ndarray, measurements: Dict) -> str:
        """Write a minimal ASCII DXF fallback when ezdxf is unavailable."""
        boundary = DXFExporter._ensure_polygon(boundary)
        lines: List[str] = [
            "0",
            "SECTION",
            "2",
            "HEADER",
            "9",
            "$INSUNITS",
            "70",
            "6",  # meters
            "0",
            "ENDSEC",
            "0",
            "SECTION",
            "2",
            "ENTITIES",
            "0",
            "LWPOLYLINE",
            "8",
            "A-WALL-EXT",
            "90",
            str(max(len(boundary) - 1, 0)),
            "70",
            "1",
        ]

        for x, y in boundary[:-1]:
            lines.extend(["10", f"{float(x):.6f}", "20", f"{float(y):.6f}"])

        area_txt = measurements.get("area_sqft", measurements.get("area_sqm", 0))
        lines.extend(
            [
                "0",
                "TEXT",
                "8",
                "A-ANNO",
                "10",
                "0.2",
                "20",
                "0.2",
                "40",
                "0.2",
                "1",
                f"AREA {area_txt}",
                "0",
                "ENDSEC",
                "0",
                "EOF",
            ]
        )

        with open(path, "w", encoding="ascii") as f:
            f.write("\n".join(lines))

        return path

    def _add_layers(self, doc) -> None:
        for layer in DEFAULT_LAYERS:
            if layer not in doc.layers:
                doc.layers.add(layer)

    def export_floor_plan(
        self,
        floor_plan_data: Dict,
        output_path: str,
        metadata: Dict | None = None,
    ) -> str:
        """Export floor plan to DXF using world-meter coordinates."""
        boundary = np.asarray(floor_plan_data.get("boundary_world_m", []), dtype=float)
        if len(boundary) < 3:
            raise ValueError("Cannot export DXF without a valid boundary polygon.")

        boundary = self._ensure_polygon(boundary)
        measurements = floor_plan_data.get("measurements", {})

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if not EZDXF_AVAILABLE:
            return self._manual_dxf(output_path, boundary, measurements)

        doc = ezdxf.new(dxfversion="R2018")
        doc.units = ezdxf.units.M
        self._add_layers(doc)
        msp = doc.modelspace()

        msp.add_lwpolyline(
            [(float(p[0]), float(p[1])) for p in boundary[:-1]],
            close=True,
            dxfattribs={"layer": "A-WALL-EXT"},
        )

        width_m = float(measurements.get("width_m", 0.0))
        depth_m = float(measurements.get("depth_m", 0.0))
        area_sqm = float(measurements.get("area_sqm", 0.0))
        area_sqft = float(measurements.get("area_sqft", 0.0))

        text_lines = [
            f"WIDTH: {width_m:.3f} m",
            f"DEPTH: {depth_m:.3f} m",
            f"AREA: {area_sqm:.2f} sqm / {area_sqft:.1f} sqft",
        ]

        if metadata:
            status = metadata.get("compliance_status")
            profile = metadata.get("compliance_profile")
            if status:
                text_lines.append(f"COMPLIANCE: {status}")
            if profile:
                text_lines.append(f"PROFILE: {profile}")

        min_xy = boundary.min(axis=0)
        text_x = float(min_xy[0])
        text_y = float(min_xy[1]) - 0.2

        for i, line in enumerate(text_lines):
            msp.add_text(
                line,
                dxfattribs={"height": 0.12, "layer": "A-ANNO"},
            ).set_placement((text_x, text_y - i * 0.14))

        doc.saveas(output_path)
        return output_path
