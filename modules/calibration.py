"""Metric calibration utilities for two-pass floor plan reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationInput:
    """User-provided calibration constraints."""

    point1_px: Tuple[float, float]
    point2_px: Tuple[float, float]
    known_distance: float
    known_distance_unit: str = "m"
    pixel_uncertainty: float = 2.0



def convert_distance_to_meters(distance: float, unit: str) -> float:
    """Convert distance to meters."""
    unit_norm = unit.strip().lower()
    if unit_norm in {"m", "meter", "meters"}:
        return float(distance)
    if unit_norm in {"ft", "feet", "foot"}:
        return float(distance) * 0.3048
    if unit_norm in {"in", "inch", "inches"}:
        return float(distance) * 0.0254
    raise ValueError(f"Unsupported calibration unit: {unit}")



def _pixel_to_model_coords(
    point_px: Sequence[float],
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Project a 2D floor plan pixel coordinate to model-space floor coordinates."""
    x_px, y_px = float(point_px[0]), float(point_px[1])
    h, w = image_shape

    x_u = x_range[0] + (x_px / max(w - 1, 1)) * (x_range[1] - x_range[0])
    z_u = z_range[1] - (y_px / max(h - 1, 1)) * (z_range[1] - z_range[0])
    return np.array([x_u, z_u], dtype=np.float64)



def _local_support(occupancy: np.ndarray, point_px: Sequence[float], radius: int = 6) -> float:
    """Estimate local support around clicked point from occupancy evidence."""
    if occupancy is None:
        return 0.5

    occupancy = np.asarray(occupancy)
    if occupancy.size == 0:
        return 0.5

    x_px = int(round(point_px[0]))
    y_px = int(round(point_px[1]))

    h, w = occupancy.shape[:2]
    x0 = max(0, x_px - radius)
    x1 = min(w, x_px + radius + 1)
    y0 = max(0, y_px - radius)
    y1 = min(h, y_px + radius + 1)

    window = occupancy[y0:y1, x0:x1]
    if window.size == 0:
        return 0.0

    if window.max() > 1.0:
        window = window / 255.0

    return float(np.clip(np.mean(window), 0.0, 1.0))



def compute_scale_factor(
    calibration_input: CalibrationInput,
    floor_plan_data: Dict,
    image_shape: Tuple[int, int],
) -> Dict:
    """Compute metric scale factor from two clicked points and known distance."""
    known_distance_m = convert_distance_to_meters(
        calibration_input.known_distance,
        calibration_input.known_distance_unit,
    )

    if known_distance_m <= 0:
        raise ValueError("Known distance must be positive.")

    x_range = floor_plan_data.get("x_range")
    z_range = floor_plan_data.get("z_range")
    if x_range is None or z_range is None:
        raise ValueError("Floor plan data missing x/z ranges for calibration.")

    p1_model = _pixel_to_model_coords(
        calibration_input.point1_px,
        x_range,
        z_range,
        image_shape,
    )
    p2_model = _pixel_to_model_coords(
        calibration_input.point2_px,
        x_range,
        z_range,
        image_shape,
    )

    model_distance = float(np.linalg.norm(p2_model - p1_model))
    if model_distance <= 1e-8:
        raise ValueError("Selected calibration points are too close together.")

    scale_factor = known_distance_m / model_distance

    model_width = float(abs(x_range[1] - x_range[0]))
    model_depth = float(abs(z_range[1] - z_range[0]))
    meters_per_pixel_x = (model_width * scale_factor) / max(image_shape[1] - 1, 1)
    meters_per_pixel_y = (model_depth * scale_factor) / max(image_shape[0] - 1, 1)
    meters_per_pixel = (meters_per_pixel_x + meters_per_pixel_y) / 2.0

    occupancy = floor_plan_data.get("occupancy")
    support_a = _local_support(occupancy, calibration_input.point1_px)
    support_b = _local_support(occupancy, calibration_input.point2_px)
    support = min(support_a, support_b)

    click_uncertainty_m = (
        calibration_input.pixel_uncertainty * np.sqrt(2.0) * meters_per_pixel
    )
    support_penalty_m = max(0.0, 0.5 - support) * 0.02
    uncertainty_m = click_uncertainty_m + support_penalty_m
    uncertainty_mm = float(uncertainty_m * 1000.0)

    return {
        "scale_factor": float(scale_factor),
        "known_distance_m": float(known_distance_m),
        "model_distance_units": float(model_distance),
        "calibration_segment_model": [p1_model.tolist(), p2_model.tolist()],
        "support_score": float(support),
        "uncertainty_mm": uncertainty_mm,
        "meters_per_pixel": float(meters_per_pixel),
    }



def scale_points(points: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale points globally to metric units."""
    return points * float(scale_factor)



def parse_calibration_input(raw: Dict) -> CalibrationInput:
    """Parse calibration input dictionary into strongly typed object."""
    point1 = raw.get("point1_px")
    point2 = raw.get("point2_px")
    if point1 is None or point2 is None:
        raise ValueError("Calibration input must include point1_px and point2_px.")

    return CalibrationInput(
        point1_px=(float(point1[0]), float(point1[1])),
        point2_px=(float(point2[0]), float(point2[1])),
        known_distance=float(raw.get("known_distance")),
        known_distance_unit=str(raw.get("known_distance_unit", "m")),
        pixel_uncertainty=float(raw.get("pixel_uncertainty", 2.0)),
    )
