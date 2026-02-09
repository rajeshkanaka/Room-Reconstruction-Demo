"""
Depth Calibration Module

Provides scale correction for metric depth estimates using:
- User-supplied known wall measurement
- Cross-view consistency checks
- Bounding box sanity checks
"""

import numpy as np
from typing import Optional, Tuple, List
from termcolor import colored


class DepthCalibrator:
    """
    Calibrates metric depth estimates against known reference dimensions.

    Even metric depth models can have systematic bias. This module computes
    a single scale correction factor applied globally to depth values.
    """

    def __init__(self):
        self.scale_factor = 1.0
        self.confidence = "uncalibrated"

    def calibrate_from_known_dimension(
        self,
        points: np.ndarray,
        axis: int,
        known_meters: float,
    ) -> float:
        """
        Compute scale correction from a known room dimension.

        Measures the spread of the point cloud along the specified axis
        and computes the factor needed to match the known distance.

        Args:
            points: Nx3 point cloud in model units
            axis: 0=X (width), 1=Y (height), 2=Z (depth)
            known_meters: The true dimension in meters

        Returns:
            Scale correction factor (multiply all coordinates by this)
        """
        if len(points) == 0 or known_meters <= 0:
            return 1.0

        # Use percentiles to ignore outliers
        lo = np.percentile(points[:, axis], 2)
        hi = np.percentile(points[:, axis], 98)
        current_spread = hi - lo

        if current_spread < 1e-6:
            print(colored("[Calibrator] Warning: near-zero spread on axis", "yellow"))
            return 1.0

        self.scale_factor = known_meters / current_spread
        self.confidence = "user_reference"

        print(
            colored(
                f"[Calibrator] Axis {axis}: {current_spread:.2f}m -> {known_meters:.2f}m "
                f"(factor: {self.scale_factor:.3f})",
                "green",
            )
        )

        return self.scale_factor

    def calibrate_cross_view(
        self,
        per_view_spreads: List[Tuple[float, float, float]],
    ) -> Tuple[float, str]:
        """
        Cross-validate measurements from multiple views.

        Compares the bounding box dimensions seen from each view.
        If they're consistent, confidence is high. If they diverge,
        flag for user attention.

        Args:
            per_view_spreads: List of (x_spread, y_spread, z_spread) per view

        Returns:
            (consistency_score, confidence_level)
            consistency_score: 0-1 where 1 = perfectly consistent
            confidence_level: "high", "medium", "low"
        """
        if len(per_view_spreads) < 2:
            return 1.0, "low"

        spreads = np.array(per_view_spreads)

        # Coefficient of variation per axis
        cv_per_axis = []
        for axis in range(3):
            axis_vals = spreads[:, axis]
            if axis_vals.mean() > 1e-6:
                cv = axis_vals.std() / axis_vals.mean()
                cv_per_axis.append(cv)

        if not cv_per_axis:
            return 0.0, "low"

        avg_cv = np.mean(cv_per_axis)

        # Lower CV = more consistent
        consistency = max(0.0, 1.0 - avg_cv)

        if consistency > 0.8:
            confidence = "high"
        elif consistency > 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        print(
            colored(
                f"[Calibrator] Cross-view consistency: {consistency:.2f} ({confidence})",
                "green" if confidence == "high" else "yellow",
            )
        )

        return consistency, confidence

    def sanity_check(self, points: np.ndarray) -> bool:
        """
        Basic sanity check on point cloud dimensions.

        Returns True if the bounding box is within plausible indoor room range.
        """
        if len(points) == 0:
            return False

        lo = np.percentile(points, 2, axis=0)
        hi = np.percentile(points, 98, axis=0)
        extents = hi - lo

        # Indoor rooms: 1.5-15m wide, 2-5m tall, 1.5-15m deep
        width_ok = 1.0 < extents[0] < 20.0
        height_ok = 1.5 < extents[1] < 6.0
        depth_ok = 1.0 < extents[2] < 20.0

        plausible = width_ok and height_ok and depth_ok

        if not plausible:
            print(
                colored(
                    f"[Calibrator] Warning: dimensions outside plausible range: "
                    f"{extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f}m",
                    "yellow",
                )
            )

        return plausible

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the current scale factor to a point cloud."""
        return points * self.scale_factor
