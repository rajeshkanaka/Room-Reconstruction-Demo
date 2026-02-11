"""
Depth Calibration Module

Provides scale correction for metric depth estimates using:
- User-supplied known wall measurement
- Cross-view consistency checks
- Bounding box sanity checks
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
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
        self.last_solution: Dict[str, Any] = {
            "scale_factor": 1.0,
            "confidence": "uncalibrated",
            "consistency": 0.0,
            "cues": [],
        }

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

    def _estimate_pairwise_baseline(
        self,
        camera_poses: Optional[Dict],
    ) -> Optional[float]:
        """
        Estimate median pairwise camera baseline from camera poses.

        Args:
            camera_poses: Mapping of camera index -> {"transform": 4x4 ndarray}

        Returns:
            Median pairwise baseline in model units, or None if unavailable.
        """
        if not camera_poses or len(camera_poses) < 2:
            return None

        centers = []
        for _, pose_data in camera_poses.items():
            pose = pose_data.get("transform") if isinstance(pose_data, dict) else None
            if pose is None:
                continue
            pose = np.asarray(pose)
            if pose.shape != (4, 4):
                continue
            centers.append(pose[:3, 3].astype(np.float64))

        if len(centers) < 2:
            return None

        baselines = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = float(np.linalg.norm(centers[i] - centers[j]))
                if dist > 1e-6:
                    baselines.append(dist)

        if not baselines:
            return None

        return float(np.median(np.array(baselines, dtype=np.float64)))

    def solve_scale_from_cues(
        self,
        points: np.ndarray,
        assumed_room_width: Optional[float] = None,
        camera_poses: Optional[Dict] = None,
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Solve a global scale factor from multiple weak geometric cues.

        Cues:
        - Ceiling-height prior (strong in auto mode)
        - Assumed room width prior (strong only in user_reference mode)
        - Camera baseline prior (weak, optional)

        Args:
            points: Nx3 point cloud in model units
            assumed_room_width: Optional known width prior in meters
            camera_poses: Optional camera pose dict for baseline cue
            mode: "auto" or "user_reference"

        Returns:
            Dict with scale_factor, confidence, consistency, and cue details.
        """
        result: Dict[str, Any] = {
            "scale_factor": 1.0,
            "confidence": "low",
            "consistency": 0.0,
            "cues": [],
            "mode": mode,
        }

        if points is None or len(points) == 0:
            self.scale_factor = 1.0
            self.confidence = "low"
            self.last_solution = result
            return result

        lo = np.percentile(points, 2, axis=0)
        hi = np.percentile(points, 98, axis=0)
        extents = hi - lo

        horizontal_span = float(max(extents[0], extents[2]))
        height_span = float(extents[1])

        cues: List[Dict[str, Any]] = []

        def _add_cue(name: str, scale: float, weight: float, source: str) -> None:
            if not np.isfinite(scale) or scale <= 0 or weight <= 0:
                return
            cues.append(
                {
                    "name": name,
                    "scale": float(scale),
                    "weight": float(weight),
                    "source": source,
                }
            )

        # Height prior: strongest cue in auto mode to avoid overfitting to default width.
        target_height = 2.7  # meters
        if height_span > 1e-6:
            height_weight = 0.70 if mode == "auto" else 0.25
            _add_cue(
                "ceiling_height_prior",
                target_height / height_span,
                height_weight,
                "architectural_prior",
            )

        # Assumed room width prior (strong only when explicitly requested).
        if assumed_room_width is not None and assumed_room_width > 0 and horizontal_span > 1e-6:
            width_weight = 0.20 if mode == "auto" else 0.85
            _add_cue(
                "room_width_prior",
                float(assumed_room_width) / horizontal_span,
                width_weight,
                "user_input",
            )

        # Camera baseline prior (weak cue, helps stabilize ambiguous scenes).
        baseline = self._estimate_pairwise_baseline(camera_poses)
        if baseline is not None and baseline > 1e-6:
            target_baseline = 0.9  # meters; typical handheld spacing across a capture set
            _add_cue(
                "camera_baseline_prior",
                target_baseline / baseline,
                0.10 if mode == "auto" else 0.05,
                "camera_pose_statistics",
            )

        if not cues:
            self.scale_factor = 1.0
            self.confidence = "low"
            self.last_solution = result
            return result

        logs = np.array([np.log(c["scale"]) for c in cues], dtype=np.float64)
        weights = np.array([c["weight"] for c in cues], dtype=np.float64)

        # Drop strong outliers in cue space when we have enough cues.
        if len(cues) >= 3:
            median_log = float(np.median(logs))
            mad = float(np.median(np.abs(logs - median_log))) + 1e-6
            threshold = max(0.35, 2.5 * mad)
            keep = np.abs(logs - median_log) <= threshold
            if int(np.sum(keep)) >= 2:
                cues = [c for c, k in zip(cues, keep) if bool(k)]
                logs = np.array([np.log(c["scale"]) for c in cues], dtype=np.float64)
                weights = np.array([c["weight"] for c in cues], dtype=np.float64)

        weights = weights / max(float(np.sum(weights)), 1e-12)
        solved_log_scale = float(np.sum(weights * logs))
        solved_scale = float(np.exp(solved_log_scale))
        solved_scale = float(np.clip(solved_scale, 0.25, 8.0))

        if len(cues) > 1:
            log_sigma = float(np.sqrt(np.sum(weights * (logs - solved_log_scale) ** 2)))
        else:
            # Single cue is weakly constrained by definition.
            log_sigma = 0.35

        consistency = float(max(0.0, min(1.0, 1.0 - (log_sigma / 0.45))))
        if len(cues) >= 3 and log_sigma < 0.12:
            confidence = "high"
        elif log_sigma < 0.28:
            confidence = "medium"
        else:
            confidence = "low"

        if mode == "user_reference" and any(c["name"] == "room_width_prior" for c in cues):
            confidence = "high" if confidence != "low" else "medium"

        result = {
            "scale_factor": solved_scale,
            "confidence": confidence,
            "consistency": consistency,
            "cues": cues,
            "mode": mode,
            "extents_before_m": [float(extents[0]), float(extents[1]), float(extents[2])],
        }

        self.scale_factor = solved_scale
        self.confidence = confidence
        self.last_solution = result

        cue_summary = ", ".join(f"{c['name']}={c['scale']:.2f}x" for c in cues)
        print(
            colored(
                f"[Calibrator] Multi-cue scale: {solved_scale:.3f}x "
                f"({confidence}, consistency={consistency:.2f}; {cue_summary})",
                "green" if confidence == "high" else "yellow",
            )
        )

        return result

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
