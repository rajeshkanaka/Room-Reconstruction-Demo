"""QA report generation for reconstruction compliance outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from modules.compliance_profile import ComplianceProfile



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default



def build_accuracy_metrics(
    profile: ComplianceProfile,
    calibration: Optional[Dict],
    sfm_result: Optional[Dict],
) -> Dict[str, Any]:
    """Build proxy accuracy metrics for pass/fail gating."""
    calibration_uncertainty_mm = _safe_float(
        (calibration or {}).get("uncertainty_mm", 999.0),
        999.0,
    )

    reproj_errors = []
    if sfm_result:
        reproj_errors = sfm_result.get("reprojection_errors", []) or []

    if reproj_errors:
        mean_reproj_px = float(sum(reproj_errors) / len(reproj_errors))
    else:
        mean_reproj_px = 0.0

    predicted_critical_mm = calibration_uncertainty_mm + mean_reproj_px * 1.5
    predicted_overall_mm = calibration_uncertainty_mm + mean_reproj_px * 3.0

    critical_pass = predicted_critical_mm <= profile.tolerance.critical_mm
    overall_pass = predicted_overall_mm <= profile.tolerance.overall_mm

    return {
        "predicted_critical_error_mm": round(predicted_critical_mm, 3),
        "predicted_overall_error_mm": round(predicted_overall_mm, 3),
        "mean_reprojection_error_px": round(mean_reproj_px, 4),
        "critical_threshold_mm": profile.tolerance.critical_mm,
        "overall_threshold_mm": profile.tolerance.overall_mm,
        "critical_pass": critical_pass,
        "overall_pass": overall_pass,
    }



def build_qa_report(
    profile: ComplianceProfile,
    compliance_status: str,
    quality_result: Optional[Any],
    calibration: Optional[Dict],
    sfm_result: Optional[Dict],
    floor_plan_measurements: Optional[Dict],
    notes: Optional[list] = None,
) -> Dict[str, Any]:
    """Build QA report payload."""
    accuracy = build_accuracy_metrics(profile, calibration, sfm_result)

    if quality_result is None:
        quality_payload = {}
        quality_failures = []
        quality_pass = False
    else:
        quality_payload = quality_result.metrics
        quality_failures = quality_result.failures
        quality_pass = quality_result.passed

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "compliance_profile": profile.name,
        "compliance_profile_display": profile.display_name,
        "standards": profile.standards,
        "compliance_status": compliance_status,
        "capture_quality": {
            "passed": quality_pass,
            "metrics": quality_payload,
            "failures": quality_failures,
        },
        "calibration": calibration or {},
        "accuracy_metrics": accuracy,
        "floor_plan_measurements": floor_plan_measurements or {},
        "sfm": {
            "success": bool((sfm_result or {}).get("success", False)),
            "num_registered": int((sfm_result or {}).get("num_registered", 0)),
            "stats": (sfm_result or {}).get("stats", {}),
        },
        "pass_flags": {
            "quality_pass": quality_pass,
            "calibration_pass": _safe_float((calibration or {}).get("uncertainty_mm", 999))
            <= profile.tolerance.max_calibration_uncertainty_mm,
            "critical_tolerance_pass": accuracy["critical_pass"],
            "overall_tolerance_pass": accuracy["overall_pass"],
        },
        "notes": notes or [],
    }

    return report



def save_qa_report(report: Dict[str, Any], output_path: str) -> str:
    """Persist QA report as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return output_path
