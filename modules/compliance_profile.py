"""
Compliance profile definitions for reconstruction-grade deliverables.

Profiles centralize thresholds for capture quality, calibration quality, and
reported tolerance targets. The current v1 profile is US residential focused.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CaptureThresholds:
    """Capture quality thresholds before calibration/final output."""

    min_images: int
    target_images: int
    min_blur_laplacian_var: float
    min_overlap_ratio: float
    min_registration_ratio: float


@dataclass(frozen=True)
class ToleranceThresholds:
    """Accuracy and calibration thresholds."""

    critical_mm: float
    overall_mm: float
    area_error_pct: float
    max_calibration_uncertainty_mm: float


@dataclass(frozen=True)
class ComplianceProfile:
    """Compliance profile container."""

    name: str
    display_name: str
    standards: List[str]
    capture: CaptureThresholds
    tolerance: ToleranceThresholds
    imperial_rounding_ft: float
    metric_rounding_m: int
    require_known_distance: bool


US_RESIDENTIAL_V1 = ComplianceProfile(
    name="us_residential_v1",
    display_name="US Residential v1 (ANSI/Fannie aligned)",
    standards=[
        "ANSI Z765 ecosystem (context)",
        "Fannie software-generated floor plan expectations (2025+)",
    ],
    capture=CaptureThresholds(
        min_images=6,
        target_images=8,
        min_blur_laplacian_var=80.0,
        min_overlap_ratio=0.12,
        min_registration_ratio=0.70,
    ),
    tolerance=ToleranceThresholds(
        critical_mm=15.0,
        overall_mm=30.0,
        area_error_pct=2.5,
        max_calibration_uncertainty_mm=8.0,
    ),
    imperial_rounding_ft=0.1,
    metric_rounding_m=3,
    require_known_distance=True,
)


COMMERCIAL_BOMA_V1 = ComplianceProfile(
    name="commercial_boma_v1",
    display_name="Commercial BOMA v1 (future)",
    standards=["ANSI/BOMA Z65 family"],
    capture=CaptureThresholds(
        min_images=8,
        target_images=12,
        min_blur_laplacian_var=90.0,
        min_overlap_ratio=0.15,
        min_registration_ratio=0.75,
    ),
    tolerance=ToleranceThresholds(
        critical_mm=12.0,
        overall_mm=25.0,
        area_error_pct=2.0,
        max_calibration_uncertainty_mm=6.0,
    ),
    imperial_rounding_ft=0.1,
    metric_rounding_m=3,
    require_known_distance=True,
)


GLOBAL_IPMS_V1 = ComplianceProfile(
    name="global_ipms_v1",
    display_name="Global IPMS v1 (future)",
    standards=["IPMS all buildings"],
    capture=CaptureThresholds(
        min_images=8,
        target_images=12,
        min_blur_laplacian_var=90.0,
        min_overlap_ratio=0.15,
        min_registration_ratio=0.75,
    ),
    tolerance=ToleranceThresholds(
        critical_mm=12.0,
        overall_mm=25.0,
        area_error_pct=2.0,
        max_calibration_uncertainty_mm=6.0,
    ),
    imperial_rounding_ft=0.1,
    metric_rounding_m=3,
    require_known_distance=True,
)


PROFILES: Dict[str, ComplianceProfile] = {
    US_RESIDENTIAL_V1.name: US_RESIDENTIAL_V1,
    COMMERCIAL_BOMA_V1.name: COMMERCIAL_BOMA_V1,
    GLOBAL_IPMS_V1.name: GLOBAL_IPMS_V1,
}


def get_compliance_profile(name: str) -> ComplianceProfile:
    """Resolve a compliance profile by name."""
    if name in PROFILES:
        return PROFILES[name]
    return US_RESIDENTIAL_V1
