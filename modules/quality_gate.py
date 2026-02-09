"""Capture quality gates for reconstruction-grade output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from modules.compliance_profile import ComplianceProfile

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class QualityGateResult:
    """Quality gate output."""

    passed: bool
    status: str
    failures: List[str]
    metrics: Dict[str, float]



def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale in float32."""
    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[:, :, :3].astype(np.float32)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    else:
        gray = image.astype(np.float32)
    return gray



def compute_blur_score(image: np.ndarray) -> float:
    """Compute blur score using Laplacian variance."""
    gray = _to_gray(image)
    if CV2_AVAILABLE:
        try:
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
            return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        except Exception:
            pass

    gy, gx = np.gradient(gray)
    lap = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    return float(np.var(lap))



def _orb_overlap_ratio(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Overlap ratio using ORB keypoint matching."""
    if not CV2_AVAILABLE:
        return _fallback_overlap_ratio(image_a, image_b)

    gray_a = _to_gray(image_a).astype(np.uint8)
    gray_b = _to_gray(image_b).astype(np.uint8)

    orb = cv2.ORB_create(1200)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)

    if not matches:
        return 0.0

    matches = sorted(matches, key=lambda m: m.distance)
    good = [m for m in matches if m.distance < 60]
    denom = max(min(len(kp_a), len(kp_b)), 1)
    return float(len(good)) / float(denom)



def _fallback_overlap_ratio(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Fallback overlap estimate when ORB is unavailable."""
    gray_a = _to_gray(image_a)
    gray_b = _to_gray(image_b)

    h = min(gray_a.shape[0], gray_b.shape[0], 256)
    w = min(gray_a.shape[1], gray_b.shape[1], 256)

    small_a = gray_a[:h, :w]
    small_b = gray_b[:h, :w]

    a = (small_a - small_a.mean()) / (small_a.std() + 1e-8)
    b = (small_b - small_b.mean()) / (small_b.std() + 1e-8)

    corr = float(np.mean(a * b))
    corr = max(min(corr, 1.0), -1.0)
    return max(0.0, corr)



def compute_overlap_ratio(images: List[np.ndarray]) -> float:
    """Estimate mean overlap ratio across adjacent and skip pairs."""
    n = len(images)
    if n < 2:
        return 0.0

    ratios: List[float] = []
    for i in range(n - 1):
        ratios.append(_orb_overlap_ratio(images[i], images[i + 1]))
    for i in range(n - 2):
        ratios.append(_orb_overlap_ratio(images[i], images[i + 2]))

    if not ratios:
        return 0.0
    return float(np.mean(ratios))



def evaluate_capture_quality(
    images: List[np.ndarray],
    profile: ComplianceProfile,
    sfm_result: Optional[Dict] = None,
    min_registration_ratio_override: Optional[float] = None,
) -> QualityGateResult:
    """Run capture quality checks required before accurate finalization."""
    failures: List[str] = []

    num_images = len(images)
    blur_scores = [compute_blur_score(img) for img in images]
    mean_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
    min_blur = float(np.min(blur_scores)) if blur_scores else 0.0

    overlap_ratio = compute_overlap_ratio(images)

    num_registered = 0
    registration_ratio = 0.0
    sfm_success = False
    if sfm_result:
        sfm_success = bool(sfm_result.get("success", False))
        num_registered = int(sfm_result.get("num_registered", 0))
        if num_images > 0:
            registration_ratio = num_registered / float(num_images)

    required_registration_ratio = (
        float(min_registration_ratio_override)
        if min_registration_ratio_override is not None
        else float(profile.capture.min_registration_ratio)
    )

    if num_images < profile.capture.min_images:
        failures.append(
            f"Need at least {profile.capture.min_images} images (got {num_images})."
        )

    if mean_blur < profile.capture.min_blur_laplacian_var:
        failures.append(
            f"Image sharpness too low (mean blur score {mean_blur:.1f})."
        )

    if overlap_ratio < profile.capture.min_overlap_ratio:
        failures.append(
            f"Insufficient image overlap ({overlap_ratio:.3f})."
        )

    if not sfm_success:
        failures.append("SfM registration did not succeed.")
    elif registration_ratio < required_registration_ratio:
        failures.append(
            "Registered image ratio too low "
            f"({registration_ratio:.2%}, require {required_registration_ratio:.0%})."
        )

    passed = len(failures) == 0
    status = "PASS" if passed else "INSUFFICIENT_CAPTURE"

    metrics = {
        "num_images": float(num_images),
        "mean_blur_laplacian": mean_blur,
        "min_blur_laplacian": min_blur,
        "overlap_ratio": float(overlap_ratio),
        "num_registered": float(num_registered),
        "registration_ratio": float(registration_ratio),
        "registration_ratio_required": float(required_registration_ratio),
    }

    return QualityGateResult(
        passed=passed,
        status=status,
        failures=failures,
        metrics=metrics,
    )
