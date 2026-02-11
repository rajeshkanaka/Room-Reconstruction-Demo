"""
Phase 5 Tests: Quality scoring and output gating.
"""

import os
import sys
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_detection_result(
    used_rectangular_fallback=False,
    opening_fusion=None,
    wall_multiview_support=None,
    closure_score=0.96,
    closure_mean_gap_m=0.03,
):
    from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon

    boundary = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
    walls = [
        WallSegment(start=boundary[0], end=boundary[1]),
        WallSegment(start=boundary[1], end=boundary[2]),
        WallSegment(start=boundary[2], end=boundary[3]),
        WallSegment(start=boundary[3], end=boundary[0]),
    ]
    model = FloorPlanModel(walls=walls, rooms=[RoomPolygon(boundary=boundary, name="Room")])
    return {
        "floor_plan_model": model,
        "measurements": {"warnings": []},
        "quality_flags": {
            "used_depth_augmentation": False,
            "used_rectangular_fallback": used_rectangular_fallback,
            "used_sparse_wall_fallback": False,
            "opening_fusion": opening_fusion or {},
            "wall_multiview_support": wall_multiview_support or {},
            "closure_score": closure_score,
            "closure_mean_gap_m": closure_mean_gap_m,
            "closure_p95_gap_m": closure_mean_gap_m,
            "closure_unpaired_ratio": max(0.0, 1.0 - closure_score),
        },
    }


def _make_reconstructor_stub():
    from modules.room_reconstructor import RoomReconstructor

    recon = RoomReconstructor.__new__(RoomReconstructor)
    recon.scale_solution = {
        "scale_factor": 1.0,
        "confidence": "high",
        "consistency": 0.9,
        "mode": "auto",
        "cues": [],
    }
    return recon


def test_quality_mode_high_confidence():
    recon = _make_reconstructor_stub()
    detection = _make_detection_result(used_rectangular_fallback=False)
    measurements = {
        "width_m": 3.0,
        "depth_m": 2.0,
        "area_sqm": 6.0,
        "scale_confidence": "high",
        "scale_consistency": 0.92,
    }

    quality = recon._assess_plan_quality(
        detection_result=detection,
        measurements=measurements,
        n_images=8,
    )

    assert quality["mode"] == "high_confidence"
    assert quality["score"] >= 0.75
    assert quality["export_policy"] == "normal_export"


def test_quality_mode_approximate_with_fallback():
    recon = _make_reconstructor_stub()
    detection = _make_detection_result(used_rectangular_fallback=True)
    measurements = {
        "width_m": 2.0,
        "depth_m": 1.4,
        "area_sqm": 2.8,
        "scale_confidence": "medium",
        "scale_consistency": 0.7,
    }

    quality = recon._assess_plan_quality(
        detection_result=detection,
        measurements=measurements,
        n_images=4,
    )

    assert quality["mode"] in ("approximate", "needs_more_images")
    assert quality["score"] < 0.85
    assert any("fallback" in w.lower() for w in quality["warnings"])
    assert quality["export_policy"] in ("annotate_as_approximate", "needs_more_images")


def test_quality_mode_needs_more_images_without_geometry():
    recon = _make_reconstructor_stub()
    measurements = {
        "width_m": 0.0,
        "depth_m": 0.0,
        "area_sqm": 0.0,
        "scale_confidence": "low",
        "scale_consistency": 0.2,
    }

    quality = recon._assess_plan_quality(
        detection_result=None,
        measurements=measurements,
        n_images=3,
    )

    assert quality["mode"] == "needs_more_images"
    assert quality["recommended_next_steps"]
    assert quality["export_policy"] == "needs_more_images"


def test_measurement_summary_includes_scale_metadata():
    recon = _make_reconstructor_stub()
    detection = {
        "measurements": {
            "overall": {"width_m": 3.0, "depth_m": 2.0},
            "rooms": [{"area_sqm": 6.0}],
        }
    }
    scale_solution = {
        "scale_factor": 2.41,
        "confidence": "medium",
        "consistency": 0.88,
        "mode": "auto",
        "cues": [{"name": "ceiling_height_prior"}],
    }

    summary = recon._summarize_measurements(
        detection_result=detection,
        points=np.zeros((10, 3)),
        scale_solution=scale_solution,
    )

    assert summary["scale_factor"] == 2.41
    assert summary["scale_confidence"] == "medium"
    assert summary["scale_cues"] == ["ceiling_height_prior"]


def test_quality_penalizes_single_view_openings():
    recon = _make_reconstructor_stub()
    detection = _make_detection_result(
        opening_fusion={
            "raw_door_observations": 3,
            "raw_window_observations": 0,
            "fused_doors": 1,
            "fused_windows": 0,
            "max_support_views": 1,
        }
    )
    measurements = {
        "width_m": 3.0,
        "depth_m": 2.0,
        "area_sqm": 6.0,
        "scale_confidence": "high",
        "scale_consistency": 0.92,
    }

    quality = recon._assess_plan_quality(
        detection_result=detection,
        measurements=measurements,
        n_images=6,
    )

    assert any("single-view" in w.lower() for w in quality["warnings"])


def test_quality_penalizes_weak_wall_multiview_support():
    recon = _make_reconstructor_stub()
    detection = _make_detection_result(
        wall_multiview_support={
            "enabled": True,
            "input_segments": 6,
            "retained_segments": 3,
            "dropped_segments": 3,
            "min_support_views": 2,
            "avg_support_views": 1.0,
        }
    )
    measurements = {
        "width_m": 3.0,
        "depth_m": 2.0,
        "area_sqm": 6.0,
        "scale_confidence": "high",
        "scale_consistency": 0.92,
    }

    quality = recon._assess_plan_quality(
        detection_result=detection,
        measurements=measurements,
        n_images=8,
    )

    assert any("multi-view support" in w.lower() for w in quality["warnings"])


def test_quality_penalizes_low_wall_closure_score():
    recon = _make_reconstructor_stub()
    detection = _make_detection_result(
        closure_score=0.42,
        closure_mean_gap_m=0.26,
    )
    measurements = {
        "width_m": 3.0,
        "depth_m": 2.0,
        "area_sqm": 6.0,
        "scale_confidence": "high",
        "scale_consistency": 0.92,
    }

    quality = recon._assess_plan_quality(
        detection_result=detection,
        measurements=measurements,
        n_images=8,
    )

    assert quality["score"] < 0.9
    assert any("closure" in w.lower() for w in quality["warnings"])
