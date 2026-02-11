"""
Phase 9 Regression Pack: benchmark assertions for the 9-image sample room.

Run live benchmark explicitly:
  RUN_SAMPLE_REGRESSION=1 uv run pytest -q tests/test_phase9_regression_pack.py -s
"""

import glob
import json
import os
import random
import sys

import numpy as np
import pytest


# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _baseline_path() -> str:
    return os.path.join(_project_root(), "tests", "fixtures", "t9_sample_room_baseline.json")


def _load_baseline() -> dict:
    with open(_baseline_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _collect_sample_images(glob_expr: str) -> list[str]:
    root = _project_root()
    return sorted(glob.glob(os.path.join(root, glob_expr)))


def _set_deterministic_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _build_report(result: dict) -> dict:
    from modules.room_reconstructor import RoomReconstructor

    detection = result.get("data", {}).get("detection", {}) or {}
    model = detection.get("floor_plan_model")
    quality_flags = detection.get("quality_flags", {}) or {}
    quality = result.get("quality", {}) or {}

    walls = list(getattr(model, "walls", []) or [])
    rooms = list(getattr(model, "rooms", []) or [])
    closure_metrics = RoomReconstructor._compute_wall_closure_metrics(walls)

    closure_score = float(
        quality_flags.get("closure_score", closure_metrics.get("closure_score", 0.0))
    )
    return {
        "wall_count": len(walls),
        "room_count": len(rooms),
        "closure_score": closure_score,
        "used_rectangular_fallback": bool(quality_flags.get("used_rectangular_fallback", False)),
        "quality_mode": str(quality.get("mode", "")),
        "export_policy": str(quality.get("export_policy", "")),
        "industry_ready": bool(quality.get("industry_ready", False)),
        "quality_score": float(quality.get("score", 0.0)),
        "backend": str(result.get("backend", "")),
    }


def _assert_guardrails(report: dict, baseline: dict) -> None:
    acceptance = baseline["acceptance"]

    assert report["wall_count"] >= int(acceptance["min_wall_count"])
    assert report["closure_score"] >= float(acceptance["min_closure_score"])
    assert report["quality_mode"] in set(acceptance["allowed_quality_modes"])

    if report["used_rectangular_fallback"] and acceptance.get(
        "require_non_industry_ready_if_fallback", False
    ):
        assert not report["industry_ready"]
        assert report["quality_mode"] != "high_confidence"

    if report["industry_ready"]:
        assert report["export_policy"] == "normal_export"
    elif report["quality_mode"] in ("high_confidence", "approximate"):
        assert report["export_policy"] == "annotate_as_approximate"
    else:
        assert report["export_policy"] == "needs_more_images"


def test_t9_guardrail_logic_rejects_high_confidence_with_fallback():
    baseline = _load_baseline()
    bad = {
        "wall_count": 4,
        "room_count": 1,
        "closure_score": 0.95,
        "used_rectangular_fallback": True,
        "quality_mode": "high_confidence",
        "export_policy": "normal_export",
        "industry_ready": True,
        "quality_score": 0.9,
        "backend": "vggt",
    }
    with pytest.raises(AssertionError):
        _assert_guardrails(bad, baseline)


@pytest.mark.slow
def test_t9_sample_room_live_regression():
    if os.getenv("RUN_SAMPLE_REGRESSION", "0") != "1":
        pytest.skip("Set RUN_SAMPLE_REGRESSION=1 to run the live 9-image benchmark.")

    baseline = _load_baseline()
    image_paths = _collect_sample_images(baseline["sample_glob"])
    assert len(image_paths) >= int(baseline["acceptance"]["min_images"])

    _set_deterministic_seed(7)

    import modules.room_reconstructor as recon_mod
    from modules.room_reconstructor import RoomReconstructor

    # Keep live benchmark runnable on environments without Open3D/VGGT.
    recon_mod.ENABLE_REGISTRATION = False

    reconstructor = RoomReconstructor()
    reconstructor.use_vggt = False
    # Keep benchmark deterministic and independent of API quotas/timeouts.
    reconstructor.scene_analyzer = None
    reconstructor._postprocess_point_cloud = lambda points, colors=None: (points, colors)
    reconstructor.visualizer.create_plotly_visualization = lambda *args, **kwargs: None
    reconstructor.visualizer.export_html = lambda *args, **kwargs: None
    reconstructor.visualizer.save_point_cloud = lambda *args, **kwargs: None
    reconstructor.visualizer.create_mesh_from_points = lambda *args, **kwargs: None
    reconstructor.visualizer.save_mesh = lambda *args, **kwargs: None

    def _stub_render_floor_plan(_detection, _timestamp):
        stub_png = os.path.join(_project_root(), "outputs", "t9_stub_floor_plan.png")
        if not os.path.exists(stub_png):
            # 1x1 transparent PNG
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
                b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            with open(stub_png, "wb") as fh:
                fh.write(png_bytes)
        return {
            "floor_plan_arch_png": stub_png,
            "floor_plan_svg": None,
            "floor_plan_dxf": None,
        }

    reconstructor._render_floor_plan_model = _stub_render_floor_plan

    result = reconstructor.reconstruct(image_paths)
    assert result.get("success") is True

    report = _build_report(result)
    _assert_guardrails(report, baseline)

    output_path = os.path.join(_project_root(), "outputs", "t9_sample_room_report.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    assert os.path.exists(output_path)
