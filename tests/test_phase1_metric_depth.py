"""
Phase 1 Integration Tests: Metric Depth Pipeline

Tests the full metric depth pipeline:
- MetricDepthEstimator model loading and inference
- Metric depth-to-3D projection (no inverse-depth hack)
- DepthCalibrator scale correction
- FloorPlanGenerator metric mode (scale_factor=1.0)
- Config constants
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Step 1.5: Verify metric depth config constants."""

    def test_metric_config_importable(self):
        from config import (
            ENABLE_METRIC_DEPTH,
            METRIC_DEPTH_MODEL,
            METRIC_DEPTH_MODEL_FALLBACK,
            CALIBRATION_METHOD,
        )

        assert isinstance(ENABLE_METRIC_DEPTH, bool)
        assert isinstance(METRIC_DEPTH_MODEL, str)
        assert isinstance(METRIC_DEPTH_MODEL_FALLBACK, str)
        assert CALIBRATION_METHOD in ("auto", "user_reference", "none")

    def test_legacy_config_still_works(self):
        from config import DEPTH_MODEL, DEPTH_MODEL_FALLBACK, ASSUMED_ROOM_WIDTH_METERS

        assert isinstance(DEPTH_MODEL, str)
        assert isinstance(DEPTH_MODEL_FALLBACK, str)
        assert ASSUMED_ROOM_WIDTH_METERS > 0


class TestDepthCalibrator:
    """Step 1.4: DepthCalibrator unit tests."""

    def test_calibrate_from_known_dimension(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        # Points spanning ~5m on X axis
        rng = np.random.default_rng(42)
        points = rng.random((1000, 3)) * 5
        factor = cal.calibrate_from_known_dimension(points, axis=0, known_meters=4.0)

        assert 0.1 < factor < 10, f"Unreasonable factor: {factor}"
        calibrated = cal.apply(points)
        lo = np.percentile(calibrated[:, 0], 2)
        hi = np.percentile(calibrated[:, 0], 98)
        x_range = hi - lo
        assert abs(x_range - 4.0) < 0.5, f"Calibration failed: width={x_range:.2f}m"

    def test_cross_view_consistency(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        # Consistent views
        spreads = [(4.0, 2.5, 3.0), (4.1, 2.4, 3.1), (3.9, 2.6, 2.9)]
        consistency, confidence = cal.calibrate_cross_view(spreads)
        assert consistency > 0.8
        assert confidence == "high"

    def test_cross_view_inconsistent(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        # Wildly different views
        spreads = [(4.0, 2.5, 3.0), (10.0, 1.0, 8.0)]
        consistency, confidence = cal.calibrate_cross_view(spreads)
        assert consistency < 0.8
        assert confidence in ("medium", "low")

    def test_sanity_check_plausible(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        # Room-shaped point cloud: 4m x 2.5m x 3m
        rng = np.random.default_rng(42)
        points = rng.random((1000, 3))
        points[:, 0] *= 4  # width
        points[:, 1] *= 2.5  # height
        points[:, 2] *= 3  # depth
        assert cal.sanity_check(points) == True

    def test_sanity_check_implausible(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        # Tiny point cloud: 0.1m x 0.1m x 0.1m
        rng = np.random.default_rng(42)
        points = rng.random((1000, 3)) * 0.1
        assert cal.sanity_check(points) == False

    def test_solve_scale_from_cues_auto_prefers_height_prior(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        rng = np.random.default_rng(7)
        points = rng.random((2000, 3))
        points[:, 0] *= 1.0  # horizontal span
        points[:, 1] *= 1.1  # height span
        points[:, 2] *= 0.9  # horizontal span

        solution = cal.solve_scale_from_cues(
            points,
            assumed_room_width=4.0,
            camera_poses=None,
            mode="auto",
        )

        assert 2.0 < solution["scale_factor"] < 3.5
        cue_names = {cue["name"] for cue in solution["cues"]}
        assert "ceiling_height_prior" in cue_names
        assert "room_width_prior" in cue_names

    def test_solve_scale_from_cues_user_reference_favors_width(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        rng = np.random.default_rng(11)
        points = rng.random((2000, 3))
        points[:, 0] *= 1.0
        points[:, 1] *= 1.1
        points[:, 2] *= 0.9

        solution = cal.solve_scale_from_cues(
            points,
            assumed_room_width=4.0,
            camera_poses=None,
            mode="user_reference",
        )

        assert solution["scale_factor"] > 3.0
        assert solution["confidence"] in ("high", "medium")

    def test_solve_scale_from_cues_uses_camera_baseline(self):
        from modules.depth.depth_calibrator import DepthCalibrator

        cal = DepthCalibrator()
        rng = np.random.default_rng(19)
        points = rng.random((1200, 3))
        points[:, 0] *= 1.0
        points[:, 1] *= 1.1
        points[:, 2] *= 0.9

        camera_poses = {
            0: {"transform": np.eye(4)},
            1: {"transform": np.array(
                [[1.0, 0.0, 0.0, 0.4], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )},
            2: {"transform": np.array(
                [[1.0, 0.0, 0.0, 0.8], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )},
        }

        solution = cal.solve_scale_from_cues(
            points,
            assumed_room_width=None,
            camera_poses=camera_poses,
            mode="auto",
        )

        cue_names = {cue["name"] for cue in solution["cues"]}
        assert "camera_baseline_prior" in cue_names
        assert solution["scale_factor"] > 1.5


class TestFloorPlanModel:
    """Step 2.1 (already created in 1.1): FloorPlanModel dataclass tests."""

    def test_wall_segment(self):
        from modules.geometry.floor_plan_model import WallSegment

        w = WallSegment(start=np.array([0, 0]), end=np.array([4, 0]))
        assert w.thickness == 0.15
        assert w.material == "interior"

    def test_room_polygon_area(self):
        from modules.geometry.floor_plan_model import RoomPolygon

        room = RoomPolygon(
            boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
            name="Living Room",
        )
        assert abs(room.area - 12.0) < 0.01

    def test_floor_plan_model(self):
        from modules.geometry.floor_plan_model import (
            FloorPlanModel,
            WallSegment,
            RoomPolygon,
        )

        w1 = WallSegment(start=np.array([0, 0]), end=np.array([4, 0]))
        w2 = WallSegment(start=np.array([4, 0]), end=np.array([4, 3]))
        room = RoomPolygon(
            boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
            name="Room",
        )
        model = FloorPlanModel(walls=[w1, w2], rooms=[room])
        assert len(model.walls) == 2
        assert len(model.rooms) == 1
        assert model.scale == 50.0


class TestFloorPlanGeneratorMetric:
    """Step 1.7: FloorPlanGenerator metric mode tests."""

    def _make_room_points(self, width=4.0, depth=3.0, height=2.5, n=1000):
        """Create a synthetic room point cloud with known dimensions."""
        rng = np.random.default_rng(42)
        points = []
        # Floor
        for _ in range(n // 2):
            points.append(
                [
                    rng.uniform(0, width),
                    rng.uniform(0, 0.3),
                    rng.uniform(0, depth),
                ]
            )
        # Walls
        for _ in range(n // 8):
            points.append([0, rng.uniform(0, height), rng.uniform(0, depth)])
            points.append([width, rng.uniform(0, height), rng.uniform(0, depth)])
            points.append([rng.uniform(0, width), rng.uniform(0, height), 0])
            points.append([rng.uniform(0, width), rng.uniform(0, height), depth])
        return np.array(points)

    def test_metric_mode_no_scaling(self):
        """Metric mode: scale_factor=1.0, measurements from coordinates."""
        from modules.floor_plan_generator import FloorPlanGenerator

        points = self._make_room_points(width=4.0, depth=3.0)
        gen = FloorPlanGenerator(assumed_width=99.0)  # should be ignored
        result = gen.generate_floor_plan(points, is_metric=True)
        m = result["measurements"]

        assert m["scale_factor"] == 1.0
        assert abs(m["width_m"] - 4.0) < 1.0, f"Width: {m['width_m']}"
        assert abs(m["depth_m"] - 3.0) < 1.0, f"Depth: {m['depth_m']}"

    def test_legacy_mode_uses_assumed_width(self):
        """Legacy mode: scale_factor derived from assumed_width."""
        from modules.floor_plan_generator import FloorPlanGenerator

        points = self._make_room_points(width=4.0, depth=3.0)
        gen = FloorPlanGenerator(assumed_width=4.0)
        result = gen.generate_floor_plan(points, is_metric=False)
        m = result["measurements"]

        assert abs(m["width_m"] - 4.0) < 0.5

    def test_metric_different_room_sizes(self):
        """Metric mode handles different room sizes correctly."""
        from modules.floor_plan_generator import FloorPlanGenerator

        gen = FloorPlanGenerator(assumed_width=99.0)

        for width, depth in [(3.0, 2.5), (6.0, 4.5), (10.0, 8.0)]:
            points = self._make_room_points(width=width, depth=depth)
            result = gen.generate_floor_plan(points, is_metric=True)
            m = result["measurements"]
            assert (
                abs(m["width_m"] - width) < 1.5
            ), f"Width {m['width_m']} far from {width}"
            assert (
                abs(m["depth_m"] - depth) < 1.5
            ), f"Depth {m['depth_m']} far from {depth}"


class TestMetricDepthEstimatorImport:
    """Step 1.2-1.3: Verify MetricDepthEstimator is importable and structurally correct."""

    def test_import(self):
        from modules.depth.metric_depth import MetricDepthEstimator

        assert MetricDepthEstimator is not None

    def test_has_required_methods(self):
        from modules.depth.metric_depth import MetricDepthEstimator

        assert hasattr(MetricDepthEstimator, "estimate_depth")
        assert hasattr(MetricDepthEstimator, "depth_to_3d_points_metric")
        assert hasattr(MetricDepthEstimator, "get_focal_length")


class TestMetricDepthInference:
    """Step 1.2-1.3: Actual model inference tests (requires model download).

    These tests are marked slow since they require loading a large model.
    Run with: pytest -m slow
    """

    @pytest.mark.slow
    def test_estimate_depth_shape(self):
        from modules.depth.metric_depth import MetricDepthEstimator

        est = MetricDepthEstimator()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = est.estimate_depth(img)
        assert depth.shape == (480, 640), f"Wrong shape: {depth.shape}"
        assert depth.min() >= 0, "Negative depth"
        assert depth.max() < 100, f"Unreasonable max: {depth.max()}"

    @pytest.mark.slow
    def test_depth_to_3d_points(self):
        from modules.depth.metric_depth import MetricDepthEstimator

        est = MetricDepthEstimator()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = est.estimate_depth(img)
        points, colors = est.depth_to_3d_points_metric(img, depth)
        assert points.shape[1] == 3
        assert len(points) > 100
        z_range = points[:, 2].max() - points[:, 2].min()
        assert z_range > 0.01, "Z range too small"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
