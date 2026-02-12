"""
Tests for Learned Floor Plan Detection (CAGE / RoomFormer).

Tests the model-agnostic components:
- Density map projection from point cloud
- Polygon-to-FloorPlanModel conversion
- Metric scaling (pixel -> meter coordinates)
- Wall deduplication and closure scoring
- Fallback behavior when model unavailable
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDensityMapProjection:
    """Test point cloud -> density map projection."""

    def test_basic_projection_shape(self):
        """Density map output has correct shape and dtype."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)
        rng = np.random.default_rng(42)

        # Synthetic room: 4m x 3m, floor at Y=0
        pts = np.column_stack(
            [
                rng.uniform(-2, 2, 2000),  # X
                rng.uniform(0.0, 2.7, 2000),  # Y (height)
                rng.uniform(-1.5, 1.5, 2000),  # Z
            ]
        )

        density, meta = detector.project_to_density_map(
            pts, floor_height=0.0, resolution=256
        )

        assert density.shape == (256, 256)
        assert density.dtype == np.float32
        assert 0.0 <= density.min()
        assert density.max() <= 1.0

    def test_projection_meta_has_required_keys(self):
        """Projection metadata contains all required keys for coordinate conversion."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)
        rng = np.random.default_rng(42)
        pts = rng.uniform(-2, 2, (1000, 3))

        _, meta = detector.project_to_density_map(pts, floor_height=0.0)

        required_keys = [
            "x_range",
            "z_range",
            "meters_per_pixel_x",
            "meters_per_pixel_z",
            "origin",
            "resolution",
            "num_points_projected",
        ]
        for key in required_keys:
            assert key in meta, f"Missing key: {key}"

    def test_projection_respects_height_band(self):
        """Only points within the height band appear in density map."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        # Floor points (should be excluded from default band 0.1-2.5m)
        floor = np.column_stack(
            [
                np.zeros(500),
                np.zeros(500),  # Y=0 (below band)
                np.zeros(500),
            ]
        )
        # Wall points (should be included)
        walls = np.column_stack(
            [
                np.full(500, 1.0),
                np.full(500, 1.0),  # Y=1.0 (within band)
                np.full(500, 1.0),
            ]
        )
        pts = np.vstack([floor, walls])

        density, meta = detector.project_to_density_map(pts, floor_height=0.0)
        # Density should have non-zero values (from wall points)
        assert density.sum() > 0

    def test_projection_metric_scale(self):
        """Meters-per-pixel correctly reflects the physical extent."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        # Room exactly 4m x 3m, walls at Y=1m
        pts = np.array(
            [
                [-2, 1.0, -1.5],
                [2, 1.0, -1.5],
                [2, 1.0, 1.5],
                [-2, 1.0, 1.5],
            ]
            * 250
        )  # repeat for density

        density, meta = detector.project_to_density_map(
            pts, floor_height=0.0, resolution=256
        )

        # X extent ~4m, Z extent ~3m (plus padding)
        x_range = meta["x_range"]
        z_range = meta["z_range"]
        x_extent = x_range[1] - x_range[0]
        z_extent = z_range[1] - z_range[0]

        # Should be close to 4m and 3m (with some padding)
        assert 3.5 < x_extent < 5.0, f"X extent out of range: {x_extent}"
        assert 2.5 < z_extent < 4.0, f"Z extent out of range: {z_extent}"

    def test_empty_point_cloud(self):
        """Handles empty or near-empty point clouds gracefully."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        # Very few points
        pts = np.array([[0, 1.0, 0], [1, 1.0, 1], [2, 1.0, 2]])
        density, meta = detector.project_to_density_map(pts, floor_height=0.0)
        assert density.shape == (256, 256)


class TestPolygonToModelConversion:
    """Test raw model output -> FloorPlanModel conversion."""

    def test_converts_rectangular_room(self):
        """Single rectangular room polygon converts to correct FloorPlanModel."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)
        detector.model_type = "cage"

        # Simulate model output: rectangle in pixel coords
        raw_output = {
            "pred_rooms": [
                np.array([[10, 10], [200, 10], [200, 180], [10, 180]], dtype=np.float64)
            ],
            "pred_types": [0],  # living_room
            "scores": [0.95],
            "pred_doors": [],
            "pred_windows": [],
        }

        projection_meta = {
            "origin": np.array([0.0, 0.0]),
            "meters_per_pixel_x": 0.02,  # 256px * 0.02 = 5.12m
            "meters_per_pixel_z": 0.02,
            "resolution": 256,
        }

        model = detector._convert_to_floor_plan_model(raw_output, projection_meta)

        assert len(model.rooms) == 1
        assert model.rooms[0].room_type == "living_room"
        assert model.rooms[0].room_shape == "rectangular"
        assert len(model.walls) == 4  # 4 edges of rectangle
        assert model.rooms[0].area > 0

    def test_filters_low_confidence_rooms(self):
        """Rooms below confidence threshold are excluded."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)
        detector.model_type = "cage"

        raw_output = {
            "pred_rooms": [
                np.array(
                    [[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float64
                ),
                np.array(
                    [[110, 10], [200, 10], [200, 100], [110, 100]], dtype=np.float64
                ),
            ],
            "scores": [0.9, 0.1],  # second room is low confidence
            "pred_types": [0, 1],
            "pred_doors": [],
            "pred_windows": [],
        }

        projection_meta = {
            "origin": np.array([0.0, 0.0]),
            "meters_per_pixel_x": 0.02,
            "meters_per_pixel_z": 0.02,
            "resolution": 256,
        }

        model = detector._convert_to_floor_plan_model(raw_output, projection_meta)
        assert len(model.rooms) == 1  # only high-confidence room

    def test_converts_doors_and_windows(self):
        """Doors and windows are converted to correct metric positions."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)
        detector.model_type = "cage"

        raw_output = {
            "pred_rooms": [
                np.array([[10, 10], [200, 10], [200, 180], [10, 180]], dtype=np.float64)
            ],
            "scores": [0.9],
            "pred_types": [0],
            "pred_doors": [np.array([100.0, 10.0, 45.0])],  # x, z, width in pixels
            "pred_windows": [np.array([200.0, 90.0, 60.0])],
        }

        projection_meta = {
            "origin": np.array([-2.0, -1.5]),
            "meters_per_pixel_x": 0.02,
            "meters_per_pixel_z": 0.02,
            "resolution": 256,
        }

        model = detector._convert_to_floor_plan_model(raw_output, projection_meta)
        assert len(model.doors) == 1
        assert len(model.windows) == 1
        assert model.doors[0].source == "learned_model"
        assert model.windows[0].source == "learned_model"
        # Door width: 45px * 0.02 m/px = 0.9m
        assert abs(model.doors[0].width - 0.9) < 0.01

    def test_metric_coordinate_conversion(self):
        """Pixel coordinates are correctly converted to metric coordinates."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)
        detector.model_type = "cage"

        # Room at pixel (50, 50) to (200, 150) with known scale
        raw_output = {
            "pred_rooms": [
                np.array([[50, 50], [200, 50], [200, 150], [50, 150]], dtype=np.float64)
            ],
            "scores": [0.9],
            "pred_types": [0],
            "pred_doors": [],
            "pred_windows": [],
        }

        projection_meta = {
            "origin": np.array([-3.0, -2.0]),  # offset
            "meters_per_pixel_x": 0.025,
            "meters_per_pixel_z": 0.025,
            "resolution": 256,
        }

        model = detector._convert_to_floor_plan_model(raw_output, projection_meta)

        # First vertex: pixel (50, 50) -> meters (50*0.025 - 3.0, 50*0.025 - 2.0)
        #                                      = (-1.75, -0.75)
        expected_x = 50 * 0.025 + (-3.0)
        expected_z = 50 * 0.025 + (-2.0)
        np.testing.assert_allclose(model.rooms[0].boundary[0, 0], expected_x, atol=1e-6)
        np.testing.assert_allclose(model.rooms[0].boundary[0, 1], expected_z, atol=1e-6)


class TestWallDeduplication:
    """Test wall segment deduplication."""

    def test_removes_shared_edges(self):
        """Shared edges between adjacent rooms are deduplicated."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )
        from modules.geometry.floor_plan_model import WallSegment

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        # Two walls with nearly identical midpoints
        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([0.01, 0.01]), end=np.array([3.99, 0.01])),
        ]

        unique = detector._deduplicate_walls(walls, threshold=0.1)
        assert len(unique) == 1

    def test_keeps_distinct_walls(self):
        """Walls with different positions are kept."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )
        from modules.geometry.floor_plan_model import WallSegment

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([4, 0]), end=np.array([4, 3])),
            WallSegment(start=np.array([4, 3]), end=np.array([0, 3])),
            WallSegment(start=np.array([0, 3]), end=np.array([0, 0])),
        ]

        unique = detector._deduplicate_walls(walls)
        assert len(unique) == 4


class TestClosureScoring:
    """Test wall closure score computation."""

    def test_perfect_closure(self):
        """Perfectly closed rectangle scores near 1.0."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )
        from modules.geometry.floor_plan_model import WallSegment

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([4, 0]), end=np.array([4, 3])),
            WallSegment(start=np.array([4, 3]), end=np.array([0, 3])),
            WallSegment(start=np.array([0, 3]), end=np.array([0, 0])),
        ]

        result = detector._compute_closure_score(walls)
        assert result["closure_score"] > 0.9
        assert result["mean_gap_m"] < 0.01

    def test_open_walls_low_score(self):
        """Walls with large gaps score low."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )
        from modules.geometry.floor_plan_model import WallSegment

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([5, 1]), end=np.array([5, 3])),  # big gap
            WallSegment(start=np.array([6, 4]), end=np.array([0, 4])),  # big gap
        ]

        result = detector._compute_closure_score(walls)
        assert result["closure_score"] < 0.5


class TestRoomTypeLabels:
    """Test room type label conversion."""

    def test_known_room_types(self):
        """Known Structured3D room type indices map correctly."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        assert LearnedFloorplanDetector._room_type_label(0) == "living_room"
        assert LearnedFloorplanDetector._room_type_label(2) == "kitchen"
        assert LearnedFloorplanDetector._room_type_label(3) == "bathroom"

    def test_unknown_room_type(self):
        """Unknown indices return fallback label."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        label = LearnedFloorplanDetector._room_type_label(99)
        assert label == "room_99"


class TestRoomShapeClassification:
    """Test room shape classification from vertex count."""

    def test_shape_classification(self):
        """Vertex counts map to correct room shapes."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)

        rect = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert detector._classify_room_shape(rect) == "rectangular"

        l_shape = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
        assert detector._classify_room_shape(l_shape) == "l_shaped"


class TestFloorHeightDetection:
    """Test auto floor height detection."""

    def test_detects_floor_at_lowest_cluster(self):
        """Floor height detected at approximately the 5th percentile of Y values."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector.__new__(LearnedFloorplanDetector)
        rng = np.random.default_rng(42)

        # Floor at Y~0, walls/ceiling at Y>0.5
        pts = np.column_stack(
            [
                rng.uniform(-2, 2, 1000),
                np.concatenate(
                    [rng.uniform(-0.05, 0.05, 500), rng.uniform(0.5, 2.7, 500)]
                ),
                rng.uniform(-1.5, 1.5, 1000),
            ]
        )

        fh = detector._detect_floor_height(pts)
        assert -0.1 < fh < 0.15, f"Floor height should be near 0, got {fh}"


class TestModelFallback:
    """Test that model unavailability raises NotImplementedError for fallback."""

    @pytest.mark.slow
    def test_cage_loads_and_detects(self):
        """CAGE model loads checkpoint and runs inference (requires weights)."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector(model_type="cage")
        rng = np.random.default_rng(42)
        pts = rng.uniform(-2, 2, (1000, 3))

        result = detector.detect(pts, floor_height=0.0)
        # With random data the model may return None (no rooms detected)
        # or a valid dict -- either is acceptable as long as it doesn't crash
        if result is not None:
            assert "floor_plan_model" in result
            assert "measurements" in result
            assert "quality_flags" in result

    def test_roomformer_raises_not_implemented(self):
        """RoomFormer model raises NotImplementedError when not installed."""
        from modules.detection.learned_floorplan_detector import (
            LearnedFloorplanDetector,
        )

        detector = LearnedFloorplanDetector(model_type="roomformer")
        rng = np.random.default_rng(42)
        pts = rng.uniform(-2, 2, (1000, 3))

        with pytest.raises(NotImplementedError):
            detector.detect(pts, floor_height=0.0)
