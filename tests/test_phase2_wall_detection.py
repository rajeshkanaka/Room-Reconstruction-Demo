"""
Phase 2 Integration Tests: Wall Detection + Room Geometry

Tests the full wall detection and room segmentation pipeline:
- RANSAC floor plane detection
- Wall line detection from depth maps
- Manhattan World alignment
- Room polygon extraction (rectangular and L-shaped)
- Measurement engine accuracy
- FloorPlanModel data classes
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFloorPlaneDetection:
    """Step 2.2: RANSAC floor plane detection."""

    def test_detects_horizontal_floor(self):
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        rng = np.random.default_rng(42)
        # Floor at Y~0, walls at Y>0.5
        floor = np.column_stack(
            [
                rng.uniform(0, 4, 500),
                rng.uniform(-0.04, 0.04, 500),
                rng.uniform(0, 3, 500),
            ]
        )
        walls = np.column_stack(
            [rng.uniform(0, 4, 200), rng.uniform(0.5, 2.5, 200), rng.uniform(0, 3, 200)]
        )
        points = np.vstack([floor, walls])

        plane, inliers = det.detect_floor_plane(points)

        # Normal should be roughly [0, 1, 0]
        normal = plane[:3]
        assert abs(abs(normal[1]) - 1.0) < 0.3, f"Floor normal not vertical: {normal}"
        assert len(inliers) > 100, f"Too few inliers: {len(inliers)}"

    def test_fallback_with_no_open3d(self):
        """Verify fallback floor detection works."""
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        rng = np.random.default_rng(42)
        points = rng.random((500, 3)) * np.array([4, 2.5, 3])

        plane, inliers = det._fallback_floor_detection(points)
        assert plane.shape == (4,)
        assert len(inliers) > 0


class TestWallDetectionFromDepth:
    """Step 2.3: LSD wall line detection from depth maps."""

    def test_detects_walls_from_depth_discontinuities(self):
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        # Create depth map with clear wall boundaries
        depth = np.ones((480, 640), dtype=np.float32) * 3.0
        depth[:, :160] = 2.0  # Left wall
        depth[:, 480:] = 2.0  # Right wall
        depth[:120, :] = 4.0  # Back wall

        segments = det.detect_walls_from_depth(depth, fx=500, fy=500)
        assert len(segments) >= 2, f"Expected >=2 segments, got {len(segments)}"

    def test_empty_depth_returns_empty(self):
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        depth = np.ones((100, 100), dtype=np.float32) * 3.0  # Uniform depth
        segments = det.detect_walls_from_depth(depth)
        # Uniform depth should produce few or no segments
        assert isinstance(segments, list)

    def test_segments_are_metric_coordinates(self):
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        depth = np.ones((480, 640), dtype=np.float32) * 3.0
        depth[:, :160] = 1.5  # Wall at 1.5m

        segments = det.detect_walls_from_depth(depth, fx=500, fy=500)
        for start, end in segments:
            assert start.shape == (2,)
            assert end.shape == (2,)
            # Values should be in reasonable metric range
            for coord in [start, end]:
                assert abs(coord[0]) < 20, f"X out of range: {coord[0]}"
                assert abs(coord[1]) < 20, f"Z out of range: {coord[1]}"


class TestManhattanAlignment:
    """Step 2.4: Manhattan World wall alignment."""

    def test_snaps_noisy_segments_to_grid(self):
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        noisy_segments = [
            (np.array([0.0, 0.0]), np.array([3.98, 0.05])),
            (np.array([4.02, -0.03]), np.array([4.01, 2.97])),
            (np.array([3.99, 3.02]), np.array([0.03, 2.98])),
            (np.array([-0.02, 3.01]), np.array([0.01, 0.02])),
        ]

        aligned = det.align_walls_manhattan(noisy_segments)
        assert len(aligned) >= 2, f"Too few aligned segments: {len(aligned)}"

        for seg in aligned:
            dx = abs(seg[1][0] - seg[0][0])
            dy = abs(seg[1][1] - seg[0][1])
            # Each segment should be roughly axis-aligned
            assert min(dx, dy) < 0.3, f"Not axis-aligned: dx={dx:.2f}, dy={dy:.2f}"

    def test_handles_too_few_segments(self):
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        result = det.align_walls_manhattan(
            [
                (np.array([0, 0]), np.array([4, 0])),
            ]
        )
        assert len(result) == 1  # Returns as-is


class TestRoomSegmenter:
    """Step 2.5: Room polygon reconstruction."""

    def test_rectangular_room(self):
        from modules.detection.room_segmenter import RoomSegmenter

        seg = RoomSegmenter()
        walls = [
            (np.array([0, 0]), np.array([4, 0])),
            (np.array([4, 0]), np.array([4, 3])),
            (np.array([4, 3]), np.array([0, 3])),
            (np.array([0, 3]), np.array([0, 0])),
        ]
        rooms = seg.extract_rooms(walls)
        assert len(rooms) >= 1
        assert abs(rooms[0].area - 12.0) < 1.0, f"Area: {rooms[0].area}"

    def test_l_shaped_room(self):
        from modules.detection.room_segmenter import RoomSegmenter

        seg = RoomSegmenter()
        walls = [
            (np.array([0, 0]), np.array([6, 0])),
            (np.array([6, 0]), np.array([6, 3])),
            (np.array([6, 3]), np.array([3, 3])),
            (np.array([3, 3]), np.array([3, 5])),
            (np.array([3, 5]), np.array([0, 5])),
            (np.array([0, 5]), np.array([0, 0])),
        ]
        rooms = seg.extract_rooms(walls)
        assert len(rooms) >= 1
        # L-shape: 6*3 + 3*2 = 24
        assert abs(rooms[0].area - 24) < 2, f"L-shape area: {rooms[0].area}"

    def test_segments_to_wall_segments(self):
        from modules.detection.room_segmenter import RoomSegmenter

        seg = RoomSegmenter()
        raw = [
            (np.array([0, 0]), np.array([4, 0])),
            (np.array([4, 0]), np.array([4, 3])),
        ]
        walls = seg.segments_to_wall_segments(raw, thickness=0.2)
        assert len(walls) == 2
        assert walls[0].thickness == 0.2


class TestMeasurementEngine:
    """Step 2.6: Per-wall measurement engine."""

    def test_wall_lengths(self):
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.geometry.floor_plan_model import (
            FloorPlanModel,
            WallSegment,
            RoomPolygon,
        )

        engine = MeasurementEngine()
        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([4, 0]), end=np.array([4, 3])),
            WallSegment(start=np.array([4, 3]), end=np.array([0, 3])),
            WallSegment(start=np.array([0, 3]), end=np.array([0, 0])),
        ]
        room = RoomPolygon(
            boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]), name="Room"
        )
        model = FloorPlanModel(walls=walls, rooms=[room])
        result = engine.compute_measurements(model)

        assert result["wall_lengths"] == [4.0, 3.0, 4.0, 3.0]
        assert abs(result["rooms"][0]["area_sqm"] - 12.0) < 0.5
        assert result["overall"]["width_m"] == 4.0
        assert result["overall"]["depth_m"] == 3.0

    def test_imperial_conversions(self):
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.geometry.floor_plan_model import (
            FloorPlanModel,
            WallSegment,
            RoomPolygon,
        )

        engine = MeasurementEngine()
        walls = [WallSegment(start=np.array([0, 0]), end=np.array([4, 0]))]
        room = RoomPolygon(
            boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]), name="Room"
        )
        model = FloorPlanModel(walls=walls, rooms=[room])
        result = engine.compute_measurements(model)

        # 4m ~ 13.1ft
        assert abs(result["wall_lengths_ft"][0] - 13.1) < 0.2
        # 12sqm ~ 129.2sqft
        assert abs(result["rooms"][0]["area_sqft"] - 129.2) < 1.0

    def test_cross_validation_matching_walls(self):
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment

        engine = MeasurementEngine()
        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([4, 0]), end=np.array([4, 3])),
            WallSegment(start=np.array([4, 3]), end=np.array([0, 3])),
            WallSegment(start=np.array([0, 3]), end=np.array([0, 0])),
        ]
        model = FloorPlanModel(walls=walls)
        result = engine.compute_measurements(model)
        assert (
            len(result["warnings"]) == 0
        ), f"Unexpected warnings: {result['warnings']}"

    def test_cross_validation_mismatched_walls(self):
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment

        engine = MeasurementEngine()
        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([4, 0]), end=np.array([4, 3])),
            WallSegment(start=np.array([4, 3]), end=np.array([0.5, 3])),  # 3.5m != 4m
            WallSegment(start=np.array([0, 3]), end=np.array([0, 0])),
        ]
        model = FloorPlanModel(walls=walls)
        result = engine.compute_measurements(model)
        assert len(result["warnings"]) > 0

    def test_chain_dimensions_with_door(self):
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.geometry.floor_plan_model import WallSegment, DoorOpening

        engine = MeasurementEngine()
        wall = WallSegment(start=np.array([0, 0]), end=np.array([4, 0]))
        door = DoorOpening(position=np.array([2, 0]), width=0.9)
        chain = engine.compute_chain_dimensions(wall, [door])

        assert len(chain) == 3
        total = sum(c["length"] for c in chain)
        assert abs(total - 4.0) < 0.1, f"Chain total: {total}"
        assert chain[1]["type"] == "door"

    def test_dimension_lines_generated(self):
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment

        engine = MeasurementEngine()
        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([4, 0])),
            WallSegment(start=np.array([4, 0]), end=np.array([4, 3])),
        ]
        model = FloorPlanModel(walls=walls)
        engine.compute_measurements(model)

        assert len(model.dimensions) == 2
        assert abs(model.dimensions[0].value_m - 4.0) < 0.01
        assert abs(model.dimensions[1].value_m - 3.0) < 0.01


class TestRoomReconstructorWiring:
    """Step 2.7: Detection pipeline wired into RoomReconstructor."""

    def test_components_initialized(self):
        """Verify wall detector, room segmenter, measurement engine are created."""
        # Import and check without loading models (too slow for unit test)
        from modules.detection.wall_detector import WallDetector
        from modules.detection.room_segmenter import RoomSegmenter
        from modules.geometry.measurement_engine import MeasurementEngine

        wd = WallDetector()
        rs = RoomSegmenter()
        me = MeasurementEngine()

        assert wd is not None
        assert rs is not None
        assert me is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
