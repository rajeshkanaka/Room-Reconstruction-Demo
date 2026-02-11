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

    def test_prefers_horizontal_plane_over_dominant_wall_plane(self):
        """Floor detection should not pick a vertical wall even if wall has more points."""
        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        rng = np.random.default_rng(42)

        # Sparse floor.
        floor = np.column_stack(
            [
                rng.uniform(-2, 2, 200),
                rng.uniform(-0.02, 0.02, 200),
                rng.uniform(0, 3, 200),
            ]
        )
        # Dense wall plane at x ~= -2 (would dominate vanilla RANSAC).
        wall = np.column_stack(
            [
                rng.uniform(-2.02, -1.98, 800),
                rng.uniform(0.0, 2.6, 800),
                rng.uniform(0, 3, 800),
            ]
        )
        points = np.vstack([floor, wall])

        plane, _ = det.detect_floor_plane(points)
        assert abs(plane[1]) > 0.7, f"Expected horizontal floor plane, got {plane}"


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


class TestPointCloudWallPlaneExtraction:
    """Step 2.x: Plane-first wall extraction from 3D point cloud."""

    def test_detects_vertical_wall_planes_as_segments(self):
        try:
            import open3d  # noqa: F401
        except Exception:
            pytest.skip("Open3D unavailable")

        from modules.detection.wall_detector import WallDetector

        det = WallDetector()
        rng = np.random.default_rng(42)

        # Synthetic room: 4m x 3m footprint, ~2.7m height.
        floor = np.column_stack(
            [
                rng.uniform(-2.0, 2.0, 1200),
                rng.uniform(-0.02, 0.02, 1200),
                rng.uniform(-1.5, 1.5, 1200),
            ]
        )
        wall_left = np.column_stack(
            [
                rng.uniform(-2.02, -1.98, 900),
                rng.uniform(0.0, 2.7, 900),
                rng.uniform(-1.5, 1.5, 900),
            ]
        )
        wall_right = np.column_stack(
            [
                rng.uniform(1.98, 2.02, 900),
                rng.uniform(0.0, 2.7, 900),
                rng.uniform(-1.5, 1.5, 900),
            ]
        )
        wall_back = np.column_stack(
            [
                rng.uniform(-2.0, 2.0, 850),
                rng.uniform(0.0, 2.7, 850),
                rng.uniform(-1.52, -1.48, 850),
            ]
        )
        wall_front = np.column_stack(
            [
                rng.uniform(-2.0, 2.0, 850),
                rng.uniform(0.0, 2.7, 850),
                rng.uniform(1.48, 1.52, 850),
            ]
        )
        # Clutter points (non-wall geometry).
        clutter = np.column_stack(
            [
                rng.uniform(-0.5, 0.5, 500),
                rng.uniform(0.1, 1.1, 500),
                rng.uniform(-0.4, 0.4, 500),
            ]
        )
        points = np.vstack([floor, wall_left, wall_right, wall_back, wall_front, clutter])

        _, floor_inliers = det.detect_floor_plane(points)
        floor_h = float(np.median(points[floor_inliers, 1]))
        segments = det.detect_walls_from_point_cloud_planes(points, floor_h)

        assert len(segments) >= 4, f"Expected >=4 wall segments, got {len(segments)}"
        lengths = [float(np.linalg.norm(end - start)) for start, end in segments]
        assert max(lengths) > 2.5, f"Wall segments are too short: {lengths}"


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

    def test_rooms_to_wall_segments_closed_loop(self):
        from modules.detection.room_segmenter import RoomSegmenter
        from modules.geometry.floor_plan_model import RoomPolygon

        seg = RoomSegmenter()
        room = RoomPolygon(boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]))
        walls = seg.rooms_to_wall_segments([room], thickness=0.15)

        assert len(walls) == 4
        assert np.allclose(walls[0].start, np.array([0, 0]))
        assert np.allclose(walls[-1].end, np.array([0, 0]))

    def test_optimize_wall_graph_closes_gapped_rectangle(self):
        from modules.detection.room_segmenter import RoomSegmenter

        seg = RoomSegmenter(snap_tolerance=0.25)
        # Rectangle with small endpoint gaps and split segments.
        raw = [
            (np.array([0.0, 0.0]), np.array([1.9, 0.0])),
            (np.array([2.1, 0.0]), np.array([4.0, 0.0])),
            (np.array([4.0, 0.0]), np.array([4.0, 1.4])),
            (np.array([4.0, 1.6]), np.array([4.0, 3.0])),
            (np.array([4.0, 3.0]), np.array([2.2, 3.0])),
            (np.array([1.8, 3.0]), np.array([0.0, 3.0])),
            (np.array([0.0, 3.0]), np.array([0.0, 1.7])),
            (np.array([0.0, 1.3]), np.array([0.0, 0.0])),
        ]

        optimized = seg.optimize_wall_graph(raw)
        rooms = seg.extract_rooms(optimized)

        assert len(optimized) >= 4
        assert len(rooms) >= 1
        assert abs(rooms[0].area - 12.0) < 2.0, f"Area: {rooms[0].area}"

    def test_optimize_wall_graph_removes_tiny_fragments(self):
        from modules.detection.room_segmenter import RoomSegmenter

        seg = RoomSegmenter(snap_tolerance=0.2)
        raw = [
            (np.array([0, 0]), np.array([4, 0])),
            (np.array([4, 0]), np.array([4, 3])),
            (np.array([4, 3]), np.array([0, 3])),
            (np.array([0, 3]), np.array([0, 0])),
            (np.array([1.0, 1.0]), np.array([1.05, 1.02])),  # noise fragment
        ]
        optimized = seg.optimize_wall_graph(raw, min_length=0.2)
        lengths = [np.linalg.norm(e - s) for s, e in optimized]

        assert all(l >= 0.2 - 1e-6 for l in lengths)

    def test_optimize_wall_graph_snaps_perpendicular_corner_gaps(self):
        from modules.detection.room_segmenter import RoomSegmenter

        seg = RoomSegmenter(snap_tolerance=0.18)
        # Rectangle where each corner has a small perpendicular gap.
        raw = [
            (np.array([0.05, 0.0]), np.array([4.0, 0.0])),
            (np.array([4.0, 0.07]), np.array([4.0, 3.0])),
            (np.array([3.95, 3.0]), np.array([0.0, 3.0])),
            (np.array([0.0, 2.95]), np.array([0.0, 0.05])),
        ]

        optimized = seg.optimize_wall_graph(raw, min_length=0.2)
        rooms = seg.extract_rooms(optimized)

        assert len(rooms) >= 1
        assert abs(rooms[0].area - 12.0) < 2.0, f"Area: {rooms[0].area}"

    def test_points_to_rectangular_room_from_noisy_points(self):
        from modules.detection.room_segmenter import RoomSegmenter

        seg = RoomSegmenter()
        rng = np.random.default_rng(42)

        # True room: 4m x 3m, rotated.
        u = rng.uniform(-2.0, 2.0, 1500)
        v = rng.uniform(-1.5, 1.5, 1500)
        base = np.column_stack([u, v])

        theta = np.radians(28.0)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        points = base @ rot.T + np.array([1.5, -0.8])

        # Add outliers to verify percentile trimming.
        outliers = rng.uniform(-8.0, 8.0, (50, 2))
        points = np.vstack([points, outliers])

        room = seg.points_to_rectangular_room(points, room_name="Test")
        assert room is not None
        assert room.boundary.shape == (4, 2)
        assert 8.0 <= room.area <= 16.0, f"Unexpected area: {room.area}"


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

    def test_orientation_normalization_aligns_longest_wall(self):
        from modules.room_reconstructor import RoomReconstructor
        from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon

        recon = RoomReconstructor.__new__(RoomReconstructor)

        # 4x3 rectangle rotated by 30 degrees.
        theta = np.radians(30.0)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rect = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float64)
        rect = rect @ rot.T + np.array([2.0, -1.0])

        walls = [
            WallSegment(start=rect[0].copy(), end=rect[1].copy()),
            WallSegment(start=rect[1].copy(), end=rect[2].copy()),
            WallSegment(start=rect[2].copy(), end=rect[3].copy()),
            WallSegment(start=rect[3].copy(), end=rect[0].copy()),
        ]
        model = FloorPlanModel(
            walls=walls,
            rooms=[RoomPolygon(boundary=rect.copy(), name="Room")],
        )

        out = recon._normalize_model_orientation(model)
        lengths = [np.linalg.norm(w.end - w.start) for w in out.walls]
        longest = out.walls[int(np.argmax(lengths))]
        vec = longest.end - longest.start
        angle = abs(np.degrees(np.arctan2(vec[1], vec[0])))
        assert angle < 5.0, f"Longest wall should be near horizontal, got {angle:.2f}°"

    def test_multiview_support_filter_drops_unstable_segments(self):
        from modules.room_reconstructor import RoomReconstructor

        recon = RoomReconstructor.__new__(RoomReconstructor)
        segments = [
            (np.array([0.0, 0.0]), np.array([4.0, 0.0])),  # stable horizontal
            (np.array([4.0, 0.0]), np.array([4.0, 3.0])),  # stable vertical
            (np.array([1.0, -0.2]), np.array([2.2, 2.4])),  # unstable diagonal
        ]
        depth_segments_by_view = {
            0: [
                (np.array([0.0, 0.02]), np.array([3.9, 0.01])),
                (np.array([4.02, 0.0]), np.array([4.01, 2.9])),
            ],
            1: [
                (np.array([0.1, -0.02]), np.array([4.0, 0.03])),
                (np.array([3.98, -0.02]), np.array([4.02, 3.0])),
            ],
            2: [
                (np.array([0.0, 0.0]), np.array([4.1, 0.0])),
                (np.array([4.0, 0.1]), np.array([4.0, 3.1])),
            ],
        }

        filtered, stats = recon._filter_segments_by_multiview_support(
            segments=segments,
            depth_segments_by_view=depth_segments_by_view,
            min_support_views=2,
            min_segments_to_keep=2,
        )

        assert stats["enabled"] is True
        assert len(filtered) == 2
        assert stats["dropped_segments"] >= 1

    def test_regularize_rooms_reduces_noisy_polygon(self):
        from modules.room_reconstructor import RoomReconstructor
        from modules.detection.room_segmenter import RoomSegmenter
        from modules.geometry.floor_plan_model import RoomPolygon

        recon = RoomReconstructor.__new__(RoomReconstructor)
        recon.room_segmenter = RoomSegmenter()

        noisy_hex = np.array(
            [
                [0.0, 0.0],
                [3.0, 0.1],
                [3.6, 1.1],
                [3.0, 2.0],
                [0.2, 2.1],
                [-0.4, 1.0],
            ],
            dtype=np.float64,
        )
        room = RoomPolygon(boundary=noisy_hex, name="Room")

        regularized = recon._regularize_rooms([room], floor_points_2d=noisy_hex)
        assert len(regularized) == 1
        assert len(regularized[0].boundary) == 4

    def test_regularize_rooms_triangle_promotes_rectangular(self):
        from modules.room_reconstructor import RoomReconstructor
        from modules.detection.room_segmenter import RoomSegmenter
        from modules.geometry.floor_plan_model import RoomPolygon

        recon = RoomReconstructor.__new__(RoomReconstructor)
        recon.room_segmenter = RoomSegmenter()

        triangle = np.array(
            [
                [0.0, 0.0],
                [3.8, 0.0],
                [0.8, 1.3],
            ],
            dtype=np.float64,
        )
        # Floor support indicates broader rectangular footprint.
        floor_points = np.array(
            [
                [0.0, 0.0],
                [3.8, 0.0],
                [3.8, 1.8],
                [0.0, 1.8],
                [1.9, 0.9],
                [3.2, 1.4],
                [0.5, 1.3],
                [2.7, 0.4],
                [1.1, 1.6],
                [2.2, 1.1],
                [0.2, 0.7],
                [3.6, 0.8],
                [1.5, 0.2],
                [2.9, 1.6],
                [0.9, 1.0],
                [3.1, 0.3],
                [2.0, 1.7],
                [0.4, 0.4],
                [1.3, 1.5],
                [2.5, 0.6],
                [1.7, 1.2],
            ],
            dtype=np.float64,
        )
        room = RoomPolygon(boundary=triangle, name="Room")

        regularized = recon._regularize_rooms([room], floor_points_2d=floor_points)
        assert len(regularized) == 1
        assert len(regularized[0].boundary) == 4

    def test_wall_closure_metrics_closed_loop(self):
        from modules.room_reconstructor import RoomReconstructor
        from modules.geometry.floor_plan_model import WallSegment

        walls = [
            WallSegment(start=np.array([0.0, 0.0]), end=np.array([4.0, 0.0])),
            WallSegment(start=np.array([4.0, 0.0]), end=np.array([4.0, 3.0])),
            WallSegment(start=np.array([4.0, 3.0]), end=np.array([0.0, 3.0])),
            WallSegment(start=np.array([0.0, 3.0]), end=np.array([0.0, 0.0])),
        ]

        metrics = RoomReconstructor._compute_wall_closure_metrics(walls, tolerance=0.15)
        assert metrics["closure_score"] >= 0.95
        assert metrics["closure_mean_gap_m"] <= 0.01
        assert metrics["closure_unpaired_ratio"] <= 0.05

    def test_wall_closure_metrics_fragmented_graph(self):
        from modules.room_reconstructor import RoomReconstructor
        from modules.geometry.floor_plan_model import WallSegment

        walls = [
            WallSegment(start=np.array([0.0, 0.0]), end=np.array([3.6, 0.1])),
            WallSegment(start=np.array([4.2, 0.4]), end=np.array([4.1, 2.8])),
            WallSegment(start=np.array([3.9, 3.4]), end=np.array([0.4, 3.1])),
            WallSegment(start=np.array([-0.3, 2.6]), end=np.array([0.2, 0.8])),
        ]

        metrics = RoomReconstructor._compute_wall_closure_metrics(walls, tolerance=0.15)
        assert metrics["closure_score"] < 0.7
        assert metrics["closure_mean_gap_m"] > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
