"""
Phase 4 Integration Tests: Door/Window Detection

Tests the OpeningDetector with both SegFormer and fallback modes,
projection to floor plan coordinates, and rendering integration.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.geometry.floor_plan_model import (
    FloorPlanModel,
    WallSegment,
    RoomPolygon,
    DoorOpening,
    WindowOpening,
)


# ---------- OpeningDetector Tests ----------


class TestOpeningDetector:
    def test_ade20k_door_class_constant(self):
        from modules.detection.opening_detector import ADE20K_DOOR

        assert ADE20K_DOOR == 14

    def test_import(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()
        assert det is not None

    def test_detect_returns_dict(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = det.detect_openings(img)
        assert isinstance(result, dict)
        assert "doors" in result
        assert "windows" in result

    def test_detect_doors_list_format(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = det.detect_openings(img)
        for door in result["doors"]:
            assert "x" in door
            assert "y" in door
            assert "w" in door
            assert "h" in door
            assert "confidence" in door

    def test_detect_handles_grayscale(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        # Fallback should handle grayscale
        det.model = None  # Force fallback
        result = det.detect_openings(gray)
        assert "doors" in result

    def test_detect_handles_empty_image(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect_openings(img)
        assert isinstance(result, dict)

    def test_maps_bbox_from_image_to_depth_space(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector.__new__(OpeningDetector)
        bbox = {"x": 100, "y": 200, "w": 80, "h": 160}
        mapped = det._map_bbox_to_depth(
            bbox=bbox,
            image_shape=(1280, 640, 3),
            depth_shape=(512, 256),
        )

        assert mapped["x"] == 40
        assert mapped["y"] == 80
        assert mapped["w"] == 32
        assert mapped["h"] == 64


# ---------- Projection Tests ----------


class TestProjection:
    def test_project_door_to_floor_plan(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()

        bbox = {
            "x": 200,
            "y": 100,
            "w": 100,
            "h": 300,
            "area": 30000,
            "confidence": 0.8,
        }
        depth = np.ones((480, 640), dtype=np.float32) * 3.0
        wall = WallSegment(start=np.array([0, 0]), end=np.array([4, 0]), thickness=0.15)

        opening = det.project_to_floor_plan(
            bbox, depth, fx=500, fy=500, parent_wall=wall, opening_type="door"
        )
        assert opening is not None
        assert isinstance(opening, DoorOpening)
        assert 0.4 < opening.width < 2.0

    def test_project_window_to_floor_plan(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()

        bbox = {"x": 150, "y": 50, "w": 150, "h": 200, "area": 30000, "confidence": 0.7}
        depth = np.ones((480, 640), dtype=np.float32) * 3.0
        wall = WallSegment(start=np.array([0, 3]), end=np.array([5, 3]), thickness=0.15)

        opening = det.project_to_floor_plan(
            bbox, depth, fx=500, fy=500, parent_wall=wall, opening_type="window"
        )
        assert opening is not None
        assert isinstance(opening, WindowOpening)
        assert 0.3 < opening.width < 3.5

    def test_project_rejects_zero_depth(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()

        bbox = {
            "x": 200,
            "y": 100,
            "w": 100,
            "h": 300,
            "area": 30000,
            "confidence": 0.8,
        }
        depth = np.zeros((480, 640), dtype=np.float32)
        wall = WallSegment(start=np.array([0, 0]), end=np.array([4, 0]), thickness=0.15)

        opening = det.project_to_floor_plan(
            bbox, depth, fx=500, fy=500, parent_wall=wall
        )
        assert opening is None

    def test_deduplicate_openings(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector()

        openings = [
            DoorOpening(
                position=np.array([2.0, 0.0]), width=0.9, swing_direction="left"
            ),
            DoorOpening(
                position=np.array([2.1, 0.0]), width=0.9, swing_direction="left"
            ),  # Duplicate
            DoorOpening(
                position=np.array([5.0, 0.0]), width=0.9, swing_direction="left"
            ),  # Unique
        ]
        unique = det._deduplicate_openings(openings, threshold=0.5)
        assert len(unique) == 2

    def test_project_uses_camera_pose_for_wall_attachment(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector.__new__(OpeningDetector)

        bbox = {"x": 275, "y": 120, "w": 90, "h": 240, "confidence": 0.9}
        depth = np.ones((480, 640), dtype=np.float32) * 3.0
        wall = WallSegment(start=np.array([0.0, 3.0]), end=np.array([4.0, 3.0]))

        pose = np.eye(4)
        pose[0, 3] = 3.2
        pose[2, 3] = 0.0

        opening = det.project_to_floor_plan(
            bbox,
            depth,
            fx=500,
            fy=500,
            parent_wall=wall,
            opening_type="door",
            camera_pose=pose,
            wall_index=0,
            confidence=0.9,
            view_index=0,
        )
        assert opening is not None
        assert 2.8 < opening.position[0] < 3.6
        assert getattr(opening, "wall_index", -1) == 0
        assert getattr(opening, "support_views", 0) == 1

    def test_detect_and_project_fuses_multi_view_observations(self):
        from modules.detection.opening_detector import OpeningDetector

        det = OpeningDetector.__new__(OpeningDetector)
        det.last_fusion_stats = {}

        def mock_detect_openings(_image):
            return {
                "doors": [
                    {"x": 275, "y": 120, "w": 90, "h": 240, "confidence": 0.95}
                ],
                "windows": [],
            }

        det.detect_openings = mock_detect_openings

        images = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        depths = [
            np.ones((480, 640), dtype=np.float32) * 3.0,
            np.ones((480, 640), dtype=np.float32) * 3.0,
        ]
        walls = [WallSegment(start=np.array([0.0, 3.0]), end=np.array([4.0, 3.0]))]

        pose0 = np.eye(4)
        pose0[0, 3] = 1.0
        pose0[2, 3] = 0.0
        pose1 = np.eye(4)
        pose1[0, 3] = 1.2
        pose1[2, 3] = 0.0

        doors, windows = det.detect_and_project(
            images=images,
            depth_maps=depths,
            walls=walls,
            fx=500,
            fy=500,
            camera_poses={0: {"transform": pose0}, 1: {"transform": pose1}},
        )

        assert len(windows) == 0
        assert len(doors) == 1
        door = doors[0]
        assert getattr(door, "source", "") == "fused_multi_view"
        assert getattr(door, "support_views", 0) >= 2
        assert getattr(door, "wall_index", -1) == 0
        assert det.last_fusion_stats["raw_door_observations"] >= 2
        assert det.last_fusion_stats["fused_doors"] == 1


# ---------- Rendering Integration Tests ----------


class TestOpeningRendering:
    def test_svg_renders_door_arc(self):
        from modules.rendering.svg_renderer import SVGRenderer

        model = FloorPlanModel(
            walls=[
                WallSegment(
                    start=np.array([0, 0]), end=np.array([4, 0]), thickness=0.15
                ),
                WallSegment(
                    start=np.array([4, 0]), end=np.array([4, 3]), thickness=0.15
                ),
                WallSegment(
                    start=np.array([4, 3]), end=np.array([0, 3]), thickness=0.15
                ),
                WallSegment(
                    start=np.array([0, 3]), end=np.array([0, 0]), thickness=0.15
                ),
            ],
            rooms=[
                RoomPolygon(
                    boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]), name="Room"
                )
            ],
            doors=[
                DoorOpening(
                    position=np.array([2, 0]), width=0.9, swing_direction="left"
                )
            ],
        )

        svg = SVGRenderer().render_to_string(model)
        assert "path" in svg.lower()  # Door arc is an SVG path

    def test_svg_renders_window_lines(self):
        from modules.rendering.svg_renderer import SVGRenderer

        model = FloorPlanModel(
            walls=[
                WallSegment(
                    start=np.array([0, 0]), end=np.array([4, 0]), thickness=0.15
                ),
                WallSegment(
                    start=np.array([4, 0]), end=np.array([4, 3]), thickness=0.15
                ),
                WallSegment(
                    start=np.array([4, 3]), end=np.array([0, 3]), thickness=0.15
                ),
                WallSegment(
                    start=np.array([0, 3]), end=np.array([0, 0]), thickness=0.15
                ),
            ],
            rooms=[
                RoomPolygon(
                    boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]), name="Room"
                )
            ],
            windows=[WindowOpening(position=np.array([2, 3]), width=1.2)],
        )

        svg = SVGRenderer().render_to_string(model)
        assert "line" in svg.lower()

    def test_dxf_has_door_layer_entities(self):
        import tempfile
        import ezdxf
        from modules.rendering.dxf_renderer import DXFRenderer

        model = FloorPlanModel(
            walls=[
                WallSegment(
                    start=np.array([0, 0]), end=np.array([4, 0]), thickness=0.15
                )
            ],
            rooms=[
                RoomPolygon(
                    boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]), name="Room"
                )
            ],
            doors=[
                DoorOpening(
                    position=np.array([2, 0]), width=0.9, swing_direction="left"
                )
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
            path = f.name

        DXFRenderer().render(model, path)
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        door_entities = [e for e in msp if e.dxf.layer == "A-DOOR"]
        assert len(door_entities) > 0
        os.unlink(path)

    def test_chain_dimensions_with_door(self):
        from modules.geometry.measurement_engine import MeasurementEngine

        engine = MeasurementEngine()
        wall = WallSegment(start=np.array([0, 0]), end=np.array([4, 0]), thickness=0.15)
        door = DoorOpening(position=np.array([2, 0]), width=0.9, swing_direction="left")

        chain = engine.compute_chain_dimensions(wall, [door])
        assert len(chain) == 3, f"Expected 3 chain segments, got {len(chain)}"
        total = sum(c["length"] for c in chain)
        assert abs(total - 4.0) < 0.1, f"Chain total {total} != 4.0"


# ---------- Pipeline Wiring Test ----------


class TestPipelineWiring:
    def test_reconstructor_has_opening_detector(self):
        """RoomReconstructor should initialize with an opening detector."""
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        from modules.room_reconstructor import RoomReconstructor

        r = RoomReconstructor()
        assert r.opening_detector is not None
