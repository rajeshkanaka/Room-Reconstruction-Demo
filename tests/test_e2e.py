"""
End-to-End Integration Tests

Tests the full pipeline: images -> depth -> point cloud -> wall detection ->
room polygon -> measurement -> rendering (all formats).
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestFullPipeline:
    """Test the complete reconstruction pipeline with synthetic data."""

    def test_floor_plan_model_full_lifecycle(self):
        """Build a model, measure it, render it in all formats."""
        from modules.geometry.floor_plan_model import (
            FloorPlanModel,
            WallSegment,
            RoomPolygon,
            DimensionLine,
            DoorOpening,
            WindowOpening,
        )
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.rendering.svg_renderer import SVGRenderer
        from modules.rendering.dxf_renderer import DXFRenderer
        from modules.rendering.png_renderer import PNGRenderer
        import tempfile
        import matplotlib.pyplot as plt

        # 1. Build model
        walls = [
            WallSegment(start=np.array([0, 0]), end=np.array([5, 0]), thickness=0.15),
            WallSegment(start=np.array([5, 0]), end=np.array([5, 4]), thickness=0.15),
            WallSegment(start=np.array([5, 4]), end=np.array([0, 4]), thickness=0.15),
            WallSegment(start=np.array([0, 4]), end=np.array([0, 0]), thickness=0.15),
        ]
        room = RoomPolygon(
            boundary=np.array([[0, 0], [5, 0], [5, 4], [0, 4]]), name="Master Bedroom"
        )
        door = DoorOpening(
            position=np.array([2.5, 0]), width=0.9, swing_direction="left"
        )
        window = WindowOpening(position=np.array([2.5, 4]), width=1.2)

        model = FloorPlanModel(
            walls=walls, rooms=[room], doors=[door], windows=[window]
        )

        # 2. Compute measurements
        engine = MeasurementEngine()
        measurements = engine.compute_measurements(model)

        assert len(measurements["wall_lengths"]) == 4
        assert abs(measurements["wall_lengths"][0] - 5.0) < 0.1
        assert abs(measurements["wall_lengths"][1] - 4.0) < 0.1
        assert len(model.dimensions) > 0  # Dimensions populated

        # 3. Render SVG
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name
        SVGRenderer().render(model, svg_path)
        assert os.path.exists(svg_path)
        assert os.path.getsize(svg_path) > 100

        # 4. Render DXF
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
            dxf_path = f.name
        DXFRenderer().render(model, dxf_path)
        assert os.path.exists(dxf_path)

        # 5. Render PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name
        fig = PNGRenderer().render(model, output_path=png_path)
        plt.close(fig)
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 1000

        # 6. Verify SVG content
        with open(svg_path) as f:
            svg_content = f.read()
        assert "Master Bedroom" in svg_content
        assert "5.00m" in svg_content
        assert "path" in svg_content.lower()  # Door arc

        # Cleanup
        for p in [svg_path, dxf_path, png_path]:
            os.unlink(p)

    def test_wall_detection_to_rendering_pipeline(self):
        """Test: synthetic depth -> wall detection -> model -> rendering."""
        from modules.detection.wall_detector import WallDetector
        from modules.detection.room_segmenter import RoomSegmenter
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.geometry.floor_plan_model import FloorPlanModel
        from modules.rendering.svg_renderer import SVGRenderer
        import tempfile

        # 1. Synthetic depth map with clear wall boundaries
        depth = np.ones((480, 640), dtype=np.float32) * 3.0
        depth[:, :160] = 2.0  # Left wall
        depth[:, 480:] = 2.0  # Right wall
        depth[:120, :] = 4.0  # Back wall

        # 2. Detect walls
        detector = WallDetector()
        segments = detector.detect_walls_from_depth(depth, fx=500, fy=500)
        assert len(segments) >= 2, f"Expected >= 2 wall segments, got {len(segments)}"

        # 3. Align to Manhattan grid
        aligned = detector.align_walls_manhattan(segments)
        assert len(aligned) >= 2

        # 4. Extract rooms
        segmenter = RoomSegmenter()
        rooms = segmenter.extract_rooms(aligned)

        # 5. Build model
        wall_segments = segmenter.segments_to_wall_segments(aligned)
        model = FloorPlanModel(walls=wall_segments, rooms=rooms)

        # 6. Measurements
        engine = MeasurementEngine()
        measurements = engine.compute_measurements(model)
        assert len(measurements["wall_lengths"]) > 0

        # 7. Render SVG
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name
        SVGRenderer().render(model, svg_path)
        assert os.path.exists(svg_path)
        assert os.path.getsize(svg_path) > 100
        os.unlink(svg_path)

    def test_opening_detector_integration(self):
        """Test opening detection with synthetic data."""
        from modules.detection.opening_detector import OpeningDetector
        from modules.geometry.floor_plan_model import WallSegment, DoorOpening

        det = OpeningDetector()

        # Create a simple test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = det.detect_openings(img)

        assert isinstance(result, dict)
        assert "doors" in result
        assert "windows" in result

        # Test projection with known values
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
        assert isinstance(opening, DoorOpening)
        assert opening.width > 0

    def test_gradio_interface_creates(self):
        """Test that the Gradio interface creates without errors."""
        from app import create_demo_interface

        demo = create_demo_interface()
        assert demo is not None

    def test_all_module_imports(self):
        """Verify all modules import cleanly."""
        from modules.depth.metric_depth import MetricDepthEstimator
        from modules.depth.depth_calibrator import DepthCalibrator
        from modules.detection.wall_detector import WallDetector
        from modules.detection.room_segmenter import RoomSegmenter
        from modules.detection.opening_detector import OpeningDetector
        from modules.geometry.floor_plan_model import FloorPlanModel
        from modules.geometry.measurement_engine import MeasurementEngine
        from modules.rendering.svg_renderer import SVGRenderer
        from modules.rendering.dxf_renderer import DXFRenderer
        from modules.rendering.png_renderer import PNGRenderer
        from modules.rendering.symbol_library import SymbolLibrary

        assert FloorPlanModel is not None
        assert SVGRenderer is not None
        assert DXFRenderer is not None
        assert PNGRenderer is not None
        assert WallDetector is not None
        assert OpeningDetector is not None

    def test_reconstructor_initialization(self):
        """Test that RoomReconstructor initializes all components."""
        from modules.room_reconstructor import RoomReconstructor

        r = RoomReconstructor()
        assert r.wall_detector is not None
        assert r.room_segmenter is not None
        assert r.measurement_engine is not None
        assert r.opening_detector is not None
        assert r.floor_plan_gen is not None
        assert r.depth_estimator is not None
