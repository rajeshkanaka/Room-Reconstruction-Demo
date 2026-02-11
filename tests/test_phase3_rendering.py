"""
Phase 3 Integration Tests: Architectural Rendering

Tests SVG, DXF, and PNG renderers with FloorPlanModel.
Verifies output file creation, layer structure, and content.
"""

import pytest
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.geometry.floor_plan_model import (
    FloorPlanModel,
    WallSegment,
    RoomPolygon,
    DimensionLine,
    DoorOpening,
    WindowOpening,
)
from modules.rendering.svg_renderer import SVGRenderer
from modules.rendering.dxf_renderer import DXFRenderer
from modules.rendering.png_renderer import PNGRenderer
from modules.rendering.symbol_library import SymbolLibrary


# ---------- Fixtures ----------


@pytest.fixture
def simple_model():
    """A simple rectangular room with 4 walls."""
    walls = [
        WallSegment(start=np.array([0, 0]), end=np.array([4, 0]), thickness=0.15),
        WallSegment(start=np.array([4, 0]), end=np.array([4, 3]), thickness=0.15),
        WallSegment(start=np.array([4, 3]), end=np.array([0, 3]), thickness=0.15),
        WallSegment(start=np.array([0, 3]), end=np.array([0, 0]), thickness=0.15),
    ]
    room = RoomPolygon(
        boundary=np.array([[0, 0], [4, 0], [4, 3], [0, 3]]), name="Living Room"
    )
    return FloorPlanModel(walls=walls, rooms=[room])


@pytest.fixture
def full_model():
    """A room with walls, door, window, and dimensions."""
    walls = [
        WallSegment(start=np.array([0, 0]), end=np.array([5, 0]), thickness=0.15),
        WallSegment(start=np.array([5, 0]), end=np.array([5, 4]), thickness=0.15),
        WallSegment(start=np.array([5, 4]), end=np.array([0, 4]), thickness=0.15),
        WallSegment(start=np.array([0, 4]), end=np.array([0, 0]), thickness=0.15),
    ]
    room = RoomPolygon(
        boundary=np.array([[0, 0], [5, 0], [5, 4], [0, 4]]), name="Bedroom"
    )
    dims = [
        DimensionLine(
            start=np.array([0, 0]), end=np.array([5, 0]), value_m=5.0, offset=0.5
        ),
        DimensionLine(
            start=np.array([5, 0]), end=np.array([5, 4]), value_m=4.0, offset=0.5
        ),
    ]
    door = DoorOpening(position=np.array([2.5, 0]), width=0.9, swing_direction="left")
    window = WindowOpening(position=np.array([2.5, 4]), width=1.2)
    return FloorPlanModel(
        walls=walls, rooms=[room], dimensions=dims, doors=[door], windows=[window]
    )


@pytest.fixture
def output_dir():
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------- SVG Renderer Tests ----------


class TestSVGRenderer:
    def test_render_creates_file(self, simple_model, output_dir):
        renderer = SVGRenderer()
        path = os.path.join(output_dir, "test.svg")
        result = renderer.render(simple_model, path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 100

    def test_render_to_string(self, simple_model):
        renderer = SVGRenderer()
        svg = renderer.render_to_string(simple_model)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_svg_contains_room_label(self, simple_model):
        renderer = SVGRenderer()
        svg = renderer.render_to_string(simple_model)
        assert "Living Room" in svg

    def test_svg_contains_scale_elements(self, simple_model):
        renderer = SVGRenderer()
        svg = renderer.render_to_string(simple_model)
        assert "1m" in svg  # Scale bar
        assert "N" in svg  # North arrow

    def test_svg_with_full_model(self, full_model, output_dir):
        renderer = SVGRenderer()
        path = os.path.join(output_dir, "full.svg")
        result = renderer.render(full_model, path)
        assert os.path.exists(result)

        svg = renderer.render_to_string(full_model)
        assert "Bedroom" in svg
        assert "path" in svg.lower()  # Door arc

    def test_svg_dimension_values(self, full_model):
        renderer = SVGRenderer()
        svg = renderer.render_to_string(full_model)
        assert "5.00m" in svg
        assert "4.00m" in svg

    def test_svg_empty_model(self):
        renderer = SVGRenderer()
        model = FloorPlanModel()
        svg = renderer.render_to_string(model)
        assert "<svg" in svg  # Should return valid but empty SVG

    def test_svg_quality_banner_for_approximate_export(self, simple_model):
        renderer = SVGRenderer()
        simple_model.quality_mode = "approximate"
        simple_model.export_policy = "annotate_as_approximate"
        svg = renderer.render_to_string(simple_model)
        assert "QUALITY: APPROXIMATE - NOT FOR CONSTRUCTION" in svg


# ---------- DXF Renderer Tests ----------


class TestDXFRenderer:
    def test_render_creates_file(self, simple_model, output_dir):
        renderer = DXFRenderer()
        path = os.path.join(output_dir, "test.dxf")
        result = renderer.render(simple_model, path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 100

    def test_dxf_layer_structure(self, simple_model, output_dir):
        import ezdxf

        renderer = DXFRenderer()
        path = os.path.join(output_dir, "layers.dxf")
        renderer.render(simple_model, path)

        doc = ezdxf.readfile(path)
        layers = [l.dxf.name for l in doc.layers]
        assert "A-WALL" in layers
        assert "A-WALL-FILL" in layers
        assert "A-AREA" in layers
        assert "A-DIMS" in layers
        assert "A-ANNO" in layers

    def test_dxf_with_openings(self, full_model, output_dir):
        import ezdxf

        renderer = DXFRenderer()
        path = os.path.join(output_dir, "openings.dxf")
        renderer.render(full_model, path)

        doc = ezdxf.readfile(path)
        layers = [l.dxf.name for l in doc.layers]
        assert "A-DOOR" in layers
        assert "A-GLAZ" in layers

    def test_dxf_entities_on_correct_layers(self, full_model, output_dir):
        import ezdxf

        renderer = DXFRenderer()
        path = os.path.join(output_dir, "entities.dxf")
        renderer.render(full_model, path)

        doc = ezdxf.readfile(path)
        msp = doc.modelspace()

        wall_entities = [e for e in msp if e.dxf.layer == "A-WALL"]
        dim_entities = [e for e in msp if e.dxf.layer == "A-DIMS"]
        door_entities = [e for e in msp if e.dxf.layer == "A-DOOR"]
        window_entities = [e for e in msp if e.dxf.layer == "A-GLAZ"]

        assert len(wall_entities) > 0, "No wall entities on A-WALL layer"
        assert len(dim_entities) > 0, "No dimension entities on A-DIMS layer"
        assert len(door_entities) > 0, "No door entities on A-DOOR layer"
        assert len(window_entities) > 0, "No window entities on A-GLAZ layer"

    def test_dxf_quality_banner_for_approximate_export(self, simple_model, output_dir):
        import ezdxf

        renderer = DXFRenderer()
        simple_model.quality_mode = "approximate"
        simple_model.export_policy = "annotate_as_approximate"
        path = os.path.join(output_dir, "quality_banner.dxf")
        renderer.render(simple_model, path)

        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        anno_texts = [
            str(getattr(e.dxf, "text", ""))
            for e in msp
            if e.dxftype() == "TEXT" and e.dxf.layer == "A-ANNO"
        ]
        joined = " | ".join(anno_texts)
        assert "QUALITY: APPROXIMATE - NOT FOR CONSTRUCTION" in joined


# ---------- PNG Renderer Tests ----------


class TestPNGRenderer:
    def test_render_returns_figure(self, simple_model):
        import matplotlib.pyplot as plt

        renderer = PNGRenderer()
        fig = renderer.render(simple_model)
        assert fig is not None
        plt.close(fig)

    def test_render_to_image_creates_file(self, simple_model, output_dir):
        renderer = PNGRenderer()
        path = os.path.join(output_dir, "test.png")
        result = renderer.render_to_image(simple_model, path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 1000

    def test_png_with_full_model(self, full_model, output_dir):
        renderer = PNGRenderer()
        path = os.path.join(output_dir, "full.png")
        result = renderer.render_to_image(full_model, path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 1000

    def test_png_empty_model(self):
        import matplotlib.pyplot as plt

        renderer = PNGRenderer()
        model = FloorPlanModel()
        fig = renderer.render(model)
        assert fig is not None
        plt.close(fig)

    def test_png_quality_banner_for_approximate_export(self, simple_model):
        import matplotlib.pyplot as plt

        renderer = PNGRenderer()
        simple_model.quality_mode = "approximate"
        simple_model.export_policy = "annotate_as_approximate"
        fig = renderer.render(simple_model)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any(
            "QUALITY: APPROXIMATE - NOT FOR CONSTRUCTION" in txt for txt in texts
        )
        plt.close(fig)


# ---------- Symbol Library Tests ----------


class TestSymbolLibrary:
    def test_door_symbol_left_swing(self):
        lib = SymbolLibrary()
        door = lib.door_symbol(
            position=np.array([2, 0]),
            width=0.9,
            wall_direction=np.array([1, 0]),
            swing="left",
        )
        assert "arc" in door
        assert "gap" in door
        assert door["swing"] == "left"
        assert door["width"] == 0.9

    def test_door_symbol_right_swing(self):
        lib = SymbolLibrary()
        door = lib.door_symbol(
            position=np.array([2, 0]),
            width=0.9,
            wall_direction=np.array([1, 0]),
            swing="right",
        )
        assert door["swing"] == "right"

    def test_window_symbol(self):
        lib = SymbolLibrary()
        window = lib.window_symbol(
            position=np.array([1.5, 3]),
            width=1.2,
            wall_direction=np.array([1, 0]),
        )
        assert "lines" in window
        assert len(window["lines"]) == 3  # Triple parallel lines

    def test_dimension_tick(self):
        lib = SymbolLibrary()
        tick = lib.dimension_tick(
            position=np.array([0, 0]),
            direction=np.array([1, 0]),
        )
        assert "line" in tick

    def test_scale_bar(self):
        lib = SymbolLibrary()
        bar = lib.scale_bar(
            origin=np.array([0, 0]),
            direction=np.array([1, 0]),
            total_length=2.0,
            divisions=4,
        )
        assert "lines" in bar
        assert "fills" in bar
        assert "labels" in bar
        assert len(bar["labels"]) == 5  # 0, 0.5, 1.0, 1.5, 2.0

    def test_north_arrow(self):
        lib = SymbolLibrary()
        arrow = lib.north_arrow(
            center=np.array([0, 0]),
            size=0.5,
        )
        assert "polygon" in arrow
        assert "label" in arrow
        assert arrow["label"]["text"] == "N"


# ---------- Cross-Renderer Consistency Tests ----------


class TestRendererConsistency:
    def test_all_renderers_handle_same_model(self, full_model, output_dir):
        """All three renderers should successfully render the same model."""
        import matplotlib.pyplot as plt

        svg_path = os.path.join(output_dir, "test.svg")
        dxf_path = os.path.join(output_dir, "test.dxf")
        png_path = os.path.join(output_dir, "test.png")

        SVGRenderer().render(full_model, svg_path)
        DXFRenderer().render(full_model, dxf_path)
        fig = PNGRenderer().render(full_model, output_path=png_path)
        plt.close(fig)

        assert os.path.exists(svg_path)
        assert os.path.exists(dxf_path)
        assert os.path.exists(png_path)

    def test_renderers_handle_no_openings(self, simple_model, output_dir):
        """Renderers should work when model has no doors/windows."""
        import matplotlib.pyplot as plt

        svg_path = os.path.join(output_dir, "simple.svg")
        dxf_path = os.path.join(output_dir, "simple.dxf")
        png_path = os.path.join(output_dir, "simple.png")

        SVGRenderer().render(simple_model, svg_path)
        DXFRenderer().render(simple_model, dxf_path)
        fig = PNGRenderer().render(simple_model, output_path=png_path)
        plt.close(fig)

        assert os.path.exists(svg_path)
        assert os.path.exists(dxf_path)
        assert os.path.exists(png_path)

    def test_renderers_handle_minimal_model(self, output_dir):
        """Single wall, no room."""
        import matplotlib.pyplot as plt

        model = FloorPlanModel(
            walls=[
                WallSegment(
                    start=np.array([0, 0]), end=np.array([3, 0]), thickness=0.15
                )
            ]
        )

        svg_path = os.path.join(output_dir, "minimal.svg")
        dxf_path = os.path.join(output_dir, "minimal.dxf")
        png_path = os.path.join(output_dir, "minimal.png")

        SVGRenderer().render(model, svg_path)
        DXFRenderer().render(model, dxf_path)
        fig = PNGRenderer().render(model, output_path=png_path)
        plt.close(fig)

        assert os.path.exists(svg_path)
        assert os.path.exists(dxf_path)
        assert os.path.exists(png_path)
