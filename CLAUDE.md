# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

AI-powered room reconstruction from 4-5 photos. Primary backend: VGGT (CVPR 2025 Best Paper) produces metric depth, camera poses, focal lengths, and aligned point cloud in a single forward pass. Gemini 3 Flash adds semantic scene analysis (room type, doors, windows) in parallel. Generates 2D floor plans (SVG, DXF, PNG) with measurements and interactive 3D visualizations.

## Commands

```bash
# Install dependencies (uv recommended)
uv pip install -e ../vggt          # VGGT from local clone
uv pip install -r requirements.txt

# Gemini 3 requires Vertex AI env vars
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_CLOUD_PROJECT="adktalentpulse360"

# Run web interface (Gradio, opens at http://localhost:7850)
uv run python app.py

# Run CLI
uv run python run_cli.py sample_images/*.jpeg --room-width 5.0 --visualize

# Tests (61 tests across 7 suites, pytest)
uv run pytest                                    # All tests (excludes @slow by default)
uv run pytest tests/test_phase3_rendering.py     # Single test file
uv run pytest tests/test_e2e.py::TestFullPipeline::test_floor_plan_model_full_lifecycle  # Single test
uv run pytest -m slow                            # Only slow/integration tests (require models + sample images)
uv run pytest -m "not slow"                      # Fast tests only (synthetic data, no model loading)
```

First run downloads VGGT-1B model (~4GB) from Hugging Face automatically. No linting or formatting tools are configured.

## Architecture

**Primary Pipeline:** Photos -> VGGT (metric depth + camera poses + point cloud) || Gemini 3 (semantic analysis) -> Wall detection -> Floor plan + 3D visualization

**Legacy Fallback:** Photos -> SfM (COLMAP) -> Depth estimation (per image) -> Multi-view fusion -> Floor plan

**Entry points:**
- `app.py` -- Gradio web UI on port 7850. `process_images()` -> `RoomReconstructor.reconstruct_from_arrays()` -> returns floor plan + Plotly figure + measurements markdown.
- `run_cli.py` -- CLI. Supports glob patterns, `--room-width`, `--visualize` (Open3D viewer), `--output-dir`.

**Core orchestrator:** `modules/room_reconstructor.py` (`RoomReconstructor` class, ~3500 lines)
- `reconstruct()` / `reconstruct_from_arrays()` -- main entry methods
- VGGT path (primary): single forward pass replaces SfM + depth + registration
- Gemini runs in parallel via `ThreadPoolExecutor` (adds zero latency)
- Legacy fallback chain: SfM+TSDF -> SfM+alignment -> ICP/RANSAC
- Postprocessing: statistical outlier removal + voxel downsampling (Open3D)

**Processing modules (all in `modules/`):**

| Module | Class | Role |
|--------|-------|------|
| `vggt_reconstructor.py` | `VGGTReconstructor` | **Primary.** VGGT-1B single-pass: metric depth + camera poses + focal lengths + aligned point cloud. |
| `scene_analyzer.py` | `SceneAnalyzer` | Gemini 3 Flash via Vertex AI. Room type, doors, windows, room shape. Structured JSON output. |
| `depth_estimator.py` | `DepthEstimator` | **Legacy.** Monocular relative depth from Depth-Anything-V2. Normalized [0,1]. |
| `depth/metric_depth.py` | `MetricDepthEstimator` | **Legacy.** Metric depth in meters from Apple DepthPro or DA-V2 Metric Indoor. |
| `depth/depth_calibrator.py` | `DepthCalibrator` | Cross-model depth calibration and scale alignment. |
| `sfm_processor.py` | `SfMProcessor` | **Legacy.** COLMAP SfM via pycolmap. |
| `dense_reconstructor.py` | `DenseReconstructor` | TSDF volumetric fusion with SfM poses (Open3D). |
| `detection/wall_detector.py` | `WallDetector` | Depth gradient -> Canny -> HoughLinesP -> Manhattan alignment. |
| `detection/room_segmenter.py` | `RoomSegmenter` | Wall topology graph -> closed polygon extraction -> room shape classification. |
| `detection/opening_detector.py` | `OpeningDetector` | SegFormer (ADE20K, door=class 25, window=class 8) + Hough fallback. Projects openings onto walls. |
| `geometry/floor_plan_model.py` | `FloorPlanModel` | Central data model: `WallSegment`, `DoorOpening` (door_type, source), `WindowOpening` (sill_height, source), `RoomPolygon` (room_type, room_shape), `DimensionLine`. |
| `geometry/measurement_engine.py` | `MeasurementEngine` | Per-wall lengths, room areas, chain dimensions, bounding box. |
| `rendering/svg_renderer.py` | `SVGRenderer` | SVG with AIA layers (A-WALL, A-DOOR, A-GLAZ, A-DIMS, A-AREA, A-ANNO). |
| `rendering/dxf_renderer.py` | `DXFRenderer` | AutoCAD-compatible DXF R2018 with AIA layers. |
| `rendering/png_renderer.py` | `PNGRenderer` | Matplotlib architectural rendering. |
| `rendering/symbol_library.py` | `SymbolLibrary` | Shared door arcs, sliding doors, windows, scale bar, north arrow. |
| `visualizer_3d.py` | `Visualizer3D` | Plotly 3D scatter, Open3D viewer, PLY export, Poisson mesh, standalone HTML. |
| `floor_plan_generator.py` | `FloorPlanGenerator` | **Legacy.** Horizontal slice -> 2D density grid -> Hough wall detection. Superseded by modular detection pipeline. |

**Configuration:** `config.py` -- all parameters in one file. Key feature flags: `ENABLE_VGGT`, `ENABLE_GEMINI_ANALYSIS`, `ENABLE_METRIC_DEPTH`, `ENABLE_SFM`, `ENABLE_MVS`.

**Outputs written to `./outputs/`:** floor plan PNG/SVG/DXF, interactive 3D HTML, PLY point cloud, optional PLY mesh. All filenames include timestamps.

## Key Design Decisions

- **VGGT as primary backend** -- single forward pass produces metric depth + camera poses + aligned 3D points. No `ASSUMED_ROOM_WIDTH` hack needed.
- **Gemini 3 in parallel** -- semantic analysis runs concurrently with reconstruction via `ThreadPoolExecutor`. Zero additional latency.
- **Graceful degradation** -- if VGGT unavailable, falls back to legacy pipeline (SfM + relative depth). If Gemini unavailable, floor plan still generated without semantic labels. Each module catches its own exceptions and returns `{"success": False}` dicts.
- **FloorPlanModel as contract** -- detection populates the model, renderers consume it. Adding a new output format only requires a new renderer.
- **Gemini door/window enrichment** -- merged into FloorPlanModel with 0.5m deduplication threshold against detected openings.
- **Config-driven feature flags** -- `ENABLE_VGGT` and `ENABLE_GEMINI_ANALYSIS` toggle both backends independently.
- **Lazy model loading** -- all ML models loaded on first use, not at import time. Prevents startup crashes and reduces memory for unused backends.
- **Colored logging** -- uses `termcolor.colored` with `[MODULE]` prefixes throughout (e.g., `[VGGT]`, `[Gemini]`, `[WallDetect]`).

## Code Patterns

**Imports:** standard library -> third-party -> local (`from config import ...` / `from modules...`). Group with blank lines.

**Error handling:** try/except returning `{"success": bool, ...}` dicts. Print warnings with `termcolor.colored(..., "yellow")`. Never crash the pipeline -- degrade gracefully.

**Backend switching:**
```python
if ENABLE_VGGT:
    from modules.vggt_reconstructor import VGGTReconstructor
else:
    # Legacy path
```

**Data structures:** `@dataclass` for all structured types. `FloorPlanModel` is the central contract between detection and rendering.

**Tests:** Use synthetic data (numpy arrays, manually constructed `FloorPlanModel` instances) to avoid model loading. Mark slow/integration tests with `@pytest.mark.slow`. Test files in `tests/` use `sys.path.insert` for imports.

## Test Suites

| Suite | Count | What it covers |
|-------|-------|----------------|
| `test_phase1_metric_depth.py` | 8 | Metric depth estimation, calibration, focal length |
| `test_phase2_wall_detection.py` | 9 | Wall detection, room segmentation, measurement engine |
| `test_phase3_rendering.py` | 24 | SVG, DXF, PNG renderers, symbol library |
| `test_phase4_openings.py` | 14 | Opening detection, projection, rendering integration |
| `test_phase5_quality_scoring.py` | 6 | Quality assessment modes, export policies |
| `test_phase9_regression_pack.py` | 1 | Regression test against 9-sample-image baseline (`tests/fixtures/t9_sample_room_baseline.json`) |
| `test_e2e.py` | 6 | Full pipeline lifecycle, all module imports |

## Known Limitations

- VGGT model is ~4GB, requires GPU for reasonable performance (MPS/CUDA)
- Gemini requires Vertex AI credentials (`GOOGLE_CLOUD_PROJECT` env var)
- Floor plan grid is 100x100 -- coarse resolution for larger rooms
- Convex hull boundary cannot represent non-convex room shapes
- Gemini door/window placement is approximate (placed at 25%/50%/75% along detected walls)
- pycolmap may crash with SIGABRT on some macOS configs (auto-disabled when VGGT enabled)
- `room_reconstructor.py` is ~3500 lines -- the largest module, handles all pipeline orchestration
