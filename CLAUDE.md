# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

AI-powered room reconstruction from 4-5 photos. Primary backend: VGGT (CVPR 2025 Best Paper) produces metric depth, camera poses, focal lengths, and aligned point cloud in a single forward pass. Gemini 3 Flash adds semantic scene analysis (room type, doors, windows) in parallel. Generates 2D floor plans (SVG, DXF, PNG) with measurements and interactive 3D visualizations.

## Commands

```bash
# Install dependencies (uv recommended)
uv pip install -e ../vggt  # VGGT from local clone
uv pip install -r requirements.txt

# Gemini 3 requires Vertex AI env vars
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_CLOUD_PROJECT="adktalentpulse360"

# Run web interface (Gradio, opens at http://localhost:7850)
uv run python app.py

# Run CLI
uv run python run_cli.py sample_images/test_room_*.jpg
uv run python run_cli.py photos/*.jpg --room-width 5.0 --visualize

# No test suite or linting configured yet
```

First run downloads VGGT-1B model (~4GB) from Hugging Face automatically.

## Architecture

**Primary Pipeline:** Photos → VGGT (metric depth + camera poses + point cloud) ‖ Gemini 3 (semantic analysis) → Wall detection → Floor plan + 3D visualization

**Legacy Fallback:** Photos → SfM (COLMAP) → Depth estimation (per image) → Multi-view fusion → Floor plan

**Entry points:**
- `app.py` — Gradio web UI. `process_images()` collects uploaded images, calls `RoomReconstructor.reconstruct_from_arrays()`, returns floor plan image + Plotly figure + measurements markdown.
- `run_cli.py` — CLI. Parses args, calls `RoomReconstructor.reconstruct()` with file paths.

**Core orchestrator:** `modules/room_reconstructor.py` (`RoomReconstructor` class)
- `reconstruct()` / `reconstruct_from_arrays()` — main entry methods
- VGGT path (primary): single forward pass replaces SfM + depth + registration
- Gemini runs in parallel via ThreadPoolExecutor (adds zero latency)
- Legacy fallback: TSDF fusion → SfM alignment → ICP/RANSAC
- Postprocessing: statistical outlier removal + voxel downsampling

**Processing modules (all in `modules/`):**

| Module | Class | Role |
|--------|-------|------|
| `vggt_reconstructor.py` | `VGGTReconstructor` | **Primary.** VGGT-1B single-pass: metric depth + camera poses + focal lengths + aligned point cloud. Replaces SfM + DepthEstimator + registration. |
| `scene_analyzer.py` | `SceneAnalyzer` | Gemini 3 Flash via Vertex AI. Room type, doors, windows, room shape classification. Structured JSON output. |
| `depth_estimator.py` | `DepthEstimator` | **Legacy fallback.** Monocular depth from Depth-Anything-V2. Returns normalized [0,1] depth maps. |
| `sfm_processor.py` | `SfMProcessor` | **Legacy fallback.** COLMAP SfM via pycolmap. |
| `dense_reconstructor.py` | `DenseReconstructor` | TSDF volumetric fusion (used with SfM poses). |
| `floor_plan_generator.py` | `FloorPlanGenerator` | Horizontal slice → 2D density grid → Hough line wall detection → measurements. |
| `visualizer_3d.py` | `Visualizer3D` | Plotly 3D scatter, Open3D viewer, PLY export, Poisson mesh. |

**Data model:** `modules/geometry/floor_plan_model.py` — `FloorPlanModel` with `WallSegment`, `DoorOpening` (door_type, source), `WindowOpening` (sill_height, source), `RoomPolygon` (room_type, room_shape), `DimensionLine`.

**Renderers:** `modules/rendering/` — `SVGRenderer`, `PNGRenderer`, `SymbolLibrary` (including sliding door symbol).

**Configuration:** `config.py` — all parameters including `ENABLE_VGGT`, `VGGT_MODEL`, `ENABLE_GEMINI_ANALYSIS`, `GEMINI_MODEL`.

**Outputs written to `./outputs/`:** floor plan PNG/SVG/DXF, interactive 3D HTML, PLY point cloud, optional PLY mesh.

## Key Design Decisions

- **VGGT as primary backend** — single forward pass produces metric depth + camera poses + aligned 3D points. No ASSUMED_ROOM_WIDTH hack needed.
- **Gemini 3 in parallel** — semantic analysis runs concurrently with reconstruction via ThreadPoolExecutor. Zero additional latency.
- **Graceful degradation** — if VGGT unavailable, falls back to legacy pipeline (SfM + relative depth). If Gemini unavailable, floor plan still generated without semantic labels.
- **FloorPlanModel enrichment** — Gemini-detected doors/windows merged into the model with deduplication (0.5m threshold).
- **Config-driven feature flags** — `ENABLE_VGGT` and `ENABLE_GEMINI_ANALYSIS` toggle both backends independently.

## Known Limitations

- VGGT model is ~4GB, requires GPU for reasonable performance (MPS/CUDA)
- Gemini requires Vertex AI credentials (GOOGLE_CLOUD_PROJECT env var)
- Floor plan grid is 100x100 — coarse resolution for larger rooms
- Convex hull boundary cannot represent non-convex room shapes
- Gemini door/window placement is approximate (placed at 25%/50%/75% along detected walls)
