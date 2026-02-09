# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

AI-powered room reconstruction from 4-5 photos: estimates depth per image, optionally runs COLMAP SfM for camera poses, fuses point clouds, generates 2D floor plans with measurements, and produces interactive 3D visualizations. Proof-of-concept with ~20-30% measurement accuracy.

## Commands

```bash
# Install dependencies (uv recommended)
uv pip install -r requirements.txt

# Run web interface (Gradio, opens at http://localhost:7860)
uv run python app.py

# Run CLI
uv run python run_cli.py sample_images/test_room_*.jpg
uv run python run_cli.py photos/*.jpg --room-width 5.0 --visualize

# No test suite or linting configured yet
```

First run downloads Depth-Anything-V2 model (~350MB-1GB) from Hugging Face automatically.

## Architecture

**Pipeline:** Photos → SfM (optional, COLMAP) → Depth estimation (per image) → 3D point clouds → Multi-view fusion → Floor plan extraction → 3D visualization

**Entry points:**
- `app.py` — Gradio web UI. `process_images()` collects uploaded images, calls `RoomReconstructor.reconstruct_from_arrays()`, returns floor plan image + Plotly figure + measurements markdown.
- `run_cli.py` — CLI. Parses args, calls `RoomReconstructor.reconstruct()` with file paths.

**Core orchestrator:** `modules/room_reconstructor.py` (`RoomReconstructor` class)
- `reconstruct()` / `reconstruct_from_arrays()` — main entry methods
- Fusion strategy selection: TSDF fusion (with SfM poses) → SfM-based alignment → legacy ICP/RANSAC fallback
- Postprocessing: statistical outlier removal + voxel downsampling

**Processing modules (all in `modules/`):**

| Module | Class | Role |
|--------|-------|------|
| `depth_estimator.py` | `DepthEstimator` | Monocular depth from Depth-Anything-V2 (fallback: Intel DPT). Returns normalized [0,1] depth maps. Auto-detects CUDA. |
| `sfm_processor.py` | `SfMProcessor` | COLMAP SfM via pycolmap. SIFT features → exhaustive matching → incremental mapping. Outputs camera poses (4x4 matrices) + intrinsics. |
| `dense_reconstructor.py` | `DenseReconstructor` | TSDF volumetric fusion from multiple depth maps + SfM camera poses. |
| `floor_plan_generator.py` | `FloorPlanGenerator` | Horizontal slice (10-30% height) → 2D density grid (100x100) → Hough line wall detection → convex hull boundary → bounding box measurements. Scales using `ASSUMED_ROOM_WIDTH_METERS`. |
| `visualizer_3d.py` | `Visualizer3D` | Plotly interactive 3D scatter, Open3D desktop viewer, PLY export, optional Poisson mesh. Also contains `ScaleEstimator` utility. |

**Configuration:** `config.py` — single file with all parameters (depth model, SfM settings, floor plan resolution, camera intrinsics, voxel sizes, etc.).

**Outputs written to `./outputs/`:** floor plan PNG, interactive 3D HTML, PLY point cloud, optional PLY mesh.

## Key Design Decisions

- **Relative depth model** (Depth-Anything-V2 outputs [0,1], not metric) — measurements depend on `ASSUMED_ROOM_WIDTH_METERS` user input for scale. This is the primary accuracy bottleneck.
- **Three fusion strategies** with automatic fallback: TSDF (best, needs SfM) → SfM alignment → legacy ICP/RANSAC.
- **pycolmap is optional** — SfM gracefully disabled if not installed; falls back to heuristic registration.
- **Floor plan uses convex hull** — cannot handle L-shaped or U-shaped rooms.
- **Hardcoded camera intrinsics** (fx=fy=500, cx=cy=256) used when SfM is unavailable.

## Known Limitations

- Measurement accuracy ±20-30% (relative depth + assumed room width scaling)
- Floor plan grid is 100x100 — coarse resolution for larger rooms
- No door/window/semantic detection
- Convex hull boundary cannot represent non-convex room shapes
- Raster PNG output only — no vector/CAD export (DXF/SVG)
- `FLOOR_PLAN_REVIEW.md` contains a detailed 10-week improvement roadmap addressing these issues
