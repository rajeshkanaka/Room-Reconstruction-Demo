<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/tests-61%20passed-brightgreen?style=flat-square" alt="Tests">
</p>

<h1 align="center">Room Reconstruction from Photos</h1>

<p align="center">
  <strong>AI-powered room reconstruction: transform 4-5 room photos into architectural floor plans and interactive 3D models.</strong>
</p>

<p align="center">
  Metric depth estimation &bull; Wall detection &bull; Door/window recognition &bull; SVG / DXF / PNG output &bull; Interactive 3D
</p>

---

## Technology Evaluation Matrix

The following table evaluates every technology referenced in the [3D Room Reconstruction Deep Research](3D-Room-Reconstruction-DeepResearch.pdf) document against this implementation. Each technology is assessed for inclusion with rationale.

### Depth Estimation

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **Apple Depth Pro** | **Used** | Primary metric depth model (`apple/DepthPro-hf`) | Produces absolute metric depth (meters) from a single image. 1B-parameter model that also estimates focal length, eliminating need for camera calibration. Best-in-class accuracy for indoor scenes. |
| **Depth-Anything-V2 Metric Indoor** | **Used** | Metric depth fallback model | Strong indoor metric depth when Depth Pro is unavailable. Trained specifically on indoor NYU-Depth datasets. Automatic fallback mechanism ensures robustness. |
| **Depth-Anything-V2 Large** | **Used** | Relative depth backbone | High-resolution relative depth for point cloud generation. Optimal at 518px (multiple of 14 for ViT). Used when metric models are disabled. |
| **Intel DPT-Large** | **Used** | Legacy fallback depth model | Reliable fallback when primary and secondary models fail to load. Well-tested on diverse indoor scenes. Ensures the system always produces output. |
| **MiDaS** | Not used | -- | Superseded by Depth-Anything-V2, which achieves better accuracy on indoor benchmarks. MiDaS produces only relative depth with arbitrary scale, requiring manual calibration. |

### Structure-from-Motion & Multi-View Stereo

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **COLMAP (pycolmap)** | **Used** | SfM pipeline for camera pose estimation | State-of-the-art photogrammetry toolkit. SIFT features, exhaustive matching, incremental SfM. Produces camera poses and intrinsics essential for TSDF fusion. Robust and proven. |
| **TSDF Fusion** | **Used** | Dense volumetric fusion from SfM poses | Truncated Signed Distance Function fusion via Open3D. Produces clean, consistent point clouds when SfM poses are available. Primary fusion strategy. |
| **ICP/RANSAC Registration** | **Used** | Fallback multi-view alignment | Legacy alignment when SfM is unavailable. RANSAC for coarse alignment, ICP for refinement. Ensures multi-view fusion even without COLMAP. |
| **Meshroom (AliceVision)** | Not used | -- | Requires NVIDIA GPU for dense reconstruction. Heavier dependency footprint than pycolmap. The pycolmap approach provides equivalent SfM quality with lighter integration and cross-platform support. |
| **OpenMVG + OpenMVS** | Not used | -- | Requires building from source. COLMAP via pycolmap provides the same capability with simpler pip-based installation. No added benefit for this pipeline's scope. |

### Semantic Understanding & Detection

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **SegFormer (ADE20K)** | **Used** | Door and window semantic segmentation | NVIDIA's `segformer-b2-finetuned-ade-512-512` detects doors (class 25) and windows (class 8) via pixel-level semantic segmentation. Lightweight transformer architecture suitable for real-time inference. |
| **Hough Line Detection** | **Used** | Fallback opening detection | Classical CV fallback when SegFormer model is unavailable. Canny edge detection + HoughLinesP identifies vertical line pairs as potential door/window candidates. Zero-dependency fallback. |
| **Wall Detection (custom)** | **Used** | Depth-based wall boundary extraction | Custom pipeline: depth gradient analysis, Canny edge detection, Hough line transforms, and Manhattan-world alignment. Purpose-built for floor plan wall extraction. |
| **Detectron2 / Mask R-CNN** | Not used | -- | Instance segmentation is overkill for door/window detection. SegFormer's semantic segmentation is lighter and sufficient for identifying opening regions on walls. Detectron2 adds heavy dependencies (Detectron2 + COCO weights). |
| **MIT Scene Parse** | Not used | -- | Interesting for full scene understanding but not required. SegFormer on ADE20K already covers the classes needed (doors, windows, walls). Adding another model increases latency without proportional benefit. |

### 3D Processing & Visualization

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **Open3D** | **Used** | Point cloud processing, TSDF, ICP, filtering | Industry-standard library for 3D data. Used for TSDF volumetric fusion, ICP registration, statistical outlier removal, voxel downsampling, and Poisson meshing. Core infrastructure. |
| **Plotly** | **Used** | Interactive 3D model viewer | Produces browser-based interactive 3D scatter plots. Users can rotate, zoom, pan. Exports to standalone HTML files for sharing without any software installation. |
| **Matplotlib** | **Used** | Architectural PNG floor plan rendering | Clean, print-quality floor plan rendering with dimension annotations, door arcs, window symbols. Produces publication-ready PNG output. |
| **MeshLab / CloudCompare** | Not used | -- | Desktop GUI tools for viewing 3D data. Useful for debugging but not suitable as programmatic pipeline components. Open3D covers all required 3D processing programmatically. |

### Floor Plan Rendering & CAD Export

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **svgwrite** | **Used** | Vector SVG floor plan generation | Produces scalable vector floor plans with AIA-compliant layers, dimension lines, scale bars, north arrows, door arcs, and window symbols. Zoomable, print-quality output. |
| **ezdxf** | **Used** | DXF/CAD floor plan export | Generates AutoCAD-compatible DXF files with standard AIA layers (A-WALL, A-DOOR, A-GLAZ, A-DIMS, A-AREA, A-ANNO). Opens in AutoCAD, FreeCAD, BricsCAD, and any DWG/DXF viewer. |
| **FloorNet** | Not used | -- | Research paper for floor plan extraction from point clouds. Interesting but no maintained pip-installable implementation. Our custom wall detection + room segmentation pipeline achieves the required functionality. |

### Commercial & Advanced Alternatives

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **Instant NeRF (Instant-NGP)** | Not used | -- | Produces photorealistic novel-view synthesis but does not directly output geometry or measurements. Requires NVIDIA GPU with CUDA. Measurement extraction from radiance fields adds significant complexity. Considered for future visual enhancement. |
| **3D Gaussian Splatting** | Not used | -- | State-of-the-art for real-time rendering quality. Similar limitations to NeRF: primarily a visualization technique, not a measurement tool. Point extraction is possible but adds pipeline complexity. Future roadmap candidate. |
| **Matterport** | Not used | -- | Cloud-based commercial solution requiring specialized hardware (panoramic cameras). Excellent quality but outside the "local, open-source, phone-only" scope. Considered as a production-grade alternative for enterprise deployment. |
| **Kaarta** | Not used | -- | Hardware LiDAR+SLAM solution (Stencil 2). Real-time scanning but requires proprietary device. Not suitable for a phone-photo-only workflow. |
| **Pix4D** | Not used | -- | Commercial photogrammetry platform focused on drone mapping. Overkill for indoor room reconstruction. License-based pricing model. |
| **Agisoft Metashape** | Not used | -- | Professional photogrammetry software. Excellent quality but requires commercial license. Python scripting available but the full pipeline is proprietary. |
| **Polycam** | Not used | -- | Mobile app using ARKit + photogrammetry. iOS-only. Produces good results with LiDAR-equipped iPhones but not cross-platform and not self-hosted. |

---

## How It Works

```
                        RECONSTRUCTION PIPELINE

  Photos (4-5)                                          Output Formats
  ============                                          ==============

  +-------+      +----------+      +-----------+       +-- SVG (vector)
  | img1  |----->|  Metric  |----->|   Wall    |       +-- DXF (CAD)
  | img2  |      |  Depth   |      | Detection |------>+-- PNG (arch.)
  | img3  |      | Estimate |      +-----------+       +-- 3D (Plotly)
  | img4  |      +----------+            |             +-- PLY (mesh)
  +-------+           |            +-----------+       +-- HTML (3D)
       |               |           |   Room    |
       |          +---------+      | Segmenter |
       +--------->| COLMAP  |      +-----------+
                  |   SfM   |            |
                  +---------+      +-----------+
                       |           |  Opening  |
                  +---------+      | Detector  |
                  |  TSDF   |      | (SegForm) |
                  | Fusion  |      +-----------+
                  +---------+            |
                       |           +-----------+
                  +---------+      | Measure   |
                  | Filter  |      |  Engine   |
                  | + Clean |      +-----------+
                  +---------+            |
                       |           +-----------+
                       +---------->| Renderers |-------> Files
                                   | SVG/DXF/  |
                                   | PNG/3D    |
                                   +-----------+
```

### Stage 1: Metric Depth Estimation

Each photo is processed by **Apple Depth Pro** (primary) or **Depth-Anything-V2 Metric Indoor** (fallback) to produce absolute depth maps in meters. Unlike relative depth models, metric depth preserves real-world scale, enabling accurate room measurements without manual calibration.

### Stage 2: Structure-from-Motion (Optional)

When 3+ images are provided, **COLMAP SfM** (via pycolmap) estimates camera poses through SIFT feature extraction, exhaustive matching, and incremental mapping. These poses enable geometrically consistent multi-view fusion.

### Stage 3: Point Cloud Fusion

Depth maps are back-projected to 3D using the pinhole camera model. Three fusion strategies are attempted in order:

1. **TSDF Fusion** (best) -- volumetric fusion using SfM camera poses via Open3D
2. **SfM-Based Alignment** -- direct transformation using SfM pose matrices
3. **Legacy ICP/RANSAC** -- pairwise registration when SfM is unavailable

Post-processing applies statistical outlier removal and voxel downsampling.

### Stage 4: Wall Detection & Room Segmentation

A custom pipeline extracts architectural structure from the point cloud:

- **Depth gradient analysis** identifies wall boundaries
- **Canny + HoughLinesP** detects wall line segments
- **Manhattan-world alignment** snaps walls to orthogonal axes
- **Room segmentation** extracts room polygons from wall topology

### Stage 5: Door & Window Detection

**SegFormer** (ADE20K-finetuned) performs semantic segmentation to identify door (class 25) and window (class 8) regions. Detected bounding boxes are projected from 2D image space to 3D floor plan coordinates using depth-based back-projection. A Hough-line fallback handles cases where the model is unavailable.

### Stage 6: Measurement & Rendering

The **MeasurementEngine** computes per-wall lengths, room areas, and chain dimensions (wall-to-opening-to-wall). Three renderers produce output:

| Renderer | Library | Output | Use Case |
|:---------|:--------|:-------|:---------|
| **SVGRenderer** | svgwrite | `.svg` | Scalable vector floor plan, print-quality, web embedding |
| **DXFRenderer** | ezdxf | `.dxf` | AutoCAD / FreeCAD / BricsCAD import, AIA layer standard |
| **PNGRenderer** | matplotlib | `.png` | Quick architectural preview, reports, presentations |

All renderers produce door arcs, window symbols, dimension lines, scale bars, and north arrows via a shared `SymbolLibrary`.

---

## Quick Start

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or pip
- 8 GB+ RAM (16 GB recommended for metric depth models)
- GPU with CUDA support (optional, significantly accelerates inference)

### Install

```bash
git clone <repository-url>
cd missoula

# Option A: uv (recommended)
uv sync
# or: uv pip install -r requirements.txt

# Option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### First Run -- Model Download

On first execution, depth models are downloaded automatically from Hugging Face:

| Model | Size | Downloaded When |
|:------|:-----|:----------------|
| Apple Depth Pro | ~1.5 GB | Metric depth enabled (default) |
| Depth-Anything-V2 Metric Indoor | ~1.3 GB | Depth Pro fails to load |
| Depth-Anything-V2 Large | ~1.3 GB | Metric depth disabled |
| Intel DPT-Large | ~350 MB | All above fail |
| SegFormer (ADE20K) | ~100 MB | Opening detection on first image |

No manual download steps required. Subsequent runs use the Hugging Face cache.

### Run

```bash
# Web interface (recommended)
uv run python app.py
# Open http://localhost:7860

# Command line
uv run python run_cli.py sample_images/img*.jpeg --room-width 4.0
uv run python run_cli.py path/to/photos/*.jpg --room-width 5.0 --visualize
```

### Run Tests

```bash
uv run python -m pytest tests/ -v
```

| Test Suite | Tests | Coverage |
|:-----------|------:|:---------|
| `test_phase1_metric_depth.py` | 8 | Metric depth estimation, calibration |
| `test_phase2_wall_detection.py` | 9 | Wall detection, room segmentation, measurement |
| `test_phase3_rendering.py` | 24 | SVG, DXF, PNG renderers, symbol library |
| `test_phase4_openings.py` | 14 | Opening detection, projection, rendering integration |
| `test_e2e.py` | 6 | Full pipeline lifecycle, all module imports |
| **Total** | **61** | |

---

## Photography Guide

### Recommended Technique

Capture **4-5 photos** from different corners of the room. Each photo should include the floor and at least two walls.

```
  Corner Positions                    What Each Shot Captures
  =================                   =======================

  +-------------------+
  | A >>            B |               A: Floor + North wall + West wall
  |      >>     >>    |               B: Floor + North wall + East wall
  |          X        |               C: Floor + South wall + East wall
  |      <<     <<    |               D: Floor + South wall + West wall
  | D              C  |               X: Floor + all walls (optional center shot)
  +-------------------+

  >> = camera direction from corner
```

### Best Practices

| Do | Avoid |
|:---|:------|
| Stand in corners, aim diagonally across | Shooting from the same position/angle |
| Include floor in every shot | Cropping out the floor |
| Use consistent, even lighting | Mixed lighting / very dark scenes |
| Hold camera level (no tilt) | Extreme wide-angle distortion |
| Keep scene static (no people/pets moving) | Motion blur / reflective surfaces |

---

## Output Formats

### Floor Plan Outputs

| Format | Description | Software |
|:-------|:------------|:---------|
| **SVG** | Scalable vector floor plan with layers | Any browser, Inkscape, Illustrator |
| **DXF** | CAD-standard with AIA layers | AutoCAD, FreeCAD, BricsCAD, LibreCAD |
| **PNG** | Architectural-style raster image | Any image viewer |

### DXF Layer Structure (AIA Standard)

| Layer | Color | Content |
|:------|:------|:--------|
| `A-WALL` | White | Wall centerlines |
| `A-WALL-FILL` | Gray | Wall fill polygons |
| `A-DOOR` | Green | Door arcs and swing lines |
| `A-GLAZ` | Cyan | Window symbols |
| `A-DIMS` | Red | Dimension lines and annotations |
| `A-AREA` | Blue | Room area labels |
| `A-ANNO` | White | Scale bar, north arrow, title |

### 3D & Data Outputs

| Format | Description |
|:-------|:------------|
| **Interactive 3D** | Plotly scatter plot (rotate, zoom, pan in browser) |
| **HTML** | Standalone 3D viewer file (shareable, no install needed) |
| **PLY** | Point cloud for Open3D, MeshLab, CloudCompare |

All outputs are written to the `outputs/` directory with timestamps.

---

## Configuration

All parameters are in `config.py`:

```python
# Depth estimation
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DEPTH_MAX_SIZE = 518                     # ViT-optimal (multiple of 14)

# Metric depth (Phase 1)
ENABLE_METRIC_DEPTH = True               # True = metric, False = relative
METRIC_DEPTH_MODEL = "apple/DepthPro-hf" # Primary metric model
CALIBRATION_METHOD = "auto"              # auto | user_reference | none

# SfM
ENABLE_SFM = True                        # COLMAP SfM for multi-view alignment
SFM_MIN_IMAGES = 3                       # Minimum images for SfM

# 3D reconstruction
POINT_CLOUD_DENSITY = 4                  # Sample every Nth pixel
VOXEL_SIZE = 0.05                        # Voxel downsampling (meters)
DEPTH_FUSION_METHOD = "tsdf"             # tsdf | poisson

# Floor plan
ASSUMED_ROOM_WIDTH_METERS = 4.0          # Default room width for scale
FLOOR_PLAN_RESOLUTION = 100              # Grid resolution
```

---

## Project Structure

```
missoula/
|
|-- app.py                          # Gradio web UI (entry point)
|-- run_cli.py                      # CLI entry point
|-- config.py                       # All configurable parameters
|-- requirements.txt                # Python dependencies
|
|-- modules/
|   |-- room_reconstructor.py       # Main orchestrator
|   |-- depth_estimator.py          # Depth-Anything-V2 / DPT inference
|   |-- sfm_processor.py            # COLMAP SfM (pycolmap)
|   |-- dense_reconstructor.py      # TSDF volumetric fusion
|   |-- floor_plan_generator.py     # Legacy floor plan extraction
|   |-- visualizer_3d.py            # Plotly 3D + PLY + HTML export
|   |
|   |-- depth/
|   |   |-- metric_depth.py         # Apple Depth Pro / DA-V2 Metric
|   |   |-- depth_calibrator.py     # Cross-model calibration
|   |
|   |-- detection/
|   |   |-- wall_detector.py        # Depth gradient + Hough wall detection
|   |   |-- room_segmenter.py       # Room polygon extraction
|   |   |-- opening_detector.py     # SegFormer door/window detection
|   |
|   |-- geometry/
|   |   |-- floor_plan_model.py     # Data model (walls, rooms, openings)
|   |   |-- measurement_engine.py   # Per-wall lengths, areas, chain dims
|   |
|   |-- rendering/
|       |-- svg_renderer.py         # SVG vector floor plan
|       |-- dxf_renderer.py         # DXF/CAD export (AIA layers)
|       |-- png_renderer.py         # Matplotlib architectural PNG
|       |-- symbol_library.py       # Door arcs, window lines, scale bar
|
|-- tests/
|   |-- test_phase1_metric_depth.py
|   |-- test_phase2_wall_detection.py
|   |-- test_phase3_rendering.py
|   |-- test_phase4_openings.py
|   |-- test_e2e.py
|
|-- sample_images/                  # 9 sample room photos
|-- outputs/                        # Generated floor plans, 3D models
|-- ARCHITECTURE.md                 # Detailed architecture documentation
|-- IMPLEMENTATION_PLAN.md          # 37-step implementation roadmap
|-- FLOOR_PLAN_REVIEW.md            # Improvement roadmap
```

---

## Architecture Overview

```
                   +-------------------------------------------+
                   |           RoomReconstructor                |
                   |         (Main Orchestrator)                |
                   +-------------------------------------------+
                   |                                           |
          +--------+--------+                        +---------+---------+
          |                 |                        |                   |
   +------v------+  +------v------+          +------v------+   +--------v-------+
   | MetricDepth |  |  SfM        |          |  Wall       |   | Opening        |
   | Estimator   |  |  Processor  |          | Detector    |   | Detector       |
   | (Depth Pro) |  |  (COLMAP)   |          | (Hough)     |   | (SegFormer)    |
   +-------------+  +-------------+          +-------------+   +----------------+
          |                 |                        |                   |
          v                 v                        v                   v
   +-------------+  +-------------+          +-------------+   +----------------+
   |  Depth      |  |  Dense      |          |  Room       |   | FloorPlan      |
   | Calibrator  |  | Reconstructor          | Segmenter   |   | Model          |
   +-------------+  |  (TSDF)     |          +-------------+   | (data classes) |
                    +-------------+                 |          +----------------+
                                                    v                   |
                                             +-------------+            v
                                             | Measurement |    +----------------+
                                             | Engine      |    | SVG / DXF /    |
                                             +-------------+    | PNG Renderers  |
                                                                +----------------+
```

For full architecture details including Mermaid diagrams and sequence flows, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Web Interface

The Gradio-based web interface provides:

| Tab | Content |
|:----|:--------|
| **Floor Plan (Legacy)** | Heatmap-style floor plan from point cloud density |
| **SVG Floor Plan** | Interactive vector floor plan with embedded viewer + download |
| **DXF (CAD)** | Download button for AutoCAD-compatible DXF |
| **PNG (Architectural)** | Clean architectural rendering with dimensions |
| **3D Model** | Interactive Plotly 3D scatter plot |
| **Measurements** | Per-wall dimensions, room areas, pipeline metadata |

### Calibration

The **Calibrate Measurements** panel allows post-hoc correction using a known wall dimension. Select a reference wall, enter its real length, and all measurements are re-scaled proportionally.

---

## Dependencies

| Library | Version | Purpose |
|:--------|:--------|:--------|
| **PyTorch** | 2.0+ | Deep learning inference runtime |
| **Transformers** | 4.35+ | Hugging Face model loading (depth, segmentation) |
| **timm** | 0.9+ | Vision model architectures |
| **Open3D** | 0.17+ | Point cloud processing, TSDF, ICP, filtering |
| **pycolmap** | 0.6+ | COLMAP SfM (feature extraction, matching, mapping) |
| **OpenCV** | 4.8+ | Image processing, edge detection, Hough transforms |
| **svgwrite** | 1.4+ | SVG floor plan generation |
| **ezdxf** | 1.0+ | DXF/CAD file generation |
| **Plotly** | 5.15+ | Interactive 3D visualization |
| **Matplotlib** | 3.7+ | Architectural PNG rendering |
| **Gradio** | 4.0+ | Web interface |
| **scikit-image** | 0.21+ | Image analysis, morphology |
| **SciPy** | 1.10+ | Spatial processing, interpolation |
| **Shapely** | 2.0+ | Computational geometry |
| **trimesh** | 4.0+ | Mesh utilities |

Install all with `uv sync` or `pip install -r requirements.txt`.

---

## Troubleshooting

### CUDA Out of Memory

```python
# In config.py -- reduce sizes:
DEPTH_MAX_SIZE = 384        # Down from 518
POINT_CLOUD_DENSITY = 6     # Up from 4 (fewer points)
SFM_MAX_IMAGE_SIZE = 512    # Down from 1024
```

Or force CPU:
```python
# In modules/depth_estimator.py:
self.device = "cpu"
```

### Model Download Fails

Models download from Hugging Face on first run. If downloads fail:

```bash
# Pre-download the fallback model:
uv run python -c "
from transformers import DPTImageProcessor, DPTForDepthEstimation
DPTImageProcessor.from_pretrained('Intel/dpt-large')
DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
"

# Set token for rate limits:
export HF_TOKEN=your_token_here
```

### pycolmap Crashes (macOS)

pycolmap's native library may crash with `SIGABRT` on some macOS configurations. The system handles this gracefully -- SfM is automatically disabled and the pipeline falls back to ICP/RANSAC registration. No action required.

### Poor Reconstruction Quality

- Ensure photos cover all corners with overlap
- Use even lighting (avoid harsh shadows)
- Include the floor in every shot
- Try adjusting `ASSUMED_ROOM_WIDTH_METERS` to match the actual room
- Use the calibration panel with a known wall measurement

---

## Accuracy & Limitations

### Current Accuracy

| Metric | Value | Notes |
|:-------|:------|:------|
| Measurement accuracy | Approximately +/- 15-25% | With metric depth; +/- 20-30% with relative depth |
| Room shape support | Rectangular rooms | L-shaped and irregular rooms: partial support |
| Wall detection | 2+ walls per image | Requires visible depth discontinuities |
| Opening detection | Doors and windows | Confidence threshold filtering applied |

### Known Limitations

- Floor plan grid resolution is 100x100 -- coarse for large rooms
- Convex hull boundary cannot fully represent non-convex room shapes
- Low-texture surfaces (white walls, uniform carpet) challenge feature matching
- Metric depth models are large (1-1.5 GB each) and require significant RAM
- Scale accuracy depends on depth model quality; not suitable for construction or legal purposes

### What This System Cannot Do

- Provide sub-centimeter precision measurements
- Work with a single photo (minimum 2, recommended 4-5)
- Handle outdoor scenes or very large commercial spaces
- Detect furniture, appliances, or fixtures
- Replace professional surveying equipment

---

## Roadmap

| Phase | Status | Description |
|:------|:------:|:------------|
| Phase 1: Metric Depth | Complete | Apple Depth Pro + DA-V2 Metric, depth calibrator |
| Phase 2: Wall Detection | Complete | Hough-based wall detection, room segmentation, measurement engine |
| Phase 3: Architectural Rendering | Complete | SVG, DXF, PNG renderers with symbol library |
| Phase 4: Opening Detection | Complete | SegFormer door/window detection with projection |
| Phase 5: Integration & UI | Complete | Gradio UI with format tabs, calibration, measurements |
| Phase 6: NeRF / Gaussian Splatting | Planned | Photorealistic visualization layer |
| Phase 7: Furniture Detection | Planned | Object detection and placement in floor plans |
| Phase 8: Multi-Room Support | Planned | Connected room topology and navigation |

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the detailed 37-step breakdown and [FLOOR_PLAN_REVIEW.md](FLOOR_PLAN_REVIEW.md) for the 10-week improvement roadmap.

---

## Acknowledgments

- **Apple** -- Depth Pro metric depth estimation model
- **Hugging Face** -- Model hosting and Transformers library
- **NVIDIA** -- SegFormer semantic segmentation architecture
- **ETH Zurich / COLMAP** -- Structure-from-Motion pipeline
- **Open3D** -- 3D point cloud processing toolkit
- **Depth-Anything Team** -- Depth-Anything-V2 depth estimation

---

<p align="center">
  <sub>Built for the US home renovation and property inspection market.</sub><br>
  <sub>Proof-of-concept. Not suitable for construction, legal, or professional surveying purposes.</sub>
</p>
