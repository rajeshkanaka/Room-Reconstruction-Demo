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
  VGGT single-pass reconstruction &bull; Gemini 3 scene analysis &bull; Metric depth &bull; Wall detection &bull; Door/window recognition &bull; SVG / DXF / PNG output &bull; Interactive 3D
</p>

---

## Technology Evaluation Matrix

The following table evaluates every technology referenced in the [3D Room Reconstruction Deep Research](3D-Room-Reconstruction-DeepResearch.pdf) document against this implementation. Each technology is assessed for inclusion with rationale.

### Multi-View Reconstruction (Primary)

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **VGGT (CVPR 2025 Best Paper)** | **Used** | **Primary reconstruction backend** | Visual Geometry Grounded Transformer. Single forward pass produces metric depth, camera poses, focal lengths, and aligned 3D point cloud from 2-12 images. Replaces the entire SfM + depth estimation + registration pipeline. 1B-parameter model from Meta Research. |
| **Gemini 3 Flash (Vertex AI)** | **Used** | **Semantic scene analysis** | Google's Gemini 3 multimodal model. Analyzes room photos to classify room type, detect doors/windows, and determine room shape. Runs in parallel with VGGT via ThreadPoolExecutor -- adds zero latency to the pipeline. Structured JSON output via response schema. |

### Depth Estimation (Legacy Fallback)

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **Apple Depth Pro** | **Used** | Legacy metric depth model (`apple/DepthPro-hf`) | Produces absolute metric depth (meters) from a single image. 1B-parameter model that also estimates focal length. Used when VGGT is disabled. |
| **Depth-Anything-V2 Metric Indoor** | **Used** | Metric depth fallback model | Strong indoor metric depth when Depth Pro is unavailable. Trained specifically on indoor NYU-Depth datasets. |
| **Depth-Anything-V2 Large** | **Used** | Relative depth backbone | High-resolution relative depth for point cloud generation. Optimal at 518px (multiple of 14 for ViT). Used when metric models are disabled. |
| **Intel DPT-Large** | **Used** | Legacy fallback depth model | Reliable fallback when primary and secondary models fail to load. Well-tested on diverse indoor scenes. |
| **MiDaS** | Not used | -- | Superseded by Depth-Anything-V2, which achieves better accuracy on indoor benchmarks. |

### Structure-from-Motion & Multi-View Stereo (Legacy Fallback)

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **COLMAP (pycolmap)** | **Used** | Legacy SfM pipeline for camera pose estimation | Used when VGGT is unavailable. SIFT features, exhaustive matching, incremental SfM. Produces camera poses and intrinsics for TSDF fusion. |
| **TSDF Fusion** | **Used** | Dense volumetric fusion from SfM poses | Truncated Signed Distance Function fusion via Open3D. Used in legacy path when COLMAP SfM succeeds. |
| **ICP/RANSAC Registration** | **Used** | Fallback multi-view alignment | Legacy alignment when both VGGT and SfM are unavailable. RANSAC for coarse alignment, ICP for refinement. |
| **Meshroom (AliceVision)** | Not used | -- | Requires NVIDIA GPU for dense reconstruction. VGGT supersedes the need for any SfM pipeline. |
| **OpenMVG + OpenMVS** | Not used | -- | Requires building from source. VGGT provides superior results in a single forward pass. |

### Semantic Understanding & Detection

| Technology | Status | Role in This Project | Rationale |
|:-----------|:------:|:---------------------|:----------|
| **Gemini 3 Flash** | **Used** | Room type classification, door/window counting | Multimodal analysis of room photos. Identifies room type (bedroom, kitchen, etc.), room shape, door/window locations and types (sliding, standard, etc.). Results merged into FloorPlanModel. |
| **SegFormer (ADE20K)** | **Used** | Door and window semantic segmentation | NVIDIA's `segformer-b2-finetuned-ade-512-512` detects doors (class 25) and windows (class 8) via pixel-level semantic segmentation. Provides pixel-precise opening boundaries. |
| **Hough Line Detection** | **Used** | Fallback opening detection | Classical CV fallback when SegFormer model is unavailable. Canny edge detection + HoughLinesP identifies vertical line pairs as potential door/window candidates. |
| **Wall Detection (custom)** | **Used** | Depth-based wall boundary extraction | Custom pipeline: depth gradient analysis, Canny edge detection, Hough line transforms, and Manhattan-world alignment. |
| **Detectron2 / Mask R-CNN** | Not used | -- | Instance segmentation is overkill for door/window detection. SegFormer + Gemini already covers the required detection capabilities. |

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
                     PRIMARY PIPELINE (VGGT + Gemini)

  Photos (4-5)                                          Output Formats
  ============                                          ==============

  +-------+      +-----------+      +-----------+       +-- SVG (vector)
  | img1  |----->|   VGGT    |----->|   Wall    |       +-- DXF (CAD)
  | img2  |      | (single   |      | Detection |------>+-- PNG (arch.)
  | img3  |      | forward   |      +-----------+       +-- 3D (Plotly)
  | img4  |      | pass)     |            |             +-- PLY (mesh)
  +-------+      +-----------+      +-----------+       +-- HTML (3D)
       |          Returns:          |   Room    |
       |          - metric depth    | Segmenter |
       |          - camera poses    +-----------+
       |          - focal lengths         |
       |          - point cloud     +-----------+
       |                            |  Opening  |
       |   (parallel)               | Detector  |
       +--------->+-----------+     +-----------+
                  | Gemini 3  |           |
                  | Flash     |     +-----------+
                  | (Vertex)  |     | Measure   |
                  +-----------+     |  Engine   |
                   Returns:         +-----------+
                   - room type            |
                   - doors/windows  +-----------+
                   - room shape     | Renderers |-------> Files
                                    | SVG/DXF/  |
                                    | PNG/3D    |
                                    +-----------+

  LEGACY FALLBACK (if VGGT unavailable):
  Photos --> COLMAP SfM --> Depth Estimation --> TSDF/ICP Fusion --> ...
```

### Primary Path: VGGT Reconstruction

**VGGT** (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) processes all images in a single forward pass, producing metric depth maps, camera poses with estimated focal lengths, and an aligned 3D point cloud. This replaces the entire legacy pipeline (SfM + depth estimation + registration) with a single model.

### Parallel: Gemini Scene Analysis

**Gemini 3 Flash** runs concurrently via ThreadPoolExecutor. It analyzes the room photos to classify room type (bedroom, kitchen, etc.), detect doors and windows with their types (sliding, standard, etc.), and determine room shape. Results are merged into the FloorPlanModel after wall detection.

### Legacy Fallback

When VGGT is unavailable, the system falls back to the original pipeline:
1. **Apple Depth Pro** or **Depth-Anything-V2** for per-image depth estimation
2. **COLMAP SfM** for camera pose estimation
3. **TSDF Fusion** / **SfM Alignment** / **ICP/RANSAC** for point cloud fusion

### Wall Detection & Room Segmentation

A custom pipeline extracts architectural structure from the point cloud:

- **Depth gradient analysis** identifies wall boundaries
- **Canny + HoughLinesP** detects wall line segments
- **Manhattan-world alignment** snaps walls to orthogonal axes
- **Room segmentation** extracts room polygons from wall topology

### Door & Window Detection

Two complementary detection sources:
1. **Gemini 3** -- counts and classifies doors/windows from photos, identifies types (sliding, double, etc.)
2. **SegFormer** (ADE20K-finetuned) -- pixel-level semantic segmentation for precise opening boundaries

Detected openings from both sources are merged into the FloorPlanModel with deduplication (0.5m threshold).

### Measurement & Rendering

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

- **Python 3.11+** (3.11.10 recommended)
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or pip
- 16 GB+ RAM (VGGT-1B is a large model)
- GPU with CUDA or Apple MPS support (strongly recommended for VGGT)
- **Google Cloud project** with Vertex AI enabled (for Gemini scene analysis)

### Install

```bash
git clone <repository-url>
cd missoula

# Install VGGT from local clone
uv pip install -e ../vggt

# Install remaining dependencies
uv pip install -r requirements.txt
```

### Environment Setup (Gemini 3)

Gemini scene analysis requires Vertex AI credentials:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
```

Gemini is optional -- the pipeline works without it, just without semantic labels (room type, door/window classification).

### First Run -- Model Download

On first execution, models are downloaded automatically from Hugging Face:

| Model | Size | Downloaded When |
|:------|:-----|:----------------|
| **VGGT-1B** | ~4 GB | Primary reconstruction (default) |
| Apple Depth Pro | ~1.5 GB | Legacy fallback (VGGT disabled) |
| Depth-Anything-V2 Metric Indoor | ~1.3 GB | Depth Pro fails to load |
| Intel DPT-Large | ~350 MB | All above fail |
| SegFormer (ADE20K) | ~100 MB | Opening detection on first image |

No manual download steps required. Subsequent runs use the Hugging Face cache.

### Run

```bash
# Web interface (recommended)
uv run python app.py
# Open http://localhost:7850

# Command line
uv run python run_cli.py sample_images/test_room_*.jpg
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
# VGGT (primary reconstruction backend)
ENABLE_VGGT = True                       # True = use VGGT, False = legacy pipeline
VGGT_MODEL = "facebook/VGGT-1B"         # CVPR 2025 Best Paper
VGGT_CONFIDENCE_THRESHOLD = 0.5          # Point cloud confidence filter
VGGT_MAX_SIZE = 518                      # Input image max dimension

# Gemini scene analysis
ENABLE_GEMINI_ANALYSIS = True            # True = Gemini semantic analysis
GEMINI_MODEL = "gemini-3-flash-preview"          # Vertex AI model
GEMINI_TIMEOUT = 30                      # Seconds to wait for Gemini

# Legacy depth estimation (used when VGGT disabled)
ENABLE_METRIC_DEPTH = True               # True = metric, False = relative
METRIC_DEPTH_MODEL = "apple/DepthPro-hf" # Primary metric model

# Legacy SfM (used when VGGT disabled)
ENABLE_SFM = True                        # COLMAP SfM for multi-view alignment
SFM_MIN_IMAGES = 3                       # Minimum images for SfM

# 3D reconstruction
POINT_CLOUD_DENSITY = 4                  # Sample every Nth pixel
VOXEL_SIZE = 0.05                        # Voxel downsampling (meters)

# Floor plan
ASSUMED_ROOM_WIDTH_METERS = 4.0          # Default room width (legacy only)
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
|   |-- room_reconstructor.py       # Main orchestrator (VGGT + Gemini + legacy)
|   |-- vggt_reconstructor.py       # VGGT-1B single-pass reconstruction (NEW)
|   |-- scene_analyzer.py           # Gemini 3 scene analysis (NEW)
|   |-- depth_estimator.py          # Depth-Anything-V2 / DPT (legacy fallback)
|   |-- sfm_processor.py            # COLMAP SfM (legacy fallback)
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
|   |   |-- floor_plan_model.py     # Data model (walls, rooms, openings, semantics)
|   |   |-- measurement_engine.py   # Per-wall lengths, areas, chain dims
|   |
|   |-- rendering/
|       |-- svg_renderer.py         # SVG vector floor plan (sliding doors)
|       |-- dxf_renderer.py         # DXF/CAD export (AIA layers)
|       |-- png_renderer.py         # Matplotlib architectural PNG (sliding doors)
|       |-- symbol_library.py       # Door arcs, sliding doors, windows, scale bar
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
                   |                    |                       |
          +--------v--------+  +-------v--------+     +--------v---------+
          |  PRIMARY PATH   |  |   PARALLEL     |     | LEGACY FALLBACK  |
          |                 |  |                 |     |                  |
   +------v------+         |  +-------v--------+|    +------v------+  +------v------+
   |   VGGT      |         |  | Gemini 3 Flash ||    | MetricDepth |  |  SfM        |
   | Reconstructor         |  | (Vertex AI)    ||    | Estimator   |  |  Processor  |
   | - metric depth        |  | - room type    ||    | (Depth Pro) |  |  (COLMAP)   |
   | - camera poses        |  | - doors/windows||    +-------------+  +-------------+
   | - focal lengths       |  | - room shape   ||          |                 |
   | - point cloud         |  +----------------+|    +-----v-----+  +-------v-------+
   +-------------+         |         |           |    |  Depth    |  |  TSDF/ICP     |
          |                |         |           |    | Calibrate |  |  Fusion       |
          v                |         v           |    +-----------+  +---------------+
   +-------------+         |  +-------------+    |
   | Wall        |<--------+  | FloorPlan   |    |
   | Detector    |             | Model       |<---+
   +-------------+             | (enriched)  |
          |                    +-------------+
   +------v------+                   |
   | Room        |            +------v------+
   | Segmenter   |            | SVG / DXF / |
   +-------------+            | PNG Render  |
          |                   +-------------+
   +------v------+
   | Measurement |
   | Engine      |
   +-------------+
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
| **VGGT** | 0.0.1 | Visual Geometry Grounded Transformer (local clone) |
| **google-genai** | 1.0+ | Gemini 3 Flash via Vertex AI |
| **PyTorch** | 2.0+ | Deep learning inference runtime |
| **Transformers** | 4.35+ | Hugging Face model loading (depth, segmentation) |
| **einops** | 0.8+ | Tensor operations (VGGT dependency) |
| **Open3D** | 0.17+ | Point cloud processing, TSDF, ICP, filtering |
| **pycolmap** | 0.6+ | COLMAP SfM (legacy fallback) |
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

Install VGGT separately: `uv pip install -e ../vggt`, then `uv pip install -r requirements.txt`.

---

## Troubleshooting

### VGGT Out of Memory

VGGT-1B requires significant GPU memory. If you run out of memory:

```python
# In config.py -- disable VGGT to use legacy pipeline:
ENABLE_VGGT = False

# Or reduce image size:
VGGT_MAX_SIZE = 364         # Down from 518 (must be multiple of 14)
```

### Gemini Not Working

```bash
# Verify environment variables are set:
echo $GOOGLE_GENAI_USE_VERTEXAI    # Should be "1"
echo $GOOGLE_CLOUD_LOCATION        # Should be "global"
echo $GOOGLE_CLOUD_PROJECT          # Your GCP project ID

# Or disable Gemini (pipeline still works without it):
# In config.py: ENABLE_GEMINI_ANALYSIS = False
```

### CUDA Out of Memory (Legacy Pipeline)

```python
# In config.py -- reduce sizes:
DEPTH_MAX_SIZE = 384        # Down from 518
POINT_CLOUD_DENSITY = 6     # Up from 4 (fewer points)
```

### Model Download Fails

Models download from Hugging Face on first run. If downloads fail:

```bash
# Set token for rate limits:
export HF_TOKEN=your_token_here

# Pre-download the fallback model:
uv run python -c "
from transformers import DPTImageProcessor, DPTForDepthEstimation
DPTImageProcessor.from_pretrained('Intel/dpt-large')
DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
"
```

### pycolmap Crashes (macOS)

pycolmap's native library may crash with `SIGABRT` on some macOS configurations. When VGGT is enabled (default), COLMAP is not used. If using the legacy pipeline, SfM is automatically disabled and falls back to ICP/RANSAC registration.

### Poor Reconstruction Quality

- Ensure photos cover all corners with overlap
- Use even lighting (avoid harsh shadows)
- Include the floor in every shot
- With VGGT, measurements are metric -- no room width calibration needed
- With legacy pipeline, try adjusting `ASSUMED_ROOM_WIDTH_METERS`
- Use the calibration panel with a known wall measurement

---

## Accuracy & Limitations

### Current Accuracy

| Metric | Value | Notes |
|:-------|:------|:------|
| Measurement accuracy (VGGT) | Metric depth -- model-dependent | VGGT produces metric depth; no ASSUMED_ROOM_WIDTH hack needed |
| Measurement accuracy (legacy) | Approximately +/- 15-25% | With metric depth; +/- 20-30% with relative depth |
| Room shape support | Rectangular rooms | L-shaped and irregular rooms: partial support |
| Wall detection | 2+ walls per image | Requires visible depth discontinuities |
| Opening detection | Doors and windows | SegFormer + Gemini dual-source detection |
| Semantic classification | Room type, door types | Via Gemini 3 Flash (confidence-scored) |

### Known Limitations

- VGGT model is ~4 GB and requires GPU (CUDA or MPS) for reasonable inference speed
- Gemini requires Google Cloud credentials and Vertex AI access
- Floor plan grid resolution is 100x100 -- coarse for large rooms
- Convex hull boundary cannot fully represent non-convex room shapes
- Gemini door/window placement is approximate (positioned at 25%/50%/75% along walls)
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
| **Phase 6: VGGT Integration** | **Complete** | **VGGT-1B single-pass reconstruction replaces SfM + depth + registration** |
| **Phase 7: Gemini Integration** | **Complete** | **Gemini 3 Flash semantic analysis (room type, doors, windows) in parallel** |
| Phase 8: NeRF / Gaussian Splatting | Planned | Photorealistic visualization layer |
| Phase 9: Furniture Detection | Planned | Object detection and placement in floor plans |
| Phase 10: Multi-Room Support | Planned | Connected room topology and navigation |

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the detailed 37-step breakdown and [FLOOR_PLAN_REVIEW.md](FLOOR_PLAN_REVIEW.md) for the 10-week improvement roadmap.

---

## Acknowledgments

- **Meta Research** -- VGGT (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper)
- **Google** -- Gemini 3 Flash multimodal model via Vertex AI
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
