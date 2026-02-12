# Room Reconstruction Pipeline
## From Photos to Floor Plans

**How 4-12 photos become a 2D floor plan with measurements.**

---

## Pipeline Flow

```
Input Photos (4-12, same room)
        |
1. Load Images (PIL, NumPy)
        |
       +---------+
       |         |
       v         v
2A. VGGT (3D)    2B. Gemini 3 (Semantic)
   - Camera poses      - Room type (kitchen, bedroom...)
   - Metric depth      - Door count + positions
   - 616K+ points      - Window count + positions
   - Focal lengths     - Room shape
   (PyTorch, MPS)      (Vertex AI, parallel)
       |         |
       +---------+
             |
3. Scale Calibration (multi-cue solver, 2x scaling)
             |
4. World Frame Canonicalization (align +Y = up)
             |
       +-----+-----+
       |           |
       v           v
5A. CAGE (Primary)    5B. Hough (Fallback)
   - Dense points        - Postprocessed points
     (616K raw)           (2-3K downsampled)
   - 256x256 density     - Floor plane detection
     map (log+blur)      - Depth gradient edges
   - Learned edges       - Hough line detection
     (Transformer)       - Manhattan alignment
   - Room polygons       - Room segmentation
   (NeurIPS 2025)        (OpenCV, SciPy)
       |           |
       +-----------+
             |
6. Manhattan Alignment (snap to axis-aligned rectangle)
             |
7. Gemini Merge (room type + doors + windows onto walls)
             |
8. Measurement Engine (wall lengths, room area, dimensions)
             |
9. Quality Assessment (closure score, confidence rating)
             |
10. Render Floor Plans
    +-- PNG (Matplotlib) -- architectural drawing
    +-- SVG (svgwrite)   -- vector, AIA layers
    +-- DXF (ezdxf)      -- AutoCAD R2018
             |
11. 3D Visualization (Plotly interactive HTML, PLY export)
             |
OUTPUT: Floor plans + 3D Model + Measurements
```

---

## Step Details

### Step 1: Load Images
**Library:** PIL, NumPy
**What:** Read 4-12 photos of the same room. Convert to RGB arrays. 5 images is optimal for one room.

### Step 2A: VGGT (Primary 3D Reconstruction)
**Library:** VGGT-1B (CVPR 2025 Best Paper), PyTorch
**What:** Single forward pass produces metric depth maps, camera poses, focal lengths, and aligned 3D point cloud (~616K points).
**Device:** MPS (Apple Silicon) or CUDA. Falls back to CPU.
**Key:** No COLMAP/SfM needed. One model does everything.

### Step 2B: Gemini 3 Flash (Semantic Analysis)
**Library:** Google Gemini 3 Flash via Vertex AI
**What:** Runs in parallel with VGGT (zero extra latency). Analyzes room type, door count/positions, window count/positions, room shape.
**Auth:** `GOOGLE_GENAI_USE_VERTEXAI=1`, `GOOGLE_CLOUD_LOCATION="global"`, `GOOGLE_CLOUD_PROJECT` env vars.

### Step 3: Scale Calibration
**Library:** NumPy, custom multi-cue solver
**What:** VGGT produces relative-scale coordinates. The solver applies ~2x scaling using room geometry cues to produce realistic metric dimensions.

### Step 4: World Frame Canonicalization
**Library:** NumPy
**What:** Rotates all points so +Y aligns with gravity (up). Ensures consistent floor plane detection downstream.

### Step 5A: CAGE Learned Detection (Primary)
**Library:** CAGE (NeurIPS 2025), PyTorch, Shapely
**What:** Takes **dense raw points** (616K, before downsampling) and projects them to a 256x256 top-down density map. The density map uses log normalization + Gaussian blur (sigma=1.5) to match Structured3D training data. CAGE's edge-centric Transformer predicts room polygons directly.
**Model:** ResNet-50 backbone, 40.9M params, checkpoint ~493MB.
**Fallback:** If CAGE produces 0 rooms (threshold 0.3), automatically falls back to Hough pipeline (Step 5B).

### Step 5B: Hough Pipeline (Fallback)
**Library:** OpenCV, SciPy, Open3D
**What:** Classical pipeline: floor plane detection (RANSAC) -> depth gradient edges -> Canny -> HoughLinesP -> Manhattan alignment -> room segmentation. Uses postprocessed points (2-3K after denoising + voxel downsampling).

### Step 6: Manhattan Alignment
**Library:** Shapely, NumPy
**What:** Snaps CAGE's organic polygons to axis-aligned rectangles. If the CAGE polygon covers less than 40% of the point cloud extent, uses the point cloud bounding box instead. Produces clean 4-wall rectangular rooms.

### Step 7: Gemini Merge
**Library:** Custom
**What:** Enriches the detected geometry with Gemini's semantic data:
- Sets room type label ("Kitchen", "Bedroom", etc.)
- Places doors on walls (with 0.5m deduplication)
- Places windows on walls
- Assigns wall directions (north/south/east/west)

### Step 8: Measurement Engine
**Library:** NumPy
**What:** Computes per-wall lengths, room area (m² and ft²), bounding box dimensions, chain dimensions for rendering.

### Step 9: Quality Assessment
**What:** Scores the floor plan based on wall closure, point cloud density, number of images. Ratings: APPROXIMATE, DRAFT, NEEDS MORE IMAGES.

### Step 10: Render Floor Plans
**PNG (Matplotlib):** Architectural drawing with walls, door arcs, window symbols, dimension lines (m + ft), room label, area, scale bar, quality badge.
**SVG (svgwrite):** Vector format with AIA layers (A-WALL, A-DOOR, A-GLAZ, A-DIMS, A-AREA, A-ANNO).
**DXF (ezdxf):** AutoCAD R2018 compatible with AIA layers.

### Step 11: 3D Visualization
**Library:** Plotly, Open3D
**What:** Interactive 3D HTML (Plotly scatter), PLY point cloud export, optional Poisson mesh.

---

## Summary Flow

```
Photos --> VGGT (3D points) + Gemini (semantics)  [parallel]
  --> Scale + Canonicalize
  --> CAGE density map (or Hough fallback)
  --> Manhattan alignment
  --> Gemini merge (room type, doors, windows)
  --> Measurements + Quality
  --> PNG/SVG/DXF + 3D HTML
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| **VGGT-1B** | 3D reconstruction (depth, poses, points) |
| **CAGE** | Learned floor plan detection (NeurIPS 2025) |
| **Google Gemini 3** | Room type, doors, windows (Vertex AI) |
| **PyTorch** | Model inference (MPS/CUDA/CPU) |
| **OpenCV** | Image processing, Hough fallback |
| **NumPy** | Math, arrays, coordinate transforms |
| **Open3D** | Point cloud processing, RANSAC |
| **SciPy** | Gaussian blur, scientific computing |
| **Shapely** | Polygon operations, room merging |
| **Matplotlib** | PNG architectural rendering |
| **svgwrite** | SVG floor plan files |
| **ezdxf** | DXF CAD files (AutoCAD R2018) |
| **Plotly** | Interactive 3D visualization |

---

## Key Files

| File | Role |
|------|------|
| `modules/room_reconstructor.py` | Main orchestrator (~3500 lines) |
| `modules/vggt_reconstructor.py` | VGGT-1B single-pass reconstruction |
| `modules/scene_analyzer.py` | Gemini 3 Flash semantic analysis |
| `modules/detection/learned_floorplan_detector.py` | CAGE detection + Manhattan alignment |
| `modules/detection/cage_loader.py` | CAGE model loading (CPU/MPS compatible) |
| `modules/detection/wall_detector.py` | Hough fallback wall detection |
| `modules/geometry/floor_plan_model.py` | Central data model (walls, rooms, doors, windows) |
| `modules/geometry/measurement_engine.py` | Dimension computation |
| `modules/rendering/png_renderer.py` | Matplotlib architectural rendering |
| `modules/rendering/svg_renderer.py` | SVG with AIA layers |
| `modules/rendering/dxf_renderer.py` | AutoCAD DXF export |
| `config.py` | All settings and feature flags |

---

## Output Files (in `./outputs/`)

- `floor_plan_arch_*.png` - Architectural floor plan image
- `floor_plan_*.svg` - Vector floor plan (AIA layers)
- `floor_plan_*.dxf` - AutoCAD CAD file
- `room_3d_*.html` - Interactive 3D visualization
- `room_pointcloud_*.ply` - 3D point cloud
- `room_mesh_*.ply` - Optional Poisson mesh

---

## Performance

- **VGGT model:** ~4GB download (first run only, cached after)
- **CAGE model:** ~493MB checkpoint (ResNet-50 backbone)
- **5 images on MPS:** ~60-90 seconds total
- **12 images on MPS:** ~3-5 minutes (diminishing returns for single room)
- **Gemini:** Runs in parallel (zero extra latency)
- **Recommended:** 4-5 images per room for best speed/quality tradeoff

---

## Accuracy

- **VGGT path:** Metric depth (actual meters), ~2x scale calibration applied
- **CAGE detection:** F1 99.1% on Structured3D benchmark
- **Quality score:** 0-1, with confidence rating (APPROXIMATE / DRAFT)
- **Manhattan alignment:** Ensures clean rectangular output for architectural use
