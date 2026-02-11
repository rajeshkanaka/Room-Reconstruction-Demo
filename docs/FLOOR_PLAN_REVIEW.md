# Floor Plan System: Technical Review & Industry-Standard Upgrade Plan

**Date:** 2026-02-09
**Scope:** Review of current 2D floor plan pipeline + plan to reach industry-standard output
**Goal:** Accurate measurements and 2D floor plans usable for renovation/reconstruction projects

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Analysis](#2-current-system-analysis)
3. [Industry Standard Requirements](#3-industry-standard-requirements)
4. [Gap Analysis: Current vs Required](#4-gap-analysis-current-vs-required)
5. [Improvement Plan](#5-improvement-plan)
6. [Architecture Changes](#6-architecture-changes)
7. [Implementation Phases](#7-implementation-phases)
8. [References](#8-references)

---

## 1. Executive Summary

The current floor plan output is a proof-of-concept matplotlib heatmap with ~20-30% measurement error. It cannot be used for any professional purpose. To reach industry-standard output suitable for renovation projects, three fundamental problems must be solved:

| Problem | Current State | Required State |
|---------|--------------|----------------|
| **Measurement accuracy** | +/-20-30%, based on assumed room width | +/-1-5%, metric depth or calibrated reference |
| **Drawing quality** | Matplotlib heatmap, raster PNG | Architectural vector drawing (SVG/DXF) |
| **Structural detection** | Density blob + convex hull | Walls, doors, windows, wall thickness |

The system needs changes at every layer: depth model, scale calibration, wall detection, and rendering.

---

## 2. Current System Analysis

### 2.1 Measurement Pipeline (How dimensions are computed today)

```
Photos -> Depth-Anything V2 (RELATIVE depth) -> Inverse depth -> Point cloud
       -> Top-down projection -> 2D histogram -> "assumed_width" scaling -> Dimensions
```

**Critical flaw:** The entire measurement chain depends on a single user-supplied guess (`assumed_room_width`). The depth model (Depth-Anything V2) produces relative depth only -- values between 0 and 1 with no physical units. The system then applies inverse-depth (`z = 1.0 / (depth + 1e-3) * 0.5`) which gives arbitrary-unit 3D points.

Scale is recovered by:
```python
# floor_plan_generator.py:136-137
room_width_units = x_max - x_min
self.scale_factor = self.assumed_width / room_width_units
```

This means: the X-axis spread of the point cloud is forced to equal whatever the user typed. Every other measurement (depth, area) is derived from this single assumption. If the user guesses wrong, everything is wrong proportionally.

### 2.2 Floor Plan Generation Pipeline

```
Point cloud -> Height slice (10-30% of Y range) -> Center + project to XZ plane
           -> 2D histogram (100x100 grid) -> Morphological cleanup
           -> Hough line detection -> Convex hull boundary -> Matplotlib render
```

**Problems at each step:**

| Step | Code Location | Problem |
|------|--------------|---------|
| Height slice | `floor_plan_generator.py:185-209` | Fixed 10-30% band. Doesn't adapt to actual floor geometry. Mixes floor and furniture points. |
| Grid resolution | `config.py:40` | 100x100 grid. For a 5m room, each cell = 5cm. Far too coarse for wall detection. |
| Density map | `floor_plan_generator.py:122-128` | Raw histogram counting. No distinction between wall points, furniture, floor. |
| Occupancy | `floor_plan_generator.py:211-242` | Binary threshold + morphology. Treats everything (walls + furniture + clutter) as one blob. |
| Wall detection | `floor_plan_generator.py:244-297` | Hough lines on binary blob edges. Detects blob boundary, not actual walls. Cannot find wall thickness or wall-wall junctions. |
| Room boundary | `floor_plan_generator.py:299-343` | Convex hull or simplified contour. Cannot handle L-shaped, U-shaped, or any non-convex room. Produces a single polygon. |
| Scale | `floor_plan_generator.py:136-137` | Single assumed width. No validation. No reference objects. |
| Dimensions | `floor_plan_generator.py:141-145` | Bounding box dimensions only. Cannot measure individual walls, alcoves, or recesses. |

### 2.3 Rendering Pipeline

```
Matplotlib figure (14x7 inches) -> Two panels:
  Left:  Density heatmap + occupancy overlay + wall edges + boundary polygon
  Right: Schematic rectangle with dimension arrows
-> Saved as 150 DPI PNG
```

**Problems:**

| Issue | Detail |
|-------|--------|
| Two-panel layout | Left panel is a debug visualization (heatmap), not a floor plan. Right panel is a fixed-scale schematic that doesn't represent actual room shape. |
| No wall thickness | Walls drawn as single-pixel lines. Industry standard requires showing wall thickness (typically 10-30cm). |
| No doors/windows | Not detected, not drawn. |
| No architectural symbols | No door swings, no window marks, no hatching. |
| Raster output only | PNG at 150 DPI. Cannot be imported into CAD software. Not scalable. |
| No scale bar | Only text annotation of total dimensions. No graphical scale reference. |
| No north arrow | No orientation indicator. |
| Dimension lines | Basic matplotlib arrows. Not proper architectural dimension lines with tick marks, extension lines, or standards-compliant formatting. |
| Font/weight | Default matplotlib fonts. No line weight hierarchy. |
| Room labels | Only total area in center. No room name, no per-wall dimensions. |
| Coordinate system | Pixel coordinates on axes. Should show metric grid. |

### 2.4 Depth Model Limitations

**Depth-Anything V2** (`depth_estimator.py:105-113`):
- Produces **relative** depth (affine-invariant). Output has unknown scale and shift.
- Excellent edge quality and structural understanding.
- NOT suitable for direct metric measurement without external calibration.

The depth-to-3D conversion (`depth_estimator.py:259-310`) uses:
- Hardcoded focal lengths (`fx=500, fy=500`) when SfM intrinsics aren't available.
- Inverse depth mapping: `z = 1.0 / (depth + 1e-3) * DEPTH_SCALE`
- Principal point assumed at image center.

When SfM succeeds (pycolmap), real camera intrinsics are used. But the depth values themselves remain relative -- SfM only fixes the scale of camera poses, not depth map scale.

---

## 3. Industry Standard Requirements

### 3.1 ANSI Z765-2021 (Residential Measurement Standard)

The ANSI Z765 standard governs residential floor area measurement in the US:

- Measurements to the **nearest tenth of a foot** (3cm precision)
- Area calculated to the **nearest whole square foot**
- Based on **exterior dimensions** of walls
- Separate floor plan per level
- Minimum ceiling height requirements (7ft standard, 5ft under slope)
- Mandated by Fannie Mae, HUD, Freddie Mac, VA, FHA

**Accuracy implication:** To report to 0.1 ft precision, the system needs measurement accuracy of at least +/-5cm (2 inches). Current system has +/-60cm to +/-1m error.

### 3.2 Architectural Drawing Standards

Per AIA/ANSI/NCS conventions for renovation floor plans:

**Line weights (3-tier hierarchy):**
- **Heavy (0.6-1.0mm):** Cut walls (exterior and interior structural)
- **Medium (0.3-0.5mm):** Doors, windows, stairs, fixtures
- **Light (0.1-0.2mm):** Dimensions, annotations, hatching, furniture

**Required elements:**
- Wall lines with thickness (exterior walls: 15-30cm; interior: 10-15cm)
- Door symbols with swing arcs (90-degree arc showing opening direction)
- Window symbols (parallel lines within wall rectangles)
- Dimension lines with extension lines and tick marks
- Room labels with name and area
- Scale bar (e.g., 1:50 or 1:100)
- North arrow or orientation indicator
- Title block with project info, date, scale
- Hatching for cut materials (concrete, wood framing, etc.)

**Output formats:**
- DXF (AutoCAD Drawing Exchange Format) -- universal CAD exchange
- SVG (Scalable Vector Graphics) -- web-compatible vector
- PDF (with vector content, not rasterized)

### 3.3 Industry Tool Benchmarks

| Tool | Accuracy | Output | Method |
|------|----------|--------|--------|
| Matterport | +/-1% | DXF, BIM, PDF | LiDAR + vision |
| CubiCasa | +/-1-3% | SVG, DXF, PDF | AI from single panorama |
| magicplan | +/-2-5% | DXF, PDF, PNG | LiDAR/camera scan |
| RoomSketcher | Manual | SVG, DXF, PDF | AI-convert from blueprints |
| iGUIDE | +/-0.5% | DXF, PDF | Laser scanner |
| **This project** | **+/-20-30%** | **PNG (raster)** | **Monocular depth + guess** |

---

## 4. Gap Analysis: Current vs Required

### 4.1 Measurement Accuracy

| Aspect | Current | Required | Gap |
|--------|---------|----------|-----|
| Depth type | Relative (0-1) | Metric (meters) | **Fundamental** -- need metric depth model |
| Scale source | User guess (single number) | Camera intrinsics + known reference OR metric model | **Fundamental** -- no ground truth |
| Precision | ~60cm-1m error | 3-5cm (0.1ft) | **10-20x improvement needed** |
| Validation | None | Cross-check multiple measurements | **Missing entirely** |
| Wall measurement | Bounding box only | Per-wall segments | **Missing entirely** |
| Multi-room | Not supported | Per-room measurements | **Missing entirely** |

### 4.2 Structural Detection

| Feature | Current | Required | Gap |
|---------|---------|----------|-----|
| Walls | Hough lines on density blob | Segmented wall lines with thickness | **Needs semantic understanding** |
| Doors | Not detected | Location, width, swing direction | **Missing entirely** |
| Windows | Not detected | Location, width | **Missing entirely** |
| Room shape | Convex hull (convex only) | Arbitrary polygon (L/U/T shapes) | **Algorithm replacement needed** |
| Wall junctions | Not detected | T-junctions, corners, intersections | **Missing entirely** |
| Wall thickness | Not rendered | 10-30cm shown in plan | **Missing entirely** |
| Furniture | Mixed with walls | Separated or excluded | **Needs segmentation** |

### 4.3 Drawing Quality

| Aspect | Current | Required | Gap |
|--------|---------|----------|-----|
| Format | Raster PNG, 150 DPI | Vector SVG + DXF + PDF | **Complete rewrite of renderer** |
| Line weights | Uniform 1px | 3-tier hierarchy (heavy/medium/light) | **New rendering system** |
| Dimension lines | Red matplotlib arrows | Extension lines + ticks + text | **New rendering system** |
| Symbols | None | Doors, windows, stairs per AIA | **Architectural symbol library** |
| Scale bar | None | Graphical scale bar | **Add** |
| North arrow | None | Orientation indicator | **Add** |
| Title block | "Room Floor Plan" text | Project info, date, scale, author | **Add** |
| Room labels | Single "ROOM" + area | Name + area per room | **Add** |
| Grid | None | Metric grid overlay (optional) | **Add** |
| Color scheme | Blue heatmap + green boundary | Black/white architectural standard | **Change** |
| Layers | Flat image | Walls / Dimensions / Furniture / Annotations | **Add layer support** |

---

## 5. Improvement Plan

### Phase 1: Fix Measurement Accuracy (Critical)

**Goal:** Reduce error from +/-30% to +/-5%

#### 1A. Replace depth model with metric depth estimation

**Current:** `Depth-Anything-V2-Large-hf` (relative depth)
**Replace with:** One of the following metric depth models:

| Model | Accuracy | Speed | Needs Intrinsics | Notes |
|-------|----------|-------|-------------------|-------|
| **Depth Pro (Apple)** | Best avg zero-shot metric | <1s on GPU | No (estimates focal length) | 504M params, fits 6GB VRAM. Best boundary sharpness. |
| **UniDepthV2** | Best cross-domain generalization | ~0.5s | No (self-prompted camera module) | Also outputs uncertainty. |
| **Metric3D v2** | Strong metric accuracy | ~0.5s | Yes (requires camera params) | Good when SfM provides intrinsics. |

**Recommended:** Use **Depth Pro** as primary (no intrinsics needed, best boundaries), with **UniDepthV2** as fallback. Both produce depth in **meters** directly.

**Impact on code:**
- `depth_estimator.py`: Replace model loading and inference. Output is metric depth (meters), not 0-1 normalized.
- `depth_estimator.py:293-302`: Remove inverse-depth hack (`z = 1.0 / (depth + 1e-3)`). Use depth values directly as Z coordinates.
- `config.py`: Remove `DEPTH_SCALE`, update `DEPTH_MODEL`.
- `room_reconstructor.py`: Remove `assumed_room_width` as measurement source. Keep as optional UI hint only.

#### 1B. Implement reference-object calibration (fallback)

When metric depth models aren't precise enough, use known-size objects for calibration:

1. **Standard door detection:** US interior doors are 80" x 32-36" (203cm x 81-91cm). Detect doors in images and use their pixel height to calibrate depth scale.
2. **A4/Letter paper:** If user places a sheet on the floor, detect and calibrate.
3. **User-measured wall:** Allow user to measure one wall with a tape measure and input the value. Use this as ground truth to calibrate all other measurements.

**Implementation:**
- Add `ReferenceCalibrator` module
- Integrate YOLO or similar for door/object detection
- Compute scale factor: `real_size / estimated_size` applied globally to metric depth

#### 1C. Cross-validate measurements

After computing dimensions, validate:
- Opposite walls should be roughly equal length in rectangular rooms
- Area consistency: sum of wall segments should form a closed polygon
- Depth consistency: same wall measured from two views should agree
- SfM sparse point triangulation provides independent scale check

### Phase 2: Fix Structural Detection (High Priority)

**Goal:** Detect actual walls, doors, windows instead of density blobs

#### 2A. Replace density-based wall detection with line-segment detection

**Current approach fails because:** A 2D histogram of point cloud density sees furniture, clutter, and walls as the same thing. Hough transform on this blob finds blob edges, not wall lines.

**New approach:** Wall-line extraction from depth discontinuities

```
For each image:
  1. Compute depth map (metric)
  2. Compute depth gradient (Sobel/Canny on depth)
  3. Depth discontinuities = wall-floor and wall-wall boundaries
  4. Extract line segments from depth edges (LSD or EDLines algorithm)
  5. Classify lines: vertical = wall edges, horizontal = floor/ceiling
  6. Project wall-floor intersection lines to ground plane
```

Combine from multiple views:
```
  7. Merge colinear segments from different views
  8. Snap to orthogonal grid (most rooms have 90-degree walls)
  9. Close gaps and form closed polygons
  10. Assign wall thickness (default 15cm interior, 25cm exterior)
```

**Key algorithms:**
- **LSD (Line Segment Detector):** Better than Hough for architectural lines. Detects sub-pixel line segments without thresholds.
- **Manhattan World assumption:** Most rooms have walls aligned to 2-3 dominant directions. Use vanishing point detection to find these directions and constrain wall lines.
- **RANSAC line fitting:** Fit lines to point cloud slices, not density maps.

#### 2B. Add semantic segmentation for doors and windows

Use a pre-trained indoor scene segmentation model to identify:
- Wall surfaces
- Door openings
- Window openings
- Floor / ceiling boundaries

**Options:**
- **SAM 2 (Segment Anything Model 2):** Zero-shot segmentation. Prompt with "door" / "window" to get masks.
- **ADE20K-trained models:** DeepLab or SegFormer trained on ADE20K (150 indoor categories including wall, door, window, floor, ceiling).
- **Structured3D-trained models:** Specifically trained for indoor structural parsing.

**Implementation:**
- Add `SemanticSegmenter` module
- Run segmentation on each input image
- Project door/window detections to floor plan coordinates
- Compute door width and position along wall

#### 2C. Replace convex hull with polygon reconstruction

**Current:** Convex hull cannot represent L-shaped rooms, alcoves, or recesses.

**New approach: Room polygon reconstruction**

```
1. Project all wall line segments to ground plane
2. Find line intersections (corners)
3. Build a planar graph of wall segments
4. Find the largest enclosed polygon (room boundary)
5. Handle non-convex shapes by tracing wall connectivity
6. Split into rooms if multiple enclosed areas detected
```

**Key library:** Use `shapely` for polygon operations (union, intersection, buffering for wall thickness).

### Phase 3: Fix Drawing Quality (High Priority)

**Goal:** Produce architectural-standard vector floor plans

#### 3A. Replace matplotlib with vector rendering engine

**Current:** Matplotlib generates raster images. Not suitable for CAD import.

**Options:**

| Engine | Output Formats | Pros | Cons |
|--------|---------------|------|------|
| **ezdxf** | DXF (AutoCAD) | Industry standard CAD format, layers, blocks, dimensions | DXF-only, complex API |
| **svgwrite** | SVG | Web-ready, simple API, CSS styling | No DXF |
| **ReportLab** | PDF (vector) | Print-ready, scalable | No DXF/SVG |
| **CadQuery** | DXF, STEP, SVG | Full CAD operations | Heavy dependency |

**Recommended:** Use **ezdxf** for DXF output + **svgwrite** for SVG output. Generate both from the same internal representation.

**Internal representation:**
```python
class FloorPlanDrawing:
    walls: List[WallSegment]       # line + thickness + material
    doors: List[DoorOpening]       # position, width, swing_direction
    windows: List[WindowOpening]   # position, width
    dimensions: List[DimensionLine]  # start, end, value, offset
    rooms: List[RoomPolygon]       # boundary, name, area
    scale: float                   # drawing scale (e.g., 1:50)
    orientation: float             # north angle
```

Render this to DXF, SVG, and PDF independently.

#### 3B. Implement architectural symbol library

Create reusable symbols following AIA standards:

**Walls:**
- Double-line representation with fill (concrete hatching or solid)
- Exterior walls: 25cm default thickness, heavy line weight
- Interior walls: 15cm default thickness, medium line weight
- Dashed lines for demolished walls (renovation plans)

**Doors:**
- Line gap in wall at door position
- 90-degree arc showing swing direction
- Door leaf line from hinge to arc
- Width dimension

**Windows:**
- Three parallel lines within wall thickness
- Or: gap in wall with cross-hatching

**Dimension lines (ASME Y14.5 style):**
- Extension lines from wall faces
- Dimension line between extension lines
- Tick marks (architectural style: 45-degree slash) or arrows
- Centered text with measurement value
- Chain dimensions for multiple segments
- Overall dimension spanning full wall run

#### 3C. Implement proper dimension annotation

**Per-wall dimensions:**
- Measure each wall segment individually
- Show in both metric (m) and imperial (ft-in)
- Format: `3.65 m (12'-0")`

**Room dimensions:**
- Width x Depth inside each room polygon
- Total area

**Chain dimensions:**
- When a wall has doors/windows, show segment dimensions
- Example: `1.2m | [D 0.9m] | 2.4m | [W 1.5m] | 0.8m`

**Overall dimensions:**
- Total exterior dimensions along each axis

#### 3D. Add standard drawing elements

- **Scale bar:** Graphical bar with metric markings (1m, 2m, etc.)
- **North arrow:** Simple arrow symbol with "N" label
- **Title block:** Border box with project name, date, scale, drawing number
- **Legend:** Material hatching key, symbol explanations
- **Grid:** Optional metric grid overlay at 1m intervals

### Phase 4: Improve Point Cloud Quality (Medium Priority)

#### 4A. Increase floor plan grid resolution

**Current:** 100x100 (`FLOOR_PLAN_RESOLUTION = 100`)
**Required:** At least 500x500 for a 5m room (1cm per cell)

For wall detection, we need sub-centimeter point cloud resolution in the ground plane. Increase to 1000x1000 and use adaptive resolution based on room size.

#### 4B. Improve floor slice extraction

**Current:** Fixed 10-30% height band. This is crude.

**Better approach:**
1. Compute height histogram of point cloud
2. Find floor plane using RANSAC plane fitting
3. Take a horizontal slice 80-120cm above floor (wall cross-section height)
4. This captures wall structure, not floor clutter

#### 4C. Separate walls from furniture

Point cloud points on walls vs furniture look different:
- **Wall points:** Form vertical planes, consistent depth, align with room edges
- **Furniture points:** Irregular shapes, internal to room, smaller clusters

Use normal estimation + plane segmentation to separate:
1. Estimate point normals
2. Find dominant vertical planes (walls)
3. Filter out points not belonging to wall planes

### Phase 5: Output & Integration (Medium Priority)

#### 5A. Multi-format export

```
Floor plan data -> Renderer:
  ├── DXF (ezdxf)      -> AutoCAD, BricsCAD, FreeCAD import
  ├── SVG (svgwrite)    -> Web viewing, Inkscape editing
  ├── PDF (reportlab)   -> Print at scale (1:50 on A3)
  ├── PNG (pillow)      -> Quick preview
  └── JSON              -> Room geometry for downstream tools
```

#### 5B. DXF layer structure

Standard CAD layer organization:
```
A-WALL          Walls (heavy, black)
A-WALL-DIMS     Wall dimensions
A-DOOR          Doors (medium, blue)
A-GLAZ          Windows/glazing (medium, cyan)
A-AREA          Room boundaries and labels
A-DIMS          Overall dimensions
A-ANNO          Annotations and notes
A-FURN          Furniture (light, gray) -- optional
A-GRID          Reference grid (light, gray)
A-TTLB          Title block border
```

#### 5C. Gradio UI improvements

- Replace single image output with tabbed format selector
- Add DXF/SVG/PDF download buttons
- Show interactive floor plan (SVG in browser)
- Add measurement editing capability (user corrects a known dimension)

---

## 6. Architecture Changes

### 6.1 Current Module Structure

```
modules/
  depth_estimator.py       # Relative depth model
  floor_plan_generator.py  # Density + Hough + matplotlib
  room_reconstructor.py    # Orchestrator
  sfm_processor.py         # COLMAP SfM
  dense_reconstructor.py   # TSDF fusion
  visualizer_3d.py         # 3D rendering
```

### 6.2 Proposed Module Structure

```
modules/
  depth/
    metric_depth.py          # Depth Pro / UniDepthV2 metric depth
    depth_calibrator.py      # Reference object calibration
  detection/
    wall_detector.py         # Line segment + plane-based wall detection
    opening_detector.py      # Door/window detection (segmentation)
    room_segmenter.py        # Room polygon extraction
  geometry/
    floor_plan_model.py      # Internal data model (walls, doors, rooms)
    measurement_engine.py    # Per-wall and per-room measurements
    polygon_ops.py           # Wall polygon operations (shapely)
  rendering/
    dxf_renderer.py          # DXF output (ezdxf)
    svg_renderer.py          # SVG output (svgwrite)
    pdf_renderer.py          # PDF output (reportlab)
    png_renderer.py          # Quick preview PNG
    symbol_library.py        # Architectural symbols (doors, windows, etc.)
    dimension_lines.py       # ASME-style dimension annotation
  sfm/
    sfm_processor.py         # COLMAP SfM (existing, cleaned up)
    dense_reconstructor.py   # TSDF fusion (existing, cleaned up)
  orchestrator.py            # Main pipeline
config.py                    # Configuration
```

### 6.3 Data Flow (New Pipeline)

```
Input Images
    |
    v
[Metric Depth Estimation] -- Depth Pro / UniDepthV2
    |                        Output: depth in meters per pixel
    v
[SfM] -- COLMAP (camera poses + intrinsics)
    |
    v
[Point Cloud Generation] -- Metric depth + real intrinsics
    |                        Output: point cloud in meters
    v
[Wall Detection] -- Depth gradients + line segments + plane fitting
    |               Output: wall line segments in ground plane
    v
[Opening Detection] -- Semantic segmentation (doors, windows)
    |                   Output: door/window positions along walls
    v
[Room Polygon Extraction] -- Wall graph -> enclosed polygons
    |                        Output: room boundaries
    v
[Measurement Engine] -- Per-wall lengths, room areas, overall dims
    |                   Output: FloorPlanModel with all measurements
    v
[Calibration] -- Reference object check + cross-validation
    |             Output: Calibrated FloorPlanModel
    v
[Rendering] -- DXF + SVG + PDF + PNG
               Output: Industry-standard floor plan files
```

---

## 7. Implementation Phases

### Phase 1: Metric Depth + Scale Fix (Weeks 1-2)

**Priority: CRITICAL -- without this, nothing else matters**

| Task | Files | Effort |
|------|-------|--------|
| Integrate Depth Pro model | `modules/depth/metric_depth.py` | High |
| Remove inverse-depth hack | `depth_estimator.py` refactor | Medium |
| Remove `assumed_room_width` as measurement source | `config.py`, `room_reconstructor.py`, `floor_plan_generator.py` | Medium |
| Add reference-object calibration | `modules/depth/depth_calibrator.py` | Medium |
| Add cross-validation checks | `modules/geometry/measurement_engine.py` | Low |
| Update tests with known-size room | `tests/` | Medium |

**Validation:** Process a room with known dimensions (measured by tape). Output must be within +/-5% of ground truth.

### Phase 2: Wall Detection + Room Geometry (Weeks 3-4)

**Priority: HIGH -- determines floor plan shape accuracy**

| Task | Files | Effort |
|------|-------|--------|
| Line segment detector on depth maps | `modules/detection/wall_detector.py` | High |
| Manhattan World wall alignment | `modules/detection/wall_detector.py` | Medium |
| RANSAC floor plane detection | `modules/detection/wall_detector.py` | Medium |
| Room polygon reconstruction | `modules/detection/room_segmenter.py` | High |
| Internal floor plan data model | `modules/geometry/floor_plan_model.py` | Medium |
| Per-wall measurement | `modules/geometry/measurement_engine.py` | Medium |

**Validation:** Generate floor plan polygon for a rectangular room. Corners must be within 5cm of ground truth. Polygon must be closed.

### Phase 3: Architectural Rendering (Weeks 5-6)

**Priority: HIGH -- produces the actual deliverable**

| Task | Files | Effort |
|------|-------|--------|
| SVG renderer with line weights | `modules/rendering/svg_renderer.py` | High |
| DXF renderer with layers | `modules/rendering/dxf_renderer.py` | High |
| Architectural symbol library | `modules/rendering/symbol_library.py` | Medium |
| ASME-style dimension lines | `modules/rendering/dimension_lines.py` | Medium |
| Scale bar + north arrow + title block | `modules/rendering/svg_renderer.py` | Low |
| PDF renderer | `modules/rendering/pdf_renderer.py` | Medium |
| PNG preview renderer | `modules/rendering/png_renderer.py` | Low |

**Validation:** Output DXF file opens correctly in AutoCAD/FreeCAD. Wall lines have correct thickness. Dimensions match measurement engine output.

### Phase 4: Door/Window Detection (Weeks 7-8)

**Priority: MEDIUM -- enhances plan completeness**

| Task | Files | Effort |
|------|-------|--------|
| Semantic segmentation integration | `modules/detection/opening_detector.py` | High |
| Door detection + width estimation | `modules/detection/opening_detector.py` | Medium |
| Window detection + position mapping | `modules/detection/opening_detector.py` | Medium |
| Door/window symbols in renderers | `modules/rendering/symbol_library.py` | Medium |
| Update data model for openings | `modules/geometry/floor_plan_model.py` | Low |

**Validation:** Detected door positions within 20cm of actual. Door widths within 5cm. Swing direction correct.

### Phase 5: Integration + UI (Week 9-10)

| Task | Files | Effort |
|------|-------|--------|
| New Gradio UI with format selector | `app.py` | Medium |
| Download buttons for DXF/SVG/PDF | `app.py` | Low |
| Interactive SVG viewer | `app.py` | Medium |
| User measurement correction input | `app.py` | Medium |
| End-to-end test suite | `tests/` | High |
| Documentation update | `README.md`, `ARCHITECTURE.md` | Medium |

---

## 8. References

### Standards
- [ANSI Z765-2021 Summary](https://www.accuratehomemeasuring.com/frequently-asked-questions/24-the-ansi-standard-short-summary)
- [ANSI Measurement Standards (ProEducate)](https://www.proeducate.com/courses/static_files/docs/LA/PropertyMeasurement.pdf)
- [ASME Y14.5 & Y14.100 Standards](https://blog.ansi.org/ansi/what-are-the-asme-y14-5-and-asme-y14-100-standards/)
- [Architectural Line Weight Standards](https://www.coohom.com/article/architectural-floor-plan-line-weights)
- [Floor Plan Symbols Guide](https://cedreo.com/blog/floor-plan-symbols/)
- [Architecture Drawing Conventions](https://archimash.com/articles/architecture-design-drawing-conventions/)

### Metric Depth Models
- [Depth Pro: Sharp Monocular Metric Depth (Apple, ICLR 2025)](https://learnopencv.com/depth-pro-monocular-metric-depth/)
- [UniDepthV2: Universal Metric Depth (Feb 2025)](https://arxiv.org/abs/2502.20110)
- [Metric3D v2 (IEEE, 2024)](https://ieeexplore.ieee.org/document/10638254/)
- [Survey on Monocular Metric Depth Estimation](https://arxiv.org/html/2501.11841v3)

### Industry Tools
- [Best AI Floor Plan Generators Compared (2026)](https://www.cubi.casa/best-ai-floor-plan/)
- [Matterport Schematic Floor Plans](https://support.matterport.com/s/article/FAQ-Schematic-Floor-Plans?language=en_US)
- [Room Scan Apps for Business](https://www.arcsite.com/blog/room-scan-apps)
- [SVG Floor Plans (CubiCasa)](https://www.cubi.casa/tag/svg-floor-plans/)

### Libraries for Implementation
- [ezdxf (DXF generation)](https://ezdxf.readthedocs.io/)
- [svgwrite (SVG generation)](https://svgwrite.readthedocs.io/)
- [shapely (polygon operations)](https://shapely.readthedocs.io/)
- [Depth Pro GitHub](https://github.com/apple/ml-depth-pro)
- [UniDepth GitHub](https://github.com/lpiccinelli-eth/UniDepth)
