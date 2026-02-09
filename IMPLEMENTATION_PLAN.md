# Implementation Plan: Floor Plan System Upgrade

**Source:** `FLOOR_PLAN_REVIEW.md`
**Created:** 2026-02-09
**Last Updated:** 2026-02-09
**Status:** IN PROGRESS (Phase 1)

---

## Progress Tracker

```
OVERALL: [########............] 43% (16/37 steps complete)

Phase 1 - Metric Depth + Scale:    [##########] 100% (8/8) COMPLETE
Phase 2 - Wall Detection + Geom:   [##########] 100% (8/8) COMPLETE
Phase 3 - Architectural Rendering:  [..........] 0%  (0/8) << NEXT
Phase 4 - Door/Window Detection:    [..........] 0%  (0/6)
Phase 5 - Integration + UI:         [..........] 0%  (0/7)
```

---

## Resume Protocol

**At the start of every session, do this:**

1. Read this file (`IMPLEMENTATION_PLAN.md`)
2. Find the first step with status `[ ]` (not started) or `[~]` (in progress)
3. Read the step's listed **Files** and **Dependencies** to load context
4. Continue from that step
5. After completing a step, update its status to `[x]` and update the Progress Tracker counts above
6. Commit after each completed step with message: `feat(floor-plan): step X.Y - <description>`

**Status legend:**
- `[ ]` = Not started
- `[~]` = In progress (was started in a previous session but not finished)
- `[x]` = Complete
- `[!]` = Blocked (with note explaining why)

---

## Phase 1: Metric Depth + Scale Fix (CRITICAL)

**Goal:** Replace relative depth with metric depth. Reduce measurement error from +/-30% to +/-5%.
**Validation:** Process a room with known dimensions. Output must be within +/-5% of ground truth.

### Step 1.1: Create new module directory structure
- **Status:** `[x]`
- **Description:** Create the new `modules/depth/`, `modules/detection/`, `modules/geometry/`, `modules/rendering/` package directories with `__init__.py` files. This is scaffolding only - no logic changes.
- **Files to create:**
  - `modules/depth/__init__.py`
  - `modules/depth/metric_depth.py` (empty class stub)
  - `modules/depth/depth_calibrator.py` (empty class stub)
  - `modules/detection/__init__.py`
  - `modules/geometry/__init__.py`
  - `modules/geometry/floor_plan_model.py` (empty dataclass stub)
  - `modules/rendering/__init__.py`
- **Dependencies:** None
- **Test:** `python -c "from modules.depth import metric_depth; from modules.detection import *; from modules.geometry import *; from modules.rendering import *; print('OK')"`
- **Acceptance:** All imports succeed. Existing `uv run python run_cli.py sample_images/test_room_*.jpg` still works unchanged.

### Step 1.2: Integrate Depth Pro metric depth model
- **Status:** `[x]`
- **Description:** Implement `MetricDepthEstimator` in `modules/depth/metric_depth.py`. This class wraps Apple's Depth Pro model (or UniDepthV2 as fallback) to produce depth maps in **meters**. Must handle model download, GPU/CPU detection, and batch processing. Keep the existing `DepthEstimator` untouched for now.
- **Files to modify:**
  - `modules/depth/metric_depth.py` (implement `MetricDepthEstimator`)
  - `requirements.txt` (add `depth-pro` or `unidepth` dependency)
- **Dependencies:** Step 1.1
- **Test:**
  ```python
  from modules.depth.metric_depth import MetricDepthEstimator
  import numpy as np
  est = MetricDepthEstimator()
  fake_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
  depth = est.estimate_depth(fake_img)
  assert depth.shape == (480, 640), f"Wrong shape: {depth.shape}"
  assert depth.min() >= 0, "Negative depth values"
  assert depth.max() < 100, f"Unreasonable max depth: {depth.max()}"  # meters
  print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m  OK")
  ```
- **Acceptance:** Returns depth in meters. Values in reasonable range (0.1m - 20m for indoor). Falls back gracefully if Depth Pro unavailable.

### Step 1.3: Add metric depth-to-3D projection
- **Status:** `[x]`
- **Description:** Add `depth_to_3d_points_metric()` method to `MetricDepthEstimator`. Unlike the current inverse-depth hack (`z = 1.0 / (depth + 1e-3) * DEPTH_SCALE`), this directly uses metric depth values as Z coordinates. Camera intrinsics come from SfM when available, or from the model's focal length estimate (Depth Pro estimates focal length).
- **Files to modify:**
  - `modules/depth/metric_depth.py` (add `depth_to_3d_points_metric()`)
- **Dependencies:** Step 1.2
- **Test:**
  ```python
  from modules.depth.metric_depth import MetricDepthEstimator
  import numpy as np
  est = MetricDepthEstimator()
  img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
  depth = est.estimate_depth(img)
  points, colors = est.depth_to_3d_points_metric(img, depth)
  assert points.shape[1] == 3, "Points must be Nx3"
  assert len(points) > 100, "Too few points generated"
  z_range = points[:, 2].max() - points[:, 2].min()
  assert z_range > 0.01, "Z range too small - not metric?"
  print(f"Generated {len(points)} points, Z range: {z_range:.2f}m  OK")
  ```
- **Acceptance:** 3D points have Z values in meters. No inverse-depth hack. Point cloud shape is geometrically plausible.

### Step 1.4: Implement depth calibrator with reference objects
- **Status:** `[x]`
- **Description:** Create `DepthCalibrator` in `modules/depth/depth_calibrator.py`. Supports three calibration methods: (a) user-supplied known wall measurement, (b) standard door detection (80" US interior door), (c) cross-view consistency check. Outputs a single scale correction factor applied to metric depth.
- **Files to modify:**
  - `modules/depth/depth_calibrator.py` (implement `DepthCalibrator`)
- **Dependencies:** Step 1.3
- **Test:**
  ```python
  from modules.depth.depth_calibrator import DepthCalibrator
  import numpy as np
  cal = DepthCalibrator()
  # Test user-supplied calibration
  points = np.random.rand(1000, 3) * 5  # 5m range
  factor = cal.calibrate_from_known_dimension(points, axis=0, known_meters=4.0)
  assert 0.1 < factor < 10, f"Unreasonable factor: {factor}"
  calibrated = points * factor
  x_range = calibrated[:, 0].max() - calibrated[:, 0].min()
  assert abs(x_range - 4.0) < 0.5, f"Calibration failed: width={x_range:.2f}m"
  print(f"Calibration factor: {factor:.3f}  OK")
  ```
- **Acceptance:** Scale correction factor is reasonable. Known-dimension calibration works within 5%.

### Step 1.5: Update config.py for metric depth
- **Status:** `[x]`
- **Description:** Add new config constants for metric depth. Keep old constants for backward compatibility but mark as deprecated. Add `METRIC_DEPTH_MODEL`, `METRIC_DEPTH_MODEL_FALLBACK`, `ENABLE_METRIC_DEPTH` (default True), `CALIBRATION_METHOD` (default "auto").
- **Files to modify:**
  - `config.py`
- **Dependencies:** Step 1.2
- **Test:** `python -c "from config import METRIC_DEPTH_MODEL, ENABLE_METRIC_DEPTH; print('OK')"`
- **Acceptance:** New config values importable. Old config values still work.

### Step 1.6: Wire metric depth into RoomReconstructor
- **Status:** `[x]`
- **Description:** Modify `RoomReconstructor` to use `MetricDepthEstimator` when `ENABLE_METRIC_DEPTH=True`. Update `process_single_image()` to call `depth_to_3d_points_metric()` instead of the old path. Keep fallback to old `DepthEstimator` when metric model unavailable. Remove `assumed_room_width` as the measurement source (keep as UI hint only for calibrator).
- **Files to modify:**
  - `modules/room_reconstructor.py` (conditional metric depth path)
- **Dependencies:** Steps 1.3, 1.4, 1.5
- **Test:**
  ```bash
  uv run python run_cli.py sample_images/test_room_*.jpg --room-width 4.0
  # Output measurements should now be derived from metric depth, not assumed_room_width
  ```
- **Acceptance:** Reconstruction completes. Measurements are from metric depth model. `assumed_room_width` no longer directly dictates output dimensions (used only as calibration hint).

### Step 1.7: Update FloorPlanGenerator to use metric points
- **Status:** `[x]`
- **Description:** Modify `FloorPlanGenerator.generate_floor_plan()` to detect whether input points are already in meters (metric pipeline) or arbitrary units (legacy pipeline). When metric, skip the `assumed_width / room_width_units` scaling hack at line 136-137. Compute measurements directly from point cloud coordinates.
- **Files to modify:**
  - `modules/floor_plan_generator.py`
- **Dependencies:** Step 1.6
- **Test:**
  ```python
  from modules.floor_plan_generator import FloorPlanGenerator
  import numpy as np
  # Create a synthetic "room" with known dimensions: 4m x 3m
  points = []
  # Floor: 4m wide (X), 3m deep (Z), at Y=0
  for _ in range(500):
      points.append([np.random.uniform(0, 4), np.random.uniform(0, 0.3), np.random.uniform(0, 3)])
  # Walls
  for _ in range(200):
      points.append([0, np.random.uniform(0, 2.5), np.random.uniform(0, 3)])
      points.append([4, np.random.uniform(0, 2.5), np.random.uniform(0, 3)])
  points = np.array(points)
  gen = FloorPlanGenerator(assumed_width=4.0)
  result = gen.generate_floor_plan(points)
  m = result['measurements']
  assert abs(m['width_m'] - 4.0) < 1.0, f"Width off: {m['width_m']}"
  assert abs(m['depth_m'] - 3.0) < 1.0, f"Depth off: {m['depth_m']}"
  print(f"Measured: {m['width_m']:.2f}m x {m['depth_m']:.2f}m  OK")
  ```
- **Acceptance:** With metric point clouds, measurements come from actual coordinates, not the assumed-width scaling hack.

### Step 1.8: Phase 1 integration test
- **Status:** `[x]`
- **Description:** Create `tests/test_phase1_metric_depth.py`. End-to-end test: load sample images -> metric depth -> point cloud -> floor plan measurements. Verify the full pipeline works with the new metric depth path. Compare measurements before/after to confirm improvement.
- **Files to create:**
  - `tests/__init__.py`
  - `tests/test_phase1_metric_depth.py`
- **Dependencies:** Steps 1.6, 1.7
- **Test:** `uv run python -m pytest tests/test_phase1_metric_depth.py -v`
- **Acceptance:** All tests pass. The metric path produces output. Measurements are more consistent across different `assumed_room_width` values (since they should no longer matter).

---

## Phase 2: Wall Detection + Room Geometry (HIGH PRIORITY)

**Goal:** Detect actual walls instead of density blobs. Support non-convex room shapes.
**Validation:** Floor plan polygon for rectangular room has corners within 5cm of ground truth. Polygon is closed.

### Step 2.1: Implement FloorPlanModel data classes
- **Status:** `[x]`
- **Description:** Create the internal data model in `modules/geometry/floor_plan_model.py`. Define dataclasses: `WallSegment` (start, end, thickness, material), `DoorOpening` (position, width, swing_direction), `WindowOpening` (position, width), `RoomPolygon` (boundary, name, area), `DimensionLine` (start, end, value, offset), `FloorPlanModel` (walls, doors, windows, rooms, dimensions, scale, orientation). This is the single source of truth between detection and rendering.
- **Files to modify:**
  - `modules/geometry/floor_plan_model.py`
- **Dependencies:** Step 1.1
- **Test:**
  ```python
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon
  import numpy as np
  w1 = WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15)
  w2 = WallSegment(start=np.array([4,0]), end=np.array([4,3]), thickness=0.15)
  room = RoomPolygon(boundary=np.array([[0,0],[4,0],[4,3],[0,3]]), name="Living Room")
  model = FloorPlanModel(walls=[w1, w2], rooms=[room])
  assert len(model.walls) == 2
  assert room.area > 0
  print(f"Room area: {room.area:.1f}m^2  OK")
  ```
- **Acceptance:** All dataclasses instantiate correctly. `RoomPolygon.area` computed via Shoelace formula. Serializable to dict/JSON.

### Step 2.2: Implement RANSAC floor plane detection
- **Status:** `[x]`
- **Description:** Add floor plane detection to `modules/detection/wall_detector.py`. Use Open3D's `segment_plane()` to find the dominant horizontal plane (floor). Return the floor plane equation and inlier mask. This replaces the crude fixed 10-30% height slice in `floor_plan_generator.py:185-209`.
- **Files to modify:**
  - `modules/detection/__init__.py`
  - `modules/detection/wall_detector.py` (new file, add `WallDetector` class with `detect_floor_plane()`)
- **Dependencies:** Step 1.1
- **Test:**
  ```python
  from modules.detection.wall_detector import WallDetector
  import numpy as np
  det = WallDetector()
  # Synthetic room: floor at Y=0, ceiling at Y=2.5
  floor = np.column_stack([np.random.uniform(0,4,500), np.random.uniform(-0.1,0.1,500), np.random.uniform(0,3,500)])
  walls = np.column_stack([np.random.uniform(0,4,200), np.random.uniform(0,2.5,200), np.random.uniform(0,3,200)])
  points = np.vstack([floor, walls])
  plane, inliers = det.detect_floor_plane(points)
  assert len(inliers) > 300, f"Too few floor inliers: {len(inliers)}"
  # Normal should be roughly [0, 1, 0] (vertical)
  normal = plane[:3]
  assert abs(abs(normal[1]) - 1.0) < 0.3, f"Floor normal not vertical: {normal}"
  print(f"Floor plane normal: {normal}, inliers: {len(inliers)}  OK")
  ```
- **Acceptance:** Correctly identifies horizontal floor plane. Inlier mask separates floor from walls.

### Step 2.3: Implement LSD wall line detection from depth maps
- **Status:** `[x]`
- **Description:** Add wall detection pipeline to `WallDetector`. For each depth map: compute depth gradient -> detect depth discontinuities (wall-floor, wall-wall edges) -> extract line segments using OpenCV LSD (Line Segment Detector). Project wall-floor intersection lines to ground plane. Return wall line segments in metric coordinates.
- **Files to modify:**
  - `modules/detection/wall_detector.py` (add `detect_walls_from_depth()`)
- **Dependencies:** Steps 1.3, 2.2
- **Test:**
  ```python
  from modules.detection.wall_detector import WallDetector
  import numpy as np
  det = WallDetector()
  # Create a synthetic depth map with clear wall boundaries
  depth = np.ones((480, 640), dtype=np.float32) * 3.0  # 3m depth
  depth[:, :160] = 2.0  # Left wall at 2m
  depth[:, 480:] = 2.0  # Right wall at 2m
  depth[:120, :] = 4.0  # Back wall at 4m
  segments = det.detect_walls_from_depth(depth, fx=500, fy=500)
  assert len(segments) >= 2, f"Expected >=2 wall segments, got {len(segments)}"
  print(f"Detected {len(segments)} wall segments  OK")
  ```
- **Acceptance:** Detects at least the major wall boundaries from depth discontinuities. Returns line segments, not density blobs.

### Step 2.4: Manhattan World wall alignment
- **Status:** `[x]`
- **Description:** Add wall alignment to `WallDetector`. Most rooms have walls at 90-degree angles. Detect dominant directions via vanishing point analysis or histogram of line segment angles. Snap detected wall segments to the two dominant perpendicular directions. Merge colinear segments and close gaps.
- **Files to modify:**
  - `modules/detection/wall_detector.py` (add `align_walls_manhattan()`)
- **Dependencies:** Step 2.3
- **Test:**
  ```python
  from modules.detection.wall_detector import WallDetector
  import numpy as np
  det = WallDetector()
  # Create noisy wall segments (roughly aligned to axes but not exactly)
  segments = [
      (np.array([0.0, 0.0]), np.array([3.98, 0.05])),   # ~horizontal
      (np.array([4.02, -0.03]), np.array([4.01, 2.97])), # ~vertical
      (np.array([3.99, 3.02]), np.array([0.03, 2.98])),  # ~horizontal
      (np.array([-0.02, 3.01]), np.array([0.01, 0.02])), # ~vertical
  ]
  aligned = det.align_walls_manhattan(segments)
  for seg in aligned:
      dx = abs(seg[1][0] - seg[0][0])
      dy = abs(seg[1][1] - seg[0][1])
      # Each segment should be axis-aligned (one of dx, dy should be ~0)
      assert min(dx, dy) < 0.1, f"Not axis-aligned: dx={dx:.2f}, dy={dy:.2f}"
  print(f"Aligned {len(aligned)} segments  OK")
  ```
- **Acceptance:** Noisy segments snap to perpendicular axes. Colinear segments merged. Gap tolerance configurable.

### Step 2.5: Room polygon reconstruction
- **Status:** `[x]`
- **Description:** Create `modules/detection/room_segmenter.py` with `RoomSegmenter` class. Takes wall line segments, builds a planar graph of wall connectivity, finds enclosed polygons representing rooms. Handles non-convex shapes (L, U, T rooms). Uses `shapely` for polygon operations. Replaces the convex hull boundary in current system.
- **Files to modify:**
  - `modules/detection/room_segmenter.py` (new file)
  - `requirements.txt` (add `shapely>=2.0`)
- **Dependencies:** Step 2.4
- **Test:**
  ```python
  from modules.detection.room_segmenter import RoomSegmenter
  import numpy as np
  seg = RoomSegmenter()
  # L-shaped room walls
  walls = [
      (np.array([0,0]), np.array([6,0])),
      (np.array([6,0]), np.array([6,3])),
      (np.array([6,3]), np.array([3,3])),
      (np.array([3,3]), np.array([3,5])),
      (np.array([3,5]), np.array([0,5])),
      (np.array([0,5]), np.array([0,0])),
  ]
  rooms = seg.extract_rooms(walls)
  assert len(rooms) >= 1, "Should detect at least 1 room"
  assert rooms[0].area > 20, f"Area too small: {rooms[0].area}"
  # L-shape area = 6*3 + 3*2 = 24
  assert abs(rooms[0].area - 24) < 2, f"Area wrong: {rooms[0].area} (expected ~24)"
  print(f"Room area: {rooms[0].area:.1f}m^2  OK")
  ```
- **Acceptance:** Produces closed polygon for L-shaped room. Area calculation correct. Returns `RoomPolygon` dataclass instances.

### Step 2.6: Per-wall measurement engine
- **Status:** `[x]`
- **Description:** Create `modules/geometry/measurement_engine.py` with `MeasurementEngine` class. Takes `FloorPlanModel` and computes: per-wall length, room width x depth, total area, bounding box dimensions. Results in both metric and imperial. Cross-validates: opposite walls should match in rectangular rooms. Populates `DimensionLine` objects.
- **Files to modify:**
  - `modules/geometry/measurement_engine.py` (new file)
- **Dependencies:** Steps 2.1, 2.5
- **Test:**
  ```python
  from modules.geometry.measurement_engine import MeasurementEngine
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon
  import numpy as np
  engine = MeasurementEngine()
  walls = [
      WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15),
      WallSegment(start=np.array([4,0]), end=np.array([4,3]), thickness=0.15),
      WallSegment(start=np.array([4,3]), end=np.array([0,3]), thickness=0.15),
      WallSegment(start=np.array([0,3]), end=np.array([0,0]), thickness=0.15),
  ]
  room = RoomPolygon(boundary=np.array([[0,0],[4,0],[4,3],[0,3]]), name="Room")
  model = FloorPlanModel(walls=walls, rooms=[room])
  result = engine.compute_measurements(model)
  assert abs(result['rooms'][0]['area_sqm'] - 12.0) < 0.5
  assert len(result['wall_lengths']) == 4
  assert abs(result['wall_lengths'][0] - 4.0) < 0.1
  print(f"Wall lengths: {result['wall_lengths']}  OK")
  ```
- **Acceptance:** Per-wall lengths correct. Area correct. Imperial conversions correct. Cross-validation warnings logged for mismatched opposite walls.

### Step 2.7: Wire detection pipeline into RoomReconstructor
- **Status:** `[x]`
- **Description:** Add the new wall detection + room segmentation pipeline to `RoomReconstructor`. After point cloud fusion, run: floor plane detection -> wall line extraction from depth maps -> Manhattan alignment -> room polygon extraction -> measurement engine. Store `FloorPlanModel` in the result dict alongside the old floor plan data for comparison.
- **Files to modify:**
  - `modules/room_reconstructor.py`
- **Dependencies:** Steps 2.2-2.6
- **Test:**
  ```bash
  uv run python run_cli.py sample_images/test_room_*.jpg
  # Should output both old and new measurements for comparison
  ```
- **Acceptance:** New pipeline runs without errors. `FloorPlanModel` is populated in result dict. Old pipeline still works as fallback.

### Step 2.8: Phase 2 integration test
- **Status:** `[x]`
- **Description:** Create `tests/test_phase2_wall_detection.py`. Test wall detection on synthetic depth maps with known wall positions. Test room polygon extraction for rectangular and L-shaped rooms. Verify measurement accuracy.
- **Files to create:**
  - `tests/test_phase2_wall_detection.py`
- **Dependencies:** Step 2.7
- **Test:** `uv run python -m pytest tests/test_phase2_wall_detection.py -v`
- **Acceptance:** All tests pass. Wall positions within 10cm of ground truth. Room polygons are closed. Areas correct within 5%.

---

## Phase 3: Architectural Rendering (HIGH PRIORITY)

**Goal:** Produce industry-standard vector floor plans (SVG + DXF) with line weights, dimensions, and symbols.
**Validation:** DXF opens in FreeCAD. Wall lines have correct thickness. Dimensions match measurement engine.

### Step 3.1: SVG renderer with line weight hierarchy
- **Status:** `[ ]`
- **Description:** Create `modules/rendering/svg_renderer.py` using `svgwrite`. Renders a `FloorPlanModel` to SVG with proper line weight hierarchy: heavy (0.6-1.0mm) for walls, medium (0.3-0.5mm) for openings, light (0.1-0.2mm) for dimensions. Walls drawn as double lines with thickness. Black/white architectural color scheme.
- **Files to modify:**
  - `modules/rendering/svg_renderer.py` (new file)
  - `requirements.txt` (add `svgwrite>=1.4`)
- **Dependencies:** Step 2.1
- **Test:**
  ```python
  from modules.rendering.svg_renderer import SVGRenderer
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon
  import numpy as np
  renderer = SVGRenderer()
  walls = [
      WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15),
      WallSegment(start=np.array([4,0]), end=np.array([4,3]), thickness=0.15),
      WallSegment(start=np.array([4,3]), end=np.array([0,3]), thickness=0.15),
      WallSegment(start=np.array([0,3]), end=np.array([0,0]), thickness=0.15),
  ]
  room = RoomPolygon(boundary=np.array([[0,0],[4,0],[4,3],[0,3]]), name="Living Room")
  model = FloorPlanModel(walls=walls, rooms=[room])
  svg_path = renderer.render(model, "outputs/test_floor_plan.svg")
  import os
  assert os.path.exists(svg_path), "SVG not created"
  assert os.path.getsize(svg_path) > 100, "SVG too small"
  print(f"SVG saved: {svg_path} ({os.path.getsize(svg_path)} bytes)  OK")
  ```
- **Acceptance:** SVG file renders in browser. Walls are double-line with fill. Three line weight tiers visible.

### Step 3.2: DXF renderer with CAD layers
- **Status:** `[ ]`
- **Description:** Create `modules/rendering/dxf_renderer.py` using `ezdxf`. Renders `FloorPlanModel` to DXF with standard CAD layers: A-WALL, A-WALL-DIMS, A-DOOR, A-GLAZ, A-AREA, A-DIMS, A-ANNO. Walls as polylines with thickness. Proper line weights per layer.
- **Files to modify:**
  - `modules/rendering/dxf_renderer.py` (new file)
  - `requirements.txt` (add `ezdxf>=1.0`)
- **Dependencies:** Step 2.1
- **Test:**
  ```python
  from modules.rendering.dxf_renderer import DXFRenderer
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon
  import numpy as np
  renderer = DXFRenderer()
  walls = [
      WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15),
      WallSegment(start=np.array([4,0]), end=np.array([4,3]), thickness=0.15),
  ]
  room = RoomPolygon(boundary=np.array([[0,0],[4,0],[4,3],[0,3]]), name="Room")
  model = FloorPlanModel(walls=walls, rooms=[room])
  dxf_path = renderer.render(model, "outputs/test_floor_plan.dxf")
  import os, ezdxf
  assert os.path.exists(dxf_path)
  doc = ezdxf.readfile(dxf_path)
  layers = [l.dxf.name for l in doc.layers]
  assert "A-WALL" in layers, f"Missing A-WALL layer. Layers: {layers}"
  print(f"DXF layers: {layers}  OK")
  ```
- **Acceptance:** DXF opens in FreeCAD/AutoCAD. Contains proper layer structure. Entities on correct layers.

### Step 3.3: Architectural symbol library
- **Status:** `[ ]`
- **Description:** Create `modules/rendering/symbol_library.py`. Implement reusable architectural symbols: wall (double-line with fill), door (gap + 90-degree swing arc), window (triple parallel lines in wall gap), dimension tick marks (45-degree slash). Each symbol is a function that returns SVG/DXF primitives for the respective renderer.
- **Files to modify:**
  - `modules/rendering/symbol_library.py` (new file)
- **Dependencies:** None (pure geometry)
- **Test:**
  ```python
  from modules.rendering.symbol_library import SymbolLibrary
  import numpy as np
  lib = SymbolLibrary()
  # Door symbol: position along wall, width, swing direction
  door = lib.door_symbol(position=np.array([2, 0]), width=0.9, wall_direction=np.array([1, 0]), swing="left")
  assert 'arc' in door or 'path' in str(door), "Door should have arc element"
  # Window symbol
  window = lib.window_symbol(position=np.array([1.5, 3]), width=1.2, wall_direction=np.array([1, 0]))
  assert len(window) > 0, "Window symbol empty"
  print("Symbol library OK")
  ```
- **Acceptance:** Door symbol has gap + arc. Window symbol has triple lines. Dimension ticks at 45 degrees. Symbols are renderer-agnostic (return coordinate data).

### Step 3.4: ASME-style dimension lines
- **Status:** `[ ]`
- **Description:** Create `modules/rendering/dimension_lines.py`. Implements proper architectural dimension annotation: extension lines from wall faces, dimension line between them, tick marks (45-degree architectural style), centered text with value in both metric and imperial. Supports chain dimensions for multiple wall segments.
- **Files to modify:**
  - `modules/rendering/dimension_lines.py` (new file)
- **Dependencies:** Step 2.1
- **Test:**
  ```python
  from modules.rendering.dimension_lines import DimensionLineGenerator
  import numpy as np
  gen = DimensionLineGenerator()
  dim = gen.create_dimension(
      start=np.array([0, 0]), end=np.array([4, 0]),
      offset=0.5, value_m=4.0
  )
  assert dim.text_metric == "4.00 m"
  assert dim.text_imperial == "13'-1\""
  assert len(dim.extension_lines) == 2
  print(f"Dimension: {dim.text_metric} / {dim.text_imperial}  OK")
  ```
- **Acceptance:** Dimension lines have extension lines, ticks, and centered text. Metric + imperial. Chain dimensions work.

### Step 3.5: Scale bar, north arrow, title block
- **Status:** `[ ]`
- **Description:** Add standard drawing elements to SVG and DXF renderers: graphical scale bar (e.g., 0-1m-2m marks), north arrow symbol, title block border with project name/date/scale/drawing number. These are rendered in a reserved margin area outside the floor plan content.
- **Files to modify:**
  - `modules/rendering/svg_renderer.py` (add standard elements)
  - `modules/rendering/dxf_renderer.py` (add standard elements)
  - `modules/rendering/symbol_library.py` (scale bar + north arrow symbols)
- **Dependencies:** Steps 3.1, 3.2, 3.3
- **Test:**
  ```python
  # Render a floor plan and verify standard elements exist in SVG
  from modules.rendering.svg_renderer import SVGRenderer
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon
  import numpy as np
  renderer = SVGRenderer()
  model = FloorPlanModel(
      walls=[WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15)],
      rooms=[RoomPolygon(boundary=np.array([[0,0],[4,0],[4,3],[0,3]]), name="Room")],
      scale=50  # 1:50
  )
  svg_content = renderer.render_to_string(model)
  assert "scale" in svg_content.lower(), "Missing scale bar"
  assert "N" in svg_content, "Missing north arrow"
  print("Standard drawing elements present  OK")
  ```
- **Acceptance:** Scale bar shows metric markings. North arrow visible. Title block has project info.

### Step 3.6: PNG preview renderer
- **Status:** `[ ]`
- **Description:** Create `modules/rendering/png_renderer.py`. Renders `FloorPlanModel` to a clean PNG preview using matplotlib or Pillow. Replaces the current heatmap-based rendering. Uses the same architectural style (wall fills, dimension lines) but rasterized. This is for quick preview in Gradio without SVG support.
- **Files to modify:**
  - `modules/rendering/png_renderer.py` (new file)
- **Dependencies:** Steps 2.1, 3.3, 3.4
- **Test:**
  ```python
  from modules.rendering.png_renderer import PNGRenderer
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon
  import numpy as np
  renderer = PNGRenderer()
  walls = [
      WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15),
      WallSegment(start=np.array([4,0]), end=np.array([4,3]), thickness=0.15),
      WallSegment(start=np.array([4,3]), end=np.array([0,3]), thickness=0.15),
      WallSegment(start=np.array([0,3]), end=np.array([0,0]), thickness=0.15),
  ]
  room = RoomPolygon(boundary=np.array([[0,0],[4,0],[4,3],[0,3]]), name="Room")
  model = FloorPlanModel(walls=walls, rooms=[room])
  fig = renderer.render(model)
  assert fig is not None
  print("PNG render OK")
  ```
- **Acceptance:** Clean architectural-style PNG. Walls drawn with thickness (not single-pixel). Dimensions annotated.

### Step 3.7: Wire renderers into FloorPlanGenerator
- **Status:** `[ ]`
- **Description:** Update `FloorPlanGenerator.create_floor_plan_image()` to optionally use the new renderers when a `FloorPlanModel` is available. Add `render_formats` parameter to control output (png, svg, dxf). When new pipeline provides `FloorPlanModel`, use new renderers. When fallback (old pipeline), use existing matplotlib path.
- **Files to modify:**
  - `modules/floor_plan_generator.py`
- **Dependencies:** Steps 3.1, 3.2, 3.6
- **Test:**
  ```bash
  uv run python run_cli.py sample_images/test_room_*.jpg
  # Should generate SVG and DXF alongside PNG
  ls outputs/floor_plan_*.{svg,dxf,png}
  ```
- **Acceptance:** Multiple output formats generated. Old PNG path still works as fallback.

### Step 3.8: Phase 3 integration test
- **Status:** `[ ]`
- **Description:** Create `tests/test_phase3_rendering.py`. Test SVG/DXF output validity. Verify DXF layers. Check dimension values match input model. Verify symbols are present.
- **Files to create:**
  - `tests/test_phase3_rendering.py`
- **Dependencies:** Step 3.7
- **Test:** `uv run python -m pytest tests/test_phase3_rendering.py -v`
- **Acceptance:** All renderer tests pass. DXF is valid. SVG renders in browser.

---

## Phase 4: Door/Window Detection (MEDIUM PRIORITY)

**Goal:** Detect doors and windows from images and place them on the floor plan.
**Validation:** Detected door positions within 20cm of actual. Door widths within 5cm.

### Step 4.1: Implement semantic segmentation module
- **Status:** `[ ]`
- **Description:** Create `modules/detection/opening_detector.py` with `OpeningDetector` class. Uses a pre-trained segmentation model (ADE20K-based SegFormer or SAM2) to identify door and window regions in input images. Returns bounding boxes and masks for each detected opening.
- **Files to modify:**
  - `modules/detection/opening_detector.py` (new file)
  - `requirements.txt` (add segmentation model dependency)
- **Dependencies:** Step 1.1
- **Test:**
  ```python
  from modules.detection.opening_detector import OpeningDetector
  import numpy as np
  det = OpeningDetector()
  # Test with synthetic image (real test needs actual room photo)
  img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
  result = det.detect_openings(img)
  assert isinstance(result, dict)
  assert 'doors' in result and 'windows' in result
  print(f"Detected {len(result['doors'])} doors, {len(result['windows'])} windows  OK")
  ```
- **Acceptance:** Model loads and runs inference. Returns structured detection results. Handles images with no openings gracefully.

### Step 4.2: Project door/window detections to floor plan coordinates
- **Status:** `[ ]`
- **Description:** Add projection logic to `OpeningDetector`. Given 2D bounding boxes of doors/windows in image space, plus camera intrinsics and depth map, compute their position and width in floor plan (ground plane) coordinates. Map each opening to its parent wall segment.
- **Files to modify:**
  - `modules/detection/opening_detector.py` (add `project_to_floor_plan()`)
- **Dependencies:** Steps 4.1, 2.3
- **Test:**
  ```python
  from modules.detection.opening_detector import OpeningDetector
  from modules.geometry.floor_plan_model import DoorOpening, WallSegment
  import numpy as np
  det = OpeningDetector()
  # Simulate a door detection at pixel (200, 300) with width 100px
  door_bbox = {'x': 200, 'y': 100, 'w': 100, 'h': 300}
  depth_map = np.ones((480, 640)) * 3.0  # 3m depth
  wall = WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15)
  opening = det.project_to_floor_plan(door_bbox, depth_map, fx=500, fy=500, parent_wall=wall)
  assert isinstance(opening, DoorOpening)
  assert 0.5 < opening.width < 2.0, f"Door width unreasonable: {opening.width}"
  print(f"Door width: {opening.width:.2f}m at position {opening.position}  OK")
  ```
- **Acceptance:** Door/window projected to correct wall. Width in reasonable range (0.6-1.5m doors, 0.5-3m windows).

### Step 4.3: Add door/window symbols to renderers
- **Status:** `[ ]`
- **Description:** Update SVG and DXF renderers to draw door symbols (gap + 90-degree arc) and window symbols (triple lines in wall gap) when `FloorPlanModel` contains `DoorOpening` and `WindowOpening` entries.
- **Files to modify:**
  - `modules/rendering/svg_renderer.py` (add door/window rendering)
  - `modules/rendering/dxf_renderer.py` (add door/window rendering on A-DOOR, A-GLAZ layers)
  - `modules/rendering/png_renderer.py` (add door/window rendering)
- **Dependencies:** Steps 3.1, 3.2, 3.3, 4.2
- **Test:**
  ```python
  from modules.rendering.svg_renderer import SVGRenderer
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon, DoorOpening
  import numpy as np
  renderer = SVGRenderer()
  model = FloorPlanModel(
      walls=[WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15)],
      rooms=[RoomPolygon(boundary=np.array([[0,0],[4,0],[4,3],[0,3]]), name="Room")],
      doors=[DoorOpening(position=np.array([2, 0]), width=0.9, swing_direction="left")]
  )
  svg = renderer.render_to_string(model)
  assert "arc" in svg.lower() or "path" in svg.lower(), "Missing door arc in SVG"
  print("Door/window symbols render  OK")
  ```
- **Acceptance:** Door gap visible in wall. Arc shows swing direction. Window parallel lines visible.

### Step 4.4: Update FloorPlanModel for openings
- **Status:** `[ ]`
- **Description:** Ensure `FloorPlanModel` properly stores and serializes door/window data. Add chain dimension support: when a wall has doors/windows, split into segments with individual dimensions (e.g., `1.2m | [D 0.9m] | 2.4m`).
- **Files to modify:**
  - `modules/geometry/floor_plan_model.py`
  - `modules/geometry/measurement_engine.py`
- **Dependencies:** Steps 2.1, 2.6
- **Test:**
  ```python
  from modules.geometry.measurement_engine import MeasurementEngine
  from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, DoorOpening
  import numpy as np
  engine = MeasurementEngine()
  wall = WallSegment(start=np.array([0,0]), end=np.array([4,0]), thickness=0.15)
  door = DoorOpening(position=np.array([2, 0]), width=0.9, swing_direction="left")
  model = FloorPlanModel(walls=[wall], doors=[door])
  chain = engine.compute_chain_dimensions(wall, [door])
  assert len(chain) == 3, f"Expected 3 segments (wall-door-wall), got {len(chain)}"
  total = sum(c['length'] for c in chain)
  assert abs(total - 4.0) < 0.1, f"Chain total wrong: {total}"
  print(f"Chain dimensions: {chain}  OK")
  ```
- **Acceptance:** Chain dimensions sum to wall length. Door/window widths included.

### Step 4.5: Wire opening detection into pipeline
- **Status:** `[ ]`
- **Description:** Add opening detection step to `RoomReconstructor`. After wall detection, run `OpeningDetector` on each input image, project detections to floor plan, and add to `FloorPlanModel`.
- **Files to modify:**
  - `modules/room_reconstructor.py`
- **Dependencies:** Steps 4.1-4.4
- **Test:**
  ```bash
  uv run python run_cli.py sample_images/test_room_*.jpg
  # Floor plan should now show doors/windows if detected
  ```
- **Acceptance:** Pipeline runs without errors. Detected openings appear in FloorPlanModel and rendered output.

### Step 4.6: Phase 4 integration test
- **Status:** `[ ]`
- **Description:** Create `tests/test_phase4_openings.py`. Test opening detection, projection, and rendering. Use room photos with visible doors.
- **Files to create:**
  - `tests/test_phase4_openings.py`
- **Dependencies:** Step 4.5
- **Test:** `uv run python -m pytest tests/test_phase4_openings.py -v`
- **Acceptance:** All tests pass. Door positions and widths in expected ranges.

---

## Phase 5: Integration + UI (MEDIUM PRIORITY)

**Goal:** Updated Gradio UI with format selection, download buttons, and measurement editing.
**Validation:** Full end-to-end flow works in Gradio. All output formats downloadable.

### Step 5.1: Add format selection to Gradio UI
- **Status:** `[ ]`
- **Description:** Update `app.py` to add a format selector (checkboxes for PNG, SVG, DXF, PDF). Replace single image output with tabbed format-specific outputs. Add download buttons for each format.
- **Files to modify:**
  - `app.py`
- **Dependencies:** Steps 3.7, 4.5
- **Test:** Start `uv run python app.py`, upload images, verify all format tabs appear and download buttons work.
- **Acceptance:** Format selector works. SVG/DXF/PNG tabs display results. Download buttons functional.

### Step 5.2: Interactive SVG floor plan viewer
- **Status:** `[ ]`
- **Description:** Embed the SVG floor plan directly in Gradio using `gr.HTML()` component. SVG should be pannable/zoomable in the browser. Add hover tooltips showing wall measurements.
- **Files to modify:**
  - `app.py`
- **Dependencies:** Step 5.1
- **Test:** Upload images in Gradio, switch to SVG tab, verify SVG renders interactively.
- **Acceptance:** SVG visible in browser. Pan/zoom works. Measurements readable.

### Step 5.3: User measurement correction input
- **Status:** `[ ]`
- **Description:** Add a "Calibrate" section to Gradio UI. User can input a known dimension (e.g., "Wall A = 3.5m") which triggers the `DepthCalibrator` to adjust all measurements. Reprocesses floor plan with corrected scale.
- **Files to modify:**
  - `app.py`
- **Dependencies:** Steps 1.4, 5.1
- **Test:** Upload images, see measurements, input a correction, verify measurements update.
- **Acceptance:** Calibration input adjusts all measurements proportionally. UI updates without full reprocessing.

### Step 5.4: Updated measurements display
- **Status:** `[ ]`
- **Description:** Replace the simple measurements markdown table with a detailed breakdown: per-wall dimensions, room area, overall dimensions, confidence indicators. Show both old (legacy) and new (metric) measurements during transition period.
- **Files to modify:**
  - `app.py`
- **Dependencies:** Steps 2.6, 5.1
- **Test:** Upload images, verify measurements tab shows per-wall breakdown.
- **Acceptance:** Per-wall measurements visible. Area calculated from polygon, not bounding box.

### Step 5.5: End-to-end test suite
- **Status:** `[ ]`
- **Description:** Create `tests/test_e2e.py`. Full pipeline test: images -> metric depth -> SfM -> point cloud -> wall detection -> room polygon -> measurement -> rendering (all formats). Verify each stage produces output. Performance benchmarks.
- **Files to create:**
  - `tests/test_e2e.py`
- **Dependencies:** All previous phases
- **Test:** `uv run python -m pytest tests/test_e2e.py -v`
- **Acceptance:** Full pipeline test passes. All output files created. No regressions.

### Step 5.6: Documentation update
- **Status:** `[ ]`
- **Description:** Update `README.md`, `ARCHITECTURE.md`, and `CLAUDE.md` to reflect the new pipeline. Document new config options. Update module descriptions. Add examples of new output formats.
- **Files to modify:**
  - `README.md`
  - `ARCHITECTURE.md`
  - `CLAUDE.md`
- **Dependencies:** Step 5.5
- **Test:** Manual review of documentation accuracy.
- **Acceptance:** All docs reflect current system. New modules documented. Config options listed.

### Step 5.7: Final cleanup and deprecation removal
- **Status:** `[ ]`
- **Description:** Remove deprecated code paths: old inverse-depth hack (if metric depth stable), `assumed_room_width` as measurement source (keep as calibration hint), old matplotlib-only rendering (replaced by png_renderer.py). Clean up config.py deprecated constants.
- **Files to modify:**
  - `modules/depth_estimator.py` (mark as legacy)
  - `modules/floor_plan_generator.py` (mark as legacy)
  - `config.py` (remove deprecated constants)
- **Dependencies:** Step 5.5 (all tests pass first)
- **Test:** `uv run python -m pytest tests/ -v` (all tests still pass after cleanup)
- **Acceptance:** No dead code. Clean imports. All tests pass.

---

## Dependency Graph

```
Phase 1 (Metric Depth):
1.1 ─→ 1.2 ─→ 1.3 ─→ 1.6 ─→ 1.7 ─→ 1.8
1.1 ─→ 1.5 ──────────↗
1.3 ─→ 1.4 ──────────↗

Phase 2 (Wall Detection):
1.1 ─→ 2.1
1.1 ─→ 2.2 ─→ 2.3 ─→ 2.4 ─→ 2.5 ─→ 2.7 ─→ 2.8
2.1 ──────────────────→ 2.5
2.1 ─→ 2.6 ──────────────────→ 2.7

Phase 3 (Rendering):
2.1 ─→ 3.1 ──────────→ 3.5 ─→ 3.7 ─→ 3.8
2.1 ─→ 3.2 ──────────→ 3.5
     3.3 ──────────→ 3.5
2.1 ─→ 3.4 ──────────→ 3.6
                       3.6 ──→ 3.7

Phase 4 (Openings):
1.1 ─→ 4.1 ─→ 4.2 ─→ 4.3 ─→ 4.5 ─→ 4.6
2.1 ─→ 4.4 ──────────→ 4.5

Phase 5 (Integration):
3.7 + 4.5 ─→ 5.1 ─→ 5.2
1.4 + 5.1 ─→ 5.3
2.6 + 5.1 ─→ 5.4
all ─→ 5.5 ─→ 5.6 ─→ 5.7
```

---

## Changelog

| Date | Session | Steps Completed | Notes |
|------|---------|----------------|-------|
| 2026-02-09 | Initial | - | Plan created |
| 2026-02-09 | Session 1 | 1.1, 1.2, 1.3, 1.4 | Scaffolding + MetricDepthEstimator (Depth Pro + DA V2 fallback) + DepthCalibrator + FloorPlanModel dataclasses. Depth Pro loads OK, inference test pending (CPU-only, slow). |
| 2026-02-09 | Session 2 | 1.5, 1.6, 1.7, 1.8 | Phase 1 COMPLETE. Config constants, wired metric depth into RoomReconstructor (conditional path + calibration), FloorPlanGenerator metric mode (scale_factor=1.0 bypasses assumed_width), 15/15 integration tests pass. |
| 2026-02-09 | Session 2 | 2.1-2.8 | Phase 2 COMPLETE. WallDetector (RANSAC floor plane, depth gradient LSD lines, Manhattan alignment), RoomSegmenter (Shapely polygon extraction, L-shaped rooms), MeasurementEngine (per-wall lengths, chain dimensions, cross-validation), wired into RoomReconstructor. 32/32 tests pass. Next: Phase 3 (rendering). |
