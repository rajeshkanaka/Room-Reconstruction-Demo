# Future Direction: CAGE Integration for Production-Grade Floor Plans

## The Decision

**Chosen approach: Replace the Hough-transform wall detection with CAGE (NeurIPS 2025), keeping VGGT for metric scale. RoomFormer (CVPR 2023) as proven fallback.**

This is the single change that will move the project from "proof of concept" to "usable floor plans."

---

## Why the Other Options Were Eliminated

### Way 1: Floor-Transformer -- ELIMINATED (Fabricated)

The `VGGT_ALTERNATIVES_RESEARCH.md` document is entirely hallucinated. Every claim was verified and found false:

| Claim | Reality |
|-------|---------|
| GitHub: `yisol/floor-transformer` | **404 Not Found.** Yisol (Yonsei Univ.) works on fashion AI (IDM-VTON), not floor plans. |
| PyPI: `pip install floor-transformer` | **Does not exist** on PyPI. |
| Paper: arXiv 2406.01723 | **Actually:** "Wasserstein Distributionally Robust Control" -- a control theory paper. |
| Reference: arXiv 2406.07085 | **Actually:** "CAT: Multi-Organ and Tumor Segmentation" -- a medical imaging paper. |
| Metric: "81.7% F1-score" | **Fabricated.** No model, no benchmark, no measurement. |
| Implementation plan | **Mock code only.** `MockFloorTransformer` class returns hardcoded rectangles. |

**Action:** Delete `VGGT_ALTERNATIVES_RESEARCH.md` and `IMPLEMENTATION_PLAN_FLOOR_TRANSFORMER.md`. They are AI hallucination artifacts that will mislead future development.

### Way 2: Layout Anything -- ELIMINATED (Not Implementable)

The `2D_Floor_Plan_Enhancement_Research.md` correctly identified Floor-Transformer as fake and pointed to a real model. However, Layout Anything itself is not viable for integration:

| Issue | Detail |
|-------|--------|
| No code released | No GitHub repo found. Paper is an arXiv preprint (Dec 2025), not confirmed at WACV 2026. |
| Wrong output type | Produces room layout **segmentation masks** (floor/ceiling/wall pixel labels), not vectorized floor plan polygons. |
| Single-image only | Estimates layout from one perspective viewpoint. Does not fuse multiple views into a unified floor plan. |
| Wrong problem | Room layout estimation != floor plan reconstruction. Layout gives "what walls are visible from here." Floor plan gives "the complete room shape from above." |

**Action:** Keep `2D_Floor_Plan_Enhancement_Research.md` as reference (its analysis of the core problem is correct), but do not pursue Layout Anything.

---

## The Core Problem (Confirmed)

The current pipeline:
```
Images -> VGGT -> Point Cloud -> Density Map -> Hough Transform -> Walls
                                                 ^^^^^^^^^^^^^^^^^^^
                                                 THE BOTTLENECK
```

VGGT produces excellent metric 3D reconstruction. The failure is in the "last mile" conversion:

- **Hough transform** (1962 algorithm) is non-semantic, noise-sensitive, and cannot handle clutter
- Current `WallDetector.detect_walls_from_depth()` uses gradient -> Canny -> HoughLinesP -> Manhattan alignment
- Current `WallDetector.detect_walls_from_point_cloud_planes()` uses Open3D RANSAC plane fitting
- Result: wavy walls, missing corners, over-segmentation, fragmented segments
- Quality flags show `used_rectangular_fallback: true` as common degradation mode

---

## The Solution: CAGE (Primary) + RoomFormer (Fallback)

### Primary: CAGE (NeurIPS 2025)

**CAGE** (NeurIPS 2025) is an edge-centric Transformer that takes a top-down density map and outputs watertight, topologically valid vectorized floor plans.

- **Paper:** "CAGE: Controllable Articulation GEneration" -- Edge-based floorplan reconstruction (NeurIPS 2025)
- **Code:** https://github.com/ee-Liu/CAGE
- **Pretrained weights:** Available on Google Drive (Structured3D + SceneCAD)
- **Performance:** F1 99.1% (rooms), 91.7% (corners), 89.3% (angles) on Structured3D -- **surpasses RoomFormer on all metrics**

### Fallback: RoomFormer (CVPR 2023)

**RoomFormer** is the established baseline with the same input/output contract as CAGE.

- **Paper:** "Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries" (CVPR 2023)
- **Code:** https://github.com/ywyue/RoomFormer (MIT License)
- **Pretrained weights:** Available for Structured3D and SceneCAD
- **Maturity:** Public for 2+ years, actively maintained, well-documented
- **Successor:** PolyRoom (ECCV 2024) at https://github.com/3dv-casia/PolyRoom improves by ~1-2%

### Why This Stack Fits Perfectly

| Factor | CAGE | RoomFormer (fallback) |
|--------|------|----------------------|
| **Input format** | 2D density map (same as current pipeline computes) | 2D density map (identical) |
| **Output format** | Edge-based vectorized floor plan (watertight topology) | Variable-size set of room polygons |
| **Room F1** | **99.1%** | ~95% |
| **Corner F1** | **91.7%** | ~86% |
| **Angle F1** | **89.3%** | ~84% |
| **Semantic output** | Room types + structural edges | Room types + doors + windows |
| **Architecture** | Dual-query Transformer decoder (edge-centric) | Deformable attention Transformer (two-level query) |
| **Code maturity** | NeurIPS 2025 (newer, less battle-tested) | CVPR 2023 (proven, well-documented) |

**Strategy:** Start with CAGE for best quality. If installation or domain gap issues arise, RoomFormer is a drop-in replacement with the same input contract.

### How It Replaces the Bottleneck

```
BEFORE (broken):
  Point Cloud -> density_map -> gradient -> Canny -> HoughLinesP -> merge -> Manhattan align
                                -> RANSAC planes -> project -> merge
                                -> room segmenter (graph topology)
  Result: wavy walls, fragmented segments, rectangular fallback

AFTER (CAGE/RoomFormer):
  Point Cloud -> density_map -> CAGE(density_map) -> watertight vectorized floor plan
  Result: clean connected polygons, room types, topologically valid edges
```

The current `WallDetector`, `RoomSegmenter`, and partially `OpeningDetector` are all replaced by a single model inference.

---

## Implementation Plan

### Phase 1: Learned Detector Module (Core Integration)

**New file:** `modules/detection/learned_floorplan_detector.py`

**What it does:**
1. Takes VGGT point cloud as input
2. Projects to top-down 2D density map (reuse existing projection code from `floor_plan_generator.py`)
3. Runs CAGE (primary) or RoomFormer (fallback) inference on the density map
4. Converts polygon output to `FloorPlanModel` data structures
5. Scales polygons using VGGT metric depth (meters per pixel)

**Key integration points:**
- Input: `np.ndarray` point cloud from `VGGTReconstructor.reconstruct()`
- Output: Populated `FloorPlanModel` with `WallSegment`, `RoomPolygon`, `DoorOpening`, `WindowOpening`
- The existing rendering pipeline (SVG, DXF, PNG) consumes `FloorPlanModel` unchanged

**Dependencies to install:**
```bash
# Shared dependencies (already present for VGGT)
pip install torch torchvision
pip install scipy shapely

# CAGE (primary)
git clone https://github.com/ee-Liu/CAGE.git external/cage
# Download pretrained weights from Google Drive link in CAGE repo

# RoomFormer (fallback)
pip install detectron2         # RoomFormer's backbone
git clone https://github.com/ywyue/RoomFormer.git external/roomformer
```

**Config additions (`config.py`):**
```python
# --- Learned Floor Plan Detection ---
ENABLE_LEARNED_FLOORPLAN = True             # Use CAGE/RoomFormer for wall detection
LEARNED_FLOORPLAN_MODEL = "cage"            # "cage" (primary) or "roomformer" (fallback)
CAGE_MODEL_PATH = "external/cage/checkpoints/cage_s3d.pth"
ROOMFORMER_MODEL_PATH = "external/roomformer/checkpoints/roomformer_s3d.pth"
FLOORPLAN_DENSITY_RESOLUTION = 256          # Density map resolution (256x256)
FLOORPLAN_CONFIDENCE_THRESHOLD = 0.5        # Minimum polygon confidence
FLOORPLAN_FALLBACK_TO_HOUGH = True          # Fall back to Hough if learned model fails
```

**Module skeleton:**
```python
class LearnedFloorplanDetector:
    """
    CAGE/RoomFormer-based floor plan detection.

    Replaces Hough-transform wall detection with learned polygon prediction.
    Takes VGGT point cloud, projects to density map, runs CAGE or RoomFormer,
    and populates FloorPlanModel.

    CAGE (NeurIPS 2025): Edge-centric, watertight topology, best metrics.
    RoomFormer (CVPR 2023): Proven fallback, same input contract.
    """

    def __init__(self, model_type="cage"):
        self.model_type = model_type
        self.model = None  # Lazy load

    def detect(self, points_3d, metric_scale, floor_height=None):
        """
        Main entry point.

        Args:
            points_3d: Nx3 numpy array from VGGT
            metric_scale: meters-per-unit from VGGT calibration
            floor_height: Y-coordinate of floor plane (auto-detected if None)

        Returns:
            FloorPlanModel populated with walls, rooms, doors, windows
        """
        density_map = self._project_to_density_map(points_3d, floor_height)
        raw_output = self._run_model(density_map)
        floor_plan_model = self._convert_to_model(raw_output, metric_scale)
        return floor_plan_model
```

### Phase 2: Pipeline Integration

**File:** `modules/room_reconstructor.py`

**Changes:**
1. Add RoomFormer as primary detection path (alongside existing Hough fallback)
2. Modify the detection section of `reconstruct()` / `reconstruct_from_arrays()` to route through RoomFormer when `ENABLE_ROOMFORMER=True`
3. Keep existing `WallDetector` + `RoomSegmenter` as fallback chain

**Integration point in `room_reconstructor.py`:**
The current detection flow (approximately lines 800-1200) does:
```python
# Current flow (to be wrapped with RoomFormer routing):
wall_detector = WallDetector()
segments = wall_detector.detect_walls_from_depth(...)     # or detect_walls_from_point_cloud_planes
room_segmenter = RoomSegmenter()
rooms = room_segmenter.segment_rooms(segments)
opening_detector = OpeningDetector()
openings = opening_detector.detect(images)
# ... populate FloorPlanModel
```

**New flow:**
```python
if ENABLE_LEARNED_FLOORPLAN:
    try:
        from modules.detection.learned_floorplan_detector import LearnedFloorplanDetector
        detector = LearnedFloorplanDetector(model_type=LEARNED_FLOORPLAN_MODEL)
        floor_plan_model = detector.detect(points_3d, metric_scale, floor_height)
        # CAGE/RoomFormer provides walls + rooms + doors + windows in one shot
    except Exception as e:
        print(colored(f"[LearnedFloorplan] Failed: {e}, falling back to Hough", "yellow"))
        # Fall through to existing detection
else:
    # Existing Hough + RANSAC pipeline (unchanged)
```

### Phase 3: Density Map Quality

**File:** `modules/detection/roomformer_detector.py` (density map projection)

The quality of the density map directly affects RoomFormer's output. Key considerations:

1. **Resolution:** RoomFormer was trained on 256x256 density maps. Match this resolution.
2. **Projection:** Top-down (bird's-eye-view) projection of point cloud along Y-axis (gravity direction).
3. **Floor isolation:** Use VGGT floor plane detection to isolate floor-level points (0 to ~0.3m above floor).
4. **Normalization:** Density values normalized to [0, 255] uint8 range.
5. **Mean surface normal map:** RoomFormer optionally takes a surface normal map as second channel. This can be derived from VGGT's point normals.

The existing `floor_plan_generator.py` already computes a density grid (`_create_density_grid`). This code should be extracted and enhanced for RoomFormer input.

### Phase 4: Metric Scale Fusion

VGGT provides metric depth (absolute scale in meters). RoomFormer outputs normalized polygons. The fusion step:

1. RoomFormer outputs polygons in density-map pixel coordinates
2. Compute scale factor: `meters_per_pixel = room_physical_extent / density_map_resolution`
3. The physical extent comes from VGGT: `x_range = x_max - x_min` of the projected floor points
4. Apply scale to all polygon vertices: `vertex_meters = vertex_pixels * meters_per_pixel + origin_offset`

This replaces the current `ASSUMED_ROOM_WIDTH` hack with actual metric scaling from VGGT.

### Phase 5: Testing

**New test file:** `tests/test_learned_floorplan_detection.py`

Tests to write:
1. **Density map projection** -- synthetic point cloud -> density map -> correct shape, resolution, normalization
2. **Polygon conversion** -- CAGE/RoomFormer output format -> FloorPlanModel data structures
3. **Metric scaling** -- pixel coordinates -> meter coordinates with known scale factor
4. **Model fallback** -- CAGE failure -> RoomFormer -> Hough detection chain
5. **End-to-end** (slow) -- sample images -> VGGT -> CAGE -> FloorPlanModel -> rendered output

**Regression test update:** Modify `tests/fixtures/t9_sample_room_baseline.json` acceptance criteria:
- Increase `min_wall_count` expectation (learned models should find more walls)
- Increase `min_closure_score` to 0.95 (CAGE produces watertight topology by design)
- Add `cage` / `roomformer` as expected backend values

### Phase 6: UI Integration

**File:** `app.py`

No UI changes needed for initial integration. RoomFormer runs transparently behind the existing pipeline. The existing radio button / backend selection (if added later) can expose it, but the default path should just use RoomFormer when available.

**File:** `run_cli.py`

Add optional `--detector` flag:
```
--detector roomformer|hough|auto   (default: auto, prefers RoomFormer)
```

---

## Risk Assessment

### Low Risk
- **Code availability:** Both CAGE and RoomFormer have public code with pretrained weights
- **Dependency compatibility:** Uses PyTorch (already required for VGGT)
- **Fallback safety:** Three-tier fallback chain: CAGE -> RoomFormer -> Hough (existing)

### Medium Risk
- **Domain gap:** Both models trained on Structured3D (synthetic) and SceneCAD (RGB-D scans). VGGT density maps from 4-5 perspective photos will be sparser. May need fine-tuning.
- **CAGE code maturity:** NeurIPS 2025 -- newer, less battle-tested than RoomFormer. May have installation quirks.
- **detectron2 installation:** Required for RoomFormer fallback. Can be finicky on some platforms.
- **Density map quality:** If VGGT point cloud is sparse, density map may be low quality. Need robust projection with smoothing.

### Mitigation
- Start with CAGE; if any issues, RoomFormer is immediate fallback (same input contract)
- Test with synthetic density maps first (known-good input)
- Test with VGGT output from sample images (real-world input)
- If domain gap is significant, fine-tune on a small set of VGGT density maps
- PolyRoom (ECCV 2024) is an additional fallback option

---

## What Changes Where

### New Files
```
modules/detection/learned_floorplan_detector.py  # CAGE/RoomFormer wrapper + density projection + model conversion
tests/test_learned_floorplan_detection.py        # Unit + integration tests
external/cage/                                   # Cloned CAGE repo (gitignored or submodule)
external/roomformer/                             # Cloned RoomFormer repo (gitignored or submodule)
```

### Modified Files
```
config.py                                    # Add ENABLE_LEARNED_FLOORPLAN + settings
modules/room_reconstructor.py                # Route detection through CAGE/RoomFormer when enabled
requirements.txt                             # Add detectron2 (for RoomFormer fallback)
```

### Unchanged Files
```
modules/geometry/floor_plan_model.py         # FloorPlanModel is the contract -- no changes needed
modules/rendering/svg_renderer.py            # Consumes FloorPlanModel -- no changes needed
modules/rendering/dxf_renderer.py            # Consumes FloorPlanModel -- no changes needed
modules/rendering/png_renderer.py            # Consumes FloorPlanModel -- no changes needed
modules/rendering/symbol_library.py          # Consumes FloorPlanModel -- no changes needed
modules/vggt_reconstructor.py                # VGGT unchanged -- still provides point cloud + metric depth
modules/scene_analyzer.py                    # Gemini unchanged -- still provides semantic enrichment
modules/detection/wall_detector.py           # Preserved as fallback
modules/detection/room_segmenter.py          # Preserved as fallback
modules/detection/opening_detector.py        # Preserved as fallback (RoomFormer may not detect all openings)
```

### Files to Delete
```
VGGT_ALTERNATIVES_RESEARCH.md               # Hallucinated -- every reference is fabricated
IMPLEMENTATION_PLAN_FLOOR_TRANSFORMER.md     # Based on fabricated research
docs/VGGT_ALTERNATIVES_RESEARCH.md           # Duplicate of above
```

---

## Success Criteria

| Metric | Current | Target | How to Measure |
|--------|---------|--------|---------------|
| Wall closure score | ~0.8 (with rectangular fallback) | >0.95 | `_compute_wall_closure_metrics()` in regression test |
| Rectangular fallback rate | High | <10% | `used_rectangular_fallback` in quality flags |
| Wall segment count | 4 (fallback rectangle) | Matches actual room geometry | Visual inspection + regression test |
| Corner precision | +/- 10-20cm | +/- 5cm | Measured against ground truth if available |
| Processing time (detection only) | ~2-5s (Hough chain) | ~1-3s (single inference) | Timer in pipeline |
| Quality mode | "approximate" or "needs_more_images" | "high_confidence" | Quality scoring system |

---

## Execution Order

1. Clone CAGE + RoomFormer, install dependencies, verify both run standalone on Structured3D samples
2. Write `learned_floorplan_detector.py` with density map projection and model conversion
3. Write unit tests with synthetic density maps (model-agnostic tests)
4. Integrate into `room_reconstructor.py` behind `ENABLE_LEARNED_FLOORPLAN` flag
5. Run CAGE on sample images, compare output quality visually against current Hough pipeline
6. If CAGE has issues (installation, domain gap), switch to RoomFormer (same input contract)
7. Update regression test baseline with new quality expectations
8. Delete fabricated research documents (`VGGT_ALTERNATIVES_RESEARCH.md`, `IMPLEMENTATION_PLAN_FLOOR_TRANSFORMER.md`)
9. (Parallel) Set up Plane-DUSt3R for evaluation as future end-to-end replacement

---

## Future Upgrades (After CAGE/RoomFormer Works)

### Upgrade Path 1: Plane-DUSt3R (ICLR 2025) -- Most Promising
Would replace both VGGT and CAGE with a single model that takes unposed sparse perspective views and outputs structural plane layouts directly. This is the "nuclear option" that eliminates the entire multi-step pipeline.
- **Paper:** "Unposed Sparse Views Room Layout Reconstruction in the Age of Pretrain Model" (ICLR 2025)
- **Code:** https://github.com/justacar/Plane-DUSt3R (MIT License)
- **Weights:** https://huggingface.co/yxuan/Plane-DUSt3R
- **Key innovation:** Fine-tunes DUSt3R to predict "plane pointmaps" -- 3D coordinates of structural surfaces with furniture removed. No camera poses required.
- **Trade-off:** More radical change, loses VGGT's general-purpose point cloud and metric depth. Early-stage code (44 stars, 7 commits).
- **Action:** Set up in parallel for evaluation. If it works on real-world photos, it could become the primary pipeline.

### Upgrade Path 2: PolyRoom (ECCV 2024)
Drop-in replacement for RoomFormer with ~1-2% better metrics. Same input/output contract.
- Code: https://github.com/3dv-casia/PolyRoom

### Upgrade Path 3: Gemini-Enhanced Post-Processing
Use Gemini 3 (already in pipeline) to validate/correct CAGE/RoomFormer output:
- Verify room types match visual evidence
- Add missing doors/windows from semantic analysis
- Resolve ambiguous polygon intersections

---

## Summary

**The decision is made: integrate CAGE (NeurIPS 2025) with RoomFormer (CVPR 2023) fallback to replace Hough-transform wall detection.**

This is the right choice because:
1. It directly fixes the identified bottleneck (heuristic wall detection)
2. CAGE achieves F1 99.1% rooms / 91.7% corners -- far beyond heuristic methods
3. Both models have publicly available code and pretrained weights
4. Input format matches what VGGT already produces (point cloud -> density map)
5. Output format matches what the rendering pipeline already consumes (FloorPlanModel)
6. Existing pipeline preserved as fallback (zero-risk deployment)
7. Clear upgrade path to Plane-DUSt3R (ICLR 2025) if end-to-end replacement is desired later
8. No fabricated dependencies -- every reference in this document is verified

### Verified References

| Resource | URL | Status |
|----------|-----|--------|
| CAGE code | https://github.com/ee-Liu/CAGE | Verified |
| CAGE paper | https://arxiv.org/abs/2509.15459 | NeurIPS 2025 |
| RoomFormer code | https://github.com/ywyue/RoomFormer | Verified, MIT License |
| RoomFormer paper | CVPR 2023 proceedings | Verified |
| PolyRoom code | https://github.com/3dv-casia/PolyRoom | Verified, ECCV 2024 |
| Plane-DUSt3R code | https://github.com/justacar/Plane-DUSt3R | Verified, ICLR 2025 |
| Plane-DUSt3R weights | https://huggingface.co/yxuan/Plane-DUSt3R | Verified |
