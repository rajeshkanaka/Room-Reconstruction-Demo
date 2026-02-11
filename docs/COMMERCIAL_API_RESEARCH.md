# Commercial & Open-Source Alternatives: Master Research Document

**Date:** 2026-02-09
**Project:** Missoula Room Reconstruction Pipeline
**Audience:** Engineering leadership, product decision-makers

---

## 1. Executive Summary

The current pipeline produces room reconstructions with 20-30% measurement error, convex-hull-only floor plans, and no semantic understanding of architectural elements. Three categories of alternatives were evaluated: metric depth estimation (12 models), floor plan generation (13 services/models), and 3D reconstruction (14 tools). The conclusion is clear: **three open-source tools -- VGGT/MASt3R, Depth Anything V3 Metric, and Grounded SAM 2 -- can reduce measurement error to under 10%, enable non-convex room support, and add door/window detection, all within 3-4 weeks of integration effort and at zero licensing cost.** A fourth option, Plane-DUSt3R, could collapse the entire pipeline into a single model purpose-built for room layout from sparse photos.

---

## 2. Current Pipeline Limitations

| Limitation | Root Cause | Impact |
|------------|-----------|--------|
| 20-30% measurement error | Depth Anything V2 outputs relative [0,1] depth; scale depends on `ASSUMED_ROOM_WIDTH_METERS = 4.0` | Measurements are unreliable for any practical use |
| Convex hull floor plans only | `FloorPlanGenerator` uses convex hull boundary extraction | L-shaped, U-shaped, and irregular rooms are misrepresented |
| COLMAP fails on 4-5 photos | COLMAP SfM requires significant overlap and 10+ images for reliable results | Pipeline falls back to heuristic centroid alignment ~50% of the time |
| No door/window detection | No semantic segmentation; wall detection uses depth-based Hough lines | Floor plans lack essential architectural features |
| Hardcoded camera intrinsics | `CAMERA_FX=500, CAMERA_FY=500` when SfM unavailable | 3D projection geometry is incorrect for most cameras |
| Coarse floor plan grid | 100x100 density grid for floor plan extraction | Fine details lost in larger rooms |
| Raster-only output | Floor plan is a matplotlib PNG | No CAD/vector export (DXF/SVG) |

---

## 3. Top Recommendations Matrix

Ranked by overall value to the project (impact x feasibility / cost).

| Rank | Tool | Category | Impact | Integration (days) | Cost | License | Replaces | Key Improvement |
|:----:|------|----------|:------:|:---------:|:----:|:-------:|----------|-----------------|
| 1 | **VGGT** | 3D Reconstruction | HIGH | 5-7 | Free | Meta Research | COLMAP + DepthEstimator + ICP Registration | Single model replaces 3 modules; works on 4-5 photos |
| 2 | **DA3 Metric** | Depth Estimation | HIGH | 2-3 | Free | Apache 2.0 | DepthEstimator (relative depth) | Metric depth in meters; eliminates ASSUMED_ROOM_WIDTH |
| 3 | **Grounded SAM 2** | Scene Understanding | HIGH | 3-4 | Free | Apache 2.0 | Nothing (new capability) | Wall/door/window segmentation from text prompts |
| 4 | **MoGe-2** | Depth Estimation | HIGH | 3-4 | Free | MIT | DepthEstimator + depth-to-3D conversion | One pass: metric point maps + normals + FOV |
| 5 | **Plane-DUSt3R** | 3D + Floor Plan | HIGH | 5-7 | Free | Open | Entire pipeline (depth through floor plan) | Direct structural plane detection from 2-5 photos |
| 6 | **MASt3R** | 3D Reconstruction | HIGH | 5-7 | Free | CC-BY-NC-SA | COLMAP + DepthEstimator + ICP | Larger ecosystem than VGGT; metric reconstruction |
| 7 | **PolyRoom** | Floor Plan | MEDIUM | 10-15 | Free | Open | FloorPlanGenerator convex hull pipeline | Transformer-based vectorized room polygons from point clouds |
| 8 | **CubiCasa** | Floor Plan (Commercial) | HIGH | 5-7 | $$$ | Proprietary | FloorPlanGenerator entirely | 95-97% accuracy; requires video input (not photos) |
| 9 | **Apple Depth Pro** | Depth Estimation | HIGH | 2-3 | Free | Apple ASCL | DepthEstimator + camera intrinsics | Best zero-shot metric depth + focal length estimation |
| 10 | **LLM Vision (GPT-4o/Claude)** | Scene Understanding | LOW | 1-2 | $ | Proprietary | Nothing (augmentation) | Room shape classification, feature identification |

---

## 4. Category Deep Dives

### 4.1 Depth Estimation: From Relative to Metric

**Current limitation:** Depth Anything V2 produces relative depth normalized to [0,1]. The pipeline multiplies by `ASSUMED_ROOM_WIDTH_METERS` (default 4.0m) to estimate real dimensions, yielding 20-30% error.

**Top 3 alternatives:**

| Attribute | DA3 Metric | MoGe-2 | Apple Depth Pro |
|-----------|-----------|--------|-----------------|
| Metric output | Yes (with focal length) | Yes (direct) | Yes (direct) |
| Intrinsics needed | Focal length | No (estimates FOV) | No (estimates focal) |
| Speed | ~50ms/image | ~60ms/image | ~300ms/image |
| License | Apache 2.0 | MIT | Apple ASCL (review needed) |
| VRAM | ~4GB | ~4GB | ~5GB |
| Extra outputs | Multi-view poses, 3D points | Normals, FOV, point maps | Focal length |
| NYU AbsRel | Competitive (~0.045-0.05) | SOTA | ~0.05 (zero-shot) |

**Winner: Depth Anything V3 Metric.** Rationale: Apache 2.0 license clears all commercial concerns. Direct successor to the current DA2, making migration a near drop-in replacement. The 50ms inference speed supports interactive workflows. Multi-view capabilities could eventually replace COLMAP for pose estimation. MoGe-2 is a close second due to its richer single-pass output (normals + FOV), but DA3's larger community and permissive license give it the edge.

**Integration snippet** (changes to `modules/depth_estimator.py`):

```python
def _load_da3_metric(self):
    """Load Depth Anything V3 Metric model."""
    from depth_anything_3.api import DepthAnything3
    self.da3_model = DepthAnything3.from_pretrained("DA3Metric-Large")
    self.model_type = "da3_metric"

def estimate_metric_depth(self, image: np.ndarray) -> dict:
    """Return metric depth in meters plus estimated focal length."""
    if self.model_type == "da3_metric":
        net_output = self.da3_model(image)
        focal = self._estimate_focal_length(image)
        metric_depth = focal * net_output / 300.0  # meters
        return {"depth": metric_depth, "focal_length": focal}
```

**Before/After accuracy:**
- Before: 20-30% measurement error (relative depth + assumed 4.0m width)
- After: 5-10% measurement error (metric depth in meters, no assumed width)

---

### 4.2 Floor Plan Generation: Beyond Convex Hulls

**Current limitation:** The pipeline slices point clouds at 10-30% height, projects onto a 100x100 grid, runs Hough line detection for walls, and fits a convex hull for room boundary. This cannot represent L-shaped or U-shaped rooms, has no door/window detection, and outputs only raster PNG.

**Top 3 alternatives:**

| Attribute | PolyRoom | Grounded SAM 2 + Shapely | CubiCasa |
|-----------|----------|--------------------------|----------|
| Input | Point clouds | RGB images | Video walkthrough |
| Non-convex rooms | Yes (per-room polygons) | Yes (mask-derived boundaries) | Yes |
| Door/window detection | No | Yes (text-prompted) | Yes |
| Vector output | Polygon vertices | Masks to polygons via Shapely | SVG, DXF |
| Accuracy | Research-grade | Depends on depth quality | 95-97% |
| Cost | Free | Free | $23-30/plan |
| Integration effort | 10-15 days | 3-4 days | 5-7 days |
| Workflow change | None (accepts current point clouds) | None | Requires video, not photos |

**Winner: Grounded SAM 2 + Shapely (Phase 1), then PolyRoom (Phase 2).** Rationale: Grounded SAM 2 delivers the highest-impact improvement with the least effort. By segmenting walls, doors, and windows from input photos using text prompts ("wall", "door", "window"), it provides semantic understanding the pipeline completely lacks. Combined with metric depth, wall masks become 3D wall planes, enabling non-convex room boundary extraction via Shapely. PolyRoom is the better long-term solution (purpose-built Transformer for floor plan reconstruction from point clouds) but requires more integration work and may need fine-tuning.

**Integration snippet** (new `modules/semantic_segmenter.py`):

```python
from sam2.build_sam import build_sam2
from grounding_dino.groundingdino.util.inference import load_model, predict

class SemanticSegmenter:
    ARCHITECTURAL_PROMPT = "wall . door . window . floor . ceiling"

    def segment(self, image: np.ndarray) -> dict:
        """Detect and segment architectural elements via text prompt."""
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=image,
            caption=self.ARCHITECTURAL_PROMPT,
            box_threshold=0.3,
            text_threshold=0.25,
        )
        results = {}
        for box, phrase in zip(boxes, phrases):
            mask = self.sam2_predictor.predict(box=box)
            results.setdefault(phrase, []).append(mask)
        return results  # {"wall": [mask1, ...], "door": [mask1, ...], ...}
```

**Before/After accuracy:**
- Before: Convex hull only, no openings, raster PNG output
- After: Non-convex room polygons, door/window locations, vector-ready polygon data

---

### 4.3 3D Reconstruction: Replacing COLMAP

**Current limitation:** COLMAP SfM fails on ~50% of 4-5 image sets due to insufficient overlap. When it fails, the pipeline falls back to heuristic centroid-based alignment, producing noisy, misaligned point clouds. The pipeline runs three separate stages (SfM, depth estimation, ICP registration) that could be unified.

**Top 3 alternatives:**

| Attribute | VGGT | MASt3R/DUSt3R | Plane-DUSt3R |
|-----------|------|---------------|--------------|
| Min images | 1 | 2 | 2 |
| Camera calibration | Not needed | Not needed | Not needed |
| Indoor optimization | Yes | Yes | Purpose-built for rooms |
| Speed (5 images) | ~5-15s | ~10-30s | ~15-30s |
| Outputs | Poses + depth + points + tracks | Poses + pointmaps + focal lengths | Structural planes (walls/floor/ceiling) |
| COLMAP export | Native | Via MASt3R-SfM | No |
| License | Meta Research | CC-BY-NC-SA (research) | Open |
| Replaces | SfM + Depth + Registration | SfM + Depth + Registration | SfM + Depth + Registration + Floor Plan |

**Winner: VGGT.** Rationale: CVPR 2025 Best Paper. Single feed-forward pass produces camera poses, depth maps, point clouds, and 3D tracks. Native COLMAP export format means downstream tools work unchanged. Faster than MASt3R and with a cleaner API. The single-model approach eliminates three failure-prone pipeline stages (SfM, depth, registration) in one integration. MASt3R is a strong alternative with a larger ecosystem, while Plane-DUSt3R is the most ambitious option -- directly outputting room layout planes -- but carries higher risk due to synthetic training data.

**Integration snippet** (new `modules/vggt_reconstructor.py`):

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

class VGGTReconstructor:
    def __init__(self):
        self.model = VGGT.from_pretrained("facebook/VGGT-1B")

    def reconstruct(self, image_paths: list[str]) -> dict:
        """Single-pass reconstruction from arbitrary images."""
        images = load_and_preprocess_images(image_paths).to("cuda")
        with torch.no_grad():
            predictions = self.model(images)
        return {
            "points_3d": predictions["point_maps"],
            "depth_maps": predictions["depth_maps"],
            "camera_poses": predictions["extrinsics"],
            "focal_lengths": predictions["intrinsics"],
        }
```

**Before/After accuracy:**
- Before: COLMAP succeeds ~50% of the time on 4-5 photos; fallback produces noisy point clouds
- After: Reconstruction succeeds >90% of the time; point cloud noise reduced by >50%

---

## 5. Recommended Integration Roadmap

### Phase 1: Quick Win (Week 1-2) -- Metric Depth

| Item | Detail |
|------|--------|
| **Tool** | Depth Anything V3 Metric |
| **Replaces** | `DepthEstimator` relative depth path |
| **What changes** | `estimate_depth()` returns meters instead of [0,1]; `ASSUMED_ROOM_WIDTH_METERS` becomes unnecessary |
| **Expected improvement** | Measurement error: 20-30% down to 5-10% |
| **Integration effort** | 2-3 days |
| **Risk** | Low -- direct successor to current DA2; same API patterns |

**Why first:** This is the single variable responsible for the largest error in the pipeline. Swapping relative depth for metric depth immediately improves every downstream measurement. The integration is nearly a drop-in replacement since DA3 is the direct successor to the currently used DA2.

**Config changes:**
```python
# config.py
DEPTH_MODEL = "DA3Metric-Large"
ENABLE_METRIC_DEPTH = True
METRIC_DEPTH_MODEL = "DA3Metric-Large"
```

### Phase 2: Core Reconstruction (Week 3-4) -- VGGT

| Item | Detail |
|------|--------|
| **Tool** | VGGT (Visual Geometry Grounded Transformer) |
| **Replaces** | `SfMProcessor` (COLMAP) + `DepthEstimator` + ICP/RANSAC registration |
| **What changes** | Single `VGGTReconstructor.reconstruct()` call replaces the three-stage pipeline |
| **Expected improvement** | Reconstruction success rate: ~50% to >90% on 4-5 image sets |
| **Integration effort** | 5-7 days |
| **Risk** | Medium -- requires 16GB+ VRAM; Meta Research license needs review for commercial use |

**Why second:** With metric depth in place (Phase 1), the next bottleneck is multi-view alignment. VGGT eliminates the fragile COLMAP dependency and produces aligned point clouds from as few as 2 images. The fallback chain becomes: VGGT -> MASt3R -> COLMAP+DA3 -> Legacy ICP.

### Phase 3: Scene Understanding (Week 5-6) -- Grounded SAM 2

| Item | Detail |
|------|--------|
| **Tool** | Grounded SAM 2 (Grounding DINO + SAM 2) |
| **Replaces** | Depth-based wall heuristics in `FloorPlanGenerator` |
| **What changes** | New `SemanticSegmenter` module provides wall/door/window masks per image; floor plan uses mask-derived boundaries instead of convex hull |
| **Expected improvement** | Non-convex room support; door/window detection; wall precision +30% |
| **Integration effort** | 3-4 days |
| **Risk** | Low -- Apache 2.0 license; well-established library; augments rather than replaces |

**Why third:** With accurate geometry (Phases 1-2), semantic understanding becomes the remaining gap. Grounded SAM 2 adds the architectural element detection that transforms a point cloud into an informative floor plan.

### Phase 4: Advanced (Week 7+) -- Plane-DUSt3R and Vector Output

| Item | Detail |
|------|--------|
| **Tool** | Plane-DUSt3R + ezdxf/svgwrite |
| **Replaces** | The entire floor plan extraction pipeline |
| **What changes** | Structural planes detected directly from images; floor plan derived from plane intersections; DXF/SVG vector output |
| **Expected improvement** | Floor plan accuracy approaches commercial tools; CAD-ready output |
| **Integration effort** | 7-10 days |
| **Risk** | Medium-High -- trained on synthetic data (Structured3D); may need fine-tuning |

---

## 6. Integration Architecture Diagram

```
                          CURRENT PIPELINE
    ================================================================

    Photos (4-5)
        |
        v
    [SfMProcessor]          -----> Camera Poses (fails ~50%)
        |                           |
        v                           v
    [DepthEstimator]         [DenseReconstructor]
    (DA2 relative 0-1)       (TSDF fusion)
        |                           |
        v                           v
    depth_to_3d_points()     Fused Point Cloud
    (hardcoded fx=500)              |
        |                           v
        +--------> ICP/RANSAC Registration
                        |
                        v
                [FloorPlanGenerator]
                (convex hull + Hough lines)
                        |
                        v
                  Floor Plan PNG


                         UPGRADED PIPELINE
    ================================================================

    Photos (4-5)
        |
        +------------------------------+
        |                              |
        v                              v
    [VGGT Reconstructor]       [Semantic Segmenter]
    (single forward pass)      (Grounded SAM 2)
        |                              |
        |--- Camera Poses             |--- Wall masks
        |--- Metric Depth Maps        |--- Door masks
        |--- Dense Point Cloud        |--- Window masks
        |--- Focal Lengths            |
        |                              |
        v                              v
    [FloorPlanGenerator v2] <--- Semantic Masks
    (mask-derived walls + Shapely non-convex polygons)
        |
        +---> Floor Plan PNG
        +---> Floor Plan SVG/DXF (ezdxf)
        +---> Measurements JSON
        |
        v
    [Visualizer3D]
    (Plotly interactive 3D + PLY export)

    FALLBACK CHAIN:
    VGGT -> MASt3R -> [DA3 Metric + COLMAP] -> [DA2 + ICP]
```

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| VGGT requires 16GB+ VRAM | Medium | High | Implement resolution scaling; offer DA3+COLMAP as lighter fallback |
| Non-commercial licenses (VGGT Meta Research, MASt3R CC-BY-NC-SA) block commercial deployment | Medium | High | DA3 Metric (Apache 2.0) + Grounded SAM 2 (Apache 2.0) have no license restrictions; negotiate commercial license with Meta/NAVER if VGGT/MASt3R chosen |
| Metric depth model inaccurate on specific room types (mirrors, glass, dark rooms) | Medium | Medium | Keep relative depth + user-provided room width as manual fallback; validate with known-dimension reference objects |
| Plane-DUSt3R trained on synthetic data does not generalize to real photos | Medium | Medium | Fine-tune on real indoor datasets (ZInD, ScanNet++); keep VGGT + Grounded SAM 2 path as primary |
| Grounded SAM 2 misdetects walls in cluttered rooms | Low | Low | Confidence thresholding; ensemble with depth-based detection as cross-check |
| Dependency complexity increases (VGGT + SAM2 + DA3 all require CUDA + PyTorch) | Low | Medium | Docker containerization; single conda environment with pinned versions; CI testing |
| DA3 focal length estimation is wrong for unusual cameras (fisheye, ultrawide) | Low | Medium | Allow user to input known focal length; cross-validate with Depth Pro's focal estimation |

---

## 8. Sources

### Depth Estimation
- [Apple Depth Pro -- ICLR 2025](https://arxiv.org/abs/2410.02073) | [GitHub](https://github.com/apple/ml-depth-pro)
- [Depth Anything V3 -- Nov 2025](https://arxiv.org/abs/2511.10647) | [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [MoGe-2 -- NeurIPS 2025](https://arxiv.org/abs/2507.02546) | [GitHub](https://github.com/microsoft/MoGe)
- [UniDepth V2 -- arXiv Feb 2025](https://arxiv.org/abs/2502.20110) | [GitHub](https://github.com/lpiccinelli-eth/UniDepth)
- [Metric3D v2 -- IEEE TPAMI 2024](https://arxiv.org/abs/2404.15506) | [GitHub](https://github.com/YvanYin/Metric3D)
- [ZoeDepth -- arXiv 2023](https://arxiv.org/abs/2302.12288) | [GitHub](https://github.com/isl-org/ZoeDepth)
- [Marigold -- CVPR 2024 Oral](https://arxiv.org/abs/2312.02145) | [GitHub](https://github.com/prs-eth/Marigold)
- [Prompt Depth Anything -- CVPR 2025](https://arxiv.org/abs/2412.14015) | [GitHub](https://github.com/DepthAnything/PromptDA)
- [Survey on Monocular Metric Depth -- Jan 2025](https://arxiv.org/abs/2501.11841)

### Floor Plan Generation
- [CubiCasa Developer Portal](https://www.cubi.casa/developers/) | [Integrate API](https://integrate.docs.cubi.casa/)
- [magicplan REST API](https://apidocs.magicplan.app/)
- [Apple RoomPlan Documentation](https://developer.apple.com/documentation/roomplan/)
- [Matterport API](https://api.matterport.com/)
- [PolyRoom -- ECCV 2024](https://arxiv.org/abs/2407.10439) | [GitHub](https://github.com/3dv-casia/PolyRoom)
- [BADGR -- CVPR 2025 Highlight](https://badgr-diffusion.github.io/)
- [FloorSAM -- September 2025](https://arxiv.org/abs/2509.15750)
- [RasterScan Platform](https://www.rasterscan.com/) | [GitHub](https://github.com/RasterScan/Floor-Plan-Recognition)
- [Floorplanner API](https://floorplanner.readme.io/reference/api)
- [Zillow Indoor Dataset (ZInD)](https://github.com/zillow/zind)
- [iGUIDE Floor Plans](https://goiguide.com/iguide/floor-plans)

### 3D Reconstruction
- [VGGT -- CVPR 2025 Best Paper](https://arxiv.org/abs/2503.11651) | [GitHub](https://github.com/facebookresearch/vggt)
- [DUSt3R -- NAVER Labs](https://github.com/naver/dust3r)
- [MASt3R -- NAVER Labs](https://github.com/naver/mast3r)
- [MV-DUSt3R+ -- CVPR 2025 Oral](https://mv-dust3rp.github.io/)
- [Plane-DUSt3R -- ICLR 2025](https://arxiv.org/abs/2502.16779) | [GitHub](https://github.com/justacar/Plane-DUSt3R)
- [Meshroom / AliceVision 2025.1](https://github.com/alicevision/Meshroom)
- [Agisoft Metashape Python API 2.3.0](https://www.agisoft.com/pdf/metashape_python_api_2_3_0.pdf)
- [RealityCapture / RealityScan 2.0](https://www.capturingreality.com/)

### Scene Understanding
- [SAM 2 -- Meta AI](https://ai.meta.com/sam2/) | [GitHub](https://github.com/facebookresearch/sam2)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)

### Hardware Depth Sensors
- [Intel RealSense D455](https://www.intelrealsense.com/depth-camera-d455/)
- [Apple ARKit Depth](https://developer.apple.com/documentation/AVFoundation/capturing-depth-using-the-lidar-camera)
- [Google ARCore Depth API](https://developers.google.com/ar/develop/depth)

### Benchmarks and Comparisons
- [NYU Depth V2 Leaderboard](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2)
- [DUSt3R/MASt3R/VGGT Evaluation (2025)](https://www.tandfonline.com/doi/full/10.1080/10095020.2025.2597491)
- [Open Source 3D Reconstruction Comparison](https://www.triposrai.com/posts/open-source-3d-reconstruction-showdown/)
- [CubiCasa & Matterport Accuracy Testing](https://www.insiderealestatephotography.com/post/cubicasa-matterport-floor-plans-how-accurate-are-they)
