# Metric Depth Estimation Alternatives: Comprehensive Research Report

**Date:** February 9, 2026
**Context:** Replacing Depth Anything V2 (relative depth [0,1]) in a room reconstruction pipeline
**Current Bottleneck:** ~20-30% measurement error due to relative depth requiring assumed room width for scaling

---

## Executive Summary

The current system uses Depth Anything V2, which produces **relative (affine-invariant) depth maps** normalized to [0,1]. This is the primary accuracy bottleneck: without metric scale, the pipeline depends on `ASSUMED_ROOM_WIDTH_METERS` (currently 4.0m) for scaling, yielding ~20-30% measurement error.

**Key Findings:**

1. **Multiple production-ready metric depth models now exist** that can produce depth in real meters from a single image, without camera intrinsics.
2. **Top recommendations for this pipeline:**
   - **Rank 1: Apple Depth Pro** -- Best balance of metric accuracy, speed (0.3s), no intrinsics needed, Python API, ~5GB VRAM. Already configured in `config.py` as `METRIC_DEPTH_MODEL`.
   - **Rank 2: Depth Anything V3 Metric** -- Apache 2.0 license, competitive accuracy, direct successor to current DA2, easiest migration path.
   - **Rank 3: MoGe-2 (Microsoft)** -- Simultaneously predicts metric point maps + normals + camera FOV in one forward pass, directly applicable to room reconstruction.
3. **Expected accuracy improvement:** From ~20-30% error down to ~5-10% error on indoor scenes, depending on model choice and calibration approach.
4. **Integration complexity is low** for top candidates -- all provide Python APIs compatible with the existing `DepthEstimator` class interface.

---

## Table of Contents

1. [Detailed Model Analysis](#1-detailed-model-analysis)
2. [Comparison Matrix](#2-comparison-matrix)
3. [Benchmark Accuracy Table](#3-benchmark-accuracy-table)
4. [Ranking and Recommendation](#4-ranking-and-recommendation)
5. [Implementation Roadmap: Top 2 Options](#5-implementation-roadmap)
6. [Integration with Current Codebase](#6-integration-with-current-codebase)
7. [Risk Assessment](#7-risk-assessment)
8. [Sources](#8-sources)

---

## 1. Detailed Model Analysis

### 1.1 Apple Depth Pro

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- absolute metric depth in meters |
| **Camera Intrinsics Required** | No -- also estimates focal length from the image |
| **Architecture** | Multi-scale ViT with DINOv2 encoder, DPT-like fusion |
| **Indoor Accuracy (NYU)** | State-of-the-art; average rank 2.5 across all benchmarks |
| **Speed** | 0.3s per 2.25MP image on high-end GPU; ~15s on RTX 4050 |
| **VRAM** | ~5GB (FP16), ~8-10GB (FP32) |
| **Output** | Depth map in meters + estimated focal length in pixels |
| **Resolution** | Up to 2.25 megapixels |
| **Python API** | Yes -- `depth_pro.create_model_and_transforms()`, HuggingFace Transformers |
| **License** | Apple Sample Code License (ASCL) -- NOT Apache/MIT. Non-standard; review needed for commercial use |
| **HuggingFace ID** | `apple/DepthPro-hf` |
| **GitHub** | github.com/apple/ml-depth-pro |
| **Strengths** | Best boundary sharpness, no intrinsics needed, fast, metric output |
| **Limitations** | Apple ASCL license may restrict commercial deployment; larger model (~1B params) |
| **Publication** | ICLR 2025 |

**Integration Notes:**
- The `config.py` already has `METRIC_DEPTH_MODEL = "apple/DepthPro-hf"` configured.
- Returns `prediction["depth"]` in meters and `prediction["focallength_px"]` for focal length.
- The predicted focal length can replace the hardcoded `CAMERA_FX=500, CAMERA_FY=500`.

---

### 1.2 UniDepth V2

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- metric 3D points directly |
| **Camera Intrinsics Required** | No -- jointly predicts intrinsics and depth |
| **Architecture** | ViT-L14 backbone, pseudo-spherical output representation |
| **Indoor Accuracy (NYU)** | Among the best on NYUv2 and KITTI (per DA3 paper comparisons) |
| **Speed** | >30% faster than V1; tested on RTX 4090 with FP16 |
| **Output** | Metric 3D point maps + predicted camera intrinsics + uncertainty maps |
| **Python API** | Yes -- `UniDepthV2.from_pretrained()` or `torch.hub.load()` |
| **ONNX Support** | Yes |
| **License** | Creative Commons BY-NC 4.0 (non-commercial only) |
| **HuggingFace ID** | `lpiccinelli/unidepth-v2-vitl14` |
| **GitHub** | github.com/lpiccinelli-eth/UniDepth |
| **Strengths** | Predicts intrinsics (no camera info needed), uncertainty output, ONNX export, edge-guided loss for sharpness |
| **Limitations** | Non-commercial license restricts deployment; requires CUDA >11.8; xFormers compatibility issues possible |
| **Publication** | UniDepth: CVPR 2024; UniDepthV2: arXiv Feb 2025 |

**Integration Notes:**
- The uncertainty output is valuable for weighting during multi-view fusion.
- Intrinsics prediction eliminates the hardcoded camera parameters.
- Non-commercial license is a hard blocker for commercial products.

---

### 1.3 Metric3D v2

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- metric depth + surface normals |
| **Camera Intrinsics Required** | Uses canonical camera space transformation to handle different cameras |
| **Architecture** | ViT-L / ViT-giant2 variants; trained on 16M images from 16 datasets |
| **Indoor Accuracy (NYU)** | AbsRel ~0.042 (ViT-L), delta1 ~0.979 -- lowest reported AbsRel on NYU |
| **Speed** | Moderate (ViT-giant2 is computationally heavy) |
| **Output** | Metric depth map + surface normal map |
| **Python API** | Yes -- PyTorch inference scripts, ONNX export supported |
| **License** | BSD-2-Clause (non-commercial usage restriction noted) |
| **GitHub** | github.com/YvanYin/Metric3D |
| **Strengths** | Best published NYU accuracy, surface normals included, large training data diversity |
| **Limitations** | Performance depends on crop size selection at test time (may violate zero-shot premise); non-commercial restriction; ViT-giant2 requires significant VRAM |
| **Publication** | IEEE TPAMI 2024 |

**Integration Notes:**
- Surface normals are highly valuable for wall detection in floor plan generation.
- The canonical camera transformation is an elegant solution for handling unknown cameras.
- Crop size sensitivity is a concern for a general-purpose room scanning tool.

---

### 1.4 ZoeDepth

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- metric depth via adaptive metric binning |
| **Camera Intrinsics Required** | No |
| **Architecture** | MiDAS backbone + metric bins module with domain routing |
| **Indoor Accuracy (NYU)** | AbsRel ~0.077 (ViT-L) -- was SOTA at publication, now surpassed |
| **Speed** | Fast inference; smaller model footprint |
| **Output** | Metric depth map in meters |
| **Python API** | Yes -- HuggingFace Transformers (`Intel/zoedepth-nyu-kitti`) |
| **License** | MIT License -- most permissive option |
| **HuggingFace IDs** | `Intel/zoedepth-nyu`, `Intel/zoedepth-nyu-kitti` |
| **GitHub** | github.com/isl-org/ZoeDepth |
| **Strengths** | MIT license, HuggingFace integration, multi-domain support (indoor+outdoor), already supported in current codebase |
| **Limitations** | Accuracy significantly behind newer models (0.077 vs 0.042-0.045); foundational work now superseded |
| **Publication** | arXiv 2023 |

**Integration Notes:**
- The current `DepthEstimator` class already has a `_load_zoedepth()` method.
- While it produces metric depth, accuracy is ~80% worse than Metric3D v2 on NYU (0.077 vs 0.042).
- Best suited as a low-risk "first metric depth upgrade" before moving to Depth Pro or DA3.

---

### 1.5 DepthCrafter

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | No -- produces relative (affine-invariant) depth sequences |
| **Camera Intrinsics Required** | No camera poses or optical flow needed |
| **Architecture** | Conditional diffusion model based on image-to-video diffusion |
| **Temporal Consistency** | Excellent -- designed for video, up to 110 frames at once |
| **Speed** | ~2.1 fps on A100 GPU |
| **Output** | Temporally consistent relative depth sequences |
| **Python API** | Yes -- PyTorch, HuggingFace model hub |
| **License** | Academic/research/education only -- commercial use prohibited |
| **GitHub** | github.com/Tencent/DepthCrafter |
| **Strengths** | Best temporal consistency for video, handles dynamic scenes |
| **Limitations** | NOT metric depth; very slow (diffusion-based); academic license only; 3GB model; high VRAM |
| **Publication** | CVPR 2025 Highlight |

**Integration Notes:**
- Does NOT solve the metric depth problem -- still relative depth.
- Could be useful if the pipeline moved to video input for temporal consistency.
- The slow speed (diffusion-based) and academic-only license make it unsuitable for production.
- Video Depth Anything (also CVPR 2025) is a better video depth option with metric variants.

---

### 1.6 Marigold

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | No -- produces relative (affine-invariant) depth by default |
| **Camera Intrinsics Required** | No |
| **Architecture** | Fine-tuned Stable Diffusion for depth estimation |
| **Indoor Accuracy (NYU)** | AbsRel ~0.055-0.059 (Marigold v1.1) |
| **Speed** | ~5.2 seconds per image (948M params, iterative denoising) |
| **Output** | High-quality relative depth maps with fine details |
| **Python API** | Yes -- HuggingFace Diffusers pipeline |
| **License** | Apache 2.0 (permissive) |
| **GitHub** | github.com/prs-eth/Marigold |
| **Strengths** | Excellent fine detail, Apache license, active development (v1.1 May 2025), depth completion variant (Marigold-DC) |
| **Limitations** | Not metric by default; very slow (diffusion-based, multi-step); can be converted to metric with additional techniques but adds complexity |
| **Publication** | CVPR 2024 Oral, Best Paper Award Candidate |

**Integration Notes:**
- Not directly useful for metric depth without additional conversion (defocus blur, sparse depth fitting).
- The Marigold-DC variant can complete sparse LiDAR/depth data into dense depth -- useful if combined with iPhone LiDAR.
- Diffusion-based speed (~5s/image) is too slow for interactive room scanning.

---

### 1.7 Google ARCore Depth API

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- per-pixel distance in meters |
| **Camera Intrinsics Required** | Handled internally by ARCore |
| **Technology** | Depth-from-motion + ML enhancement; optional ToF sensor fusion |
| **Indoor Accuracy** | Not formally published; estimated ~5-20cm at room scale |
| **Speed** | Real-time (30fps with depth) |
| **Device Coverage** | 87%+ of active Android devices (as of Oct 2025) |
| **Output** | Depth images aligned with camera frames, metric (meters) |
| **API** | Android SDK (Java/Kotlin), Unity, Unreal -- no Python/REST API |
| **License** | Google proprietary (part of ARCore SDK) |
| **Strengths** | Real-time, metric, works on most Android phones, hardware ToF sensor fusion when available |
| **Limitations** | Android-only; no desktop/Python API; requires device motion for depth-from-motion; no standalone image API; accuracy not formally benchmarked |

**Integration Notes:**
- Cannot be used in the current desktop Python pipeline.
- Would require a companion mobile app to capture depth maps, then transfer to the server.
- Interesting for a future mobile-first version of the product.

---

### 1.8 Intel RealSense Depth Sensors

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- hardware depth in meters |
| **Technology** | Stereo vision (D4xx series) or LiDAR (L515, now EOL) |
| **Indoor Accuracy** | D455: <2% error at 4m; L515: consistent across 0.25-9m |
| **Speed** | Real-time (up to 90fps depth) |
| **Range** | D455: up to 6m; D435: up to 3m; L515: 0.25-9m (EOL) |
| **Output** | Dense depth maps at up to 1280x720 @ 30fps |
| **API** | Python (pyrealsense2), C++, C#, ROS |
| **Pricing** | D455: ~$300; D435i: ~$250 |
| **License** | Apache 2.0 (librealsense SDK) |
| **Status** | Spun off from Intel in mid-2025; now independent company; D555 PoE camera recently launched |
| **Strengths** | Hardware-based metric depth, real-time, Python SDK, IMU for SLAM, well-established ecosystem |
| **Limitations** | Requires physical hardware purchase; stereo cameras struggle in textureless areas; range limited to ~4-6m for good accuracy; L515 LiDAR discontinued |

**Integration Notes:**
- Best accuracy option (~2% error vs ~5-10% for ML models).
- Requires users to have specific hardware -- limits accessibility.
- Python SDK (pyrealsense2) is mature and well-documented.
- Could be offered as an "advanced mode" for users with the hardware.

---

### 1.9 Apple iPhone/iPad LiDAR

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- per-pixel depth in meters via ARKit |
| **Technology** | dToF (direct Time-of-Flight) LiDAR scanner |
| **Indoor Accuracy** | ~1-4cm for linear measurements at <3m; ~5-20cm for full-room reconstruction |
| **Range** | Up to ~5m (iPad Pro stated), ~6.4m measured maximum |
| **Optimal Distance** | ~2 meters |
| **API** | ARKit (Swift/Objective-C), no Python API |
| **Devices** | iPhone 12 Pro and later Pro models, iPad Pro (2020+) |
| **Strengths** | Consumer hardware many users already own, metric depth, real-time |
| **Limitations** | iOS-only; no Python/desktop API; accuracy degrades beyond 3m; app development required; resolution lower than camera images |

**Integration Notes:**
- Similar to ARCore: would require a companion iOS app.
- Many potential users may already have LiDAR-capable iPhones.
- Could be used with Prompt Depth Anything (see bonus section) as a sparse depth prompt.

---

### 1.10 Azure Spatial Anchors

| Attribute | Details |
|-----------|---------|
| **Status** | RETIRED -- service was discontinued November 20, 2024 |

Not a viable option. Azure Spatial Anchors has been shut down. No replacement depth estimation service from Microsoft Azure exists under this brand.

---

### 1.11 BONUS: Depth Anything V3 (DA3) Metric

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- DA3Metric-Large produces metric depth |
| **Camera Intrinsics Required** | Needs focal length for metric conversion: `metric_depth = focal * net_output / 300.` |
| **Architecture** | Plain DINO ViT encoder, depth-ray prediction |
| **Indoor Accuracy** | Competitive with UniDepth v1/v2 on NYUv2; SOTA on ETH3D |
| **Speed** | ~20 fps at 768x1024 (ViT-L) -- very fast |
| **Output** | Metric depth (with focal length), 3D point maps, camera poses |
| **Python API** | Yes -- pip installable, API at `depth_anything_3.api` |
| **License** | Apache 2.0 -- fully permissive |
| **GitHub** | github.com/ByteDance-Seed/Depth-Anything-3 |
| **Strengths** | Apache 2.0 license, fastest metric depth, multi-view support, direct DA2 successor, 3D Gaussian splatting, can replace COLMAP for pose estimation |
| **Limitations** | Requires focal length for metric conversion (can estimate or use Depth Pro's focal length estimation); November 2025 release -- relatively new |
| **Publication** | November 2025 tech report |

**Integration Notes:**
- Direct successor to current Depth Anything V2, making migration straightforward.
- Apache 2.0 license is ideal for commercial use.
- Multi-view capabilities could replace or augment the SfM/COLMAP pipeline.
- DA3Nested-Giant-Large outputs metric depth directly in meters without needing focal length.
- Can potentially replace both depth estimation AND SfM in the pipeline.

---

### 1.12 BONUS: MoGe-2 (Microsoft)

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- metric-scale 3D point maps |
| **Camera Intrinsics Required** | No -- also predicts camera FOV |
| **Architecture** | ViT-L backbone with separate scale prediction branch |
| **Indoor Accuracy** | SOTA in both relative AND metric geometry (NeurIPS 2025) |
| **Speed** | 60ms per image on A100/RTX 3090 (FP16) -- extremely fast |
| **Output** | Metric point maps, depth maps, normal maps, camera FOV -- all in one pass |
| **Python API** | Yes -- PyTorch, HuggingFace checkpoints |
| **License** | MIT License |
| **GitHub** | github.com/microsoft/MoGe |
| **Strengths** | One forward pass for point maps + depth + normals + FOV; MIT license; very fast; supports registration across video frames |
| **Limitations** | Relatively new (NeurIPS 2025); less community adoption than Depth Pro or DA3 |
| **Publication** | MoGe: CVPR 2025 Oral; MoGe-2: NeurIPS 2025 |

**Integration Notes:**
- Outputs 3D point maps directly -- could bypass the separate depth-to-3D conversion step.
- Normal maps improve wall detection for floor plans.
- FOV estimation eliminates hardcoded camera intrinsics.
- Video frame registration capability could simplify multi-view fusion.
- MIT license is the most permissive option.

---

### 1.13 BONUS: Prompt Depth Anything (CVPR 2025)

| Attribute | Details |
|-----------|---------|
| **Metric Depth** | Yes -- uses LiDAR prompt to produce 4K metric depth |
| **Approach** | Augments Depth Anything with sparse LiDAR depth as a "prompt" |
| **Indoor Accuracy** | SOTA on ARKitScenes and ScanNet++ |
| **Resolution** | Up to 4K |
| **Requirement** | Needs sparse LiDAR depth input (e.g., from iPhone LiDAR) |
| **GitHub** | github.com/DepthAnything/PromptDA |
| **Strengths** | Highest resolution metric depth; leverages consumer LiDAR hardware |
| **Limitations** | Requires LiDAR input -- not pure monocular |

**Integration Notes:**
- Ideal for a mobile-first pipeline where iPhone LiDAR provides sparse depth prompts.
- Could be a Phase 3 upgrade after establishing basic metric depth.

---

## 2. Comparison Matrix

| Model | Metric Depth | Intrinsics Needed | NYU AbsRel | Speed | License | VRAM | Python API | Commercial OK |
|-------|:-----------:|:-----------------:|:----------:|:-----:|:-------:|:----:|:----------:|:------------:|
| **Depth Pro** | Yes | No (estimates) | ~top-3 | 0.3s | Apple ASCL | ~5GB | Yes | Review needed |
| **DA3 Metric** | Yes | Focal length | Competitive | ~50ms | Apache 2.0 | ~4GB | Yes | Yes |
| **MoGe-2** | Yes | No (estimates FOV) | SOTA | 60ms | MIT | ~4GB | Yes | Yes |
| **UniDepth V2** | Yes | No (predicts) | Top-tier | Fast | CC BY-NC 4.0 | ~6GB | Yes | No |
| **Metric3D v2** | Yes | Canonical transform | ~0.042 | Moderate | BSD-2 (NC) | ~8GB+ | Yes | No |
| **ZoeDepth** | Yes | No | ~0.077 | Fast | MIT | ~4GB | Yes | Yes |
| **Marigold** | No (relative) | No | ~0.057 | 5.2s | Apache 2.0 | ~6GB | Yes | Yes |
| **DepthCrafter** | No (relative) | No | N/A | ~0.5s/frame | Academic only | ~10GB | Yes | No |
| **ARCore Depth** | Yes | Internal | Unknown | Real-time | Proprietary | N/A | No | Yes |
| **RealSense D455** | Yes (hardware) | Calibrated | <2% @ 4m | Real-time | Apache 2.0 | N/A | Yes | Yes |
| **iPhone LiDAR** | Yes (hardware) | Internal | ~1-4cm | Real-time | Proprietary | N/A | No | Yes |

---

## 3. Benchmark Accuracy Table (NYU Depth V2 -- Indoor)

Lower AbsRel = better. Higher delta1 = better.

| Model | AbsRel | delta1 | Setting |
|-------|:------:|:------:|---------|
| Metric3D v2 (ViT-L) | 0.042 | 0.979 | Zero-shot / Fine-tuned |
| Metric3D v2 (ViT-g) | 0.043 | 0.982 | Zero-shot / Fine-tuned |
| Depth Anything V2 (ViT-L) | 0.045 | -- | Fine-tuned on NYU |
| Depth Pro | ~0.05* | -- | Zero-shot (avg rank 2.5) |
| Marigold v1.1 | 0.055-0.059 | 0.961-0.964 | Affine-invariant |
| Depth Anything V1 (ViT-L) | 0.076 | 0.947 | Affine-invariant |
| ZoeDepth (ViT-L) | 0.077 | 0.953 | Metric (fine-tuned) |
| **Current system (DA2 relative)** | **N/A** | **N/A** | **Relative depth, 20-30% error after scaling** |

*Note: Depth Pro's exact NYU number is not widely published; its strength is zero-shot generalization across many benchmarks rather than single-benchmark dominance.*

**Important caveat from the literature:** The field lacks a widely accepted benchmarking standard. Training data, model size, and inference overhead vary significantly, making strict numerical comparisons between models somewhat imprecise.

---

## 4. Ranking and Recommendation

### Weighted Scoring (10-point scale)

| Criterion (Weight) | Depth Pro | DA3 Metric | MoGe-2 | UniDepth V2 | Metric3D v2 | ZoeDepth |
|---------------------|:---------:|:----------:|:------:|:-----------:|:-----------:|:--------:|
| Metric accuracy (25%) | 9 | 8 | 9 | 9 | 10 | 6 |
| No intrinsics needed (15%) | 10 | 6 | 10 | 10 | 7 | 8 |
| Speed (10%) | 8 | 10 | 10 | 8 | 6 | 8 |
| License (15%) | 5 | 10 | 10 | 2 | 3 | 10 |
| Integration ease (15%) | 9 | 9 | 7 | 7 | 6 | 10 |
| Output richness (10%) | 8 | 9 | 10 | 9 | 9 | 5 |
| Community/maturity (10%) | 9 | 7 | 6 | 7 | 8 | 8 |
| **Weighted Total** | **8.15** | **8.35** | **8.40** | **7.10** | **6.90** | **7.55** |

### Final Rankings

| Rank | Model | Score | Rationale |
|:----:|-------|:-----:|-----------|
| 1 | **MoGe-2** | 8.40 | MIT license, 60ms speed, one-pass point maps + depth + normals + FOV. Directly outputs what the pipeline needs. |
| 2 | **DA3 Metric** | 8.35 | Apache 2.0, direct DA2 successor (easiest migration), 20fps, multi-view capabilities could replace COLMAP. |
| 3 | **Depth Pro** | 8.15 | Best zero-shot accuracy with focal length estimation; already in config.py. License needs legal review for commercial use. |
| 4 | **ZoeDepth** | 7.55 | MIT license, already integrated in codebase, lowest risk upgrade. But accuracy is significantly behind leaders. |
| 5 | **UniDepth V2** | 7.10 | Excellent technical capabilities but NC license blocks commercial use. |
| 6 | **Metric3D v2** | 6.90 | Best raw NYU accuracy but NC license, crop sensitivity, and VRAM requirements are concerns. |

### If commercial licensing is NOT a concern (research/internal use):
1. Metric3D v2 (best raw accuracy)
2. UniDepth V2 (intrinsics prediction + uncertainty)
3. Depth Pro (best overall package)

### If commercial licensing IS required:
1. DA3 Metric (Apache 2.0, best overall)
2. MoGe-2 (MIT, fastest, richest output)
3. ZoeDepth (MIT, already integrated, lowest risk)

---

## 5. Implementation Roadmap

### Option A: Depth Anything V3 Metric (Recommended for Easiest Migration)

**Phase 1: Basic Integration (1-2 days)**

1. Install DA3:
   ```bash
   pip install xformers torch>=2 torchvision
   pip install -e .  # from cloned Depth-Anything-3 repo
   ```

2. Add DA3 model loading to `DepthEstimator`:
   ```python
   def _load_da3_metric(self, model_name: str):
       from depth_anything_3.api import DepthAnything3
       self.da3_model = DepthAnything3.from_pretrained("DA3Metric-Large")
       self.model_type = "da3_metric"
   ```

3. Modify `estimate_depth()` to return metric depth:
   ```python
   if self.model_type == "da3_metric":
       net_output = self.da3_model(image)
       # Use estimated or provided focal length
       focal = self.estimate_focal_length(image) or CAMERA_FX
       metric_depth = focal * net_output / 300.0  # meters
       return metric_depth
   ```

4. Update `config.py`:
   ```python
   METRIC_DEPTH_MODEL = "DA3Metric-Large"
   ENABLE_METRIC_DEPTH = True
   ```

**Phase 2: Remove Assumed Width Dependency (1 day)**

1. Modify `FloorPlanGenerator` to use metric depth directly instead of `ASSUMED_ROOM_WIDTH_METERS`.
2. Update `depth_to_3d_points()` to skip inverse-depth conversion when metric depth is available.
3. Remove `DEPTH_SCALE` factor for metric depth paths.

**Phase 3: Multi-view Enhancement (2-3 days)**

1. Use DA3's multi-view capabilities for direct pose estimation.
2. Replace or augment COLMAP SfM with DA3 camera pose prediction.
3. Test with DA3Nested-Giant-Large for fully metric multi-view reconstruction.

**Success Metrics:**
- Measurement error < 10% (vs current ~20-30%)
- Processing time < 2s per image
- No degradation in floor plan quality

**Resource Requirements:**
- GPU with >= 4GB VRAM (RTX 3060 or better)
- Python >= 3.10
- ~2GB disk for model weights

---

### Option B: MoGe-2 (Recommended for Best Output Quality)

**Phase 1: Basic Integration (2-3 days)**

1. Install MoGe-2:
   ```bash
   pip install torch torchvision
   # Clone and install from github.com/microsoft/MoGe
   ```

2. Replace depth estimation pipeline:
   ```python
   from moge import MoGe2
   model = MoGe2.from_pretrained("Ruicheng/moge-2-vitl-normal")

   # Single forward pass produces everything needed
   output = model(image)
   point_map = output["points"]      # Metric 3D point map
   depth_map = output["depth"]       # Metric depth
   normal_map = output["normals"]    # Surface normals
   fov = output["fov"]              # Camera field of view
   ```

3. This directly outputs 3D points, bypassing the separate `depth_to_3d_points()` conversion entirely.

**Phase 2: Enhanced Floor Plan Generation (2-3 days)**

1. Use surface normals to identify walls (normals perpendicular to floor plane).
2. Use metric point maps directly for measurement.
3. Replace Hough line detection with normal-guided wall segmentation.

**Phase 3: Multi-view Fusion (2-3 days)**

1. Use MoGe-2's video frame registration for multi-view alignment.
2. Compute similarity transformations between frames using image matching (PDCNet).
3. Fuse registered metric point maps.

**Success Metrics:**
- Measurement error < 8% (improved wall detection via normals)
- Single-image processing < 100ms
- Floor plan shows wall orientation from normals

**Resource Requirements:**
- GPU with >= 4GB VRAM
- Python >= 3.8
- ~1GB disk for model weights

---

## 6. Integration with Current Codebase

The current `DepthEstimator` class at `/Users/rajesh/conductor/workspaces/room-reconstruction-demo/missoula/modules/depth_estimator.py` has the following integration points:

### Current Interface (to preserve):
```python
class DepthEstimator:
    def estimate_depth(self, image: np.ndarray, ...) -> np.ndarray:
        # Returns depth map (H, W), normalized 0-1

    def depth_to_3d_points(self, image, depth, fx, fy, ...) -> Tuple[np.ndarray, np.ndarray]:
        # Returns (points N,3), (colors N,3)
```

### Proposed Extension:
```python
class DepthEstimator:
    def estimate_depth(self, image, ..., metric=False) -> np.ndarray:
        # If metric=True, returns depth in meters
        # If metric=False, returns normalized [0,1] (backward compatible)

    def estimate_metric_depth(self, image) -> Dict[str, Any]:
        # Returns {"depth": np.ndarray,  # meters
        #          "focal_length": float,  # pixels
        #          "normals": np.ndarray,  # optional
        #          "confidence": np.ndarray}  # optional

    def depth_to_3d_points(self, image, depth, fx=None, fy=None, ...) -> Tuple:
        # If fx/fy are None, uses estimated focal length
        # If depth is metric, skips inverse-depth and DEPTH_SCALE
```

### Key Code Changes Needed:

1. **`config.py`**: Already has `ENABLE_METRIC_DEPTH` and `METRIC_DEPTH_MODEL` -- just needs model ID updated.

2. **`depth_estimator.py`**: Add metric depth model loading and inference path.

3. **`floor_plan_generator.py`**: Remove dependency on `ASSUMED_ROOM_WIDTH_METERS` when metric depth is available.

4. **`room_reconstructor.py`**: Pass `metric=True` flag through the pipeline.

5. **`visualizer_3d.py`**: Update `ScaleEstimator` to use metric depth directly.

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Metric depth inaccurate on specific room types | Medium | High | Keep relative depth fallback; calibration with known object |
| License issues with Apple ASCL | Medium | High | Use DA3 (Apache 2.0) or MoGe-2 (MIT) as primary |
| Model too large for user GPU | Low | Medium | Offer multiple model sizes (DA3 has Small/Large variants) |
| API changes in new models | Low | Low | Pin model versions; abstract behind DepthEstimator interface |
| Focal length estimation error | Medium | Medium | Allow user to input known focal length; cross-validate with multiple estimators |
| Metric depth fails on specific images | Low | Medium | Automatic fallback to relative depth with user-provided room width |
| Multi-view scale inconsistency | Medium | High | Use single model for all views; validate scale consistency across frames |

---

## 8. Sources

### Papers and Research
- [Apple Depth Pro - ICLR 2025](https://arxiv.org/abs/2410.02073)
- [UniDepth - CVPR 2024](https://arxiv.org/abs/2403.18913)
- [UniDepthV2 - arXiv Feb 2025](https://arxiv.org/abs/2502.20110)
- [Metric3D v2 - IEEE TPAMI 2024](https://arxiv.org/abs/2404.15506)
- [ZoeDepth - arXiv 2023](https://arxiv.org/abs/2302.12288)
- [DepthCrafter - CVPR 2025](https://github.com/Tencent/DepthCrafter)
- [Marigold - CVPR 2024 Oral](https://arxiv.org/abs/2312.02145)
- [Depth Anything V3 - Nov 2025](https://arxiv.org/abs/2511.10647)
- [MoGe - CVPR 2025 Oral](https://arxiv.org/abs/2410.19115)
- [MoGe-2 - NeurIPS 2025](https://arxiv.org/abs/2507.02546)
- [Prompt Depth Anything - CVPR 2025](https://arxiv.org/abs/2412.14015)
- [Survey on Monocular Metric Depth Estimation - Jan 2025](https://arxiv.org/abs/2501.11841)

### Code Repositories
- [Apple Depth Pro GitHub](https://github.com/apple/ml-depth-pro)
- [Depth Anything V3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [MoGe / MoGe-2 GitHub](https://github.com/microsoft/MoGe)
- [UniDepth GitHub](https://github.com/lpiccinelli-eth/UniDepth)
- [Metric3D GitHub](https://github.com/YvanYin/Metric3D)
- [ZoeDepth GitHub](https://github.com/isl-org/ZoeDepth)
- [Marigold GitHub](https://github.com/prs-eth/Marigold)
- [Prompt Depth Anything GitHub](https://github.com/DepthAnything/PromptDA)

### Model Hubs
- [Apple DepthPro-hf on HuggingFace](https://huggingface.co/apple/DepthPro-hf)
- [DA3 on HuggingFace](https://huggingface.co/depth-anything/DA3-BASE)
- [ZoeDepth on HuggingFace](https://huggingface.co/Intel/zoedepth-nyu-kitti)
- [UniDepth V2 on HuggingFace](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14)

### Hardware
- [Intel RealSense D455](https://www.intelrealsense.com/depth-camera-d455/)
- [RealSense Spin-off from Intel](https://www.therobotreport.com/after-intel-exit-realsense-maps-its-own-future-in-3d-vision/)
- [Apple ARKit Depth Documentation](https://developer.apple.com/documentation/AVFoundation/capturing-depth-using-the-lidar-camera)
- [Google ARCore Depth API](https://developers.google.com/ar/develop/depth)

### Industry Analysis
- [LearnOpenCV Depth Pro Explainer](https://learnopencv.com/depth-pro-monocular-metric-depth/)
- [Roboflow Depth Estimation Models Comparison](https://blog.roboflow.com/depth-estimation-models/)
- [HuggingFace Monocular Depth Estimation Guide](https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide)
- [NYU Depth V2 Benchmark Leaderboard](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2)
