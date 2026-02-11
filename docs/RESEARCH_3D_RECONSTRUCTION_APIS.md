# Research Report: 3D Reconstruction, Point Cloud Processing, and Scene Understanding APIs

**Date:** 2026-02-09
**Scope:** Commercial and advanced alternatives to improve the Missoula room reconstruction pipeline
**Current Stack:** Open3D, TSDF fusion, COLMAP SfM (optional), Depth-Anything-V2, ICP/RANSAC registration

---

## Executive Summary

The current pipeline suffers from three core weaknesses: (1) unreliable multi-view alignment from COLMAP SfM with few images, (2) noisy point clouds from relative depth estimation + heuristic registration, and (3) limited scene understanding (no semantic segmentation of walls, doors, or windows from images). This research evaluates 14 alternatives across photogrammetry, neural reconstruction, and scene understanding domains.

**Top Recommendations (Ranked):**

1. **MASt3R / DUSt3R** -- Replace COLMAP SfM and depth estimation entirely. Works with 2-10 images, no camera calibration needed. Free, open source, Python-native. **Highest impact, lowest cost.**
2. **VGGT (Visual Geometry Grounded Transformer)** -- CVPR 2025 Best Paper. Even faster than MASt3R, single feed-forward pass, exports COLMAP format. **Best if speed matters.**
3. **Grounded SAM 2** -- Add semantic segmentation of walls, doors, windows using text prompts. Augments (does not replace) the reconstruction pipeline. **Best for scene understanding.**
4. **Meshroom/AliceVision 2025.1** -- Open-source photogrammetry with new Python bindings, Gaussian Splatting plugin, and ML-based depth estimation. **Best full-pipeline replacement if more images are available.**
5. **Plane-DUSt3R** -- Purpose-built for room layout reconstruction from sparse unposed views. Directly outputs room planes. **Most directly relevant to the floor plan use case.**

---

## Detailed Analysis of Each Alternative

---

### 1. DUSt3R / MASt3R (NAVER Labs Europe)

**Core Capability:** Dense stereo 3D reconstruction from arbitrary image collections without camera calibration or pose priors.

| Attribute | Detail |
|-----------|--------|
| API Type | Python library (PyTorch). Clone from GitHub, run inference scripts. No REST API. |
| Input Requirements | Minimum 2 images, optimal 4-20. No overlap requirements (handles non-overlapping views). Resolution up to 518px (ViT input). |
| Output Format | Dense per-pixel 3D pointmaps, camera poses (4x4), focal lengths, confidence masks. Export to PLY via trimesh/Open3D (community scripts). COLMAP export available via MASt3R-SfM. |
| Indoor Accuracy | State-of-the-art on indoor benchmarks. +50% completeness over COLMAP on sparse views. Scale-invariant (no metric scale without reference). MASt3R adds metric capability. |
| Pricing | Free, open source (CC-BY-NC-SA 4.0 for research; commercial licensing via NAVER). |
| GPU Requirements | 16GB+ VRAM recommended. >120 images may exceed 16GB with one-ref pairing. |
| Integration Difficulty | **Medium.** Requires PyTorch + CUDA. Replace `SfMProcessor` and `DepthEstimator` modules. ~2-3 days integration work. |
| Replaces/Augments | **Replaces** COLMAP SfM, depth estimation, and ICP registration entirely. Single unified model for all three tasks. |

**Key API Pattern:**
```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

model = AsymmetricCroCo3DStereo.from_pretrained("DUSt3R_ViTLarge_BaseDecoder_512_dpt")
images = load_images(["img1.jpg", "img2.jpg", ...], size=512)
pairs = make_pairs(images, scene_graph="complete")
output = inference(pairs, model, device="cuda")
scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(...)

pts3d = scene.get_pts3d()       # List of per-image 3D pointmaps
masks = scene.get_masks()       # Confidence masks
poses = scene.get_im_poses()    # Camera extrinsics (4x4)
focals = scene.get_focals()     # Estimated focal lengths
```

**Ecosystem Variants:**
- **MASt3R:** Adds dense local features for robust matching. Metric reconstruction.
- **MASt3R-SfM:** Handles 200+ images via sparse graph retrieval. COLMAP-style output.
- **MV-DUSt3R+ (CVPR 2025 Oral):** Single-stage multi-view. Reconstructs a room in 0.89 seconds with 12 views.
- **MASt3R-SLAM (CVPR 2025):** Real-time SLAM from video input.
- **Plane-DUSt3R (ICLR 2025):** Room layout-specific variant (see entry below).

**Impact Assessment:** This is the single highest-impact upgrade. The current pipeline's biggest bottleneck is COLMAP failing on 4-5 casual photos and then falling back to centroid alignment. DUSt3R/MASt3R was designed precisely for this scenario (sparse, uncalibrated, indoor photos) and eliminates three separate pipeline stages in one model.

---

### 2. VGGT (Visual Geometry Grounded Transformer) -- Meta/Facebook Research

**Core Capability:** Feed-forward transformer that predicts camera parameters, depth maps, point maps, and 3D point tracks from 1 to hundreds of views in a single pass.

| Attribute | Detail |
|-----------|--------|
| API Type | Python library (PyTorch). GitHub repo with pip-installable requirements. |
| Input Requirements | 1 to hundreds of images. Works with single images (monocular) through dense multi-view. |
| Output Format | Camera extrinsics/intrinsics, depth maps, dense point clouds, 3D tracks. **Native COLMAP export** (cameras.bin, images.bin, points3D.bin). Direct integration with gsplat for Gaussian Splatting. |
| Indoor Accuracy | CVPR 2025 Best Paper Award. Outperforms DUSt3R on camera pose estimation. Competitive monocular depth (vs. DepthAnything-V2). Higher computational efficiency than MASt3R. |
| Pricing | Free, open source (Meta Research license). |
| GPU Requirements | 1.2B parameters. Requires substantial VRAM (A100 recommended for many views). |
| Integration Difficulty | **Medium.** Similar to DUSt3R integration pattern. Cleaner API with native COLMAP export. |
| Replaces/Augments | **Replaces** COLMAP SfM, depth estimation, and registration. Single model for all geometry tasks. |

**Key API Pattern:**
```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT.from_pretrained("facebook/VGGT-1B")
images = load_and_preprocess_images(["img1.jpg", "img2.jpg"]).to(device)
with torch.no_grad():
    predictions = model(images)
# predictions contains: extrinsics, intrinsics, depth maps, point maps, tracks
```

**Impact Assessment:** Marginally better than DUSt3R/MASt3R on benchmarks, with cleaner COLMAP export. The tradeoff is that VGGT is newer (March 2025) and has a smaller community/ecosystem than DUSt3R. Both are strong choices; VGGT edges ahead on speed and API cleanliness.

---

### 3. Plane-DUSt3R (ICLR 2025)

**Core Capability:** Room layout reconstruction from unposed sparse perspective views. Fine-tuned DUSt3R specifically for detecting structural planes (walls, floor, ceiling) in indoor scenes.

| Attribute | Detail |
|-----------|--------|
| API Type | Python (PyTorch). GitHub repo. |
| Input Requirements | 2-5 unposed perspective images of a room. Trained on Structured3D dataset. |
| Output Format | 3D structural plane parameters, room layout polygons. Can derive floor plans directly from plane intersections. |
| Indoor Accuracy | Outperforms prior room layout methods on Structured3D. Robust to non-overlapping views and even cartoon/rendered images. |
| Pricing | Free, open source. |
| GPU Requirements | Same as DUSt3R (16GB+ VRAM). |
| Integration Difficulty | **Medium.** Built on DUSt3R/MASt3R stack. Would replace both the reconstruction and floor plan generation stages. |
| Replaces/Augments | **Replaces** depth estimation, SfM, point cloud registration, AND floor plan generation. Purpose-built for the exact use case of this project. |

**Impact Assessment:** This is the most directly relevant model for the Missoula project's goal (room reconstruction to floor plan from 4-5 photos). It eliminates the current pipeline's weakest step (convex hull floor plan extraction) by detecting actual structural planes. The limitation is that it was trained on synthetic data (Structured3D) and may need fine-tuning for real-world photos.

---

### 4. Nerfstudio (Instant-NGP / Splatfacto / Nerfacto)

**Core Capability:** Neural radiance fields and 3D Gaussian Splatting for novel view synthesis and scene reconstruction.

| Attribute | Detail |
|-----------|--------|
| API Type | Python CLI + library. `ns-train` command-line interface. Python API via model classes. |
| Input Requirements | **30-150 images** recommended. Needs COLMAP poses or transforms.json. Minimal motion blur. Consistent exposure. |
| Output Format | NeRF/Gaussian Splat scene representation. Can export point clouds and meshes via post-processing. |
| Indoor Accuracy | Good for novel view synthesis. Geometric accuracy depends heavily on input quality. Textureless indoor surfaces (white walls) are problematic. |
| Pricing | Free, open source (Apache 2.0). |
| GPU Requirements | NVIDIA GPU with CUDA required. 8GB+ VRAM minimum. |
| Integration Difficulty | **High.** Requires 30+ images (current pipeline targets 4-5). Needs pre-computed camera poses (COLMAP). Training takes minutes to hours. |
| Replaces/Augments | **Augments** downstream visualization. Does not replace core reconstruction for sparse views. Could be added as optional high-quality rendering after DUSt3R/VGGT provides camera poses. |

**Impact Assessment:** Not a good fit for the current 4-5 image constraint. However, if the project scope expands to accept video input or 30+ photos, Nerfstudio + gsplat becomes compelling for photorealistic visualization. The combination of VGGT (for poses) + gsplat (for rendering) is an active research pipeline.

---

### 5. Meshroom / AliceVision (Open Source Photogrammetry)

**Core Capability:** Full photogrammetry pipeline: SfM, dense reconstruction, meshing, texturing. Node-based visual programming.

| Attribute | Detail |
|-----------|--------|
| API Type | Python CLI + node-based GUI. New Python bindings in v2025.1.0. Can be scripted headlessly. |
| Input Requirements | 20+ images recommended for reliable reconstruction. Overlapping coverage needed. |
| Output Format | Dense point cloud, textured mesh, camera poses. Exports OBJ, PLY, ABC. |
| Indoor Accuracy | Struggles with textureless walls (common indoors). Surface reconstruction can fill gaps. New ML plugins (mrDepthEstimation, mrGSplat) improve results. |
| Pricing | Free, open source (MPL 2.0). |
| GPU Requirements | NVIDIA CUDA GPU required for GPU-accelerated nodes. CUDA 12 for v2025.1.0. |
| Integration Difficulty | **Medium-High.** Full pipeline replacement. New Python bindings simplify scripting but the node architecture has a learning curve. |
| Replaces/Augments | **Replaces** the entire pipeline (SfM through meshing). But requires more images than the current 4-5 target. |

**Notable 2025.1.0 Features:**
- Native Python bindings for ML integration
- AI segmentation nodes using natural language prompts
- mrGSplat plugin for Gaussian Splatting
- mrDepthEstimation for monocular depth
- Plugin architecture for custom nodes

**Impact Assessment:** Best option if the project pivots to requiring more images (20+). The 2025.1 ML plugins are exciting but experimental. For 4-5 images, DUSt3R/MASt3R remains superior.

---

### 6. Pix4D Cloud API (PIX4Dengine)

**Core Capability:** Commercial cloud photogrammetry: orthophoto, DSM, point cloud, and 3D mesh generation.

| Attribute | Detail |
|-----------|--------|
| API Type | REST API (cloud-based). Also available as on-premise Python SDK. Runs on AWS infrastructure. |
| Input Requirements | Designed primarily for aerial/drone imagery. Indoor support exists but is not the primary use case. |
| Output Format | Dense point cloud, 3D textured mesh, DSM, DTM, orthomosaic, contour lines. |
| Indoor Accuracy | Optimized for aerial photogrammetry. Indoor/close-range performance is adequate but not best-in-class compared to dedicated indoor tools. |
| Pricing | **Enterprise pricing** (contact sales). Billed per gigapixel processed. Consumer PIX4Dcloud ~$199/month. No public API pricing. |
| Integration Difficulty | **Low** (REST API). Upload images, poll for results, download outputs. Language-agnostic. |
| Replaces/Augments | Could **replace** the full reconstruction pipeline, but is expensive and not optimized for indoor/room scenarios. |

**Impact Assessment:** Poor fit. Designed for drone/aerial workflows. Expensive. Not optimized for indoor scenes with 4-5 images. Not recommended.

---

### 7. Agisoft Metashape (Professional Edition)

**Core Capability:** Professional photogrammetry software with Python scripting API. Full SfM + MVS pipeline.

| Attribute | Detail |
|-----------|--------|
| API Type | Python scripting API (embedded Python 3.8). Can run headlessly via CLI. Service Provider License available for SaaS integration. |
| Input Requirements | Works with as few as 3 images but quality improves significantly with 20+. Handles indoor scenes. |
| Output Format | Dense point cloud, textured mesh, DEM, orthomosaic. Exports PLY, OBJ, FBX, DXF, LAS, and many more. |
| Indoor Accuracy | Professional-grade. Good indoor performance when sufficient images are provided. Camera self-calibration handles varying focal lengths. |
| Pricing | **Professional: $3,499** one-time purchase. Service Provider: $155.90/month minimum (pay-per-use). Educational: significant discounts. 30-day free trial. |
| Integration Difficulty | **Medium.** Python API is well-documented (2.3.0 reference, Dec 2025). Can automate full pipeline via scripts. Runs locally (no cloud dependency). |
| Replaces/Augments | **Replaces** COLMAP SfM + dense reconstruction. Outputs feed directly into existing Open3D post-processing. |

**Key Python API Pattern:**
```python
import Metashape
doc = Metashape.Document()
chunk = doc.addChunk()
chunk.addPhotos(["img1.jpg", "img2.jpg", ...])
chunk.matchPhotos(accuracy=Metashape.HighestAccuracy)
chunk.alignCameras()
chunk.buildDepthMaps()
chunk.buildDenseCloud()
chunk.buildModel()
chunk.exportPoints("output.ply")
```

**Impact Assessment:** Best commercial option if budget allows. The Python API is mature and well-documented. However, at $3,499 for a proof-of-concept, and with free alternatives (DUSt3R/VGGT) outperforming it on sparse views, this is hard to justify unless accuracy on 20+ image sets is critical.

---

### 8. RealityCapture / RealityScan 2.0 (Epic Games)

**Core Capability:** GPU-accelerated photogrammetry. Fastest commercial photogrammetry tool. Now includes LiDAR-augmented workflows.

| Attribute | Detail |
|-----------|--------|
| API Type | CLI (extensive command set), REST API (RealityCapture Node), Python scripting via subprocess/requests. |
| Input Requirements | 3+ images. Handles thousands of images efficiently. LiDAR augmentation in v2.1. |
| Output Format | Dense point cloud, textured mesh, camera poses. Exports PLY, OBJ, FBX, etc. |
| Indoor Accuracy | Excellent with sufficient images. GPU-accelerated processing is 5-10x faster than Metashape. |
| Pricing | **Free** for individuals/companies under $1M revenue. $1,250/seat/year for larger companies. $1,850/year in Unreal bundle. |
| Integration Difficulty | **Medium-High.** Windows only. CLI is powerful but requires subprocess orchestration from Python. REST API (Node) adds HTTP interface. No native Python SDK. |
| Replaces/Augments | **Replaces** COLMAP SfM + dense reconstruction. Outputs feed into existing pipeline. |

**Impact Assessment:** Attractive free pricing for small teams. The Windows-only constraint and lack of native Python SDK make integration harder than Metashape or open-source alternatives. Best for teams already using Unreal Engine.

---

### 9. Azure Spatial Anchors / Object Anchors -- RETIRED

**Status: All Azure Mixed Reality cloud services have been retired.**

- Azure Spatial Anchors: Retired November 20, 2024.
- Azure Object Anchors: Retired (retirement page live).
- Azure Remote Rendering: Retired September 30, 2025.
- Microsoft Mesh: Retired December 1, 2025.

**Impact Assessment:** Not viable. Do not consider. Microsoft has wound down its entire Azure Mixed Reality cloud platform and has not announced a replacement.

---

### 10. Google Cloud Vision / Vertex AI

**Core Capability:** Cloud-based 2D computer vision (object detection, OCR, labeling). Gemini multimodal for image understanding. No dedicated 3D or spatial API.

| Attribute | Detail |
|-----------|--------|
| API Type | REST API, Python SDK (`google-cloud-vision`). |
| Spatial Capabilities | None for 3D reconstruction. Gemini models have "imprecise spatial reasoning" per Google's own documentation. Cannot locate objects precisely in images. |
| Pricing | Pay-per-use. Vision API: $1.50-$3.50 per 1000 images. Vertex AI: varies by model and compute. |

**Impact Assessment:** Not suitable for 3D reconstruction or spatial understanding. Could potentially be used for pre-processing (scene classification, object detection as input to other tools) but provides no geometric capability.

---

### 11. Segment Anything Model 2 (SAM2) -- Meta

**Core Capability:** Universal image and video segmentation. Zero-shot segmentation of any object with point/box/mask prompts.

| Attribute | Detail |
|-----------|--------|
| API Type | Python library (PyTorch). `pip install -e .` from GitHub repo. Also available via Ultralytics: `pip install ultralytics`. |
| Input Requirements | Single image or video. Requires geometric prompts (points or bounding boxes) -- no text prompts natively. |
| Output Format | Binary segmentation masks per object. |
| Indoor Accuracy | 6x more accurate than SAM1. Excellent edge detection. Real-time (44 FPS). Zero-shot generalization. |
| Pricing | Free, open source (Apache 2.0). |
| GPU Requirements | CUDA-enabled GPU required. |
| Integration Difficulty | **Low.** Well-documented Python API. Pre-trained checkpoints available. |
| Replaces/Augments | **Augments** the pipeline. Add wall/floor/ceiling segmentation to improve floor plan extraction. Does not replace geometric reconstruction. |

**Limitation for Architectural Use:** SAM2 has no text encoder. You cannot prompt it with "wall" -- you need to provide bounding boxes or points. This is why it pairs with Grounding DINO (see below).

**Key Upcoming:** SAM 3 introduces Promptable Concept Segmentation, which would enable text-based categorical segmentation (e.g., "segment all walls"). This would eliminate the need for Grounding DINO, but SAM 3 is not yet production-ready.

---

### 12. Grounding DINO + SAM (Grounded SAM 2)

**Core Capability:** Open-vocabulary object detection (Grounding DINO) combined with precise segmentation (SAM 2). Text-prompt-based detection and segmentation of any object.

| Attribute | Detail |
|-----------|--------|
| API Type | Python library. Install SAM2 + Grounding DINO from GitHub. Also available via `pip install autodistill-grounded-sam-2`. |
| Input Requirements | Single image + text prompt (e.g., "wall", "door", "window", "floor"). |
| Output Format | Bounding boxes with confidence scores + pixel-precise segmentation masks. |
| Indoor Accuracy | Zero-shot COCO 52.5 AP. Grounding DINO 1.5/1.6 (2025) further improves detection. No architectural-specific benchmarks published, but text-prompt approach works well for common indoor elements. |
| Pricing | Free, open source. Grounding DINO 1.5/1.6 API is cloud-based (via dds-cloudapi-sdk). |
| Integration Difficulty | **Low-Medium.** Python installation straightforward. Requires defining appropriate text prompts for architectural elements. |
| Replaces/Augments | **Augments** the pipeline significantly. Provides semantic understanding of walls, doors, windows that the current pipeline completely lacks. Feed segmentation masks into floor plan generator for dramatically better wall detection. |

**Key API Pattern:**
```python
# Using Grounded SAM 2
from sam2.build_sam import build_sam2
from grounding_dino.groundingdino.util.inference import load_model, predict

# Detect architectural elements
boxes, logits, phrases = predict(
    model=grounding_model,
    image=image,
    caption="wall . door . window . floor . ceiling",
    box_threshold=0.3,
    text_threshold=0.25
)

# Segment each detected element with SAM2
for box in boxes:
    masks = sam2_predictor.predict(box=box)
```

**Impact Assessment:** This is the best available solution for adding semantic scene understanding to the pipeline. The current `WallDetector` module relies on depth-based heuristics and Hough line detection. Grounded SAM 2 would provide actual wall/door/window segmentation masks from the input images, dramatically improving floor plan accuracy. Integration with the existing `WallDetector` and `OpeningDetector` modules is straightforward.

---

### 13. Luma AI

**Core Capability:** 3D capture via NeRF/Gaussian Splatting from photos. WebGL-based rendering.

| Attribute | Detail |
|-----------|--------|
| API Type | Mobile app (iOS) + WebGL JavaScript API. No Python SDK or REST API for 3D reconstruction. |
| Input Requirements | Video capture from smartphone (iOS app). |
| Output Format | PLY (Gaussian Splat), mesh export via app. |
| Pricing | Free for capture and WebGL. Dream Machine video generation is subscription-based ($9.99/month). |

**Impact Assessment:** Not suitable for pipeline integration. No programmatic API for 3D reconstruction. The mobile app no longer supports Gaussian Splatting directly. Luma has pivoted to AI video generation (Dream Machine). Not recommended.

---

### 14. Polycam

**Core Capability:** Cloud-based 3D scanning and photogrammetry via mobile app.

| Attribute | Detail |
|-----------|--------|
| API Type | Mobile app (iOS/Android) + web platform. No public developer API. Enterprise plan offers custom integrations. |
| Input Requirements | Up to 2,000 images per model. Photo mode or LiDAR mode (iOS devices with LiDAR). |
| Output Format | OBJ, GLTF, FBX, STL, USDZ (meshes); PLY, LAS, XYZ, DXF (point clouds). |
| Pricing | Basic: ~$27/month. Enterprise: custom pricing. No standalone API access. |
| Integration Difficulty | **Very High.** No public API. Would require Enterprise agreement for any programmatic integration. |

**Impact Assessment:** Polycam is an end-user product, not a developer tool. No API available for pipeline integration. Not recommended for this project.

---

## Comparison Matrix

### Reconstruction Capability Comparison

| Tool | Min Images | Camera Cal Required | Indoor Optimization | Metric Scale | Speed (5 imgs) |
|------|-----------|-------------------|-------------------|-------------|----------------|
| **MASt3R** | 2 | No | Yes (trained on indoor data) | Yes | ~10-30s |
| **VGGT** | 1 | No | Yes | Yes | ~5-15s |
| **Plane-DUSt3R** | 2 | No | Yes (purpose-built) | No (plane-relative) | ~15-30s |
| **MV-DUSt3R+** | 5 | No | Yes | No | ~0.89s |
| COLMAP (current) | 3+ | No (self-cal) | Partial | No | ~60-300s |
| Meshroom 2025.1 | 20+ | No | Partial | No | ~300-1800s |
| Metashape | 3+ | No (self-cal) | Yes | Optional | ~120-600s |
| RealityCapture | 3+ | No | Yes | Optional | ~30-120s |
| Nerfstudio | 30+ | Needs COLMAP | Partial | No | ~300-3600s |

### Integration and Cost Comparison

| Tool | License | Cost | Python Native | Integration Effort | Current Pipeline Fit |
|------|---------|------|--------------|-------------------|---------------------|
| **MASt3R** | CC-BY-NC-SA | Free (research) | Yes | 2-3 days | Excellent (4-5 imgs) |
| **VGGT** | Meta Research | Free | Yes | 2-3 days | Excellent |
| **Plane-DUSt3R** | Open | Free | Yes | 3-5 days | Perfect (room layout) |
| **Grounded SAM 2** | Apache 2.0 | Free | Yes | 1-2 days | Augments well |
| Meshroom 2025.1 | MPL 2.0 | Free | Yes (new) | 5-7 days | Needs more images |
| Metashape | Proprietary | $3,499 | Yes (API) | 3-5 days | Good (but expensive) |
| RealityCapture | Proprietary | Free/<$1M | CLI/REST | 5-7 days | Windows only |
| Nerfstudio | Apache 2.0 | Free | Yes | 7-10 days | Needs 30+ images |
| Pix4D | Proprietary | Enterprise | REST API | 3-5 days | Wrong domain (aerial) |

### Scene Understanding Comparison

| Tool | Wall Detection | Door/Window Detection | Floor Plan Output | Text Prompting |
|------|--------------|---------------------|------------------|---------------|
| **Grounded SAM 2** | Via text prompt | Via text prompt | Masks only | Yes |
| **Plane-DUSt3R** | Structural planes | No | Plane layout | No |
| Current WallDetector | Depth heuristics | Depth heuristics | Convex hull | No |
| SAM2 (standalone) | Via box/point prompt | Via box/point prompt | Masks only | No |
| Google Vision | Object labels only | No | No | No |

---

## Weighted Scoring (0-10 scale)

Weights: Accuracy (25%), Integration Ease (20%), Cost (15%), Sparse View Performance (20%), Maintenance/Community (10%), Feature Completeness (10%)

| Tool | Accuracy | Integration | Cost | Sparse Views | Community | Features | **Weighted** |
|------|----------|-------------|------|-------------|-----------|----------|-------------|
| **MASt3R** | 9 | 7 | 10 | 10 | 8 | 8 | **8.85** |
| **VGGT** | 9 | 8 | 10 | 10 | 7 | 9 | **9.00** |
| **Plane-DUSt3R** | 8 | 7 | 10 | 10 | 6 | 9 | **8.55** |
| **Grounded SAM 2** | 8 | 8 | 10 | N/A | 9 | 7 | **8.40** |
| Meshroom 2025.1 | 7 | 5 | 10 | 4 | 7 | 8 | **6.50** |
| Metashape | 9 | 7 | 3 | 6 | 7 | 9 | **6.90** |
| RealityCapture | 9 | 4 | 7 | 6 | 6 | 9 | **6.80** |
| Nerfstudio | 7 | 4 | 10 | 2 | 8 | 7 | **5.75** |
| Pix4D | 6 | 6 | 2 | 3 | 5 | 7 | **4.65** |

---

## Implementation Roadmap

### Phase 1: Replace Core Reconstruction (Highest Impact -- 1-2 weeks)

**Recommended: VGGT or MASt3R as primary reconstruction engine**

VGGT is recommended as the primary choice due to its CVPR 2025 Best Paper status, cleaner API, native COLMAP export, and single-pass architecture. MASt3R is an equally valid alternative with a larger community.

**Step-by-step:**

1. **Install VGGT** (Day 1)
   ```bash
   git clone https://github.com/facebookresearch/vggt.git
   pip install -r vggt/requirements.txt
   ```

2. **Create `modules/vggt_reconstructor.py`** (Days 2-3)
   - New module wrapping VGGT inference
   - Input: list of RGB images (np.ndarray)
   - Output: dict with `points_3d`, `camera_poses`, `depth_maps`, `focal_lengths`, `confidence_masks`
   - Handle GPU/CPU fallback
   - Implement PLY export using existing `Visualizer3D.save_point_cloud()`

3. **Modify `RoomReconstructor.__init__`** (Day 4)
   - Add VGGT as primary reconstruction backend
   - Keep COLMAP SfM as fallback
   - Keep depth estimation as fallback for when VGGT is unavailable

4. **Replace fusion pipeline** (Days 5-6)
   - In `reconstruct()` and `reconstruct_from_arrays()`:
     - If VGGT available: single call produces aligned point cloud + poses
     - Eliminate separate depth estimation, SfM, and registration steps
     - Feed VGGT depth maps to existing floor plan generator
   - Fallback chain: VGGT -> MASt3R -> COLMAP+Depth -> Legacy ICP

5. **Update `config.py`** (Day 6)
   - Add `ENABLE_VGGT = True`
   - Add `VGGT_MODEL = "facebook/VGGT-1B"`
   - Add `VGGT_MAX_IMAGES = 50`

6. **Test and validate** (Days 7-10)
   - Compare point cloud quality: VGGT vs. current pipeline
   - Measure floor plan accuracy improvement
   - Benchmark inference time
   - Test with 2, 4, 5, 10, and 20 image sets

**Success Metrics:**
- Multi-view alignment succeeds on >90% of casual photo sets (vs. current ~50% COLMAP success rate on 4-5 images)
- Point cloud noise reduced by >50%
- Floor plan measurements within 10-15% accuracy (vs. current 20-30%)

**Potential Obstacles:**
- VGGT requires substantial GPU memory (16GB+ VRAM). Mitigation: implement batch processing and resolution scaling.
- VGGT is scale-invariant. Mitigation: use the existing `assumed_room_width` calibration or Plane-DUSt3R's structural planes for scale.
- License: VGGT is Meta Research license; review commercial use terms. MASt3R is CC-BY-NC-SA (non-commercial). For commercial use, may need to negotiate licensing or use Meshroom/COLMAP for the SfM step.

---

### Phase 2: Add Semantic Scene Understanding (Medium Impact -- 1 week)

**Recommended: Grounded SAM 2 for wall/door/window segmentation**

**Step-by-step:**

1. **Install Grounded SAM 2** (Day 1)
   ```bash
   pip install segment-anything-2
   pip install autodistill-grounded-sam-2
   # OR full install from source for latest features
   git clone https://github.com/IDEA-Research/Grounded-SAM-2
   ```

2. **Create `modules/detection/semantic_segmenter.py`** (Days 2-3)
   - Wrap Grounded SAM 2 with architectural text prompts
   - Method: `segment_architectural_elements(image) -> dict`
     - Prompts: "wall", "door", "window", "floor", "ceiling"
     - Returns: dict of element_type -> list of segmentation masks
   - Confidence thresholding for reliable detections

3. **Integrate with WallDetector** (Days 3-4)
   - Current `WallDetector.detect_walls_from_depth()` uses depth heuristics
   - New approach: use SAM2 wall masks to identify wall regions in depth maps
   - Combine semantic masks with depth data for 3D wall plane fitting
   - Use door/window masks to populate `OpeningDetector` results directly

4. **Improve floor plan extraction** (Days 5-6)
   - Use wall segmentation masks to identify actual wall boundaries
   - Replace convex hull with mask-derived wall lines
   - Wall masks + depth maps = 3D wall planes with actual room shape (handles L-shaped, U-shaped rooms)

5. **Test and validate** (Day 7)
   - Compare wall detection: semantic vs. depth-only
   - Test door/window detection accuracy
   - Validate non-convex room shape handling

**Success Metrics:**
- Door/window detection accuracy >70% (vs. current depth-heuristic approach)
- Non-convex room shapes supported (L-shaped, U-shaped)
- Wall line precision improved by >30%

---

### Phase 3: Room Layout Specialization (Optional -- 1 week)

**Recommended: Plane-DUSt3R for direct room layout estimation**

This is an alternative or complement to Phase 1. Plane-DUSt3R directly outputs room structural planes, which can be converted to floor plans without the intermediate point cloud step.

**Step-by-step:**

1. Install Plane-DUSt3R (MASt3R + fine-tuned checkpoint)
2. Create `modules/plane_reconstructor.py` wrapping the pipeline
3. Run Plane-DUSt3R to get structural planes (walls, floor, ceiling)
4. Convert plane intersections to wall segments directly
5. Feed wall segments to existing `FloorPlanModel` and renderers (SVG/DXF/PNG)

**Benefit:** Eliminates the point cloud -> density grid -> Hough line detection -> convex hull pipeline entirely, replacing it with direct geometric plane estimation.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|-----------|
| VGGT/MASt3R insufficient VRAM on consumer GPUs | Medium | Medium | Implement resolution scaling, batch processing, CPU fallback |
| Non-commercial license restricts deployment | High | Medium | Use COLMAP+Depth as commercial fallback; negotiate license with NAVER/Meta |
| Plane-DUSt3R trained on synthetic data may not generalize | Medium | Medium | Fine-tune on real indoor photos; keep legacy pipeline as fallback |
| Grounded SAM 2 misdetects architectural elements | Low | Low | Confidence thresholding, ensemble with depth-based detection |
| New dependencies increase install complexity | Low | High | Create separate conda environments; Docker containerization |

---

## Conclusion

The current pipeline's primary bottleneck is COLMAP SfM failing on 4-5 casual photos, causing cascading failures through the entire reconstruction chain. The transformer-based geometric models (VGGT, MASt3R, DUSt3R) fundamentally solve this problem by eliminating the need for separate feature detection, matching, triangulation, and bundle adjustment steps.

The recommended upgrade path is:
1. **VGGT or MASt3R** replaces COLMAP SfM + Depth Estimation + ICP Registration (3 modules with 1)
2. **Grounded SAM 2** adds semantic understanding of architectural elements (walls, doors, windows)
3. **Plane-DUSt3R** (optional) replaces the floor plan extraction pipeline with direct structural plane detection

Total estimated integration effort: 2-4 weeks for Phases 1-2, with immediate accuracy improvements expected on the first test run.

---

## Sources

- [DUSt3R GitHub -- NAVER Labs](https://github.com/naver/dust3r)
- [MASt3R GitHub -- NAVER Labs](https://github.com/naver/mast3r)
- [MASt3R -- NAVER Labs Europe Blog](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/)
- [MV-DUSt3R+ -- CVPR 2025 Oral](https://mv-dust3rp.github.io/)
- [MASt3R-SLAM -- CVPR 2025](https://github.com/rmurai0610/MASt3R-SLAM)
- [Plane-DUSt3R -- ICLR 2025](https://github.com/justacar/Plane-DUSt3R)
- [Plane-DUSt3R Paper (arXiv)](https://arxiv.org/abs/2502.16779)
- [VGGT -- CVPR 2025 Best Paper](https://github.com/facebookresearch/vggt)
- [VGGT Project Page](https://vgg-t.github.io/)
- [VGGT Paper (arXiv)](https://arxiv.org/abs/2503.11651)
- [awesome-dust3r -- Curated List](https://github.com/ruili3/awesome-dust3r)
- [Nerfstudio Documentation](https://docs.nerf.studio/)
- [Nerfstudio Instant-NGP](https://docs.nerf.studio/nerfology/methods/instant_ngp.html)
- [Meshroom / AliceVision GitHub](https://github.com/alicevision/Meshroom)
- [Meshroom 2025.1.0 Release](https://github.com/alicevision/Meshroom/releases/tag/v2025.1.0)
- [AliceVision Framework](https://alicevision.org/)
- [Pix4D Cloud API](https://developer.pix4d.com/cloud-api/index.html)
- [PIX4Dengine](https://www.pix4d.com/product/pix4dengine)
- [Agisoft Metashape Python API 2.3.0](https://www.agisoft.com/pdf/metashape_python_api_2_3_0.pdf)
- [Agisoft Metashape Pricing](https://www.agisoftmetashape.com/is-metashape-a-one-time-purchase-understanding-agisofts-licensing-model/)
- [RealityCapture Pricing Changes](https://www.capturingreality.com/pricing-changes)
- [RealityScan 2.0 Announcement](https://www.realityscan.com/en-US/news/realityscan-20-new-release-brings-powerful-new-features-to-a-rebranded-realitycapture)
- [RealityCapture CLI Scripts](https://www.capturingreality.com/realitycapture-cli-sample-scripts)
- [Azure Spatial Anchors Retirement](https://azure.microsoft.com/en-us/updates?id=azure-spatial-anchors-retirement)
- [Azure Object Anchors Retirement](https://azure.microsoft.com/en-us/updates?id=azure-object-anchors-retirement)
- [Google Cloud Vision API](https://cloud.google.com/vision)
- [Vertex AI Vision](https://cloud.google.com/vertex-ai-vision)
- [SAM 2 -- Meta AI](https://ai.meta.com/sam2/)
- [SAM 2 GitHub](https://github.com/facebookresearch/sam2)
- [SAM 2 Ultralytics Docs](https://docs.ultralytics.com/models/sam-2/)
- [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded SAM 2 GitHub](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Autodistill Grounded SAM 2](https://docs.autodistill.com/base_models/grounded-sam-2/)
- [IndoorGS -- CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Ruan_IndoorGS_Geometric_Cues_Guided_Gaussian_Splatting_for_Indoor_Scene_Reconstruction_CVPR_2025_paper.pdf)
- [PlanarGS -- NeurIPS 2025](https://arxiv.org/html/2510.23930)
- [AGS-Mesh -- 3DV 2025](https://github.com/XuqianRen/AGS_Mesh)
- [Luma AI Interactive Scenes](https://lumalabs.ai/interactive-scenes)
- [Polycam Pricing](https://poly.cam/pricing)
- [Scaniverse -- Gaussian Splat App Comparison](https://scaniverse.com/news/creating-splats-which-app-to-choose)
- [DUSt3R/MASt3R/VGGT Evaluation (2025)](https://www.tandfonline.com/doi/full/10.1080/10095020.2025.2597491)
- [LearnOpenCV -- DUSt3R Explanation](https://learnopencv.com/dust3r-geometric-3d-vision/)
- [LearnOpenCV -- MASt3R SfM](https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/)
- [LearnOpenCV -- VGGT](https://learnopencv.com/vggt-visual-geometry-grounded-transformer-3d-reconstruction/)
- [Open Source 3D Reconstruction Comparison (2025)](https://www.triposrai.com/posts/open-source-3d-reconstruction-showdown/)
- [Gaussian Splatting Guide (2026)](https://www.utsubo.com/blog/gaussian-splatting-guide)
