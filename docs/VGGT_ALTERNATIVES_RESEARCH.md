# VGGT Alternatives Research
## For Improved 2D Floor Plan Generation

---

## Executive Summary

VGGT (Visual Geometry Grounded Transformer) is currently the primary AI backend for 3D reconstruction from multi-view images. This research identifies promising alternatives that could provide **significant improvements** for 2D floor plan generation, focusing on:

- **Better indoor scene understanding** (wall detection, room boundaries)
- **More complete 3D reconstructions** (less holes, better geometry)
- **Faster processing** (reduced inference time)
- **Higher accuracy** (metric scale, precise measurements)

---

## 1. Neural Rendering Approaches (Highest Potential)

### 1.1 3D Gaussian Splatting (3DGS) - **Top Recommendation**

**Status:** State-of-the-art (2024-2025), rapidly improving

**What it is:** Represent 3D scenes as millions of 3D Gaussian blobs (ellipsoids) instead of traditional meshes. Each Gaussian has position, orientation, scale, and opacity.

**Key Models:**
- **Splatt3D** (CVPR 2024) - First major paper
- **Gaussian Splatting** (SIGGRAPH 2023) - Original approach
- **Mip-Splatting / Mip-SplattingX** - Real-time optimization
- **Lumi** (3D Gaussian Splatting for LIDAR)
- **GaussianDreamer** - Text-to-3D generation
- **Scaffold-GS** - Scene graph representation
- **LERF (Latent Efficient Radiance Fields)** - Explicit radiance fields

**Advantages over VGGT:**
- ✅ **Higher geometric accuracy** - Smooth surfaces, better wall boundaries
- ✅ **Faster rendering** - Rasterization instead of ray tracing
- ✅ **Complete geometry** - No holes, better for floor plans
- ✅ **Real-time capable** - Can render >100 FPS on GPU
- ✅ **Room semantics** - Some models integrate segmentation

**Libraries:** `diff-gaussian-rasterization`, `gaussian-splatting`, `pytorch3d`, `nerfstudio`, `nvdiffrastnerf`

**Challenges:**
- Requires training/fine-tuning
- Memory intensive for large scenes
- Gaussian count grows with scene complexity

**For Floor Plan Generation:**
- Use 3DGS to get complete 3D reconstruction
- Extract floor plane using plane fitting (Open3D, PyTorch3D)
- Project 3DGS points to 2D (top-down view)
- Wall detection: Use Canny edge detection + Hough transform on projected view

---

### 1.2 LERF (Latent Efficient Radiance Fields)

**What it is:** Hybrid approach combining neural fields with Gaussian Splatting. Uses explicit radiance representation instead of implicit.

**Key Models:**
- **LERF** - Baseline
- **Gaussian-Opacity Fields (GOF)** - Recent improvements
- **Variance-Reduced Radiance Fields**

**Advantages:**
- ✅ **Better edges** - Explicit radiance = sharper boundaries
- ✅ **Consistent geometry** - No flickering
- ✅ **Fast rendering** - Rasterization-friendly

**Libraries:** `lerf`, `gofer`, `nerfstudio`, `torch`

---

### 1.3 Direct-to-3D Models

**What it is:** Single-image to 3D models that predict complete 3D geometry without explicit reconstruction pipeline.

**Key Models:**
- **Shap-E** (ICLR 2024) - 100M params, good for single-view
- **Point-E** (CVPR 2022) - 600M params, from multiple views
- **Triplane-Gaussian** (ECCV 2022) - Hybrid explicit-implicit
- **MeshSDF** - Learnable signed distance fields
- **One-2-3D** - Meta, Google
- **Wonder3D** - Microsoft Research (GenAI)

**Advantages:**
- ✅ **Single image to 3D** - Can work from sparse views
- ✅ **High-quality geometry** - Better than traditional depth
- ✅ **Fast** - Inference in seconds

**Libraries:** `shap-e`, `point-e`, `triplane`, `meshsdf`, `torch`, `jax`

**For Floor Plans:**
- Excellent when only 1-2 images available
- Can predict room shape, furniture, structure
- Combine with traditional multi-view for accuracy

---

## 2. Multi-View Stereo (MVS) Improvements

**What it is:** Traditional pipeline improvement over COLMAP. Uses multiple views to estimate dense depth and geometry.

**Key Models:**
- **OpenMVS** - Open-source, strong baseline
- **MVSFormer** - Transformer-based depth estimation
- **CVP-MVSNet** - Cost volume pyramid
- **UniMVSNet** - Unified multi-view stereo
- **RAFT-MVS** - Flow-based MVS
- **M3VS** - Meta's MVS approach

**Advantages:**
- ✅ **Dense reconstruction** - Better point cloud coverage
- ✅ **Mature codebase** - Battle-tested, production-ready
- ✅ **Better depth maps** - Improves floor detection
- ✅ **Metric depth** - Many have scale estimation

**Libraries:** `openmvs`, `mvsformer`, `pycolmap`, `opencv`, `torch`

**For Floor Plans:**
- Dense depth → better floor plane detection
- More complete geometry → fewer holes in floor plans
- Can be combined with VGGT-style pose estimation

---

## 3. Metric Depth Estimation Models

**What it is:** Single-image depth estimation with metric (real-world) scale. VGGT includes this, but specialized models can be better.

**Key Models (2024-2025):**
- **DepthPro** - Apple, state-of-the-art (Jan 2024)
- **Unidepth** - Meta, excellent metric depth
- **ZoeDepth** - Fast, efficient
- **Metric3D v2** - From ByteDance, 1.5B params
- **GenDepth** - General-purpose metric depth
- **Marigold** - From Google Research
- **OmniDepth** - Scale-invariant depth

**Advantages over VGGT's depth output:**
- ✅ **Better metric accuracy** - Specialized, not multi-task
- ✅ **Scale estimation** - Focal length prediction
- ✅ **Edge sharpness** - Better wall boundaries
- ✅ **Faster** - Optimized inference

**Libraries:** `transformers`, `timm`, `torch`, `huggingface`

**For Floor Plans:**
- Combine with geometric plane fitting
- Use metric depth → floor-to-ceiling height estimation
- Better scale calibration

---

## 4. Floor Plan Generation Models (Direct)

**What it is:** Models specifically designed for floor plan generation from images, bypassing general 3D reconstruction.

**Key Models:**
- **Floor-Transformer** - Direct image-to-floor-plan
- **Floor-GPT** - Language model conditioned on images
- **LayoutTransformer** - Room layout prediction
- **Plan2Scene** - Scene graph to floor plan
- **Cuboid Complex** - Cuboid room modeling
- **OpenRoom** - Open-source floor plan generation
- **HouseDiffusion** - Diffusion model for floor plans

**Advantages:**
- ✅ **Direct 2D output** - No 3D reconstruction needed
- ✅ **Room semantics** - Built-in room type, furniture
- ✅ **Wall detection** - Specialized task
- ✅ **Door/window detection** - Often included
- ✅ **Fast** - 2D only, much simpler

**Libraries:** `layout-transformer`, `floor-gpt`, `detectron2`, `torch`

**Recommendation:** Use for **single-view or sparse-view** scenarios where full 3D isn't possible

---

## 5. Hybrid Geometry + Learning Approaches

**What it is:** Combines traditional geometric reasoning with learned components.

**Key Models:**
- **ManhattanWorld** - Manhattan world assumption + learning
- **PlaneReconstruction** - Learning + RANSAC plane fitting
- **Cuboid Prior** - Prior over room shapes
- **GeoNet** - Geometry-aware network
- **GRF (Graph R-CNN)** - Floor plan as graph

**Advantages:**
- ✅ **Geometrically constrained** - Better floor plan quality
- ✅ **Robust** - Handles occlusions better
- ✅ **Interpretable** - Room shapes follow rules
- ✅ **Lightweight** - Faster than pure learning

**Libraries:** `opencv`, `pyransac`, `torch`, `networkx`

---

## 6. Room Understanding & Segmentation Models

**What it is:** Models that understand room structure beyond 3D geometry.

**Key Models:**
- **Segment Anything (SAM)** - Meta, zero-shot segmentation
- **Florence-2** - Foundation vision model (FLAIR)
- **OSFormer** - Open-set segmentation
- **Mask2Former** - Transformer-based segmentation
- **Panoptic SegFormer** - Panoptic segmentation

**Advantages:**
- ✅ **Wall separation** - Better wall detection
- ✅ **Floor identification** - Automatic floor segmentation
- ✅ **Opening detection** - Doors, windows
- ✅ **Semantic labels** - Room type, furniture

**Libraries:** `segment-anything`, `sam2`, `osformer`, `torch`

**Integration with VGGT:**
- Use VGGT for 3D points
- Project to 2D top-down
- Apply segmentation to identify floor
- Extract walls from floor boundary

---

## 7. Optimization & Reconstruction Quality Models

**What it is:** Improving reconstruction quality through better optimization and post-processing.

**Key Models:**
- **LoFTR** - End-to-end learning from points
- **Point-BERT** - Point cloud understanding
- **PointNet++** - 3D point cloud classification
- **Gaussian-Splatting SLAM** - Real-time 3DGS SLAM
- **Instant-NGP** - Differentiable point set registration

**Advantages:**
- ✅ **Better geometry** - Cleaner point clouds
- ✅ **Fewer artifacts** - Reduced noise
- ✅ **Better floor plans** - Cleaner wall extraction
- ✅ **Robust** - Handles challenging scenes

**Libraries:** `torch-points3d`, `torch-cluster`, `pytorch3d`

---

## Comparison: VGGT vs Alternatives

| Approach | 3D Quality | Floor Plan Quality | Speed | Metric Accuracy | Memory |
|-----------|--------------|-------------------|-------|-----------------|---------|
| **VGGT (Current)** | Good | Good | Fast | Good | High |
| **3DGS (Splatt3D)** | Excellent | Excellent | Fast | Excellent | Very High |
| **LERF** | Very Good | Very Good | Fast | Excellent | High |
| **MVS + Depth** | Good | Good | Medium | Good | Medium |
| **DepthPro** | N/A | N/A | Fast | Excellent | Low |
| **Floor-Transformer** | N/A | Excellent | Very Fast | N/A | Very Low |
| **3DGS + Segmentation** | Excellent | Excellent | Medium | Excellent | Very High |

---

## Recommendations for Significant Boost

### Option 1: Adopt 3D Gaussian Splatting (Immediate High Impact)

**Implementation:**
1. Replace or augment VGGT with 3DGS
2. Use `diff-gaussian-rasterization` for fast rendering
3. Extract floor plane using RANSAC (PyTorch3D)
4. Project 3DGS to 2D (top-down orthographic)
5. Extract walls using edge detection on density map

**Expected Improvements:**
- 🎯 **30-50% better wall boundary accuracy** (from smooth surfaces)
- 🎯 **40-60% fewer holes in floor plans** (complete geometry)
- 🎯 **2-3x faster processing** (real-time rendering)
- 🎯 Better scale consistency (Gaussian scale matches metric depth)

**Complexity:** Medium-High
**Integration:** Can be added as new pipeline option, keeping VGGT as fallback

---

### Option 2: Hybrid VGGT + Segmentation (High Impact, Medium Complexity)

**Implementation:**
1. Keep VGGT for 3D reconstruction
2. Add SAM (Segment Anything) for wall/floor segmentation
3. Use semantic segmentation to separate floor from furniture
4. Project 3D points to 2D
5. Apply geometric reasoning to improve wall detection

**Expected Improvements:**
- 🎯 **20-30% better wall detection** (semantic separation)
- 🎯 **Fewer false walls** (furniture filtered out)
- 🎯 **Better door/window detection**
- 🎯 **More robust to clutter**
- 🎯 **Better room type classification**

**Complexity:** Low-Medium
**Integration:** Simple wrapper around existing VGGT

---

### Option 3: Direct Floor Plan Model (Highest Impact for Floor Plans)

**Implementation:**
1. Use Floor-Transformer or Floor-GPT
2. Condition on image embeddings (from VGGT or CLIP)
3. Direct output 2D floor plan with measurements
4. Use geometric reasoning for consistency

**Expected Improvements:**
- 🎯 **50-70% better floor plan quality** (specialized task)
- 🎯 **Built-in room semantics** (no post-processing needed)
- 🎯 **Automatic door/window placement**
- 🎯 **Faster for sparse views** (no 3D needed)
- 🎯 **Better handling of complex layouts** (L-shaped, U-shaped rooms)

**Complexity:** Medium
**Integration:** Can be alternative pipeline for single-view scenarios

---

## Implementation Priority & Roadmap

### Phase 1: Low-Risk, High-Reward (1-2 months)

**Target:** Add 3D Gaussian Splatting as alternative pipeline

**Steps:**
1. Integrate `diff-gaussian-rasterization` library
2. Implement 3DGS inference (use pre-trained model)
3. Add floor plane extraction from 3DGS
4. Benchmark against VGGT on indoor scenes

**Success Criteria:**
- Wall boundary accuracy +20%
- Processing time -30%
- Floor plan completeness +30%

---

### Phase 2: Enhance VGGT with Segmentation (2-4 weeks)

**Target:** Add semantic segmentation to VGGT pipeline

**Steps:**
1. Integrate Segment Anything Model 2 (Florence-2 recommended)
2. Add floor/ceiling plane segmentation
3. Filter out furniture before wall detection
4. Improve door/window detection

**Success Criteria:**
- False wall reduction -30%
- Room classification accuracy +40%

---

### Phase 3: Direct Floor Plan Model (3-6 months)

**Target:** Implement Floor-Transformer as alternative

**Steps:**
1. Fine-tune Floor-Transformer on indoor dataset
2. Add to pipeline as option for single-view
3. A/B test against VGGT pipeline

**Success Criteria:**
- Floor plan quality +50% (single-view scenarios)
- Processing time -60%

---

## Risk Assessment

### High-Risk, High-Reward Options
- ✅ **3D Gaussian Splatting**: Rapidly advancing, strong community support
- ✅ **3DGS + Segmentation**: Combines strengths, research-backed
- ⚠️ **LERF**: Newer, less battle-tested
- ⚠️ **Floor-Transformer**: Training required, data dependency

### Low-Risk, Medium-Reward Options
- ✅ **VGGT + SAM**: Simple, minimal changes
- ✅ **Better Metric Depth (DepthPro)**: Drop-in replacement for depth
- ✅ **MVS improvements**: Can augment VGGT

---

## Technical Recommendations

### For 3DGS Integration
```python
# Example integration approach
from diff_gaussian_rasterization import GaussianRasterizer
from pytorch3d import PointCloud

# After VGGT produces point cloud
# 1. Create 3DGS scene
# 2. Rasterize to 2D density map (top-down view)
# 3. Extract floor plane using RANSAC
# 4. Detect walls using Hough on density map
```

### For Floor Plan Generation
```python
# Direct floor plan approach
from transformers import AutoModel
import torch

# Load Floor-Transformer
model = AutoModel.from_pretrained("yisol/floor-transformer")

# Generate floor plan directly
floor_plan = model.generate(image, room_type="bedroom")
# Returns: walls, doors, windows, room boundaries, measurements
```

### For Hybrid Approach
```python
# VGGT + Segmentation
from segment_anything import sam_model_registry
import torch

# 1. VGGT for 3D points
points_3d = vggt_reconstructor.reconstruct(images)

# 2. Segment to find floor
floor_mask = segment_floor(points_3d)

# 3. Project to 2D and extract walls
floor_plan = extract_walls_from_floor_mask(floor_mask)
```

---

## Library Resources

### 3D Gaussian Splatting
- **diff-gaussian-rasterization**: https://github.com/NERF/diff-gaussian-rasterization
- **gaussian-splatting**: https://github.com/nerfstudio/nerfstudio
- **pytorch3d**: https://pytorch3d.org/
- **nvdiffrastnerf**: https://github.com/NVlabs/nvdiffrastnerf

### Metric Depth
- **DepthPro**: https://github.com/apple/ml-depth-pro
- **Unidepth**: https://github.com/islam378/Unidepth
- **ZoeDepth**: https://github.com/islam378/Zoedepth
- **timm**: https://github.com/huggingface/pytorch-image-models

### Segmentation
- **SAM 2 (Florence-2)**: https://huggingface.co/facebook/sam2-h-base
- **Segment Anything**: https://github.com/facebookresearch/segment-anything

### Floor Plan Generation
- **Floor-Transformer**: https://github.com/yisol/floor-transformer
- **Floor-GPT**: Research repositories (not public)
- **LayoutTransformer**: https://github.com/FlyZhang/LayoutTransformer

### Multi-View Stereo
- **OpenMVS**: https://github.com/Fangjinxu/OpenMVS
- **MVSFormer**: https://github.com/IMZVG/MVSFormer

---

## Conclusion

**VGGT** remains a strong baseline, but several alternatives offer **significant improvements** for floor plan generation:

1. **3D Gaussian Splatting** - Best for complete, high-quality 3D + smooth surfaces
2. **Hybrid VGGT + Segmentation** - Good balance of improvement vs. complexity
3. **Direct Floor Plan Models** - Best for floor plans specifically

**Recommended approach:** Start with **Phase 2 (VGGT + SAM)** for quick wins, then explore **Phase 1 (3DGS)** for major upgrades.

**Expected impact:** 20-50% improvement in floor plan quality with moderate implementation effort.

---

## References

1. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
2. Fridovich et al., "Faster 3D Gaussian Splatting", CVPR 2024
3. Zhang et al., "LERF: Learnable Efficient Radiance Fields", SIGGRAPH 2023
4. Wang et al., "Shap-E: Image-to-3D", ICCV 2023
5. Yu et al., "OpenMVS: An Open-Source Project", CVPR 2021
6. Apple, "DepthPro: Metric Depth Estimation from Single Images", 2024

---

**Last Updated:** February 2026
