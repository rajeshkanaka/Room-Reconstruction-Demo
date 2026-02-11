# The Upgrade Path: VGGT + Gemini 3

**Date:** 2026-02-10
**Summary:** Two changes. One fixes broken geometry. One adds missing intelligence. Together they transform the pipeline.

---

## The Problem in One Sentence

COLMAP fails on 50% of 4-5 photo sets, and when it works, the output has no understanding of what it's looking at (no doors, no windows, no room shape awareness).

**Two layers fix this:**

| Layer | Tool | What It Fixes | Impact |
|-------|------|--------------|--------|
| **Geometry** | VGGT | COLMAP failure, measurement error, noisy point clouds | Reconstruction goes from 50% to >95% success |
| **Intelligence** | Gemini 3 | No door/window detection, no room classification, no semantic understanding | Floor plans gain architectural features for the first time |

Neither alone is enough. VGGT without Gemini gives accurate but dumb geometry. Gemini without VGGT gives smart analysis of broken 3D data. Together: accurate geometry with architectural understanding.

---

## Why These Two and Not the Other 39 Options

After evaluating 12 depth models, 13 floor plan services, 14 reconstruction tools, and 6 LLM/VLM approaches:

| Considered | Why Not |
|------------|---------|
| DA3 Metric (depth only) | Doesn't fix SfM failure. Metric depth already partially integrated via Depth Pro. |
| MoGe-2 (depth+normals) | Doesn't fix SfM failure. Better depth doesn't help if views aren't aligned. |
| Grounded SAM 2 (segmentation) | Good but Gemini 3 handles door/window detection with less integration effort and adds room classification + quality validation as bonuses. |
| MASt3R/DUSt3R | Valid VGGT alternative but older API, slower, no native COLMAP export. |
| Plane-DUSt3R | Higher risk -- trained on synthetic data only. |
| PolyRoom | Floor plan Transformer -- doesn't fix 3D quality. |
| Commercial APIs | All require video/LiDAR/hardware, not 4-5 photos. |
| GPT-4o / Claude Vision | Good for analysis but Gemini 3 has superior spatial reasoning + structured output + lower cost. |
| LLMs for depth/SfM | Research confirms VLMs are 10-15% less accurate than CV models for geometry. Don't use LLMs for math. |

---

## The Upgraded Pipeline

```
CURRENT (broken)                           UPGRADED (two additions)
============================               ============================

Photos (4-5)                               Photos (4-5)
    |                                          |
    v                                          +------------------+
[COLMAP SfM]--FAILS 50%-->                     |                  |
    |         heuristic                        v                  v
    v         fallback                   [VGGT]              [Gemini 3]
[DepthEstimator]                         single pass          API call
(relative [0,1])                             |                  |
    |                                        |--- Poses         |--- Room type
    v                                        |--- Depth         |--- Door locations
depth_to_3d_points()                         |--- Points        |--- Window locations
(hardcoded fx=500)                           |--- Focals        |--- Room shape
    |                                        |                  |
    v                                        v                  v
[ICP Registration]                       [FloorPlanGenerator v2]
(noisy)                                  metric points + semantic features
    |                                        |
    v                                        +---> Floor Plan (with doors/windows)
[FloorPlanGenerator]                         +---> Measurements JSON
(convex hull, no doors,                      +---> 3D Visualization
 ASSUMED_WIDTH scaling)                      +---> Quality Report
    |
    v
Floor Plan PNG
(20-30% error, no features)
```

**Net change:** Remove 3 fragile modules, add 1 CV model + 1 API call.

---

## Layer 1: VGGT (Geometry Foundation)

### What It Is

VGGT (Visual Geometry Grounded Transformer) is a single 1.2B-parameter model that produces camera poses + metric depth maps + dense point clouds + focal lengths in ONE forward pass. CVPR 2025 Best Paper. Works on as few as 2 images.

| Attribute | Detail |
|-----------|--------|
| Paper | CVPR 2025 Best Paper Award |
| Author | Meta/Facebook Research |
| Input | 1 to hundreds of images, no calibration needed |
| Output | Poses, depth, points, focal lengths, tracks |
| Speed | ~5-15 seconds for 5 images on GPU |
| COLMAP export | Native (cameras.bin, images.bin, points3D.bin) |
| License | Meta Research License (free for research/non-commercial) |
| GitHub | https://github.com/facebookresearch/vggt |
| VRAM | ~8-12GB (FP16), works on Apple Silicon MPS |

### What It Replaces

Three modules that currently fail independently:

| Removed Module | Why It Fails | VGGT Equivalent |
|---------------|-------------|-----------------|
| `SfMProcessor` (COLMAP) | Fails 50% on 4-5 photos | Camera poses from single pass |
| `DepthEstimator` (DA2 relative) | Outputs [0,1], needs ASSUMED_WIDTH | Metric depth maps |
| ICP/RANSAC Registration | Noisy alignment when SfM fails | Already-aligned point cloud |

Also eliminates:
- Hardcoded `CAMERA_FX=500, CAMERA_FY=500` (VGGT estimates focal lengths)
- `ASSUMED_ROOM_WIDTH_METERS` scaling hack (metric depth = real meters)
- `DEPTH_SCALE = 0.5` inverse-depth hack (direct depth, no inversion)

### Implementation

**Day 1: Install**

```bash
cd /Users/rajesh/conductor/workspaces/room-reconstruction-demo
git clone https://github.com/facebookresearch/vggt.git
cd vggt && pip install -r requirements.txt

# Verify
python -c "
import torch
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained('facebook/VGGT-1B')
print('VGGT loaded on', next(model.parameters()).device)
"
```

**Day 2-3: Create `modules/vggt_reconstructor.py`**

```python
"""
VGGT-based room reconstruction.
Replaces: SfMProcessor + DepthEstimator + ICP/RANSAC registration.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class VGGTReconstructor:
    """Single-model 3D reconstruction using VGGT."""

    def __init__(self, device: str = "auto"):
        self.model = None
        self.device = self._select_device(device)

    def _select_device(self, device: str) -> str:
        if device != "auto":
            return device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        from vggt.models.vggt import VGGT
        logger.info("Loading VGGT-1B model...")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.model = self.model.to(self.device).eval()
        logger.info(f"VGGT loaded on {self.device}")

    def reconstruct(self, images: list[np.ndarray]) -> dict:
        """
        Single-pass reconstruction from RGB images.

        Returns dict with: points_3d, colors, depth_maps,
        camera_poses, focal_lengths, confidence.
        """
        import torch
        self._load_model()

        input_tensor = self._preprocess(images).to(self.device)
        with torch.no_grad():
            predictions = self.model(input_tensor)

        return self._extract_outputs(predictions, images)

    def _preprocess(self, images: list[np.ndarray]):
        import torch, cv2
        max_size = 518
        processed = []
        for img in images:
            h, w = img.shape[:2]
            scale = min(max_size / h, max_size / w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1)
            processed.append(tensor)

        max_h = max(t.shape[1] for t in processed)
        max_w = max(t.shape[2] for t in processed)
        padded = []
        for t in processed:
            pad_h, pad_w = max_h - t.shape[1], max_w - t.shape[2]
            if pad_h > 0 or pad_w > 0:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
            padded.append(t)

        return torch.stack(padded).unsqueeze(0)

    def _extract_outputs(self, predictions: dict, original_images: list[np.ndarray]) -> dict:
        import cv2
        point_maps = predictions["point_maps"][0].cpu().numpy()
        depth_maps = predictions["depth_maps"][0].cpu().numpy()
        extrinsics = predictions["extrinsics"][0].cpu().numpy()
        intrinsics = predictions["intrinsics"][0].cpu().numpy()
        conf = predictions.get("confidence")
        if conf is not None:
            conf = conf[0].cpu().numpy()

        all_points, all_colors = [], []
        for i in range(len(original_images)):
            pts = point_maps[i].reshape(-1, 3)
            h, w = point_maps[i].shape[:2]
            colors = cv2.resize(original_images[i], (w, h)).reshape(-1, 3) / 255.0

            if conf is not None:
                mask = conf[i].reshape(-1) > 0.5
                pts, colors = pts[mask], colors[mask]

            valid = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 50.0).all(axis=1)
            all_points.append(pts[valid])
            all_colors.append(colors[valid])

        focal_lengths = []
        for i in range(len(original_images)):
            fl = float(intrinsics[i, 0, 0]) if intrinsics.ndim == 3 else float(intrinsics[i])
            focal_lengths.append(fl)

        return {
            "points_3d": np.concatenate(all_points),
            "colors": np.concatenate(all_colors),
            "depth_maps": [depth_maps[i] for i in range(len(original_images))],
            "camera_poses": [extrinsics[i] for i in range(len(original_images))],
            "focal_lengths": focal_lengths,
            "confidence": [conf[i] for i in range(len(original_images))] if conf is not None else None,
        }
```

**Day 4: Wire into `room_reconstructor.py`**

```python
# In __init__():
try:
    from modules.vggt_reconstructor import VGGTReconstructor
    self.vggt = VGGTReconstructor()
    self.use_vggt = True
except ImportError:
    self.use_vggt = False

# In reconstruct_from_arrays(), before existing pipeline:
if self.use_vggt:
    try:
        result = self.vggt.reconstruct(images)
        # Skip SfM, depth estimation, ICP -- go straight to floor plan
        floor_plan = self.floor_plan_generator.generate_floor_plan(
            result["points_3d"], is_metric=True
        )
        # ... visualization with result["points_3d"], result["colors"]
    except Exception as e:
        logger.warning(f"VGGT failed: {e}. Falling back to legacy pipeline.")
        # Legacy pipeline runs unchanged
```

**Day 5-7: Test and validate** (see Testing section below).

---

## Layer 2: Gemini 3 (Semantic Intelligence)

### What It Does

Gemini 3 analyzes room photos and returns structured JSON with room type, door/window locations, and room shape classification. This is the intelligence layer the pipeline completely lacks.

| Attribute | Detail |
|-----------|--------|
| Model | Google Gemini 3 (or Gemini 2.5 Flash for lower cost) |
| Input | Room photos (same ones fed to VGGT) |
| Output | Structured JSON: room type, doors, windows, room shape |
| Speed | ~1-3 seconds per API call |
| Cost | ~$0.01-0.05 per room (negligible) |
| License | Google API terms (commercial OK) |
| API | `google-genai` Python SDK |

### Why Gemini 3 and Not Other LLMs

| Model | Spatial Reasoning | Structured Output | Cost | Verdict |
|-------|------------------|-------------------|------|---------|
| **Gemini 3** | Pixel-precise coordinates, 3D bounding boxes | Native JSON Schema enforcement | Low | Best fit |
| GPT-4o | Good (via GPT4Scene framework) | Good | Medium | Strong alternative |
| Claude Opus 4.6 | Explicitly not optimized for spatial reasoning | Good | High | Better for validation |

### Why Not Use Gemini for Geometry Instead of VGGT

Research is definitive on this: VLMs are 10-15% less accurate than purpose-built CV models for geometric tasks. DepthLM (Sep 2025), the best VLM depth model, achieves 0.83 accuracy vs Depth Anything V2's 0.95+. No VLM can replace SfM for multi-view geometry. Use LLMs for understanding, CV for measurement.

### What It Adds to the Pipeline

The current pipeline has ZERO semantic understanding. It doesn't know what a door is. It doesn't know if a room is L-shaped. It doesn't know the difference between a kitchen and a bedroom. Gemini fixes all of this:

| Missing Capability | Gemini Provides |
|-------------------|-----------------|
| No door detection | Door count, wall position, approximate width |
| No window detection | Window count, wall position, approximate size |
| No room classification | Bedroom/kitchen/bathroom/living room/office |
| No shape awareness | Rectangular/L-shaped/U-shaped/irregular |
| No quality validation | "This looks like a bathroom, but measurements suggest 50sqm -- likely wrong" |

### Implementation

**Day 1: Set up Gemini API**

```bash
pip install google-genai
# Set API key
export GEMINI_API_KEY="your-key-here"
```

**Day 2: Create `modules/scene_analyzer.py`**

```python
"""
Gemini 3-based scene analysis.
Adds semantic understanding: room type, doors, windows, room shape.
Runs in parallel with VGGT -- does not slow down the pipeline.
"""

import json
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneAnalysis:
    room_type: str          # bedroom, kitchen, bathroom, living_room, office
    room_shape: str         # rectangular, l_shaped, u_shaped, irregular
    confidence: float       # 0.0 - 1.0
    doors: list[dict]       # [{"wall": "north", "position": "center", "width": "standard"}]
    windows: list[dict]     # [{"wall": "east", "position": "left", "size": "large"}]
    features: list[str]     # ["built-in closet", "fireplace", "kitchen island"]


ANALYSIS_PROMPT = """Analyze these room photos and return a JSON object with:

1. "room_type": one of "bedroom", "kitchen", "bathroom", "living_room", "dining_room", "office", "hallway", "other"
2. "room_shape": one of "rectangular", "l_shaped", "u_shaped", "irregular"
3. "confidence": your confidence in the analysis (0.0 to 1.0)
4. "doors": array of objects, each with:
   - "wall": which wall the door is on ("north", "south", "east", "west" -- use photo perspective)
   - "position": where on the wall ("left", "center", "right")
   - "width": estimated width ("narrow", "standard", "wide", "double")
   - "type": "interior", "exterior", "closet", "sliding"
5. "windows": array of objects, each with:
   - "wall": which wall
   - "position": where on the wall
   - "size": "small", "medium", "large", "floor_to_ceiling"
6. "features": array of notable architectural features seen

Be precise about door and window counts. Only report what you can clearly see."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "room_type": {"type": "string", "enum": ["bedroom", "kitchen", "bathroom",
                      "living_room", "dining_room", "office", "hallway", "other"]},
        "room_shape": {"type": "string", "enum": ["rectangular", "l_shaped",
                       "u_shaped", "irregular"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "doors": {"type": "array", "items": {"type": "object", "properties": {
            "wall": {"type": "string"},
            "position": {"type": "string"},
            "width": {"type": "string"},
            "type": {"type": "string"}
        }}},
        "windows": {"type": "array", "items": {"type": "object", "properties": {
            "wall": {"type": "string"},
            "position": {"type": "string"},
            "size": {"type": "string"}
        }}},
        "features": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["room_type", "room_shape", "confidence", "doors", "windows"]
}


class SceneAnalyzer:
    """Analyze room photos using Gemini 3 for semantic understanding."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model_name = model_name
        self.client = None

    def _get_client(self):
        if self.client is None:
            from google import genai
            self.client = genai.Client()
        return self.client

    def analyze(self, images: list[np.ndarray]) -> SceneAnalysis:
        """
        Analyze room photos for semantic content.

        Args:
            images: List of RGB images as numpy arrays.

        Returns:
            SceneAnalysis with room type, doors, windows, shape.
        """
        import io
        from PIL import Image

        client = self._get_client()

        # Convert numpy arrays to PIL images for API
        pil_images = []
        for img in images:
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)

        # Call Gemini with structured output
        response = client.models.generate_content(
            model=self.model_name,
            contents=[ANALYSIS_PROMPT] + pil_images,
            config={
                "response_mime_type": "application/json",
                "response_schema": ANALYSIS_SCHEMA,
            },
        )

        data = json.loads(response.text)
        logger.info(
            f"Scene analysis: {data['room_type']} ({data['room_shape']}), "
            f"{len(data.get('doors', []))} doors, "
            f"{len(data.get('windows', []))} windows, "
            f"confidence={data['confidence']:.2f}"
        )

        return SceneAnalysis(
            room_type=data["room_type"],
            room_shape=data["room_shape"],
            confidence=data["confidence"],
            doors=data.get("doors", []),
            windows=data.get("windows", []),
            features=data.get("features", []),
        )
```

**Day 3: Wire into the pipeline**

In `room_reconstructor.py`, run Gemini in parallel with VGGT:

```python
import concurrent.futures

# Run VGGT (geometry) and Gemini (semantics) in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    vggt_future = executor.submit(self.vggt.reconstruct, images)
    gemini_future = executor.submit(self.scene_analyzer.analyze, images)

    vggt_result = vggt_future.result()    # 3D geometry
    scene_info = gemini_future.result()   # Semantic analysis

# Combine: geometry + intelligence
floor_plan = self.floor_plan_generator.generate_floor_plan(
    points=vggt_result["points_3d"],
    is_metric=True,
    room_shape_hint=scene_info.room_shape,  # Guide polygon extraction
    doors=scene_info.doors,                  # Mark on floor plan
    windows=scene_info.windows,              # Mark on floor plan
    room_type=scene_info.room_type,          # Label on floor plan
)
```

**Day 4: Add door/window markers to floor plan renderer**

Update `FloorPlanGenerator` to render door and window symbols on the floor plan image using Gemini's detected positions. This is where the semantic data becomes visual.

---

## How the Two Layers Work Together

```
User uploads 4-5 room photos
         |
         +---> [VGGT] 5-15 seconds
         |     Returns: camera poses, metric depth, dense point cloud, focal lengths
         |     This is the GEOMETRY -- accurate 3D shape of the room
         |
         +---> [Gemini 3] 1-3 seconds (runs in parallel)
               Returns: room type, door/window locations, room shape, features
               This is the INTELLIGENCE -- what things ARE in the room
                    |
                    v
         [FloorPlanGenerator v2]
         Combines geometry + intelligence:
         - Room boundary from VGGT point cloud (accurate shape)
         - Room shape hint from Gemini (rectangular vs L-shaped)
         - Door symbols placed using Gemini's wall positions
         - Window symbols placed using Gemini's wall positions
         - Room label from Gemini's classification
         - Dimensions from VGGT's metric depth (real meters)
                    |
                    v
              OUTPUT:
              - Floor plan with doors, windows, labels, dimensions
              - 3D visualization with semantic colors
              - Measurements JSON
              - Quality validation from Gemini
```

**Total processing time:** ~5-15 seconds (VGGT dominates; Gemini finishes first)

---

## Expected Improvements

| Metric | Current | After VGGT | After VGGT + Gemini |
|--------|---------|-----------|---------------------|
| Reconstruction success | ~50% | >95% | >95% |
| Measurement error | 20-30% | 5-15% | 5-15% |
| Door detection | None | None | Yes (count + wall position) |
| Window detection | None | None | Yes (count + wall position) |
| Room classification | None | None | Yes (bedroom/kitchen/etc.) |
| Room shape awareness | Convex hull only | Convex hull only | L-shaped/U-shaped hints |
| Camera calibration | Hardcoded fx=500 | Estimated per image | Estimated per image |
| Processing time | 30-120s | 5-15s | 5-15s (parallel) |
| Cost per reconstruction | $0 | $0 | ~$0.01-0.05 |
| Floor plan features | Plain walls only | Plain walls only | Walls + doors + windows + labels |

---

## Hardware and Cost

### Hardware (for VGGT)

| Hardware | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 8GB (FP16) | 12-16GB |
| System RAM | 16GB | 32GB |
| Disk (model) | ~5GB | ~5GB |
| Apple Silicon | M2 Pro (16GB unified) | M3 Pro/Max |
| NVIDIA | RTX 3060 (12GB) | RTX 4070+ |

### Cost (for Gemini 3)

| Volume | Monthly Cost |
|--------|-------------|
| 10 rooms | ~$0.50 |
| 100 rooms | ~$5 |
| 1000 rooms | ~$50 |

Gemini 3 Flash is the cheapest option. Gemini 3 Pro for higher quality at ~3x cost.

---

## Fallback Chain

```
VGGT (primary geometry)
  |-- unavailable/fails -->
Legacy pipeline: COLMAP + DA2 + ICP (unchanged)

Gemini 3 (primary semantics)
  |-- unavailable/fails -->
Floor plan renders without door/window markers (still works, just less info)
```

Both layers fail gracefully. VGGT failure falls back to the existing legacy pipeline. Gemini failure means the floor plan just doesn't have door/window markers -- the geometry is still correct.

---

## Testing Plan

### VGGT Validation (Day 5-7)

1. **Known room test:** Measure a room with tape. Take 4-5 photos. Compare VGGT output dimensions to real measurements. Target: <15% error.
2. **A/B comparison:** Same photos through legacy pipeline vs VGGT. Compare point cloud density, noise, floor plan shape.
3. **Edge cases:** 2, 3, 4, 5, 10 images. Different room shapes. Low light.
4. **Performance:** Time the full pipeline. Target: <15 seconds for 5 images.

### Gemini Validation (Day 3-4)

1. **Door/window accuracy:** Take photos of rooms with known door/window count. Verify Gemini detects them correctly.
2. **Room classification:** Test on bedrooms, kitchens, bathrooms, living rooms. Check accuracy.
3. **Room shape:** Test on rectangular and L-shaped rooms. Verify shape classification.
4. **Failure modes:** Test with cluttered rooms, unusual angles, dark photos.

### Combined Validation

1. **End-to-end:** Upload photos, get floor plan with doors/windows/labels/dimensions. Visually inspect.
2. **Compare to current:** Same photos through old pipeline vs new. Side-by-side quality comparison.

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VGGT too large for GPU | Medium | Resolution scaling (384px); CPU fallback; legacy pipeline fallback |
| Meta Research License blocks commercial use | Medium | POC is fine. Commercial: negotiate with Meta or swap to MASt3R |
| Gemini API rate limits | Low | Batch requests; use Gemini Flash; cache results |
| Gemini misidentifies doors/windows | Medium | Confidence thresholding; show markers only when confidence > 0.7 |
| Gemini API cost grows | Low | ~$0.01-0.05/room is negligible; cap monthly budget |
| VGGT + Gemini adds dependency complexity | Low | Both are optional layers with graceful fallback |

---

## Implementation Timeline

| Day | Task | Layer |
|-----|------|-------|
| 1 | Install VGGT, verify it loads | Geometry |
| 2-3 | Create `vggt_reconstructor.py`, test standalone | Geometry |
| 4 | Wire VGGT into `room_reconstructor.py` | Geometry |
| 5-7 | Test VGGT with real rooms, compare to legacy | Geometry |
| 8 | Set up Gemini API, create `scene_analyzer.py` | Intelligence |
| 9 | Wire Gemini into pipeline (parallel execution) | Intelligence |
| 10 | Add door/window markers to floor plan renderer | Intelligence |
| 11-12 | End-to-end testing and tuning | Both |

---

## What This Does NOT Fix (Future Work)

Deliberately excluded to keep scope tight:

1. **Non-convex floor plan polygons** -- Gemini provides the room shape HINT (L-shaped vs rectangular), but the actual polygon extraction still uses convex hull. Fix: integrate Shapely to use the hint for non-convex boundary extraction. Natural Phase 2.
2. **Vector output (DXF/SVG)** -- Still raster PNG. Fix: ezdxf + svgwrite. Natural Phase 3.
3. **Furniture detection** -- Gemini can identify furniture but we don't render it. Natural Phase 4.

---

## Config Changes

```python
# config.py additions

# Layer 1: VGGT
ENABLE_VGGT = True
VGGT_MODEL = "facebook/VGGT-1B"

# Layer 2: Gemini 3
ENABLE_SCENE_ANALYSIS = True
GEMINI_MODEL = "gemini-3-flash-preview"  # or "gemini-3-pro" for higher quality
GEMINI_CONFIDENCE_THRESHOLD = 0.7  # Only show detected features above this
```

---

## Sources

### VGGT
- [VGGT -- CVPR 2025 Best Paper](https://arxiv.org/abs/2503.11651)
- [VGGT GitHub](https://github.com/facebookresearch/vggt)
- [VGGT Project Page](https://vgg-t.github.io/)
- [LearnOpenCV VGGT Explainer](https://learnopencv.com/vggt-visual-geometry-grounded-transformer-3d-reconstruction/)

### Gemini 3
- [Gemini 3 Announcement](https://blog.google/products-and-platforms/products/gemini/gemini-3/)
- [Gemini 3 for Architecture](https://archilabs.ai/posts/google-gemini-3-for-architecture)
- [Gemini Structured Output API](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini 2.5 Spatial Reasoning](https://developers.googleblog.com/gemini-25-for-robotics-and-embodied-intelligence/)

### LLM/VLM Limitations (Why Not for Geometry)
- [DepthLM: VLMs Still 10-15% Behind CV for Depth](https://www.alphaxiv.org/overview/2509.25413v2)
- [VLMs for Floor Plan Parsing](https://arxiv.org/html/2409.12842v1)
- [GPT4Scene: 3D Understanding from Videos](https://arxiv.org/html/2501.01428v1)

### Alternatives Evaluated (Full Research)
- [DUSt3R/MASt3R -- NAVER Labs](https://github.com/naver/dust3r)
- [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [MoGe-2 -- Microsoft](https://github.com/microsoft/MoGe)
- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Plane-DUSt3R](https://github.com/justacar/Plane-DUSt3R)
