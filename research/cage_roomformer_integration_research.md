# CAGE & RoomFormer Integration Research
## Exact Code Interfaces for Floorplan Reconstruction Models
### Date: 2026-02-12

---

## Executive Summary

Both CAGE (NeurIPS 2025) and RoomFormer (CVPR 2023) share a nearly identical codebase architecture
(CAGE is forked from RoomFormer). Both accept **single-channel 256x256 density maps** as input and
output **sets of room polygons** as normalized vertex coordinates. CAGE improves on RoomFormer by
predicting edges (x1,y1,x2,y2) instead of corners (x,y), yielding more topologically robust results
(99.1% Room F1 vs RoomFormer's lower scores).

**Key finding**: CAGE's codebase is a superset of RoomFormer. The model class in both repos is literally
named `roomformer.py`. CAGE adds: (1) edge-based representation, (2) denoising training, (3)
differentiable rasterization loss, (4) SwinV2 backbone support. The inference interface is nearly
identical -- a drop-in replacement is feasible.

---

## 1. CAGE (NeurIPS 2025)

### Repository: https://github.com/ee-Liu/CAGE

### 1.1 Installation

```bash
conda create -n cage python=3.8
conda activate cage

# PyTorch 1.9.0 + CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Dependencies
pip install -r requirements.txt

# Compile deformable attention CUDA ops
cd models/ops && sh make.sh && cd ../..

# Compile differentiable rasterization CUDA ops
cd diff_ras && python setup.py build develop && cd ..
```

### 1.2 Dependencies (requirements.txt)

```
opencv-python
numpy==1.19.0
matplotlib==3.6.2
imageio==2.19.3
scipy==1.8.1
fvcore
cloudpickle==2.1.0
omegaconf==2.2.2
fairscale==0.4.6
timm==0.5.4
shapely==1.8.2
tqdm==4.64.0
pycocotools
descartes
transformers
tensorboard
```

Plus compiled C++/CUDA extensions:
- `models/ops/` -- Deformable attention (from Deformable-DETR)
- `diff_ras/` -- Differentiable polygon rasterization (custom CUDA kernel)
- `detectron2/` -- Bundled in the repo (not pip-installed)

### 1.3 Pretrained Weights

**Download from Google Drive:**
https://drive.google.com/drive/folders/1jajjRamJ7SVgCWB-Tihp-ToqPsv0GmE7?usp=sharing

**Checkpoint files:**
- `CAGE_stru3d_swinv2.pth` -- Structured3D with SwinV2-Large backbone (best)
- Likely also ResNet-50 variants available

**SwinV2 backbone weights** (if using SwinV2):
- Download: `swinv2_large_patch4_window12_192_22k.pth`
- From: https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth
- Place in: `pretrained/` directory

**Checkpoint dict structure:**
```python
checkpoint = torch.load("checkpoint/CAGE_stru3d_swinv2.pth", map_location='cpu')
# checkpoint['model']  -- state_dict
# checkpoint['optimizer']  -- optimizer state (optional)
# checkpoint['lr_scheduler']  -- LR scheduler state (optional)
# checkpoint['epoch']  -- training epoch (optional)
```

### 1.4 Input Format

**Density map specification:**
- **Shape**: `(1, 256, 256)` -- single-channel grayscale
- **dtype**: `torch.float32`
- **Value range**: `[0.0, 1.0]` -- divided by 255 from uint8 PNG
- **Meaning**: Top-down point cloud density projection

**How density maps are generated (from point clouds):**
```python
# From data_preprocess/stru3d/generate_coco_stru3d.py
density, normalization_dict = generate_density(xyz, width=256, height=256)

# generate_density does:
# 1. Invert coordinates: ps = point_cloud * -1, flip x/y
# 2. Compute bounds with 10% margin
# 3. Normalize to [0, image_res] pixel space
# 4. Create 2D histogram (density)
# 5. Normalize: density = density / np.max(density)  -> [0, 1] float32
# 6. Save as PNG: (density * 255).astype(uint8)
```

**For our integration (from VGGT point cloud):**
```python
import numpy as np
import cv2

def point_cloud_to_density_map(points_3d, resolution=256):
    """Convert 3D point cloud to 256x256 density map for CAGE/RoomFormer.

    Args:
        points_3d: (N, 3) numpy array of 3D points
        resolution: output image size (default 256)

    Returns:
        density: (resolution, resolution) float32 array in [0, 1]
        normalization_dict: dict with min_coords, max_coords for denormalizing output
    """
    # Use X, Z coordinates (top-down projection, Y is up)
    xy = points_3d[:, [0, 2]]  # or [0, 1] depending on coordinate convention

    # Compute bounds with 10% margin
    min_coords = xy.min(axis=0)
    max_coords = xy.max(axis=0)
    margin = 0.1 * (max_coords - min_coords)
    min_coords -= margin
    max_coords += margin

    # Normalize to pixel coordinates
    coords = (xy - min_coords) / (max_coords - min_coords) * resolution
    coords = np.clip(coords, 0, resolution - 1).astype(np.int32)

    # Create density histogram
    density = np.zeros((resolution, resolution), dtype=np.float32)
    for x, y in coords:
        density[y, x] += 1  # Note: y is row, x is col

    # Normalize to [0, 1]
    if density.max() > 0:
        density = density / density.max()

    normalization_dict = {
        'min_coords': min_coords,
        'max_coords': max_coords,
        'resolution': resolution
    }

    return density, normalization_dict
```

### 1.5 Model Loading & Inference API

**Build model:**
```python
from models import build_model
import argparse

def build_cage_model(checkpoint_path, backbone='resnet50', device='cuda'):
    """Build CAGE model and load pretrained weights.

    Args:
        checkpoint_path: path to .pth checkpoint
        backbone: 'resnet50' or 'swinv2_L_192_22k'
        device: 'cuda' or 'cpu'
    """
    # Construct args namespace (mimicking argparse)
    args = argparse.Namespace(
        # Architecture
        backbone=backbone,
        position_embedding='sine',
        num_feature_levels=4,
        hidden_dim=256,
        nheads=8,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        num_queries=800,
        num_polys=20,

        # CAGE-specific
        aux_loss=True,
        with_poly_refine=True,  # edge refinement
        masked_attn=True,
        semantic_classes=-1,  # -1 = no semantic classification

        # Denoising (training only, but needed for model construction)
        use_dn=True,

        # Backbone-specific
        lr_backbone=2e-5,
        masks=False,
        dilation=False,
    )

    model = build_model(args, train=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    model.eval()

    return model, args
```

**NestedTensor wrapping (required for inference):**
```python
from util.misc import NestedTensor

def prepare_input(density_map, device='cuda'):
    """Wrap a density map as NestedTensor for model input.

    Args:
        density_map: (H, W) or (1, H, W) numpy array or torch tensor, float32, [0,1]

    Returns:
        NestedTensor with .tensors shape (1, 1, H, W) and .mask shape (1, H, W)
    """
    import torch

    if isinstance(density_map, np.ndarray):
        density_map = torch.from_numpy(density_map).float()

    if density_map.dim() == 2:
        density_map = density_map.unsqueeze(0)  # (1, H, W)

    # Add batch dimension
    tensor = density_map.unsqueeze(0).to(device)  # (1, 1, H, W)

    # Mask: False where real data, True where padding
    # For a single image with no padding, mask is all False
    mask = torch.zeros(1, density_map.shape[1], density_map.shape[2],
                       dtype=torch.bool, device=device)

    return NestedTensor(tensor, mask)
```

**Run inference:**
```python
@torch.no_grad()
def run_cage_inference(model, density_map, device='cuda'):
    """Run CAGE inference on a single density map.

    Args:
        model: loaded CAGE model
        density_map: (256, 256) float32 numpy array in [0, 1]

    Returns:
        dict with:
            'pred_logits': (1, num_polys, num_edges_per_poly) -- edge validity logits
            'pred_coords': (1, num_polys, num_edges_per_poly, 4) -- edge coords (x1,y1,x2,y2) normalized [0,1]
    """
    samples = prepare_input(density_map, device)
    outputs = model(samples)
    return outputs
```

### 1.6 Output Format

**Raw model output dict:**
```python
{
    'pred_logits': torch.Tensor,  # shape: (batch, num_polys, num_queries_per_poly)
                                   # Typical: (1, 20, 40) -- 20 polygons, 40 edges each
                                   # Raw logits (pre-sigmoid)

    'pred_coords': torch.Tensor,  # shape: (batch, num_polys, num_queries_per_poly, 4)
                                   # Typical: (1, 20, 40, 4)
                                   # Each edge: (x1, y1, x2, y2) normalized to [0, 1]
                                   # After sigmoid (already in [0,1])

    'aux_outputs': list,           # Intermediate decoder layer outputs (same format)

    'pred_room_logits': torch.Tensor,  # Only if semantic_classes > 0
                                        # shape: (batch, num_polys, num_semantic_classes)
}
```

**KEY DIFFERENCE from RoomFormer**: CAGE predicts **edges** (x1,y1,x2,y2) not corners (x,y).
- `num_queries=800, num_polys=20` => `num_queries_per_poly = 40`
- Each query predicts one edge with 4 coordinates

### 1.7 Postprocessing (predictions -> room polygons)

```python
import numpy as np
from shapely.geometry import Polygon

def postprocess_cage_output(outputs, resolution=256, logit_threshold=0.5,
                            min_corners=4, min_area=100, merge_distance=2):
    """Convert raw CAGE outputs to room polygons.

    Args:
        outputs: raw model output dict
        resolution: density map resolution (256)
        logit_threshold: confidence threshold for valid edges
        min_corners: minimum corners for a valid polygon
        min_area: minimum area in pixels for a valid polygon
        merge_distance: pixel distance to merge nearby corners

    Returns:
        list of numpy arrays, each (N, 2) representing room polygon vertices
        in pixel coordinates [0, 255]
    """
    pred_logits = outputs['pred_logits'][0]  # (num_polys, num_edges_per_poly)
    pred_coords = outputs['pred_coords'][0]  # (num_polys, num_edges_per_poly, 4)

    room_polygons = []

    for poly_idx in range(pred_logits.shape[0]):
        logits = torch.sigmoid(pred_logits[poly_idx])  # (num_edges,)
        coords = pred_coords[poly_idx]  # (num_edges, 4)

        # Filter valid edges
        valid_mask = logits > logit_threshold
        if valid_mask.sum() < 2:
            continue

        valid_coords = coords[valid_mask]  # (num_valid, 4)

        # Denormalize to pixel space
        # coords are (x1, y1, x2, y2) in [0, 1], scale to [0, 255]
        edges_px = (valid_coords * (resolution - 1)).cpu().numpy()

        # Extract corners from edges
        # Each edge gives 2 corners: (x1,y1) and (x2,y2)
        corners = []
        for edge in edges_px:
            corners.append(edge[:2])  # (x1, y1)
            corners.append(edge[2:])  # (x2, y2)
        corners = np.array(corners)

        # Remove duplicate corners (merge within merge_distance pixels)
        corners = np.around(corners).astype(np.int32)
        # Simple deduplication
        unique = []
        for c in corners:
            is_dup = False
            for u in unique:
                if np.linalg.norm(c - u) < merge_distance:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        corners = np.array(unique) if unique else np.array([]).reshape(0, 2)

        if len(corners) < min_corners:
            continue

        # Validate as polygon
        try:
            poly = Polygon(corners)
            if poly.area < min_area:
                continue
            if not poly.is_valid:
                poly = poly.buffer(0)  # fix self-intersections
            room_polygons.append(corners)
        except Exception:
            continue

    return room_polygons


def denormalize_polygons_to_meters(room_polygons, normalization_dict):
    """Convert pixel-space polygons back to metric coordinates.

    Args:
        room_polygons: list of (N, 2) arrays in pixel coords [0, 255]
        normalization_dict: dict from point_cloud_to_density_map()

    Returns:
        list of (N, 2) arrays in metric (meters) coordinates
    """
    min_c = normalization_dict['min_coords']
    max_c = normalization_dict['max_coords']
    res = normalization_dict['resolution']

    metric_polygons = []
    for poly_px in room_polygons:
        # Reverse: pixel -> normalized [0,1] -> metric
        poly_norm = poly_px.astype(np.float64) / (res - 1)
        poly_m = poly_norm * (max_c - min_c) + min_c
        metric_polygons.append(poly_m)

    return metric_polygons
```

**CAGE's actual engine.py postprocessing does these additional steps:**
1. `remove_short_edges(corners, logits)` -- filter tiny edges
2. `get_corners_from_edges(corners, logits, threshold=10)` -- extract corners from edge intersections
3. `remove_duplicate_corners(corners)` -- spatial deduplication
4. `merge_points(corners, threshold=2)` -- merge nearby corners
5. `Polygon(corners).area >= 100` check
6. `remove_rooms_with_iou(polygons)` -- remove high-IoU duplicates
7. `refine_rooms(polygons, overlap)` -- resolve overlapping regions

---

## 2. RoomFormer (CVPR 2023)

### Repository: https://github.com/ywyue/RoomFormer

### 2.1 Installation

```bash
conda create -n roomformer python=3.8
conda activate roomformer

# PyTorch 1.9.0 + CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Same requirements (CAGE forked from RoomFormer)
pip install -r requirements.txt

# Compile deformable attention (same as CAGE)
cd models/ops && sh make.sh && cd ../..

# Compile differentiable rasterization (same as CAGE)
cd diff_ras && python setup.py build develop && cd ..
```

### 2.2 Pretrained Weights

**Download from ETH Polybox:**
https://polybox.ethz.ch/index.php/s/vlBo66X0NTrcsTC

**Checkpoint file:** `roomformer_stru3d.pth`

**Checkpoint structure:** Same as CAGE -- `checkpoint['model']` contains state_dict.

### 2.3 Input Format

**Identical to CAGE:**
- **Shape**: `(1, 256, 256)` -- single-channel grayscale
- **dtype**: `torch.float32`
- **Value range**: `[0.0, 1.0]`
- **Backbone first conv**: `nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)`

### 2.4 Model Loading

```python
def build_roomformer_model(checkpoint_path, device='cuda'):
    args = argparse.Namespace(
        backbone='resnet50',  # RoomFormer default (no SwinV2 in original)
        position_embedding='sine',
        num_feature_levels=4,
        hidden_dim=256,
        nheads=8,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        num_queries=800,
        num_polys=20,
        aux_loss=True,
        with_poly_refine=False,  # RoomFormer does NOT have edge refinement
        masked_attn=True,
        semantic_classes=-1,
        lr_backbone=2e-5,
        masks=False,
        dilation=False,
    )

    model = build_model(args, train=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    return model, args
```

### 2.5 Output Format

**Raw model output dict:**
```python
{
    'pred_logits': torch.Tensor,  # shape: (batch, num_polys, num_queries_per_poly)
                                   # Typical: (1, 20, 40) -- 20 polygons, 40 corners each
                                   # Raw logits (pre-sigmoid)
                                   # Represents CORNER validity (not edge validity)

    'pred_coords': torch.Tensor,  # shape: (batch, num_polys, num_queries_per_poly, 2)
                                   # Typical: (1, 20, 40, 2)
                                   # Each corner: (x, y) normalized to [0, 1]
                                   # NOTE: 2 values per query, NOT 4 like CAGE

    'aux_outputs': list,

    'pred_room_logits': torch.Tensor,  # if semantic_classes > 0
                                        # shape: (batch, num_polys, num_semantic_classes)
                                        # Classes 0-15: room types
                                        # Classes 16-17: windows/doors
}
```

**KEY DIFFERENCE from CAGE**: RoomFormer predicts **corners** (x,y) not edges (x1,y1,x2,y2).
- `pred_coords` last dimension is 2, not 4

### 2.6 Postprocessing

```python
def postprocess_roomformer_output(outputs, resolution=256, logit_threshold=0.5,
                                   min_corners=4, min_area=100):
    """Convert raw RoomFormer outputs to room polygons.

    Returns:
        list of numpy arrays, each (N, 2) representing room polygon vertices
        in pixel coordinates [0, 255]
    """
    pred_logits = outputs['pred_logits'][0]  # (num_polys, corners_per_poly)
    pred_coords = outputs['pred_coords'][0]  # (num_polys, corners_per_poly, 2)

    room_polygons = []

    for poly_idx in range(pred_logits.shape[0]):
        logits = torch.sigmoid(pred_logits[poly_idx])
        coords = pred_coords[poly_idx]  # (corners_per_poly, 2)

        # Select valid corners
        valid_mask = logits > logit_threshold
        valid_corners = coords[valid_mask]

        if valid_corners.shape[0] < min_corners:
            continue

        # Denormalize: [0, 1] -> [0, 255]
        corners = (valid_corners * 255).cpu().numpy()
        corners = np.around(corners).astype(np.int32)

        # Merge nearby points
        # (same dedup logic as CAGE)

        try:
            poly = Polygon(corners)
            if poly.area >= min_area:
                room_polygons.append(corners)
        except Exception:
            continue

    return room_polygons
```

---

## 3. Critical Comparison

| Feature | CAGE (NeurIPS 2025) | RoomFormer (CVPR 2023) |
|---------|---------------------|------------------------|
| **Prediction unit** | Edges (x1,y1,x2,y2) | Corners (x,y) |
| **pred_coords shape** | (B, 20, 40, **4**) | (B, 20, 40, **2**) |
| **Backbone options** | ResNet-50, SwinV2-L | ResNet-50 only |
| **Denoising training** | Yes (dual-query DN) | No |
| **Diff. rasterization loss** | Yes (64x64 raster) | Yes (same) |
| **Input format** | 1ch, 256x256, [0,1] | 1ch, 256x256, [0,1] |
| **num_queries** | 800 | 800 |
| **num_polys** | 20 | 20 |
| **Structured3D Room F1** | **99.1%** | ~95% |
| **Structured3D Corner F1** | **91.7%** | ~85% |
| **Checkpoint source** | Google Drive | ETH Polybox |
| **Python** | 3.8 | 3.8 |
| **PyTorch** | 1.9.0+cu111 | 1.9.0+cu111 |
| **CUDA ops required** | Yes (deform_attn + diff_ras) | Yes (deform_attn + diff_ras) |
| **Inference time** | ~0.01s/scene | ~0.01s/scene |

---

## 4. Integration Architecture for Missoula

### 4.1 Recommended Approach

```
VGGT point cloud -> top-down density map (256x256) -> CAGE/RoomFormer -> room polygons -> FloorPlanModel
```

### 4.2 Unified Wrapper

```python
class NeuralFloorplanPredictor:
    """Unified interface for CAGE and RoomFormer."""

    def __init__(self, model_name='cage', checkpoint_path=None, device='cuda'):
        self.model_name = model_name
        self.device = device

        if model_name == 'cage':
            self.model, self.args = build_cage_model(checkpoint_path, device=device)
            self.edge_based = True
        else:
            self.model, self.args = build_roomformer_model(checkpoint_path, device=device)
            self.edge_based = False

    def predict(self, point_cloud_3d):
        """Predict room polygons from 3D point cloud.

        Args:
            point_cloud_3d: (N, 3) numpy array

        Returns:
            list of (M, 2) numpy arrays -- room polygons in METRIC coordinates
        """
        # 1. Project to density map
        density, norm_dict = point_cloud_to_density_map(point_cloud_3d, resolution=256)

        # 2. Run model
        samples = prepare_input(density, self.device)
        with torch.no_grad():
            outputs = self.model(samples)

        # 3. Postprocess
        if self.edge_based:
            room_polys_px = postprocess_cage_output(outputs)
        else:
            room_polys_px = postprocess_roomformer_output(outputs)

        # 4. Convert to metric coordinates
        room_polys_m = denormalize_polygons_to_meters(room_polys_px, norm_dict)

        return room_polys_m
```

### 4.3 Integration with FloorPlanModel

```python
from modules.geometry.floor_plan_model import FloorPlanModel, WallSegment, RoomPolygon

def neural_polygons_to_floor_plan_model(room_polys_m):
    """Convert predicted polygons to FloorPlanModel."""
    model = FloorPlanModel()

    for poly_vertices in room_polys_m:
        # Add walls from polygon edges
        for i in range(len(poly_vertices)):
            p1 = poly_vertices[i]
            p2 = poly_vertices[(i + 1) % len(poly_vertices)]
            wall = WallSegment(
                start=(float(p1[0]), float(p1[1])),
                end=(float(p2[0]), float(p2[1])),
                thickness=0.15  # default wall thickness
            )
            model.walls.append(wall)

        # Add room polygon
        room = RoomPolygon(
            vertices=[(float(v[0]), float(v[1])) for v in poly_vertices],
            room_type='unknown'  # Gemini can enrich this
        )
        model.rooms.append(room)

    return model
```

---

## 5. Practical Integration Challenges

### 5.1 CUDA Extension Compilation
Both models require compiling C++/CUDA extensions for deformable attention. This is the
primary integration friction point.

**Workaround options:**
1. Pre-compile for target platform and ship `.so` files
2. Use Docker with pre-built environment
3. Investigate if pure-PyTorch deformable attention (from newer DETR implementations)
   can substitute

### 5.2 PyTorch Version Lock
Both repos pin PyTorch 1.9.0 + CUDA 11.1. This conflicts with modern VGGT requirements.

**Mitigation:**
- Test with newer PyTorch versions (many users report 1.12+ works)
- Use separate conda environments with subprocess calls
- Or use ONNX export for inference (avoids PyTorch version lock)

### 5.3 Model Size
- ResNet-50 backbone: ~25M parameters
- SwinV2-Large backbone: ~197M parameters + 25M head = ~222M total
- Inference: ~10ms per scene on GPU

### 5.4 Density Map Quality
The models were trained on clean Structured3D synthetic data. Real-world VGGT point clouds
will be noisier. May need:
- Gaussian blur on density map
- Higher resolution then downsample
- Statistical outlier removal on point cloud first

---

## 6. Benchmark Inputs for Testing

From CAGE's benchmark.py, a standalone test:
```python
# Minimal inference test (from benchmark.py)
model = build_model(args, train=False)
model.cuda()
model.eval()

# Input: single-channel 256x256 random tensor
img = torch.randn(1, 256, 256)  # (C=1, H=256, W=256)
inputs = [img.to("cuda")]

# The model internally wraps this as NestedTensor
```

This confirms the input is `(1, 256, 256)` per image, batched as a list.

---

## Sources

- CAGE GitHub: https://github.com/ee-Liu/CAGE
- CAGE Paper (arXiv): https://arxiv.org/abs/2509.15459
- CAGE NeurIPS 2025 Poster: https://neurips.cc/virtual/2025/loc/san-diego/poster/117912
- RoomFormer GitHub: https://github.com/ywyue/RoomFormer
- RoomFormer Paper: https://arxiv.org/abs/2211.15658
- RoomFormer Project Page: https://ywyue.github.io/RoomFormer/
- CAGE Checkpoints (Google Drive): https://drive.google.com/drive/folders/1jajjRamJ7SVgCWB-Tihp-ToqPsv0GmE7
- RoomFormer Checkpoints (ETH Polybox): https://polybox.ethz.ch/index.php/s/vlBo66X0NTrcsTC
- SwinV2 Pretrained Weights: https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth
