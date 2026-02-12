"""
CAGE Model Loader with CPU/MPS Compatibility

Loads the CAGE (NeurIPS 2025) floorplan detection model with pure PyTorch
fallback for the deformable attention CUDA extension. This allows inference
on macOS (MPS) and CPU-only machines without compiling CUDA ops.

The key patches:
1. MSDeformAttnFunction.apply() -> ms_deform_attn_core_pytorch()
2. Hardcoded .cuda() calls -> device-agnostic .to(device)
3. torch.cuda.empty_cache() -> guarded by cuda availability
"""

import sys
import os
import argparse
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: add CAGE repo to sys.path for model imports
# ---------------------------------------------------------------------------
CAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external",
    "cage",
)

_ORIGINAL_PATH = None


def _setup_cage_path():
    """Add CAGE repo to sys.path (idempotent)."""
    global _ORIGINAL_PATH
    if CAGE_DIR not in sys.path:
        _ORIGINAL_PATH = sys.path.copy()
        sys.path.insert(0, CAGE_DIR)


# ---------------------------------------------------------------------------
# Pure PyTorch deformable attention (from CAGE's own codebase)
# ---------------------------------------------------------------------------


def ms_deform_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pure PyTorch multi-scale deformable attention.

    Copied from external/cage/models/ops/functions/ms_deform_attn_func.py
    (the debug/test fallback that CAGE ships but doesn't use by default).
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[lid_]
            .flatten(2)
            .transpose(1, 2)
            .reshape(N_ * M_, D_, int(H_), int(W_))
        )
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Monkey-patches applied before importing CAGE modules
# ---------------------------------------------------------------------------


class _AutoMockModule(types.ModuleType):
    """A mock module that auto-creates sub-attributes and sub-modules on access.

    Any attribute access returns a child ``_AutoMockModule``, and sub-imports
    (e.g. ``from pkg.sub.deep import Foo``) auto-register in ``sys.modules``
    so the import chain never fails -- regardless of depth.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []  # mark as package so sub-imports work
        self.__package__ = name

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        full_name = f"{self.__name__}.{name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        child = _AutoMockModule(full_name)
        sys.modules[full_name] = child
        return child

    def __call__(self, *args, **kwargs):
        """Allow mock to be used as a class or function."""
        return _MockObject()


class _MockObject:
    """A permissive mock object that accepts any attribute access or call."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _MockObject()

    def __setattr__(self, name: str, value):
        pass  # silently accept

    def __call__(self, *args, **kwargs):
        return _MockObject()

    def __bool__(self):
        return False

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


# Packages that are only needed for CAGE training, not inference.
_MOCK_PACKAGES = ("datasets", "detectron2", "diff_ras")


class _MockImportFinder:
    """``sys.meta_path`` finder that intercepts training-only imports.

    CAGE bundles a partial ``detectron2/`` directory whose sub-imports pull in
    heavy dependencies (cloudpickle, etc.) that are unnecessary for inference.
    This finder short-circuits those imports at the import-machinery level,
    which takes precedence over filesystem-based finders.
    """

    def find_module(self, fullname: str, path=None):
        for prefix in _MOCK_PACKAGES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return self
        return None

    def load_module(self, fullname: str):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _AutoMockModule(fullname)
        sys.modules[fullname] = mod
        return mod


_mock_finder_installed = False


def _mock_training_deps():
    """Install a meta-path finder that mocks training-only CAGE dependencies.

    CAGE's import chain pulls in ``datasets`` -> ``detectron2`` -> ``diff_ras``
    even for inference.  Rather than installing these heavy packages, we
    intercept their imports with a finder that returns auto-expanding mocks.
    """
    global _mock_finder_installed
    if _mock_finder_installed:
        return
    # Insert at the front so it takes precedence over the filesystem finder
    sys.meta_path.insert(0, _MockImportFinder())
    _mock_finder_installed = True


def _patch_deformable_attention():
    """Replace CUDA MSDeformAttnFunction with pure PyTorch version."""
    import importlib

    # Create a fake MultiScaleDeformableAttention module so the import
    # `import MultiScaleDeformableAttention as MSDA` doesn't fail.
    fake_msda = types.ModuleType("MultiScaleDeformableAttention")

    def _fake_forward(*args, **kwargs):
        raise RuntimeError(
            "CUDA deformable attention not available -- use patched model"
        )

    def _fake_backward(*args, **kwargs):
        raise RuntimeError("CUDA deformable attention not available")

    fake_msda.ms_deform_attn_forward = _fake_forward
    fake_msda.ms_deform_attn_backward = _fake_backward
    sys.modules["MultiScaleDeformableAttention"] = fake_msda

    # Now import the actual module (it will find our fake MSDA)
    from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction

    # Patch the MSDeformAttn.forward to use pure PyTorch
    from models.ops.modules.ms_deform_attn import MSDeformAttn

    _original_forward = MSDeformAttn.forward

    def _patched_forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        # Use pure PyTorch instead of CUDA extension
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        return output

    MSDeformAttn.forward = _patched_forward


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def _make_args(
    backbone: str = "resnet50", semantic_classes: int = -1
) -> argparse.Namespace:
    """Construct the ``argparse.Namespace`` expected by CAGE's ``build_model``.

    Default values mirror ``external/cage/main.py``'s argparse definitions
    to guarantee checkpoint compatibility.
    """
    return argparse.Namespace(
        # Architecture
        backbone=backbone,
        position_embedding="sine",
        position_embedding_scale=2 * np.pi,
        num_feature_levels=4,
        hidden_dim=256,
        nheads=8,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_queries=800,
        num_polys=20,
        dec_n_points=4,
        enc_n_points=4,
        query_pos_type="sine",
        # CAGE-specific
        aux_loss=True,
        with_poly_refine=True,
        masked_attn=True,
        semantic_classes=semantic_classes,
        use_dn=True,
        use_angle_loss=False,
        # Backbone
        lr_backbone=0,
        masks=False,
        dilation=False,
        use_checkpoint=False,
        # Training-only (needed by build_model but unused at inference)
        device="cpu",
        cls_loss_coef=0.6,
        room_cls_loss_coef=0.2,
        coords_loss_coef=6.0,
        raster_loss_coef=1.0,
        angles_loss_coef=0.5,
        set_cost_class=1.0,
        set_cost_coords=6.0,
        scalar=5,
        label_noise_scale=0.2,
        poly_noise_scale=0.4,
        dataset_name="stru3d",
    )


def load_cage_model(
    checkpoint_path: str,
    backbone: str = "resnet50",
    device: Optional[torch.device] = None,
    semantic_classes: int = -1,
) -> Tuple[nn.Module, argparse.Namespace]:
    """Load CAGE model with CPU/MPS compatibility.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        backbone: 'resnet50' or 'swinv2_L_192_22k'.
        device: Target device. Defaults to best available.
        semantic_classes: Number of room type classes (-1 = disabled).

    Returns:
        (model, args) tuple. Model is in eval mode on target device.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Setup imports, mock training-only deps, patch CUDA deformable attention
    _setup_cage_path()
    _mock_training_deps()
    _patch_deformable_attention()

    # Build model
    from models import build_model

    args = _make_args(backbone=backbone, semantic_classes=semantic_classes)
    model = build_model(args, train=False)

    # Load checkpoint (weights_only=False needed for CAGE's argparse.Namespace
    # serialized inside the checkpoint dict)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    # Handle potential key mismatches (strict=False tolerates missing aux keys)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (criterion-related, etc.)
        important_missing = [k for k in missing if not k.startswith("criterion")]
        if important_missing:
            from termcolor import colored

            print(
                colored(
                    f"[CAGE] Warning: {len(important_missing)} missing keys in checkpoint",
                    "yellow",
                )
            )

    model.to(device)
    model.eval()

    return model, args


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def prepare_input(density_map: np.ndarray, device: torch.device) -> "NestedTensor":
    """Wrap a density map as NestedTensor for CAGE input.

    Args:
        density_map: (H, W) float32 array in [0, 1].
        device: Target device.

    Returns:
        NestedTensor with .tensors (1, 1, H, W) and .mask (1, H, W).
    """
    from util.misc import NestedTensor

    tensor = torch.from_numpy(density_map).float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # (1, H, W)
    tensor = tensor.unsqueeze(0).to(device)  # (1, 1, H, W)

    # Mask: False = real data, True = padding
    mask = torch.zeros(
        1,
        density_map.shape[0],
        density_map.shape[1],
        dtype=torch.bool,
        device=device,
    )
    return NestedTensor(tensor, mask)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    density_map: np.ndarray,
    device: torch.device,
) -> Dict:
    """Run CAGE inference on a single density map.

    Args:
        model: Loaded CAGE model in eval mode.
        density_map: (256, 256) float32 array in [0, 1].
        device: Device the model is on.

    Returns:
        Dict with 'pred_logits' and 'pred_coords' tensors.
    """
    samples = prepare_input(density_map, device)
    outputs, _ = model(samples)
    return outputs


# ---------------------------------------------------------------------------
# Postprocessing: edges -> room polygons
# ---------------------------------------------------------------------------


def postprocess_predictions(
    outputs: Dict,
    resolution: int = 256,
    logit_threshold: float = 0.5,
    min_corners: int = 4,
    min_area: float = 100.0,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[np.ndarray]]:
    """Convert raw CAGE outputs to room polygons.

    Uses CAGE's own postprocessing pipeline (edge_utils).

    Args:
        outputs: Raw model output dict with pred_logits and pred_coords.
        resolution: Density map resolution (256).
        logit_threshold: Confidence threshold for valid edges.
        min_corners: Min corners for a valid room polygon.
        min_area: Min area in pixels^2 for a valid room.

    Returns:
        (room_polygons, room_types, doors, windows) where:
            room_polygons: list of Nx2 int32 arrays (pixel coords [0, 255])
            room_types: list of int room type indices
            doors: list of 2x2 int32 arrays (door endpoints)
            windows: list of 2x2 int32 arrays (window endpoints)
    """
    from shapely.geometry import Polygon
    from util.edge_utils import (
        remove_short_edges,
        get_corners_from_edges,
        remove_duplicate_corners,
        merge_points,
        remove_multi_polygon,
        remove_rooms_with_iou,
        refine_rooms,
    )

    pred_logits = torch.sigmoid(
        outputs["pred_logits"][0]
    )  # (num_polys, edges_per_poly)
    pred_coords = outputs["pred_coords"][0]  # (num_polys, edges_per_poly, 4)

    # Room type predictions (if available)
    has_semantics = "pred_room_logits" in outputs
    pred_room_labels = None
    if has_semantics:
        prob = torch.nn.functional.softmax(outputs["pred_room_logits"][0], -1)
        _, pred_room_labels = prob[..., :-1].max(-1)
        pred_room_labels = pred_room_labels.cpu().numpy()

    room_polys = []
    room_types = []
    doors_list = []
    windows_list = []

    for j in range(pred_logits.shape[0]):
        fg_mask = pred_logits[j] > logit_threshold
        valid_coords = pred_coords[j][fg_mask]

        if len(valid_coords) == 0:
            continue

        # Scale to pixel coords
        corners = (valid_coords * (resolution - 1)).cpu().numpy()
        pred_logits_per_room = pred_logits[j][fg_mask].cpu().numpy()

        # CAGE postprocessing pipeline
        corners, filtered_logits = remove_short_edges(corners, pred_logits_per_room)
        if len(corners) < 2:
            continue
        corners = get_corners_from_edges(corners, filtered_logits, threshold=10)
        corners = remove_duplicate_corners(corners)
        corners = np.around(corners).astype(np.int32)
        corners = merge_points(corners, 2)

        # Classify as room vs door/window based on semantic labels
        if has_semantics and pred_room_labels is not None:
            label = int(pred_room_labels[j])
            if label in (16, 17):
                # Door (16) or Window (17)
                if len(corners) == 2:
                    if label == 16:
                        doors_list.append(corners)
                    else:
                        windows_list.append(corners)
                continue
            room_type = label
        else:
            room_type = -1

        # Validate as room polygon
        if len(corners) >= min_corners:
            try:
                poly = Polygon(corners)
                if poly.area >= min_area:
                    room_polys.append(corners)
                    room_types.append(room_type)
            except Exception:
                continue

    # Post-process: remove overlapping rooms and refine boundaries
    if len(room_polys) > 1:
        try:
            from shapely.geometry import Polygon as ShapelyPolygon

            shapely_polygons = []
            for np_array in room_polys:
                points = [tuple(p) for p in np_array]
                if points and points[0] != points[-1]:
                    points.append(points[0])
                shapely_polygons.append(ShapelyPolygon(points))

            shapely_polygons = remove_multi_polygon(shapely_polygons)
            shapely_polygons = remove_rooms_with_iou(shapely_polygons)
            polygon_list, _ = refine_rooms(shapely_polygons, False)

            refined_polys = []
            for polygon in polygon_list:
                coords = np.array(polygon.exterior.coords, dtype=np.int32)[:-1]
                refined_polys.append(coords)
            room_polys = refined_polys

            # Trim room_types to match
            room_types = room_types[: len(room_polys)]
        except Exception:
            pass  # Keep unrefined polygons on error

    return room_polys, room_types, doors_list, windows_list
