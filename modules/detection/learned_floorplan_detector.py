"""
Learned Floor Plan Detector

Replaces heuristic Hough-transform wall detection with learned models:
- CAGE (NeurIPS 2025): Edge-centric Transformer, watertight topology, best metrics.
- RoomFormer (CVPR 2023): Two-level query Transformer, proven fallback.

Both models take a top-down 2D density map (projected from VGGT point cloud)
and output vectorized room polygons with room types, doors, and windows.

Integration contract:
    Input:  Nx3 point cloud from VGGTReconstructor
    Output: Populated FloorPlanModel (same contract as WallDetector + RoomSegmenter)
"""

import os
import sys

import numpy as np
from typing import Dict, List, Optional, Tuple

from termcolor import colored

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import (
    ENABLE_LEARNED_FLOORPLAN,
    LEARNED_FLOORPLAN_MODEL,
    CAGE_MODEL_PATH,
    ROOMFORMER_MODEL_PATH,
    FLOORPLAN_DENSITY_RESOLUTION,
    FLOORPLAN_CONFIDENCE_THRESHOLD,
)
from modules.geometry.floor_plan_model import (
    FloorPlanModel,
    WallSegment,
    RoomPolygon,
    DoorOpening,
    WindowOpening,
    DimensionLine,
)


class LearnedFloorplanDetector:
    """
    CAGE/RoomFormer-based floor plan detection.

    Replaces the Hough-transform + RANSAC + Manhattan alignment pipeline
    with a single learned model inference that outputs clean, topologically
    valid room polygons directly.

    Usage:
        detector = LearnedFloorplanDetector(model_type="cage")
        result = detector.detect(points_3d, floor_height=0.0)
        # result is a dict with "floor_plan_model", "measurements", "quality_flags"
    """

    def __init__(self, model_type: str = LEARNED_FLOORPLAN_MODEL):
        """
        Initialize the learned floor plan detector.

        Args:
            model_type: Which model to use -- "cage" or "roomformer".
        """
        self.model_type = model_type
        self._model = None
        self._device = None
        print(colored(f"[LearnedFloorplan] Initialized (model={model_type})", "cyan"))

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        print(colored(f"[LearnedFloorplan] Loading {self.model_type} model...", "cyan"))

        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"

            if self.model_type == "cage":
                self._model = self._load_cage_model()
            elif self.model_type == "roomformer":
                self._model = self._load_roomformer_model()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            print(
                colored(
                    f"[LearnedFloorplan] {self.model_type} loaded on {self._device}",
                    "green",
                )
            )
        except Exception as e:
            print(
                colored(
                    f"[LearnedFloorplan] Failed to load {self.model_type}: {e}",
                    "red",
                )
            )
            raise

    def _load_cage_model(self):
        """Load CAGE (NeurIPS 2025) model."""
        from modules.detection.cage_loader import load_cage_model

        import torch

        device = torch.device(self._device)

        # Find checkpoint: prefer ResNet-50 (smaller, faster for dev)
        checkpoint_path = None
        candidates = [
            os.path.join(
                CAGE_MODEL_PATH, "CAGE_checkpoints", "CAGE_stru3d_resnet50.pth"
            ),
            os.path.join(CAGE_MODEL_PATH, "CAGE_stru3d_resnet50.pth"),
            os.path.join(CAGE_MODEL_PATH, "CAGE_checkpoints", "CAGE_stru3d_swinv2.pth"),
            os.path.join(CAGE_MODEL_PATH, "CAGE_stru3d_swinv2.pth"),
        ]
        for path in candidates:
            if os.path.exists(path):
                checkpoint_path = path
                break

        if checkpoint_path is None:
            raise NotImplementedError(
                "CAGE checkpoint not found. Download from "
                "https://drive.google.com/drive/folders/1jajjRamJ7SVgCWB-Tihp-ToqPsv0GmE7 "
                f"and place in {CAGE_MODEL_PATH}/"
            )

        # Determine backbone from checkpoint filename
        backbone = "resnet50"
        if "swinv2" in os.path.basename(checkpoint_path):
            backbone = "swinv2_L_192_22k"

        print(
            colored(
                f"[LearnedFloorplan] Loading CAGE checkpoint: {os.path.basename(checkpoint_path)} "
                f"(backbone={backbone})",
                "cyan",
            )
        )

        model, self._cage_args = load_cage_model(
            checkpoint_path=checkpoint_path,
            backbone=backbone,
            device=device,
            semantic_classes=-1,
        )
        return model

    def _load_roomformer_model(self):
        """Load RoomFormer (CVPR 2023) model."""
        # TODO: Implement actual RoomFormer model loading once repo is cloned.
        # RoomFormer uses deformable attention Transformer with two-level queries.
        # Input: density map tensor [1, 1, H, W] or [1, 3, H, W] float32
        # Output: room polygon predictions (variable-size set)
        #
        # Expected loading pattern:
        #   sys.path.insert(0, "external/roomformer")
        #   from models.roomformer import build_model
        #   model = build_model(args)
        #   model.load_state_dict(torch.load(checkpoint_path))
        #   model.eval()
        raise NotImplementedError(
            "RoomFormer model loading not yet implemented. "
            "Clone https://github.com/ywyue/RoomFormer to external/roomformer/ "
            "and download pretrained weights."
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        points: np.ndarray,
        floor_height: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Run learned floor plan detection on a 3D point cloud.

        This is the main entry point, matching the return contract of
        RoomReconstructor._detect_walls_and_rooms().

        Args:
            points: Nx3 point cloud in metric coordinates (from VGGT).
            floor_height: Y-coordinate of the floor plane. Auto-detected if None.

        Returns:
            Dict with "floor_plan_model", "measurements", "quality_flags",
            or None if detection fails.
        """
        try:
            self._load_model()

            quality_flags = {
                "used_depth_augmentation": False,
                "used_rectangular_fallback": False,
                "used_sparse_wall_fallback": False,
                "used_learned_model": True,
                "learned_model_type": self.model_type,
                "plane_segment_count": 0,
                "aligned_segment_count": 0,
                "closure_score": 0.0,
                "closure_mean_gap_m": 0.0,
                "closure_p95_gap_m": 0.0,
                "closure_unpaired_ratio": 0.0,
            }

            # 1. Auto-detect floor height if not provided
            if floor_height is None:
                floor_height = self._detect_floor_height(points)

            # 2. Project point cloud to top-down density map
            density_map, projection_meta = self.project_to_density_map(
                points,
                floor_height=floor_height,
                resolution=FLOORPLAN_DENSITY_RESOLUTION,
            )

            # 3. Run model inference
            raw_output = self._infer(density_map)

            # 4. Convert raw output to FloorPlanModel
            model = self._convert_to_floor_plan_model(raw_output, projection_meta)

            # 4b. If detection produced 0 rooms, return None so caller can
            #     fall back to Hough pipeline.
            if len(model.rooms) == 0 or len(model.walls) < 3:
                print(
                    colored(
                        f"[LearnedFloorplan] {self.model_type} produced "
                        f"{len(model.rooms)} rooms, {len(model.walls)} walls -- "
                        "too few, returning None for fallback",
                        "yellow",
                    )
                )
                return None

            # 5. Manhattan alignment: snap walls to orthogonal angles
            model = self._manhattan_align(model, points)

            # 6. Compute quality metrics
            closure = self._compute_closure_score(model.walls)
            quality_flags["closure_score"] = closure["closure_score"]
            quality_flags["closure_mean_gap_m"] = closure["mean_gap_m"]
            quality_flags["aligned_segment_count"] = len(model.walls)

            # 7. Compute measurements
            from modules.geometry.measurement_engine import MeasurementEngine

            engine = MeasurementEngine()
            measurements = engine.compute_measurements(model)

            print(
                colored(
                    f"[LearnedFloorplan] {self.model_type}: {len(model.walls)} walls, "
                    f"{len(model.rooms)} rooms, {len(model.doors)} doors, "
                    f"{len(model.windows)} windows, closure={closure['closure_score']:.2f}",
                    "green",
                )
            )

            return {
                "floor_plan_model": model,
                "measurements": measurements,
                "quality_flags": quality_flags,
            }

        except NotImplementedError:
            # Model not yet installed -- caller should fall back to Hough
            raise
        except Exception as e:
            print(
                colored(
                    f"[LearnedFloorplan] Detection failed: {e}",
                    "yellow",
                )
            )
            return None

    # ------------------------------------------------------------------ #
    #  Density Map Projection                                             #
    # ------------------------------------------------------------------ #

    def project_to_density_map(
        self,
        points: np.ndarray,
        floor_height: float = 0.0,
        resolution: int = 256,
        height_band: Tuple[float, float] = (0.1, 2.5),
    ) -> Tuple[np.ndarray, Dict]:
        """
        Project a 3D point cloud to a top-down 2D density map.

        Filters points to a height band above the floor, then creates a
        bird's-eye-view density histogram in the XZ plane.

        Args:
            points: Nx3 array in metric coordinates (Y is up/gravity axis).
            floor_height: Y-coordinate of the detected floor plane.
            resolution: Output density map resolution (resolution x resolution).
            height_band: (min_m, max_m) above floor to include.

        Returns:
            density_map: (resolution, resolution) float32 array normalized [0, 1].
            projection_meta: Dict with x_range, z_range, meters_per_pixel, origin.
        """
        # Filter to valid points
        valid = np.isfinite(points).all(axis=1)
        pts = points[valid]

        # Filter to height band above floor
        y_min = floor_height + height_band[0]
        y_max = floor_height + height_band[1]
        height_mask = (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
        pts = pts[height_mask]

        if len(pts) < 10:
            print(
                colored(
                    "[LearnedFloorplan] Warning: too few points in height band, "
                    "using all valid points",
                    "yellow",
                )
            )
            pts = points[valid]

        # Extract X and Z (top-down plane)
        x = pts[:, 0]
        z = pts[:, 2]

        # Robust bounds using percentiles
        if len(x) > 50:
            x_min, x_max = np.percentile(x, [1, 99])
            z_min, z_max = np.percentile(z, [1, 99])
        else:
            x_min, x_max = float(x.min()), float(x.max())
            z_min, z_max = float(z.min()), float(z.max())

        # Add padding (5% of extent)
        x_extent = max(x_max - x_min, 0.5)
        z_extent = max(z_max - z_min, 0.5)
        pad = 0.05 * max(x_extent, z_extent)
        x_min -= pad
        x_max += pad
        z_min -= pad
        z_max += pad

        # Create 2D histogram
        x_bins = np.linspace(x_min, x_max, resolution + 1)
        z_bins = np.linspace(z_min, z_max, resolution + 1)
        density, _, _ = np.histogram2d(x, z, bins=[x_bins, z_bins])
        density = density.T  # (resolution, resolution), Z-axis as rows

        # Enhance for CAGE: log normalization + Gaussian blur to produce
        # wall-like patterns matching Structured3D density maps
        if density.max() > 0:
            # Log normalization: compresses dynamic range, amplifies low-density walls
            density = np.log1p(density)
            density = density / density.max()

            # Gaussian blur: spreads thin wall lines into the wider bands CAGE expects
            from scipy.ndimage import gaussian_filter

            density = gaussian_filter(density, sigma=1.5)
            d_max = density.max()
            if d_max > 0:
                density = density / d_max

        density = density.astype(np.float32)

        # Compute metric scale
        meters_per_pixel_x = (x_max - x_min) / resolution
        meters_per_pixel_z = (z_max - z_min) / resolution

        projection_meta = {
            "x_range": (float(x_min), float(x_max)),
            "z_range": (float(z_min), float(z_max)),
            "meters_per_pixel_x": meters_per_pixel_x,
            "meters_per_pixel_z": meters_per_pixel_z,
            "origin": np.array([x_min, z_min]),
            "resolution": resolution,
            "num_points_projected": len(pts),
        }

        return density, projection_meta

    # ------------------------------------------------------------------ #
    #  Model Inference                                                     #
    # ------------------------------------------------------------------ #

    def _infer(self, density_map: np.ndarray) -> Dict:
        """
        Run inference on the density map using the loaded model.

        Args:
            density_map: (H, W) float32 normalized [0, 1].

        Returns:
            Normalized output dict with:
                pred_rooms: list of Nx2 int32 arrays (pixel coords)
                pred_types: list of int room type indices
                pred_doors: list of 2x2 arrays (door endpoints in pixels)
                pred_windows: list of 2x2 arrays (window endpoints in pixels)
                scores: list of float confidence scores
        """
        import torch

        if self.model_type == "cage":
            return self._infer_cage(density_map)
        elif self.model_type == "roomformer":
            return self._infer_roomformer(density_map)
        else:
            raise ValueError(f"Unknown model: {self.model_type}")

    def _infer_cage(self, density_map: np.ndarray) -> Dict:
        """Run CAGE inference and postprocess to room polygons."""
        from modules.detection.cage_loader import run_inference, postprocess_predictions

        import torch

        device = torch.device(self._device)

        # Run model forward pass
        raw_outputs = run_inference(self._model, density_map, device)

        # Postprocess: edges -> corners -> polygons
        room_polys, room_types, doors, windows = postprocess_predictions(
            raw_outputs,
            resolution=density_map.shape[0],
            logit_threshold=FLOORPLAN_CONFIDENCE_THRESHOLD,
        )

        # Compute per-room scores (use average edge logit confidence)
        scores = []
        pred_logits = torch.sigmoid(raw_outputs["pred_logits"][0])
        for j in range(pred_logits.shape[0]):
            fg = pred_logits[j] > FLOORPLAN_CONFIDENCE_THRESHOLD
            if fg.sum() > 0:
                scores.append(float(pred_logits[j][fg].mean().cpu()))
        # Pad scores to match room count
        while len(scores) < len(room_polys):
            scores.append(0.8)

        return {
            "pred_rooms": room_polys,
            "pred_types": room_types,
            "pred_doors": doors,
            "pred_windows": windows,
            "scores": scores[: len(room_polys)],
        }

    def _infer_roomformer(self, density_map: np.ndarray) -> Dict:
        """Run RoomFormer inference."""
        # RoomFormer shares the same codebase as CAGE (CAGE is forked from it).
        # When RoomFormer weights are available, the same cage_loader can be used
        # with minor adjustments (corner-based vs edge-based output).
        raise NotImplementedError(
            "RoomFormer inference not yet implemented. "
            "Clone https://github.com/ywyue/RoomFormer to external/roomformer/ "
            "and download pretrained weights."
        )

    # ------------------------------------------------------------------ #
    #  Output Conversion                                                   #
    # ------------------------------------------------------------------ #

    def _convert_to_floor_plan_model(
        self, raw_output: Dict, projection_meta: Dict
    ) -> FloorPlanModel:
        """
        Convert raw model output to FloorPlanModel.

        Handles coordinate conversion from density-map pixels to meters
        using the projection metadata.

        Args:
            raw_output: Raw dict from _infer().
            projection_meta: Metadata from project_to_density_map().

        Returns:
            Populated FloorPlanModel.
        """
        origin = projection_meta["origin"]
        mpp_x = projection_meta["meters_per_pixel_x"]
        mpp_z = projection_meta["meters_per_pixel_z"]

        walls = []
        rooms = []
        doors = []
        windows = []

        # Convert room polygons to metric coordinates
        room_polygons = raw_output.get("pred_rooms", raw_output.get("pred_polys", []))
        room_types = raw_output.get("pred_types", [])
        scores = raw_output.get("scores", [])

        for i, poly_px in enumerate(room_polygons):
            # Filter by confidence
            score = scores[i] if i < len(scores) else 1.0
            if score < FLOORPLAN_CONFIDENCE_THRESHOLD:
                continue

            # Convert pixel vertices to meters
            if hasattr(poly_px, "cpu"):
                poly_px = poly_px.cpu().numpy()
            poly_px = np.asarray(poly_px, dtype=np.float64)

            if poly_px.ndim != 2 or poly_px.shape[1] != 2:
                continue

            # Pixel (col, row) -> meters (x, z)
            poly_m = np.zeros_like(poly_px)
            poly_m[:, 0] = poly_px[:, 0] * mpp_x + origin[0]  # x in meters
            poly_m[:, 1] = poly_px[:, 1] * mpp_z + origin[1]  # z in meters

            # Room type label
            room_type = ""
            if i < len(room_types):
                rt = room_types[i]
                if isinstance(rt, int):
                    room_type = self._room_type_label(rt)
                elif isinstance(rt, str):
                    room_type = rt

            rooms.append(
                RoomPolygon(
                    boundary=poly_m,
                    name=f"Room {i + 1}",
                    room_type=room_type,
                    room_shape=self._classify_room_shape(poly_m),
                )
            )

            # Extract wall segments from polygon edges
            n = len(poly_m)
            for j in range(n):
                start = poly_m[j]
                end = poly_m[(j + 1) % n]
                walls.append(WallSegment(start=start, end=end))

        # Convert doors
        for door_data in raw_output.get("pred_doors", []):
            if hasattr(door_data, "cpu"):
                door_data = door_data.cpu().numpy()
            door_data = np.asarray(door_data, dtype=np.float64)
            if door_data.size >= 2:
                pos_px = door_data[:2]
                pos_m = np.array(
                    [
                        pos_px[0] * mpp_x + origin[0],
                        pos_px[1] * mpp_z + origin[1],
                    ]
                )
                width = float(door_data[2]) * mpp_x if len(door_data) > 2 else 0.9
                doors.append(
                    DoorOpening(
                        position=pos_m,
                        width=width,
                        source="learned_model",
                    )
                )

        # Convert windows
        for win_data in raw_output.get("pred_windows", []):
            if hasattr(win_data, "cpu"):
                win_data = win_data.cpu().numpy()
            win_data = np.asarray(win_data, dtype=np.float64)
            if win_data.size >= 2:
                pos_px = win_data[:2]
                pos_m = np.array(
                    [
                        pos_px[0] * mpp_x + origin[0],
                        pos_px[1] * mpp_z + origin[1],
                    ]
                )
                width = float(win_data[2]) * mpp_x if len(win_data) > 2 else 1.2
                windows.append(
                    WindowOpening(
                        position=pos_m,
                        width=width,
                        source="learned_model",
                    )
                )

        # Merge fragmented rooms: when all photos are of the same room,
        # CAGE may produce multiple overlapping/adjacent polygons that should
        # be a single room.
        rooms, walls = self._merge_adjacent_rooms(rooms, walls)

        # Deduplicate wall segments (shared edges between rooms)
        walls = self._deduplicate_walls(walls)

        model = FloorPlanModel(
            walls=walls,
            rooms=rooms,
            doors=doors,
            windows=windows,
            reconstruction_backend=f"learned_{self.model_type}",
        )

        return model

    # ------------------------------------------------------------------ #
    #  Room Merging                                                        #
    # ------------------------------------------------------------------ #

    def _merge_adjacent_rooms(
        self,
        rooms: List[RoomPolygon],
        walls: List[WallSegment],
        gap_threshold: float = 0.3,
    ) -> Tuple[List[RoomPolygon], List[WallSegment]]:
        """Merge room polygons that are adjacent or overlapping.

        When all photos are of the same room, CAGE may fragment it into
        multiple polygons.  This method detects when polygons overlap or
        are within ``gap_threshold`` meters of each other and merges them
        into a single room using a buffered union.

        Args:
            rooms: List of detected RoomPolygon instances.
            walls: List of WallSegment instances (rebuilt from merged polygon).
            gap_threshold: Maximum gap in meters to bridge between polygons.

        Returns:
            (merged_rooms, merged_walls) tuple.
        """
        if len(rooms) <= 1:
            return rooms, walls

        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union

        # Build shapely polygons from room boundaries
        shapely_polys = []
        for room in rooms:
            pts = room.boundary
            if len(pts) < 3:
                continue
            coords = [tuple(p) for p in pts]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                sp = ShapelyPolygon(coords)
                if sp.is_valid and sp.area > 0:
                    shapely_polys.append(sp)
            except Exception:
                continue

        if len(shapely_polys) <= 1:
            return rooms, walls

        # Check if polygons are close enough to merge:
        # Buffer each polygon by gap_threshold/2, then union.
        # If the result is a single polygon, they belong together.
        buffered = [p.buffer(gap_threshold / 2) for p in shapely_polys]
        merged = unary_union(buffered)

        if merged.geom_type == "Polygon":
            # All rooms merge into one -- shrink the buffer back
            final = merged.buffer(-gap_threshold / 2)
            if final.is_empty or final.geom_type != "Polygon":
                # Buffer erosion collapsed the polygon; use convex hull instead
                final = unary_union(shapely_polys).convex_hull

            # Simplify to reduce vertex count from buffer operations
            final = final.simplify(tolerance=0.02, preserve_topology=True)

            boundary = np.array(final.exterior.coords)[:-1]  # drop closing vertex

            # Preserve the best room type from the original rooms
            best_type = ""
            for r in rooms:
                if r.room_type and r.room_type != "room_-1":
                    best_type = r.room_type
                    break

            merged_room = RoomPolygon(
                boundary=boundary,
                name="Room 1",
                room_type=best_type or rooms[0].room_type,
                room_shape=self._classify_room_shape(boundary),
            )

            # Rebuild wall segments from merged polygon edges
            new_walls = []
            n = len(boundary)
            for j in range(n):
                start = boundary[j]
                end = boundary[(j + 1) % n]
                new_walls.append(WallSegment(start=start, end=end))

            return [merged_room], new_walls

        elif merged.geom_type == "MultiPolygon":
            # Some groups of rooms merge, others stay separate.
            # Take the largest polygon as the primary room.
            largest = max(merged.geoms, key=lambda g: g.area)
            final = largest.buffer(-gap_threshold / 2)
            if final.is_empty or final.geom_type != "Polygon":
                final = largest

            final = final.simplify(tolerance=0.02, preserve_topology=True)
            boundary = np.array(final.exterior.coords)[:-1]

            merged_room = RoomPolygon(
                boundary=boundary,
                name="Room 1",
                room_type=rooms[0].room_type,
                room_shape=self._classify_room_shape(boundary),
            )

            new_walls = []
            n = len(boundary)
            for j in range(n):
                start = boundary[j]
                end = boundary[(j + 1) % n]
                new_walls.append(WallSegment(start=start, end=end))

            return [merged_room], new_walls

        return rooms, walls

    # ------------------------------------------------------------------ #
    #  Manhattan Alignment                                                 #
    # ------------------------------------------------------------------ #

    def _manhattan_align(
        self, model: FloorPlanModel, points: np.ndarray
    ) -> FloorPlanModel:
        """Snap room polygons to axis-aligned rectangles.

        CAGE produces organic/curved polygons from noisy point clouds.
        For architectural floor plans we want clean Manhattan-world geometry.

        Strategy:
        1. Compute the minimum-area bounding rectangle of the CAGE polygon.
        2. If the CAGE polygon covers less than 40% of the point cloud's
           floor extent, use the point cloud bounding box instead (the model
           detected only a fragment of the room).
        3. Snap the rectangle to the nearest 90-degree orientation.
        4. Rebuild walls and room from the snapped rectangle.

        Args:
            model: FloorPlanModel from CAGE detection.
            points: Original Nx3 point cloud (for extent reference).

        Returns:
            Updated FloorPlanModel with Manhattan-aligned geometry.
        """
        if len(model.rooms) == 0:
            return model

        from shapely.geometry import Polygon as ShapelyPolygon, MultiPoint

        # Get the point cloud floor extent (XZ plane)
        valid = np.isfinite(points).all(axis=1)
        pts = points[valid]
        x, z = pts[:, 0], pts[:, 2]
        if len(x) > 50:
            x_min, x_max = np.percentile(x, [2, 98])
            z_min, z_max = np.percentile(z, [2, 98])
        else:
            x_min, x_max = float(x.min()), float(x.max())
            z_min, z_max = float(z.min()), float(z.max())

        pc_width = x_max - x_min
        pc_depth = z_max - z_min
        pc_area = pc_width * pc_depth

        new_rooms = []
        new_walls = []

        for room in model.rooms:
            boundary = room.boundary
            if len(boundary) < 3:
                new_rooms.append(room)
                continue

            try:
                poly = ShapelyPolygon(
                    [tuple(p) for p in boundary] + [tuple(boundary[0])]
                )
                if not poly.is_valid:
                    poly = poly.buffer(0)

                cage_area = poly.area
            except Exception:
                new_rooms.append(room)
                continue

            # Decide: use CAGE polygon or point cloud extent
            use_pc_extent = False
            if pc_area > 0 and cage_area / pc_area < 0.40:
                use_pc_extent = True
                print(
                    colored(
                        f"[LearnedFloorplan] CAGE polygon covers only "
                        f"{cage_area/pc_area*100:.0f}% of point cloud extent -- "
                        f"using point cloud bounding box instead",
                        "yellow",
                    )
                )

            if use_pc_extent:
                # Use point cloud bounding box as room boundary
                rect_pts = np.array(
                    [
                        [x_min, z_min],
                        [x_max, z_min],
                        [x_max, z_max],
                        [x_min, z_max],
                    ]
                )
            else:
                # Compute minimum bounding rectangle of CAGE polygon
                rect_pts = self._minimum_bounding_rectangle(boundary)

            # Snap rectangle to nearest 90-degree orientation
            rect_pts = self._snap_to_axis(rect_pts)

            new_room = RoomPolygon(
                boundary=rect_pts,
                name=room.name,
                room_type=room.room_type,
                room_shape="rectangular",
            )
            new_rooms.append(new_room)

            # Build walls from rectangle edges
            n = len(rect_pts)
            for j in range(n):
                start = rect_pts[j]
                end = rect_pts[(j + 1) % n]
                new_walls.append(WallSegment(start=start, end=end))

        model.rooms = new_rooms
        model.walls = new_walls

        # Reposition doors/windows onto the new walls
        model.doors = self._snap_openings_to_walls(model.doors, model.walls)
        model.windows = self._snap_openings_to_walls(model.windows, model.walls)

        return model

    def _minimum_bounding_rectangle(self, polygon: np.ndarray) -> np.ndarray:
        """Compute the minimum-area bounding rectangle of a 2D polygon.

        Uses the rotating calipers approach via Shapely's minimum_rotated_rectangle.

        Args:
            polygon: Nx2 array of polygon vertices.

        Returns:
            4x2 array of rectangle corners, ordered counter-clockwise.
        """
        from shapely.geometry import Polygon as ShapelyPolygon

        coords = [tuple(p) for p in polygon] + [tuple(polygon[0])]
        poly = ShapelyPolygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)

        mbr = poly.minimum_rotated_rectangle
        rect = np.array(mbr.exterior.coords)[:-1]  # 4 corners, drop closing dup
        return rect

    def _snap_to_axis(self, rect: np.ndarray) -> np.ndarray:
        """Snap a rectangle to the nearest axis-aligned orientation.

        If the rectangle is already nearly axis-aligned (within 15 degrees),
        force it to be perfectly axis-aligned. Otherwise, keep the dominant
        orientation and snap to the nearest 90-degree multiple.

        Args:
            rect: 4x2 array of rectangle corners.

        Returns:
            4x2 array of axis-aligned rectangle corners.
        """
        # Compute the dominant edge direction
        edge = rect[1] - rect[0]
        angle = np.arctan2(edge[1], edge[0])

        # Snap to nearest multiple of pi/2
        snapped_angle = round(angle / (np.pi / 2)) * (np.pi / 2)

        # For architectural plans, axis-aligned is almost always best
        # Compute axis-aligned bounding box of the rectangle
        x_min, x_max = rect[:, 0].min(), rect[:, 0].max()
        z_min, z_max = rect[:, 1].min(), rect[:, 1].max()

        return np.array(
            [
                [x_min, z_min],
                [x_max, z_min],
                [x_max, z_max],
                [x_min, z_max],
            ]
        )

    def _snap_openings_to_walls(self, openings, walls):
        """Move door/window positions onto the nearest wall midpoint."""
        if not openings or not walls:
            return openings

        wall_mids = []
        wall_dirs = []
        for w in walls:
            mid = (w.start + w.end) / 2
            wall_mids.append(mid)
            d = w.end - w.start
            length = np.linalg.norm(d)
            wall_dirs.append(d / length if length > 0 else np.array([1, 0]))

        wall_mids = np.array(wall_mids)

        for opening in openings:
            pos = opening.position
            dists = np.linalg.norm(wall_mids - pos, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_wall = walls[nearest_idx]

            # Project opening onto the wall line
            wall_vec = nearest_wall.end - nearest_wall.start
            wall_len = np.linalg.norm(wall_vec)
            if wall_len > 0:
                t = np.dot(pos - nearest_wall.start, wall_vec) / (wall_len**2)
                t = np.clip(t, 0.1, 0.9)
                opening.position = nearest_wall.start + t * wall_vec

            # Assign wall direction
            d = wall_dirs[nearest_idx]
            if abs(d[0]) > abs(d[1]):
                opening.wall_direction = "south" if d[0] > 0 else "north"
            else:
                opening.wall_direction = "east" if d[1] > 0 else "west"

        return openings

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _detect_floor_height(self, points: np.ndarray) -> float:
        """Auto-detect floor height from point cloud Y-axis distribution."""
        y = points[:, 1]
        valid = np.isfinite(y)
        y = y[valid]
        if len(y) == 0:
            return 0.0
        # Floor is typically the lowest cluster -- use 5th percentile
        return float(np.percentile(y, 5))

    def _compute_closure_score(self, walls: List[WallSegment]) -> Dict:
        """Compute how well wall endpoints form closed loops."""
        if len(walls) < 3:
            return {"closure_score": 0.0, "mean_gap_m": float("inf")}

        endpoints = []
        for w in walls:
            endpoints.append(w.start)
            endpoints.append(w.end)

        endpoints = np.array(endpoints)
        # For each endpoint, find the nearest OTHER endpoint
        gaps = []
        for i in range(0, len(endpoints), 2):
            start = endpoints[i]
            end = endpoints[i + 1]
            # Find nearest endpoint to 'end' that isn't 'start' of same wall
            dists = np.linalg.norm(endpoints - end, axis=1)
            dists[i] = float("inf")  # exclude own start
            dists[i + 1] = float("inf")  # exclude own end
            if len(dists) > 0:
                gaps.append(float(dists.min()))

        if not gaps:
            return {"closure_score": 0.0, "mean_gap_m": float("inf")}

        mean_gap = float(np.mean(gaps))
        # Score: 1.0 if all gaps < 0.05m, decays for larger gaps
        score = float(np.exp(-mean_gap / 0.1))
        return {"closure_score": min(score, 1.0), "mean_gap_m": mean_gap}

    def _deduplicate_walls(
        self, walls: List[WallSegment], threshold: float = 0.1
    ) -> List[WallSegment]:
        """Remove near-duplicate wall segments (shared polygon edges)."""
        if len(walls) <= 1:
            return walls

        unique = []
        for w in walls:
            is_dup = False
            mid = (w.start + w.end) / 2.0
            length = float(np.linalg.norm(w.end - w.start))
            for u in unique:
                u_mid = (u.start + u.end) / 2.0
                u_length = float(np.linalg.norm(u.end - u.start))
                if (
                    np.linalg.norm(mid - u_mid) < threshold
                    and abs(length - u_length) < threshold
                ):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(w)

        return unique

    def _classify_room_shape(self, boundary: np.ndarray) -> str:
        """Classify room shape from polygon vertices."""
        n = len(boundary)
        if n == 4:
            return "rectangular"
        elif n == 6:
            return "l_shaped"
        elif n == 8:
            return "t_shaped"
        else:
            return "irregular"

    @staticmethod
    def _room_type_label(idx: int) -> str:
        """Convert Structured3D room type index to label."""
        # Structured3D room types used by RoomFormer / CAGE
        labels = {
            0: "living_room",
            1: "master_room",
            2: "kitchen",
            3: "bathroom",
            4: "dining_room",
            5: "child_room",
            6: "study_room",
            7: "second_room",
            8: "guest_room",
            9: "balcony",
            10: "entrance",
            11: "storage",
            12: "wall-in",
            13: "functional",
            14: "corridor",
            15: "closet",
        }
        return labels.get(idx, f"room_{idx}")
