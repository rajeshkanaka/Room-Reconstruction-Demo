"""
Opening Detector Module

Detects doors and windows in room images using semantic segmentation
(SegFormer with ADE20K classes). Projects detections to floor plan coordinates.
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from termcolor import colored

from modules.geometry.floor_plan_model import DoorOpening, WindowOpening, WallSegment


# ADE20K class indices for openings
# segformer-b2-finetuned-ade-512-512: door=14, windowpane=8
ADE20K_DOOR = 14
ADE20K_WINDOW = 8


class OpeningDetector:
    """
    Detects doors and windows in room images using semantic segmentation.

    Uses SegFormer (ADE20K) for pixel-level classification, then extracts
    bounding boxes for door/window regions and projects them to floor plan
    coordinates using depth maps and camera intrinsics.
    """

    def __init__(self, model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
        """
        Initialize the opening detector.

        Args:
            model_name: HuggingFace model name for semantic segmentation.
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.last_fusion_stats: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self):
        """Load the segmentation model."""
        try:
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
            import torch

            token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN"
            )
            self.processor = SegformerImageProcessor.from_pretrained(
                self.model_name, token=token
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name, token=token
            )

            if torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model.eval()
            print(colored(f"[OpeningDetector] Loaded {self.model_name}", "green"))
        except Exception as e:
            print(
                colored(
                    f"[OpeningDetector] Model load failed: {e}. "
                    "Door/window detection will use fallback.",
                    "yellow",
                )
            )
            self.model = None

    def detect_openings(self, image: np.ndarray) -> Dict:
        """
        Detect doors and windows in a room image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Dict with 'doors' and 'windows' lists of bounding boxes:
            {'doors': [{'x': int, 'y': int, 'w': int, 'h': int, 'confidence': float}],
             'windows': [...]}
        """
        if self.model is None:
            return self._fallback_detect(image)

        try:
            return self._segment_detect(image)
        except Exception as e:
            print(
                colored(
                    f"[OpeningDetector] Segmentation failed: {e}, using fallback",
                    "yellow",
                )
            )
            return self._fallback_detect(image)

    def _segment_detect(self, image: np.ndarray) -> Dict:
        """Detect openings using SegFormer semantic segmentation."""
        import torch

        h, w = image.shape[:2]

        inputs = self.processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Upsample logits to original resolution
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).cpu().numpy()[0]

        doors = self._extract_bboxes(seg_map, ADE20K_DOOR, min_area=500)
        windows = self._extract_bboxes(seg_map, ADE20K_WINDOW, min_area=300)

        return {"doors": doors, "windows": windows, "segmentation_map": seg_map}

    def _extract_bboxes(
        self, seg_map: np.ndarray, class_id: int, min_area: int = 300
    ) -> List[Dict]:
        """Extract bounding boxes for a given class from segmentation map."""
        mask = (seg_map == class_id).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(1.0, area / (w * h))  # Fill ratio as proxy confidence

            bboxes.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "confidence": float(confidence),
                }
            )

        return bboxes

    def _fallback_detect(self, image: np.ndarray) -> Dict:
        """
        Fallback detection using edge/geometry heuristics.

        Looks for rectangular regions with strong vertical edges
        (characteristic of doors and windows).
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find vertical lines (doors/windows tend to have strong verticals)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=int(h * 0.2),
            maxLineGap=int(h * 0.05),
        )

        doors = []
        windows = []

        if lines is not None:
            # Group vertical lines into pairs (potential door/window edges)
            verticals = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1))
                if angle > np.radians(70):  # Near-vertical
                    verticals.append(
                        (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    )

            # Sort by x position and group nearby verticals
            verticals.sort(key=lambda v: v[0])

            for i in range(len(verticals) - 1):
                gap = verticals[i + 1][0] - verticals[i][0]
                # Door-width gap: 60-200 pixels (roughly 0.6-1.2m at typical scales)
                if 40 < gap < 250:
                    x = verticals[i][0]
                    y_top = min(verticals[i][1], verticals[i + 1][1])
                    y_bot = max(verticals[i][3], verticals[i + 1][3])
                    bbox_h = y_bot - y_top

                    bbox = {
                        "x": int(x),
                        "y": int(y_top),
                        "w": int(gap),
                        "h": int(bbox_h),
                        "area": int(gap * bbox_h),
                        "confidence": 0.3,  # Low confidence for heuristic
                    }

                    # Doors are typically taller than wide, start below mid-height
                    if bbox_h > gap * 1.5 and y_bot > h * 0.6:
                        doors.append(bbox)
                    # Windows tend to be in upper portion
                    elif y_top < h * 0.5:
                        windows.append(bbox)

        return {"doors": doors, "windows": windows}

    def project_to_floor_plan(
        self,
        bbox: Dict,
        depth_map: np.ndarray,
        fx: float,
        fy: float,
        parent_wall: WallSegment,
        opening_type: str = "door",
        camera_pose: Optional[np.ndarray] = None,
        wall_index: Optional[int] = None,
        confidence: float = 0.5,
        view_index: int = -1,
    ) -> Optional[object]:
        """
        Project a 2D bounding box to floor plan coordinates.

        Args:
            bbox: Bounding box dict with x, y, w, h
            depth_map: Depth map (H, W) in meters
            fx, fy: Camera focal lengths
            parent_wall: Wall segment this opening belongs to
            opening_type: "door" or "window"
            camera_pose: Optional 4x4 world-from-camera transform
            wall_index: Optional wall index for metadata
            confidence: Detection confidence
            view_index: Source view index

        Returns:
            DoorOpening or WindowOpening, or None if projection fails
        """
        h, w = depth_map.shape[:2]
        cx = w / 2.0

        # Get depth at center of bounding box
        box_cx = int(bbox["x"] + bbox["w"] // 2)
        box_cy = int(bbox["y"] + bbox["h"] // 2)

        # Sample depth in a small region around center for robustness
        y_lo = max(0, box_cy - 5)
        y_hi = min(h, box_cy + 5)
        x_lo = max(0, box_cx - 5)
        x_hi = min(w, box_cx + 5)

        depth_region = depth_map[y_lo:y_hi, x_lo:x_hi]
        valid = depth_region[depth_region > 0]
        if len(valid) == 0:
            return None

        depth_at_opening = float(np.median(valid))
        if depth_at_opening < 0.1 or depth_at_opening > 20.0:
            return None

        wall_projection = self._project_point_to_wall(
            self._project_cam_point_to_plan(
                (box_cx - cx) * depth_at_opening / max(fx, 1e-6),
                depth_at_opening,
                camera_pose,
            ),
            parent_wall,
        )
        if wall_projection is None:
            return None

        wall_unit = wall_projection["wall_unit"]
        wall_len = wall_projection["wall_length"]
        t = wall_projection["t"]
        position = wall_projection["closest"]

        # Estimate width along wall using left/right edge projections.
        left_plan = self._project_cam_point_to_plan(
            (bbox["x"] - cx) * depth_at_opening / max(fx, 1e-6),
            depth_at_opening,
            camera_pose,
        )
        right_plan = self._project_cam_point_to_plan(
            (bbox["x"] + bbox["w"] - cx) * depth_at_opening / max(fx, 1e-6),
            depth_at_opening,
            camera_pose,
        )
        opening_width = abs(float(np.dot(right_plan - left_plan, wall_unit)))
        if opening_width < 0.05:
            opening_width = float(abs(bbox["w"]) * depth_at_opening / max(fx, 1e-6))

        # Sanity checks by opening type.
        if opening_type == "door":
            if not (0.4 <= opening_width <= 2.0):
                return None
            opening = DoorOpening(
                position=position.copy(),
                width=float(opening_width),
                swing_direction="left",
            )
        else:
            if not (0.3 <= opening_width <= 3.5):
                return None
            opening = WindowOpening(
                position=position.copy(),
                width=float(opening_width),
            )

        # Attach fusion metadata for downstream quality scoring.
        opening.source = "detected_single_view"
        opening.wall_index = int(wall_index) if wall_index is not None else -1
        opening.support_views = 1
        opening.fusion_confidence = float(np.clip(confidence, 0.0, 1.0))
        opening.view_index = int(view_index)
        opening.wall_t = float(t)

        return opening

    def _project_cam_point_to_plan(
        self,
        x_cam: float,
        z_cam: float,
        camera_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Project a camera-frame point to plan (x,z) coordinates.
        """
        if camera_pose is not None:
            pose = np.asarray(camera_pose)
            if pose.shape == (4, 4):
                pt_cam = np.array([x_cam, 0.0, z_cam, 1.0], dtype=np.float64)
                pt_world = pose @ pt_cam
                return np.array([pt_world[0], pt_world[2]], dtype=np.float64)
        return np.array([x_cam, z_cam], dtype=np.float64)

    def _project_point_to_wall(
        self,
        point_2d: np.ndarray,
        wall: WallSegment,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Project a point to a wall segment.
        """
        a = np.asarray(wall.start, dtype=np.float64)
        b = np.asarray(wall.end, dtype=np.float64)
        ab = b - a
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-6:
            return None

        wall_unit = ab / ab_len
        t = float(np.dot(point_2d - a, ab) / max(ab_len * ab_len, 1e-12))
        t_clamped = float(np.clip(t, 0.0, 1.0))
        closest = a + ab * t_clamped
        distance = float(np.linalg.norm(point_2d - closest))
        return {
            "t": t_clamped,
            "closest": closest,
            "distance": distance,
            "wall_unit": wall_unit,
            "wall_length": ab_len,
        }

    def _visibility_score(
        self,
        camera_pose: Optional[np.ndarray],
        opening_pos: np.ndarray,
        wall_unit: np.ndarray,
    ) -> float:
        """
        Estimate wall-visibility quality from camera viewpoint.

        Lower scores indicate near-grazing views, which are unstable for
        width and placement estimates.
        """
        if camera_pose is None:
            return 1.0

        pose = np.asarray(camera_pose)
        if pose.shape != (4, 4):
            return 1.0

        cam_pos = np.array([pose[0, 3], pose[2, 3]], dtype=np.float64)
        ray = opening_pos - cam_pos
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm < 1e-6:
            return 0.0

        ray_unit = ray / ray_norm
        grazing = abs(float(np.dot(ray_unit, wall_unit)))
        return float(np.clip(1.0 - grazing, 0.0, 1.0))

    def _resolve_intrinsics(
        self,
        view_idx: int,
        image_shape: Tuple[int, int, int],
        depth_shape: Tuple[int, int],
        camera_intrinsics: Optional[Dict[int, Dict[str, float]]],
        default_fx: float,
        default_fy: float,
    ) -> Tuple[float, float]:
        """
        Resolve focal lengths for a specific view in depth-map coordinates.
        """
        if camera_intrinsics and view_idx in camera_intrinsics:
            intr = camera_intrinsics[view_idx]
            fx = float(intr.get("fx", default_fx))
            fy = float(intr.get("fy", fx))
        else:
            fx = float(default_fx)
            fy = float(default_fy)

        img_h, img_w = image_shape[:2]
        dep_h, dep_w = depth_shape[:2]
        if img_w > 0 and img_h > 0:
            fx *= dep_w / float(img_w)
            fy *= dep_h / float(img_h)

        return max(fx, 1e-6), max(fy, 1e-6)

    def _bbox_confidence(self, bbox: Dict) -> float:
        """Normalize bbox confidence to [0,1]."""
        conf = float(bbox.get("confidence", 0.5))
        area = float(max(1, bbox.get("w", 1) * bbox.get("h", 1)))
        if area < 80:
            conf *= 0.7
        return float(np.clip(conf, 0.0, 1.0))

    def _collect_observations_for_view(
        self,
        bboxes: List[Dict],
        opening_type: str,
        image: np.ndarray,
        depth: np.ndarray,
        walls: List[WallSegment],
        view_idx: int,
        fx: float,
        fy: float,
        camera_pose: Optional[np.ndarray],
    ) -> List[Dict]:
        """
        Collect projected opening observations for one view.
        """
        observations = []
        for bbox in bboxes:
            bbox_conf = self._bbox_confidence(bbox)
            if bbox_conf < 0.2:
                continue

            mapped_bbox = self._map_bbox_to_depth(bbox, image.shape, depth.shape)
            h, w = depth.shape[:2]
            cx = w / 2.0

            box_cx = int(mapped_bbox["x"] + mapped_bbox["w"] // 2)
            box_cy = int(mapped_bbox["y"] + mapped_bbox["h"] // 2)
            y_lo = max(0, box_cy - 5)
            y_hi = min(h, box_cy + 5)
            x_lo = max(0, box_cx - 5)
            x_hi = min(w, box_cx + 5)
            region = depth[y_lo:y_hi, x_lo:x_hi]
            valid = region[region > 0]
            if len(valid) == 0:
                continue

            d = float(np.median(valid))
            if d < 0.1 or d > 20.0:
                continue

            center_plan = self._project_cam_point_to_plan(
                (box_cx - cx) * d / max(fx, 1e-6),
                d,
                camera_pose,
            )

            best_match = None
            for wall_idx, wall in enumerate(walls):
                proj = self._project_point_to_wall(center_plan, wall)
                if proj is None:
                    continue
                visibility = self._visibility_score(
                    camera_pose,
                    proj["closest"],
                    proj["wall_unit"],
                )
                score = proj["distance"] + (0.35 if visibility < 0.08 else 0.0)
                if best_match is None or score < best_match["score"]:
                    best_match = {
                        "wall_idx": wall_idx,
                        "wall": wall,
                        "projection": proj,
                        "visibility": visibility,
                        "score": score,
                    }

            if best_match is None:
                continue

            if best_match["projection"]["distance"] > max(
                1.2, 0.3 * best_match["projection"]["wall_length"]
            ):
                continue
            if best_match["visibility"] < 0.03:
                continue

            fused_conf = float(np.clip(bbox_conf * best_match["visibility"], 0.0, 1.0))
            opening = self.project_to_floor_plan(
                mapped_bbox,
                depth,
                fx,
                fy,
                best_match["wall"],
                opening_type=opening_type,
                camera_pose=camera_pose,
                wall_index=best_match["wall_idx"],
                confidence=fused_conf,
                view_index=view_idx,
            )
            if opening is None:
                continue

            observations.append(
                {
                    "opening_type": opening_type,
                    "wall_idx": best_match["wall_idx"],
                    "t": float(best_match["projection"]["t"]),
                    "width": float(opening.width),
                    "confidence": fused_conf,
                    "view_idx": view_idx,
                    "visibility": float(best_match["visibility"]),
                    "opening": opening,
                }
            )

        return observations

    def _fuse_opening_observations(
        self,
        observations: List[Dict],
        walls: List[WallSegment],
        opening_type: str,
    ) -> List[object]:
        """
        Fuse opening observations from multiple views into stable openings.
        """
        if not observations or not walls:
            return []

        by_wall: Dict[int, List[Dict]] = {}
        for obs in observations:
            by_wall.setdefault(int(obs["wall_idx"]), []).append(obs)

        fused_openings = []

        max_gap_m = 0.65 if opening_type == "door" else 0.9
        max_spread_m = 0.55 if opening_type == "door" else 0.8
        max_width_spread = 0.45 if opening_type == "door" else 0.9
        single_view_min_conf = 0.65 if opening_type == "door" else 0.55

        for wall_idx, wall_obs in by_wall.items():
            if wall_idx < 0 or wall_idx >= len(walls):
                continue

            wall = walls[wall_idx]
            wall_vec = wall.end - wall.start
            wall_len = float(np.linalg.norm(wall_vec))
            if wall_len < 1e-6:
                continue
            wall_unit = wall_vec / wall_len

            wall_obs = sorted(wall_obs, key=lambda o: o["t"])
            clusters: List[List[Dict]] = []

            for obs in wall_obs:
                best_cluster_idx = -1
                best_gap = float("inf")
                for ci, cluster in enumerate(clusters):
                    center_t = float(np.mean([c["t"] for c in cluster]))
                    gap_m = abs(obs["t"] - center_t) * wall_len
                    if gap_m <= max_gap_m and gap_m < best_gap:
                        best_gap = gap_m
                        best_cluster_idx = ci

                if best_cluster_idx < 0:
                    clusters.append([obs])
                else:
                    clusters[best_cluster_idx].append(obs)

            for cluster in clusters:
                views = {int(c["view_idx"]) for c in cluster}
                supports = len(views)

                t_vals = np.array([float(c["t"]) for c in cluster], dtype=np.float64)
                widths = np.array([float(c["width"]) for c in cluster], dtype=np.float64)
                weights = np.array(
                    [max(0.05, float(c["confidence"])) for c in cluster],
                    dtype=np.float64,
                )
                weights /= max(float(weights.sum()), 1e-12)

                t_fused = float(np.sum(weights * t_vals))
                spread_m = float(np.std(t_vals) * wall_len)
                width_spread = float(np.std(widths))
                conf = float(np.sum(weights * np.array([c["confidence"] for c in cluster])))

                if supports < 2 and conf < single_view_min_conf:
                    continue
                if spread_m > max_spread_m and supports < 3:
                    continue
                if width_spread > max_width_spread and supports < 3:
                    continue

                width = float(np.median(widths))
                if opening_type == "door":
                    width = float(np.clip(width, 0.55, 1.8))
                else:
                    width = float(np.clip(width, 0.4, 3.5))

                position = np.asarray(wall.start, dtype=np.float64) + wall_unit * (
                    t_fused * wall_len
                )

                if opening_type == "door":
                    opening = DoorOpening(
                        position=position.astype(np.float64),
                        width=width,
                        swing_direction="left",
                    )
                else:
                    opening = WindowOpening(
                        position=position.astype(np.float64),
                        width=width,
                    )

                opening.source = "fused_multi_view" if supports >= 2 else "detected_single_view"
                opening.support_views = int(supports)
                opening.fusion_confidence = float(np.clip(conf, 0.0, 1.0))
                opening.wall_index = int(wall_idx)
                opening.wall_t = float(np.clip(t_fused, 0.0, 1.0))
                fused_openings.append(opening)

        return fused_openings

    def detect_and_project(
        self,
        images: List[np.ndarray],
        depth_maps: List[np.ndarray],
        walls: List[WallSegment],
        fx: float = 500.0,
        fy: float = 500.0,
        camera_intrinsics: Optional[Dict[int, Dict[str, float]]] = None,
        camera_poses: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    ) -> Tuple[List[DoorOpening], List[WindowOpening]]:
        """
        Detect openings in all images and project to floor plan.

        Args:
            images: List of RGB images
            depth_maps: List of depth maps
            walls: List of wall segments for projection
            fx, fy: Camera focal lengths
            camera_intrinsics: Optional per-view intrinsics
            camera_poses: Optional per-view world-from-camera poses

        Returns:
            (doors, windows) lists of projected openings
        """
        if not walls:
            self.last_fusion_stats = {
                "raw_door_observations": 0,
                "raw_window_observations": 0,
                "fused_doors": 0,
                "fused_windows": 0,
                "max_support_views": 0,
            }
            return [], []

        door_observations: List[Dict] = []
        window_observations: List[Dict] = []

        for i, (image, depth) in enumerate(zip(images, depth_maps)):
            result = self.detect_openings(image)
            fx_i, fy_i = self._resolve_intrinsics(
                i,
                image.shape,
                depth.shape,
                camera_intrinsics=camera_intrinsics,
                default_fx=fx,
                default_fy=fy,
            )
            pose = None
            if camera_poses and i in camera_poses:
                pose = camera_poses[i].get("transform")

            door_observations.extend(
                self._collect_observations_for_view(
                    result.get("doors", []),
                    opening_type="door",
                    image=image,
                    depth=depth,
                    walls=walls,
                    view_idx=i,
                    fx=fx_i,
                    fy=fy_i,
                    camera_pose=pose,
                )
            )
            window_observations.extend(
                self._collect_observations_for_view(
                    result.get("windows", []),
                    opening_type="window",
                    image=image,
                    depth=depth,
                    walls=walls,
                    view_idx=i,
                    fx=fx_i,
                    fy=fy_i,
                    camera_pose=pose,
                )
            )

        all_doors = self._fuse_opening_observations(
            door_observations, walls, opening_type="door"
        )
        all_windows = self._fuse_opening_observations(
            window_observations, walls, opening_type="window"
        )

        # Preserve high-confidence single-view detections if fusion was too strict.
        if not all_doors and door_observations:
            all_doors = self._deduplicate_openings(
                [obs["opening"] for obs in door_observations], threshold=0.45
            )
        if not all_windows and window_observations:
            all_windows = self._deduplicate_openings(
                [obs["opening"] for obs in window_observations], threshold=0.55
            )

        max_support = 0
        for op in all_doors + all_windows:
            max_support = max(max_support, int(getattr(op, "support_views", 1)))

        self.last_fusion_stats = {
            "raw_door_observations": len(door_observations),
            "raw_window_observations": len(window_observations),
            "fused_doors": len(all_doors),
            "fused_windows": len(all_windows),
            "max_support_views": int(max_support),
        }

        print(
            colored(
                f"[OpeningDetector] Detected {len(all_doors)} doors, {len(all_windows)} windows "
                f"across {len(images)} images "
                f"(raw obs: {len(door_observations)} doors / {len(window_observations)} windows)",
                "green",
            )
        )

        return all_doors, all_windows

    def _map_bbox_to_depth(
        self,
        bbox: Dict,
        image_shape: Tuple[int, int, int],
        depth_shape: Tuple[int, int],
    ) -> Dict:
        """
        Map a bbox from image pixel space to depth-map pixel space.

        VGGT depth maps are often lower-resolution than source images.
        """
        img_h, img_w = image_shape[:2]
        dep_h, dep_w = depth_shape[:2]

        if img_w <= 0 or img_h <= 0 or dep_w <= 0 or dep_h <= 0:
            return bbox

        sx = dep_w / float(img_w)
        sy = dep_h / float(img_h)

        x = int(np.clip(round(bbox["x"] * sx), 0, dep_w - 1))
        y = int(np.clip(round(bbox["y"] * sy), 0, dep_h - 1))
        w = int(
            np.clip(round(max(1, bbox["w"]) * sx), 1, max(1, dep_w - x))
        )
        h = int(
            np.clip(round(max(1, bbox["h"]) * sy), 1, max(1, dep_h - y))
        )

        mapped = dict(bbox)
        mapped.update({"x": x, "y": y, "w": w, "h": h})
        return mapped

    def _find_nearest_wall(
        self,
        bbox: Dict,
        depth: np.ndarray,
        fx: float,
        fy: float,
        walls: List[WallSegment],
    ) -> Optional[WallSegment]:
        """Find the wall closest to a projected bounding box center."""
        if not walls:
            return None

        h, w = depth.shape[:2]
        cx = w / 2
        box_cx = bbox["x"] + bbox["w"] // 2
        box_cy = bbox["y"] + bbox["h"] // 2

        # Sample depth
        y_lo = max(0, box_cy - 5)
        y_hi = min(h, box_cy + 5)
        x_lo = max(0, box_cx - 5)
        x_hi = min(w, box_cx + 5)
        region = depth[y_lo:y_hi, x_lo:x_hi]
        valid = region[region > 0]
        if len(valid) == 0:
            return walls[0]  # Fallback to first wall

        d = float(np.median(valid))
        point_3d = np.array([(box_cx - cx) * d / fx, d])

        best_wall = None
        best_dist = float("inf")

        for wall in walls:
            # Point-to-segment distance in 2D (x, z plane)
            a = np.array([wall.start[0], wall.start[1]])
            b = np.array([wall.end[0], wall.end[1]])
            ab = b - a
            ap = point_3d - a
            t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            closest = a + t * ab
            dist = np.linalg.norm(point_3d - closest)

            if dist < best_dist:
                best_dist = dist
                best_wall = wall

        return best_wall

    def _deduplicate_openings(self, openings, threshold: float = 0.5):
        """Remove duplicate openings that are too close together."""
        if len(openings) <= 1:
            return openings

        unique = [openings[0]]
        for opening in openings[1:]:
            is_dup = False
            replace_idx = -1
            for i, existing in enumerate(unique):
                existing_wall = int(getattr(existing, "wall_index", -1))
                opening_wall = int(getattr(opening, "wall_index", -1))
                if existing_wall >= 0 and opening_wall >= 0 and existing_wall != opening_wall:
                    continue

                dist = np.linalg.norm(opening.position - existing.position)
                if dist < threshold:
                    is_dup = True
                    # Keep the stronger fused estimate.
                    existing_conf = float(getattr(existing, "fusion_confidence", 0.5))
                    opening_conf = float(getattr(opening, "fusion_confidence", 0.5))
                    existing_support = int(getattr(existing, "support_views", 1))
                    opening_support = int(getattr(opening, "support_views", 1))
                    if (opening_support, opening_conf) > (existing_support, existing_conf):
                        replace_idx = i
                    break
            if not is_dup:
                unique.append(opening)
            elif replace_idx >= 0:
                unique[replace_idx] = opening

        return unique
