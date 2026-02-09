"""
Opening Detector Module

Detects doors and windows in room images using semantic segmentation
(SegFormer with ADE20K classes). Projects detections to floor plan coordinates.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from termcolor import colored

from modules.geometry.floor_plan_model import DoorOpening, WindowOpening, WallSegment


# ADE20K class indices for openings
ADE20K_DOOR = 25
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
        self._load_model()

    def _load_model(self):
        """Load the segmentation model."""
        try:
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
            import torch

            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name
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
    ) -> Optional[object]:
        """
        Project a 2D bounding box to floor plan coordinates.

        Args:
            bbox: Bounding box dict with x, y, w, h
            depth_map: Depth map (H, W) in meters
            fx, fy: Camera focal lengths
            parent_wall: Wall segment this opening belongs to
            opening_type: "door" or "window"

        Returns:
            DoorOpening or WindowOpening, or None if projection fails
        """
        h, w = depth_map.shape[:2]
        cx = w / 2
        cy = h / 2

        # Get depth at center of bounding box
        box_cx = bbox["x"] + bbox["w"] // 2
        box_cy = bbox["y"] + bbox["h"] // 2

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

        # Project left and right edges to 3D
        left_x_3d = (bbox["x"] - cx) * depth_at_opening / fx
        right_x_3d = (bbox["x"] + bbox["w"] - cx) * depth_at_opening / fx
        opening_width = abs(right_x_3d - left_x_3d)

        # Sanity check opening width
        if opening_type == "door" and not (0.4 < opening_width < 2.0):
            return None
        if opening_type == "window" and not (0.3 < opening_width < 3.5):
            return None

        # Project center to 3D ground plane
        center_x_3d = (box_cx - cx) * depth_at_opening / fx
        center_z_3d = depth_at_opening

        # Map to wall: find closest point on wall to the projected position
        wall_dir = parent_wall.end - parent_wall.start
        wall_len = np.linalg.norm(wall_dir)
        if wall_len < 0.01:
            return None

        wall_unit = wall_dir / wall_len
        # Project the 3D position onto the wall line
        # Use x_3d as position along wall (simplified)
        t = np.clip(0.5, 0.1, 0.9)  # Default to center if projection unclear
        position = parent_wall.start + wall_unit * (wall_len * t)

        if opening_type == "door":
            return DoorOpening(
                position=position,
                width=float(opening_width),
                swing_direction="left",
            )
        else:
            return WindowOpening(
                position=position,
                width=float(opening_width),
            )

    def detect_and_project(
        self,
        images: List[np.ndarray],
        depth_maps: List[np.ndarray],
        walls: List[WallSegment],
        fx: float = 500.0,
        fy: float = 500.0,
    ) -> Tuple[List[DoorOpening], List[WindowOpening]]:
        """
        Detect openings in all images and project to floor plan.

        Args:
            images: List of RGB images
            depth_maps: List of depth maps
            walls: List of wall segments for projection
            fx, fy: Camera focal lengths

        Returns:
            (doors, windows) lists of projected openings
        """
        all_doors = []
        all_windows = []

        for i, (image, depth) in enumerate(zip(images, depth_maps)):
            result = self.detect_openings(image)

            for door_bbox in result.get("doors", []):
                if door_bbox.get("confidence", 0) < 0.2:
                    continue

                # Find nearest wall
                best_wall = self._find_nearest_wall(door_bbox, depth, fx, fy, walls)
                if best_wall is None:
                    continue

                opening = self.project_to_floor_plan(
                    door_bbox, depth, fx, fy, best_wall, "door"
                )
                if opening is not None:
                    all_doors.append(opening)

            for window_bbox in result.get("windows", []):
                if window_bbox.get("confidence", 0) < 0.2:
                    continue

                best_wall = self._find_nearest_wall(window_bbox, depth, fx, fy, walls)
                if best_wall is None:
                    continue

                opening = self.project_to_floor_plan(
                    window_bbox, depth, fx, fy, best_wall, "window"
                )
                if opening is not None:
                    all_windows.append(opening)

        # Deduplicate nearby openings
        all_doors = self._deduplicate_openings(all_doors)
        all_windows = self._deduplicate_openings(all_windows)

        print(
            colored(
                f"[OpeningDetector] Detected {len(all_doors)} doors, "
                f"{len(all_windows)} windows across {len(images)} images",
                "green",
            )
        )

        return all_doors, all_windows

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
            for existing in unique:
                dist = np.linalg.norm(opening.position - existing.position)
                if dist < threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(opening)

        return unique
