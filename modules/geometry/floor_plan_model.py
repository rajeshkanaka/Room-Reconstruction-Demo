"""
Floor Plan Data Model

Internal representation of a floor plan shared between detection and rendering.
This is the single source of truth: detectors populate it, renderers consume it.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WallSegment:
    """A single wall line segment in the floor plan."""

    start: np.ndarray  # 2D point (x, y) in meters
    end: np.ndarray  # 2D point (x, y) in meters
    thickness: float = 0.15  # meters (interior default)
    material: str = "interior"  # "interior", "exterior"


@dataclass
class DoorOpening:
    """A door opening along a wall."""

    position: np.ndarray  # 2D center point on wall line
    width: float  # meters
    swing_direction: str = "left"  # "left", "right", "double"
    door_type: str = "single"  # "single", "sliding", "double", "closet"
    source: str = "detected"  # "detected" or "gemini"


@dataclass
class WindowOpening:
    """A window opening along a wall."""

    position: np.ndarray  # 2D center point on wall line
    width: float  # meters
    sill_height: Optional[float] = None  # meters from floor
    source: str = "detected"  # "detected" or "gemini"


@dataclass
class RoomPolygon:
    """A closed room boundary polygon."""

    boundary: np.ndarray  # Nx2 array of vertices in meters
    name: str = "Room"
    room_type: str = ""  # "bedroom", "kitchen", etc.
    room_shape: str = ""  # "rectangular", "l_shaped", etc.

    @property
    def area(self) -> float:
        """Compute area via Shoelace formula."""
        n = len(self.boundary)
        if n < 3:
            return 0.0
        xs = self.boundary[:, 0]
        ys = self.boundary[:, 1]
        return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


@dataclass
class DimensionLine:
    """An architectural dimension annotation."""

    start: np.ndarray  # 2D point
    end: np.ndarray  # 2D point
    value_m: float  # measurement in meters
    offset: float = 0.5  # perpendicular offset from wall


@dataclass
class FloorPlanModel:
    """Complete floor plan data model."""

    walls: List[WallSegment] = field(default_factory=list)
    doors: List[DoorOpening] = field(default_factory=list)
    windows: List[WindowOpening] = field(default_factory=list)
    rooms: List[RoomPolygon] = field(default_factory=list)
    dimensions: List[DimensionLine] = field(default_factory=list)
    scale: float = 50.0  # drawing scale, e.g. 1:50
    orientation: float = 0.0  # north angle in degrees
    reconstruction_backend: str = ""  # "vggt" or "legacy"
    semantic_confidence: float = 0.0  # Gemini confidence score
    quality_mode: str = ""  # "high_confidence", "approximate", "needs_more_images"
    export_policy: str = ""  # "normal_export", "annotate_as_approximate", "needs_more_images"
    industry_ready: bool = False
    quality_warnings: List[str] = field(default_factory=list)
