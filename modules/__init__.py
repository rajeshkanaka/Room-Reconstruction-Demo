"""Room Reconstruction Demo Modules

This package provides the core functionality for 3D room reconstruction:
- DepthEstimator: AI-based monocular depth estimation (Depth Anything V2)
- SfMProcessor: Structure-from-Motion using COLMAP (lazy-loaded)
- DenseReconstructor: TSDF fusion for dense reconstruction (lazy-loaded)
- FloorPlanGenerator: 2D floor plan extraction with measurements
- Visualizer3D: 3D visualization and mesh generation
- RoomReconstructor: Main orchestrator combining all modules
"""

from .depth_estimator import DepthEstimator
from .floor_plan_generator import FloorPlanGenerator
from .visualizer_3d import Visualizer3D, ScaleEstimator
from .room_reconstructor import RoomReconstructor

# SfM/COLMAP modules are NOT imported here because pycolmap's native
# library may crash (SIGABRT) on some platforms, killing the process.
# They are lazy-loaded inside RoomReconstructor.__init__() instead.

__all__ = [
    "DepthEstimator",
    "FloorPlanGenerator",
    "Visualizer3D",
    "ScaleEstimator",
    "RoomReconstructor",
]
