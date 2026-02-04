"""Configuration settings for Room Reconstruction Demo"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
STATIC_DIR = os.path.join(BASE_DIR, "static")
COLMAP_WORKSPACE = os.path.join(BASE_DIR, "colmap_workspace")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(COLMAP_WORKSPACE, exist_ok=True)

# Depth Estimation Settings
# Options: "Intel/dpt-large", "depth-anything/Depth-Anything-V2-Large-hf", "Intel/zoedepth-nyu-kitti"
DEPTH_MODEL = (
    "depth-anything/Depth-Anything-V2-Large-hf"  # Upgraded to Depth Anything V2
)
DEPTH_MODEL_FALLBACK = "Intel/dpt-large"  # Fallback model if primary fails
DEPTH_MAX_SIZE = 518  # Depth Anything V2 optimal size (multiple of 14)

# 3D Reconstruction Settings
POINT_CLOUD_DENSITY = 4  # Sample every Nth pixel (higher = faster, less detail)
DEPTH_SCALE = 0.5  # Scale factor for depth values
VOXEL_SIZE = 0.05  # Voxel size for downsampling point cloud
ENABLE_REGISTRATION = True  # Use point cloud registration between views
REGISTRATION_VOXEL_SIZE = 0.08  # Downsample size for registration (meters)
REGISTRATION_RANSAC_ITERATIONS = 50000  # RANSAC iterations for coarse alignment
REGISTRATION_ICP_ITERATIONS = 50  # ICP refinement iterations
OUTLIER_NB_NEIGHBORS = 20  # Statistical outlier removal neighbors
OUTLIER_STD_RATIO = 2.0  # Statistical outlier removal threshold

# Floor Plan Settings
FLOOR_PLAN_HEIGHT_MIN = 0.1  # Min height ratio for floor detection
FLOOR_PLAN_HEIGHT_MAX = 0.3  # Max height ratio for floor detection
FLOOR_PLAN_RESOLUTION = 100  # Grid resolution for floor plan
ASSUMED_ROOM_WIDTH_METERS = 4.0  # Assumed room width for scaling

# Visualization Settings
VISUALIZATION_POINT_SIZE = 2.0
FIGURE_SIZE = (10, 8)

# Camera Intrinsics (approximate for typical smartphone)
CAMERA_FX = 500.0  # Focal length X
CAMERA_FY = 500.0  # Focal length Y
CAMERA_CX = 256.0  # Principal point X
CAMERA_CY = 256.0  # Principal point Y

# SfM (Structure-from-Motion) Settings
ENABLE_SFM = True  # Use COLMAP SfM for proper multi-view reconstruction
SFM_MIN_IMAGES = 3  # Minimum images required for SfM
SFM_FEATURE_TYPE = "SIFT"  # Feature type: SIFT, ORB, SUPERPOINT
SFM_MATCHER_TYPE = "exhaustive"  # Matcher: exhaustive, sequential, vocab_tree
SFM_MAX_IMAGE_SIZE = 1024  # Max image size for SfM processing

# Dense Reconstruction Settings
ENABLE_MVS = True  # Enable Multi-View Stereo dense reconstruction
MVS_MAX_IMAGE_SIZE = 1000  # Max image size for MVS
DEPTH_FUSION_METHOD = "tsdf"  # Options: tsdf, poisson

# Mesh Settings
MESH_DEPTH = 9  # Poisson reconstruction depth (higher = more detail)
MESH_SCALE = 1.1  # Scale for mesh bounding box
MESH_SIMPLIFY_TARGET = 100000  # Target face count for mesh simplification
