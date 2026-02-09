"""Configuration settings for Room Reconstruction Demo."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
STATIC_DIR = os.path.join(BASE_DIR, "static")
COLMAP_WORKSPACE = os.path.join(BASE_DIR, "colmap_workspace")
COLMAP_DENSE = os.path.join(COLMAP_WORKSPACE, "dense")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(COLMAP_WORKSPACE, exist_ok=True)
os.makedirs(COLMAP_DENSE, exist_ok=True)

# Depth Estimation Settings
# Options: "Intel/dpt-large", "depth-anything/Depth-Anything-V2-Large-hf", "Intel/zoedepth-nyu-kitti"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DEPTH_MODEL_FALLBACK = "Intel/dpt-large"
DEPTH_MAX_SIZE = 518

# 3D Reconstruction Settings
POINT_CLOUD_DENSITY = 4  # Sample every Nth pixel (higher = faster, less detail)
DEPTH_SCALE = 0.5  # Relative depth scaling in fallback depth-only pipeline
VOXEL_SIZE = 0.05
ENABLE_REGISTRATION = True
REGISTRATION_VOXEL_SIZE = 0.08
REGISTRATION_RANSAC_ITERATIONS = 50000
REGISTRATION_ICP_ITERATIONS = 50
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

# Floor Plan Geometry
FLOOR_PLAN_HEIGHT_MIN = 0.1
FLOOR_PLAN_HEIGHT_MAX = 0.3
FLOOR_PLAN_RESOLUTION = 160
ASSUMED_ROOM_WIDTH_METERS = 4.0  # Quick mode only; accurate mode uses calibration
MANHATTAN_SNAP_DEFAULT = True

# Visualization Settings
VISUALIZATION_POINT_SIZE = 2.0
FIGURE_SIZE = (10, 8)

# Camera Intrinsics fallback (used if SfM intrinsics unavailable)
CAMERA_FX = 500.0
CAMERA_FY = 500.0
CAMERA_CX = 256.0
CAMERA_CY = 256.0

# SfM (Structure-from-Motion)
ENABLE_SFM = True
SFM_MIN_IMAGES = 3
SFM_FEATURE_TYPE = "SIFT"
SFM_MATCHER_TYPE = "exhaustive"
SFM_MAX_IMAGE_SIZE = 1024
SFM_ENABLE_DENSE_MVS = True
SFM_INIT_NUM_TRIALS = 600
SFM_MIN_NUM_MATCHES = 12
SFM_MIN_MODEL_SIZE = 3

# SfM matching/verification tolerance (helps with low-parallax indoor capture)
SFM_MATCH_GUIDED = True
SFM_MATCH_CROSS_CHECK = False
SFM_MATCH_MAX_RATIO = 0.9
SFM_MATCH_MAX_DISTANCE = 0.85
SFM_VERIFY_MIN_INLIERS = 12
SFM_VERIFY_MIN_EF_INLIER_RATIO = 0.85
SFM_VERIFY_RANSAC_MAX_ERROR = 6.0

# SfM mapper tolerance (still quality-gated later)
SFM_INIT_MIN_NUM_INLIERS = 40
SFM_INIT_MIN_TRI_ANGLE = 4.0
SFM_INIT_MAX_ERROR = 12.0
SFM_INIT_MAX_FORWARD_MOTION = 0.999
SFM_ABS_POSE_MIN_NUM_INLIERS = 20
SFM_ABS_POSE_MIN_INLIER_RATIO = 0.15
SFM_FILTER_MIN_TRI_ANGLE = 0.5
SFM_ALLOW_TWO_VIEW_TRACKS = True

# Dense Reconstruction
ENABLE_MVS = True
MVS_MAX_IMAGE_SIZE = 1200
DEPTH_FUSION_METHOD = "tsdf"

# Mesh Settings
MESH_DEPTH = 9
MESH_SCALE = 1.1
MESH_SIMPLIFY_TARGET = 100000

# Compliance / Accuracy mode defaults
DEFAULT_COMPLIANCE_PROFILE = "us_residential_v1"
ACCURATE_MODE_DEFAULT = True

# Quality gate thresholds (profile defaults still apply)
QUALITY_MIN_IMAGES = 6
QUALITY_TARGET_IMAGES = 8
QUALITY_BLUR_LAPLACIAN_MIN = 80.0
QUALITY_MIN_OVERLAP_RATIO = 0.12
QUALITY_MIN_REGISTRATION_RATIO = 0.70

# Diagnostic mode (testing only): relax only the SfM registration ratio gate
DIAGNOSTIC_MIN_REGISTRATION_RATIO = 0.33

# Calibration / tolerance
CALIBRATION_MAX_UNCERTAINTY_MM = 8.0
CRITICAL_TOLERANCE_MM = 15.0
OVERALL_TOLERANCE_MM = 30.0
