"""
RunPod Serverless Handler for Room Reconstruction

Accepts 4-24 room photos (as base64 or URLs), runs VGGT + Gemini pipeline,
returns floor plan outputs (PNG/SVG/DXF as base64) + measurements JSON.

Input schema:
{
    "input": {
        "images": ["base64_string_1", ...],       # base64-encoded images
        "image_urls": ["https://...", ...],        # OR image URLs (alternative)
        "room_width": 4.0,                         # optional, meters (default 4.0)
        "output_formats": ["png", "svg", "dxf"]   # optional (default all)
    }
}

Output schema:
{
    "floor_plan_png": "base64...",
    "floor_plan_svg": "<svg>...</svg>",
    "floor_plan_dxf": "base64...",
    "floor_plan_arch_png": "base64...",
    "measurements": { ... },
    "quality": { ... },
    "gemini_analysis": { ... },
    "num_images": 5,
    "num_points": 123456,
    "backend": "vggt",
    "status": "success"
}
"""

import os
import sys
import base64
import io
import traceback
from urllib.request import urlopen
from urllib.error import URLError

# Fix OpenMP on some CUDA images
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod

# --- Constants ---
MIN_IMAGES = 4
MAX_IMAGES = 24
DEFAULT_ROOM_WIDTH = 4.0
ALL_FORMATS = {"png", "svg", "dxf", "arch_png"}

# Global reconstructor (persists across warm invocations)
_reconstructor = None


def get_reconstructor():
    """Lazy-load the RoomReconstructor (loads VGGT model on first call)."""
    global _reconstructor
    if _reconstructor is None:
        from modules.room_reconstructor import RoomReconstructor

        print("[RunPod] Loading RoomReconstructor + models...")
        _reconstructor = RoomReconstructor()
        print("[RunPod] Models loaded, ready for inference.")
    return _reconstructor


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image to a numpy RGB array."""
    # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64_string[:100]:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def download_image(url: str) -> np.ndarray:
    """Download an image from URL and return as numpy RGB array."""
    with urlopen(url, timeout=30) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    return np.array(img)


def encode_file_base64(filepath: str) -> str:
    """Read a file and return its base64-encoded content."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    """
    RunPod serverless handler.

    Receives job dict with 'input' key, processes images,
    returns floor plan outputs + measurements.
    """
    job_input = job["input"]

    # --- Parse inputs ---
    images_b64 = job_input.get("images", [])
    image_urls = job_input.get("image_urls", [])
    room_width = float(job_input.get("room_width", DEFAULT_ROOM_WIDTH))
    requested_formats = set(job_input.get("output_formats", ALL_FORMATS))

    # --- Load images ---
    image_arrays = []
    errors = []

    # From base64
    for i, b64 in enumerate(images_b64):
        try:
            image_arrays.append(decode_base64_image(b64))
        except Exception as e:
            errors.append(f"Failed to decode base64 image {i}: {e}")

    # From URLs
    for i, url in enumerate(image_urls):
        try:
            image_arrays.append(download_image(url))
        except (URLError, Exception) as e:
            errors.append(f"Failed to download image {i} from URL: {e}")

    # --- Validate ---
    n = len(image_arrays)
    if n < MIN_IMAGES:
        return {
            "error": f"Need at least {MIN_IMAGES} images, got {n}. {'; '.join(errors)}",
            "status": "error",
        }

    if n > MAX_IMAGES:
        return {
            "error": f"Maximum {MAX_IMAGES} images allowed, got {n}.",
            "status": "error",
        }

    # --- Run reconstruction ---
    try:
        reconstructor = get_reconstructor()
        reconstructor.assumed_room_width = room_width

        result = reconstructor.reconstruct_from_arrays(image_arrays)

        if not result.get("success", False):
            return {
                "error": result.get("error", "Reconstruction failed"),
                "status": "error",
            }

        # --- Build response ---
        response = {
            "status": "success",
            "num_images": result.get("num_images", n),
            "num_points": result.get("num_points", 0),
            "backend": result.get("backend", "unknown"),
            "measurements": result.get("measurements", {}),
        }

        # Quality assessment
        if result.get("quality"):
            response["quality"] = result["quality"]

        # Gemini analysis
        if result.get("gemini_analysis"):
            response["gemini_analysis"] = result["gemini_analysis"]

        # Detection measurements (per-wall, rooms)
        detection = result.get("data", {}).get("detection")
        if detection and detection.get("measurements"):
            response["detection_measurements"] = detection["measurements"]

        # --- Encode output files ---
        outputs = result.get("outputs", {})

        if "png" in requested_formats:
            path = outputs.get("floor_plan_image")
            if path and os.path.exists(path):
                response["floor_plan_png"] = encode_file_base64(path)

        if "svg" in requested_formats:
            path = outputs.get("floor_plan_svg")
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    response["floor_plan_svg"] = f.read()

        if "dxf" in requested_formats:
            path = outputs.get("floor_plan_dxf")
            if path and os.path.exists(path):
                response["floor_plan_dxf"] = encode_file_base64(path)

        if "arch_png" in requested_formats:
            path = outputs.get("floor_plan_arch_png")
            if path and os.path.exists(path):
                response["floor_plan_arch_png"] = encode_file_base64(path)

        # Warnings from image loading
        if errors:
            response["warnings"] = errors

        return response

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error",
        }


# --- Entry point ---
if __name__ == "__main__":
    print("[RunPod] Starting Room Reconstruction serverless worker...")
    runpod.serverless.start({"handler": handler})
