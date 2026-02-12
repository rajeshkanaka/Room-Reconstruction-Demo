"""
Room Reconstruction Demo Application

A web-based interface for reconstructing rooms from photographs.
Uses depth estimation AI to create 3D models and floor plans.

Usage:
    python app.py

This will launch a Gradio web interface accessible at http://localhost:7860
"""

import os
import sys

# Fix OpenMP issue on macOS (multiple libiomp loaded)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, ASSUMED_ROOM_WIDTH_METERS
from modules.room_reconstructor import RoomReconstructor

MIN_IMAGES = 4
MAX_IMAGES = 24

# Global reconstructor instance (lazy loaded)
_reconstructor = None
_last_result = None


def get_reconstructor():
    """Get or create the reconstructor instance (lazy loading)."""
    global _reconstructor
    if _reconstructor is None:
        print("\n" + "=" * 60)
        print("Room Reconstruction Demo - Initializing...")
        print("=" * 60 + "\n")
        _reconstructor = RoomReconstructor()
    return _reconstructor


def process_images(files, room_width, progress=gr.Progress()):
    """
    Process uploaded image files and generate reconstruction.

    Args:
        files: List of uploaded file paths from gr.File component.
        room_width: Room width in meters for scale calibration.

    Returns:
        Tuple of (floor_plan_image, 3d_plot, measurements_text,
                  svg_html, svg_file, dxf_file, arch_png_file, status)
    """
    global _last_result

    _error = lambda msg: (None, None, msg, "", None, None, None, f"Error: {msg}")

    if not files:
        return _error(f"Please upload {MIN_IMAGES}-{MAX_IMAGES} room photos.")

    if len(files) < MIN_IMAGES:
        return _error(
            f"Need at least {MIN_IMAGES} images, got {len(files)}. "
            f"Upload {MIN_IMAGES}-{MAX_IMAGES} photos from different angles."
        )

    if len(files) > MAX_IMAGES:
        return _error(
            f"Maximum {MAX_IMAGES} images allowed, got {len(files)}. "
            f"Please remove {len(files) - MAX_IMAGES} image(s)."
        )

    try:
        progress(0.05, desc="Loading images...")

        # Load images from file paths
        image_arrays = []
        for i, f in enumerate(files):
            file_path = f if isinstance(f, str) else f.name
            progress(
                0.05 + 0.1 * (i / len(files)),
                desc=f"Loading image {i+1}/{len(files)}...",
            )
            img = Image.open(file_path).convert("RGB")
            image_arrays.append(np.array(img))

        progress(0.15, desc="Initializing AI model...")
        reconstructor = get_reconstructor()
        reconstructor.assumed_room_width = float(room_width)

        # Run reconstruction
        def progress_callback(p, msg):
            progress(0.2 + 0.7 * p, desc=msg)

        result = reconstructor.reconstruct_from_arrays(image_arrays, progress_callback)

        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error occurred")
            return (
                None,
                None,
                f"Error: {error_msg}",
                "",
                None,
                None,
                None,
                f"Error: {error_msg}",
            )

        _last_result = result
        progress(0.95, desc="Preparing outputs...")

        # Primary floor plan image (architectural PNG)
        outputs = result.get("outputs", {})
        figures = result.get("figures", {})

        floor_plan_path = outputs.get("floor_plan_image")
        if not floor_plan_path or not os.path.exists(floor_plan_path):
            return _error("Floor plan generation failed -- no output image produced.")
        floor_plan_img = Image.open(floor_plan_path)

        # 3D plot
        plotly_fig = figures.get("plotly_3d")

        # Format measurements
        measurements_text = _format_measurements(result, room_width)

        # SVG viewer
        svg_html = ""
        svg_file = None
        svg_path = outputs.get("floor_plan_svg")
        if svg_path and os.path.exists(svg_path):
            with open(svg_path, "r") as f:
                svg_content = f.read()
            svg_html = f"""
            <div style="background:white; padding:10px; border:1px solid #ddd; border-radius:8px; overflow:auto; max-height:500px;">
                {svg_content}
            </div>
            """
            svg_file = svg_path

        # DXF file
        dxf_file = outputs.get("floor_plan_dxf")
        if dxf_file and not os.path.exists(dxf_file):
            dxf_file = None

        # Architectural PNG
        arch_png_file = outputs.get("floor_plan_arch_png")
        if arch_png_file and not os.path.exists(arch_png_file):
            arch_png_file = None

        n_img = result.get("num_images", len(files))
        n_pts = result.get("num_points", 0)
        status = f"Processed {n_img} images, {n_pts:,} 3D points"
        progress(1.0, desc="Complete!")

        return (
            floor_plan_img,
            plotly_fig,
            measurements_text,
            svg_html,
            svg_file,
            dxf_file,
            arch_png_file,
            status,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_msg = str(e)
        return (
            None,
            None,
            f"Error: {error_msg}",
            "",
            None,
            None,
            None,
            f"Error: {error_msg}",
        )


def _format_measurements(result, room_width):
    """Format detailed measurements markdown."""
    m = result["measurements"]
    is_metric = result.get("is_metric", False)
    detection = result.get("data", {}).get("detection")

    text = "### Room Measurements\n\n"
    text += "| Dimension | Metric | Imperial |\n"
    text += "|-----------|--------|----------|\n"
    text += f"| **Width** | {m['width_m']:.2f} m | {m['width_m']*3.28084:.1f} ft |\n"
    text += f"| **Depth** | {m['depth_m']:.2f} m | {m['depth_m']*3.28084:.1f} ft |\n"
    text += f"| **Area** | {m['area_sqm']:.1f} m2 | {m['area_sqft']:.0f} sq ft |\n\n"
    if "scale_confidence" in m:
        text += f"- **Scale confidence:** {m.get('scale_confidence', 'unknown')}\n"
        text += f"- **Scale consistency:** {m.get('scale_consistency', 0):.2f}\n"
        cues = m.get("scale_cues", [])
        if cues:
            text += f"- **Scale cues:** {', '.join(cues)}\n"
        text += "\n"

    quality = result.get("quality")
    if quality:
        text += "### Quality Assessment\n\n"
        text += f"- **Mode:** {quality.get('mode', 'unknown')}\n"
        text += f"- **Score:** {quality.get('score', 0):.2f}\n"
        text += f"- **Industry ready:** {'yes' if quality.get('industry_ready') else 'no'}\n"
        text += f"- **Export policy:** {quality.get('export_policy', 'unknown')}\n"
        for warning in quality.get("warnings", [])[:5]:
            text += f"- {warning}\n"
        if quality.get("recommended_next_steps"):
            text += "\n**Recommended next steps:**\n"
            for step in quality.get("recommended_next_steps", [])[:3]:
                text += f"- {step}\n"
        text += "\n"

    if detection and detection.get("measurements"):
        det_m = detection["measurements"]
        if det_m.get("wall_lengths"):
            text += "### Per-Wall Dimensions\n\n"
            text += "| Wall | Length (m) | Length (ft) |\n"
            text += "|------|-----------|-------------|\n"
            for i, wl in enumerate(det_m["wall_lengths"]):
                text += f"| Wall {i+1} | {wl:.2f} | {wl*3.28084:.1f} |\n"
            text += "\n"

        if det_m.get("rooms"):
            text += "### Room Details\n\n"
            for room in det_m["rooms"]:
                text += f"- **{room.get('name', 'Room')}**: "
                text += f"{room.get('area_sqm', 0):.1f} m2 "
                text += f"({room.get('area_sqft', 0):.0f} sq ft)\n"
            text += "\n"

        if det_m.get("warnings"):
            text += "### Warnings\n\n"
            for w in det_m["warnings"]:
                text += f"- {w}\n"
            text += "\n"

    # Gemini scene analysis
    gemini = result.get("gemini_analysis")
    if gemini:
        text += "### Scene Analysis (Gemini)\n\n"
        text += f"| Property | Value |\n"
        text += f"|----------|-------|\n"
        text += f"| **Room Type** | {gemini.get('room_type', 'unknown').replace('_', ' ').title()} |\n"
        text += f"| **Room Shape** | {gemini.get('room_shape', 'unknown').replace('_', ' ').title()} |\n"
        text += f"| **Doors** | {gemini.get('door_count', 0)} |\n"
        text += f"| **Windows** | {gemini.get('window_count', 0)} |\n"
        text += f"| **Confidence** | {gemini.get('confidence', 0):.0%} |\n\n"

        if gemini.get("features"):
            text += "**Features:** " + ", ".join(gemini["features"]) + "\n\n"

    # Pipeline info
    backend = result.get("backend", "unknown")
    text += "---\n"
    text += f"**Backend:** {backend.upper()}"
    if backend == "vggt":
        text += " (VGGT -- single-pass metric reconstruction)\n\n"
    elif is_metric:
        text += " (Metric depth)\n\n"
    else:
        text += " (Relative depth)\n\n"

    text += f"**Images:** {result['num_images']} | "
    text += f"**Points:** {result['num_points']:,} | "
    text += f"**SfM:** {'Yes' if result.get('sfm_success') else 'N/A' if backend == 'vggt' else 'No'}\n\n"

    if backend == "vggt":
        text += "*Measurements derived from VGGT metric depth estimation.*\n"
    elif not is_metric:
        text += f"*Scale assumes room width = {room_width} m. "
        text += "Measurements are approximate.*\n"
    else:
        text += "*Measurements derived from metric depth estimation.*\n"

    return text


def calibrate_measurements(reference_wall, known_length):
    """Re-calibrate measurements using a known wall dimension."""
    global _last_result
    if _last_result is None:
        return "No reconstruction data available. Process images first."

    if not known_length or known_length <= 0:
        return "Enter a valid wall length in meters."

    try:
        wall_idx = int(reference_wall.split(" ")[-1]) - 1
    except (ValueError, IndexError):
        wall_idx = 0

    detection = _last_result.get("data", {}).get("detection")
    if detection and detection.get("measurements"):
        wall_lengths = detection["measurements"].get("wall_lengths", [])
        if 0 <= wall_idx < len(wall_lengths):
            current = wall_lengths[wall_idx]
            factor = known_length / current if current > 0 else 1.0

            # Apply calibration factor to all measurements
            for i in range(len(wall_lengths)):
                wall_lengths[i] *= factor

            if detection["measurements"].get("rooms"):
                for room in detection["measurements"]["rooms"]:
                    room["area_sqm"] *= factor**2
                    room["area_sqft"] = room["area_sqm"] * 10.764

            return f"Calibrated: Wall {wall_idx+1} set to {known_length:.2f}m (factor: {factor:.3f}). All measurements updated."

    # Fallback: calibrate summary measurements when per-wall data is unavailable
    m = _last_result["measurements"]
    current_width = m.get("width_m", 1.0)
    if current_width > 0:
        factor = known_length / current_width
        m["width_m"] *= factor
        m["depth_m"] *= factor
        m["area_sqm"] *= factor**2
        m["area_sqft"] = m["area_sqm"] * 10.764
        return f"Calibrated with factor {factor:.3f}. Width: {m['width_m']:.2f}m, Depth: {m['depth_m']:.2f}m"

    return "Calibration failed."


def create_demo_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="Room Reconstruction Demo") as demo:
        # Header
        gr.Markdown(
            f"""
        # Room Reconstruction from Photos

        Upload **{MIN_IMAGES}-{MAX_IMAGES} photos** of a room to generate 2D floor plans
        (SVG, DXF, PNG) and interactive 3D models. Drag & drop or click to bulk-upload.
        """
        )

        with gr.Accordion("How to Use", open=False):
            gr.Markdown(
                f"""
            ### Getting Best Results

            1. **Upload {MIN_IMAGES}-{MAX_IMAGES} photos** from different corners/angles of the room
            2. **Include floor and walls** in each shot
            3. **Overlap between shots** — each photo should share ~30% with another
            4. **Good lighting** improves depth accuracy
            5. **Set room width** if known, for better scale

            ### Output Formats
            - **PNG**: Architectural floor plan preview
            - **SVG**: Vector floor plan (zoomable, print-quality)
            - **DXF**: CAD-compatible (opens in AutoCAD, FreeCAD)
            """
            )

        with gr.Row():
            # Left column: Image uploads
            with gr.Column(scale=1):
                gr.Markdown(
                    f"### Upload Room Photos ({MIN_IMAGES}-{MAX_IMAGES} images)"
                )

                file_upload = gr.File(
                    label=f"Drop {MIN_IMAGES}-{MAX_IMAGES} room photos here (bulk upload supported)",
                    file_count="multiple",
                    file_types=["image"],
                    interactive=True,
                )

                upload_gallery = gr.Gallery(
                    label="Uploaded Photos Preview",
                    columns=4,
                    rows=3,
                    height=300,
                    interactive=False,
                )

                room_width = gr.Slider(
                    minimum=2.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.5,
                    label="Room Width (meters)",
                    info="Approximate room width for scale calibration",
                )

                process_btn = gr.Button(
                    "Generate Floor Plan & 3D Model",
                    variant="primary",
                    size="lg",
                )

                status_text = gr.Textbox(label="Status", interactive=False)

                # Calibration section
                with gr.Accordion("Calibrate Measurements", open=False):
                    gr.Markdown(
                        "If you know the exact length of a wall, enter it here to recalibrate all measurements."
                    )
                    with gr.Row():
                        ref_wall = gr.Dropdown(
                            choices=["Wall 1", "Wall 2", "Wall 3", "Wall 4"],
                            value="Wall 1",
                            label="Reference Wall",
                        )
                        known_length = gr.Number(
                            label="Known Length (m)", value=0, precision=2
                        )
                        calibrate_btn = gr.Button("Calibrate")
                    calibration_result = gr.Textbox(
                        label="Calibration Result", interactive=False
                    )

            # Right column: Results
            with gr.Column(scale=1):
                gr.Markdown("### Results")

                with gr.Tabs():
                    with gr.TabItem("2D Floor Plan"):
                        floor_plan_output = gr.Image(
                            label="Architectural 2D Floor Plan", type="pil", height=400
                        )

                    with gr.TabItem("SVG Floor Plan"):
                        svg_viewer = gr.HTML(
                            value="<p style='color:#999; text-align:center;'>Process images to see SVG floor plan</p>"
                        )
                        svg_download = gr.File(label="Download SVG")

                    with gr.TabItem("DXF (CAD)"):
                        gr.Markdown(
                            "DXF floor plan for CAD software (AutoCAD, FreeCAD, etc.)"
                        )
                        dxf_download = gr.File(label="Download DXF")

                    with gr.TabItem("PNG (Architectural)"):
                        gr.Markdown("Clean architectural-style PNG rendering.")
                        arch_png_download = gr.File(label="Download Architectural PNG")

                    with gr.TabItem("3D Model"):
                        model_3d_output = gr.Plot(label="Interactive 3D Model")

                    with gr.TabItem("Measurements"):
                        measurements_output = gr.Markdown(
                            value="*Upload images and click 'Generate' to see measurements*"
                        )

        # Preview uploaded images in gallery
        def preview_uploads(files):
            if not files:
                return []
            paths = [f if isinstance(f, str) else f.name for f in files]
            return paths

        file_upload.change(
            fn=preview_uploads,
            inputs=[file_upload],
            outputs=[upload_gallery],
        )

        # Connect process button
        process_btn.click(
            fn=process_images,
            inputs=[file_upload, room_width],
            outputs=[
                floor_plan_output,
                model_3d_output,
                measurements_output,
                svg_viewer,
                svg_download,
                dxf_download,
                arch_png_download,
                status_text,
            ],
        )

        # Connect calibrate button
        calibrate_btn.click(
            fn=calibrate_measurements,
            inputs=[ref_wall, known_length],
            outputs=[calibration_result],
        )

        # Footer
        gr.Markdown(
            """
        ---
        **Pipeline:** VGGT (CVPR 2025) | Gemini 3 Flash | Open3D | SegFormer (ADE20K) | svgwrite + ezdxf

        *Proof-of-concept with metric depth via VGGT + semantic analysis via Gemini.*
        """
        )

    return demo


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Room Reconstruction Demo")
    print("=" * 60)
    print("\nStarting web interface...")
    print("Models will be loaded on first image processing.\n")

    demo = create_demo_interface()
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(server_name="0.0.0.0", server_port=7850, share=share, show_error=True)


if __name__ == "__main__":
    main()
