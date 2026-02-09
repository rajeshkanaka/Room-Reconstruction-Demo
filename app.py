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


def process_images(img1, img2, img3, img4, img5, room_width, progress=gr.Progress()):
    """
    Process uploaded images and generate reconstruction.

    Returns:
        Tuple of (floor_plan_image, 3d_plot, measurements_text,
                  svg_html, svg_file, dxf_file, arch_png_file, status)
    """
    global _last_result

    # Collect non-None images
    images = [img for img in [img1, img2, img3, img4, img5] if img is not None]

    if len(images) < 2:
        return (
            None,
            None,
            "Please upload at least 2 images.",
            "",
            None,
            None,
            None,
            "Error: Not enough images",
        )

    try:
        progress(0.1, desc="Initializing AI model...")
        reconstructor = get_reconstructor()
        reconstructor.assumed_room_width = float(room_width)
        reconstructor.floor_plan_gen.assumed_width = float(room_width)

        # Convert images to numpy arrays
        image_arrays = []
        for i, img in enumerate(images):
            progress(0.1 + 0.1 * i, desc=f"Preparing image {i+1}...")
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            image_arrays.append(img_array)

        # Run reconstruction
        def progress_callback(p, msg):
            progress(0.3 + 0.6 * p, desc=msg)

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

        # Floor plan image (legacy)
        floor_plan_path = result["outputs"]["floor_plan_image"]
        floor_plan_img = Image.open(floor_plan_path)

        # 3D plot
        plotly_fig = result["figures"]["plotly_3d"]

        # Format measurements
        measurements_text = _format_measurements(result, room_width)

        # SVG viewer
        svg_html = ""
        svg_file = None
        svg_path = result["outputs"].get("floor_plan_svg")
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
        dxf_file = result["outputs"].get("floor_plan_dxf")
        if dxf_file and not os.path.exists(dxf_file):
            dxf_file = None

        # Architectural PNG
        arch_png_file = result["outputs"].get("floor_plan_arch_png")
        if arch_png_file and not os.path.exists(arch_png_file):
            arch_png_file = None

        status = f"Processed {result['num_images']} images, {result['num_points']:,} 3D points"
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

    # Pipeline info
    text += "---\n"
    text += f"**Pipeline:** {'Metric depth' if is_metric else 'Legacy (relative depth)'}\n\n"
    text += f"**Images:** {result['num_images']} | "
    text += f"**Points:** {result['num_points']:,} | "
    text += f"**SfM:** {'Yes' if result.get('sfm_success') else 'No'}\n\n"

    if not is_metric:
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

    # Fallback: calibrate legacy measurements
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
            """
        # Room Reconstruction from Photos

        Upload 2-5 photos of a room to generate 2D floor plans (SVG, DXF, PNG)
        and interactive 3D models. Uses AI depth estimation and wall detection.
        """
        )

        with gr.Accordion("How to Use", open=False):
            gr.Markdown(
                """
            ### Getting Best Results

            1. **Upload 4-5 photos** from different corners of the room
            2. **Include floor and walls** in each shot
            3. **Good lighting** improves depth accuracy
            4. **Set room width** if known, for better scale

            ### Output Formats
            - **PNG**: Quick preview (legacy heatmap + architectural)
            - **SVG**: Vector floor plan (zoomable, print-quality)
            - **DXF**: CAD-compatible (opens in AutoCAD, FreeCAD)
            """
            )

        with gr.Row():
            # Left column: Image uploads
            with gr.Column(scale=1):
                gr.Markdown("### Upload Room Photos (2-5 images)")

                with gr.Row():
                    img1 = gr.Image(label="Photo 1", type="numpy", height=150)
                    img2 = gr.Image(label="Photo 2", type="numpy", height=150)

                with gr.Row():
                    img3 = gr.Image(label="Photo 3", type="numpy", height=150)
                    img4 = gr.Image(label="Photo 4", type="numpy", height=150)

                with gr.Row():
                    img5 = gr.Image(label="Photo 5", type="numpy", height=150)

                    with gr.Column():
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
                    with gr.TabItem("Floor Plan (Legacy)"):
                        floor_plan_output = gr.Image(
                            label="2D Floor Plan", type="pil", height=400
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

        # Connect process button
        process_btn.click(
            fn=process_images,
            inputs=[img1, img2, img3, img4, img5, room_width],
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
        **Pipeline:** Depth-Anything-V2 / Apple Depth Pro | Open3D | SegFormer (ADE20K) | svgwrite + ezdxf

        *Proof-of-concept. Measurements are approximate and not suitable for professional use.*
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
