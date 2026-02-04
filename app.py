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


def get_reconstructor():
    """Get or create the reconstructor instance (lazy loading)."""
    global _reconstructor
    if _reconstructor is None:
        print("\n" + "=" * 60)
        print("🏠 Room Reconstruction Demo - Initializing...")
        print("=" * 60 + "\n")
        _reconstructor = RoomReconstructor()
    return _reconstructor


def process_images(img1, img2, img3, img4, img5, room_width, progress=gr.Progress()):
    """
    Process uploaded images and generate reconstruction.

    Args:
        img1-img5: Uploaded images (PIL Images or numpy arrays)
        room_width: Assumed room width for scaling
        progress: Gradio progress tracker

    Returns:
        Tuple of (floor_plan_image, 3d_plot, measurements_text, status)
    """
    # Collect non-None images
    images = [img for img in [img1, img2, img3, img4, img5] if img is not None]

    if len(images) < 2:
        return (
            None,
            None,
            "⚠️ Please upload at least 2 images of the room.",
            "❌ Error: Not enough images",
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
            # Ensure RGB
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
            return (None, None, f"⚠️ {error_msg}", f"❌ Error: {error_msg}")

        progress(0.95, desc="Preparing outputs...")

        # Get floor plan image
        floor_plan_path = result["outputs"]["floor_plan_image"]
        floor_plan_img = Image.open(floor_plan_path)

        # Get 3D plot
        plotly_fig = result["figures"]["plotly_3d"]

        # Format measurements
        m = result["measurements"]
        measurements_text = f"""
### 📏 Room Measurements (Approximate)

| Dimension | Metric | Imperial |
|-----------|--------|----------|
| **Width** | {m['width_m']:.2f} m | {m['width_m']*3.28:.1f} ft |
| **Depth** | {m['depth_m']:.2f} m | {m['depth_m']*3.28:.1f} ft |
| **Area** | {m['area_sqm']:.1f} m² | {m['area_sqft']:.0f} sq ft |

---
*⚠️ Note: These are approximate measurements based on depth estimation.*  
*Actual dimensions may vary. For accurate measurements, use professional tools.*

**Processing Info:**
- Images processed: {result['num_images']}
- 3D points generated: {result['num_points']:,}
- Assumed room width: {room_width} m
"""

        status = f"✅ Successfully processed {result['num_images']} images!"

        progress(1.0, desc="Complete!")

        return (floor_plan_img, plotly_fig, measurements_text, status)

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_msg = str(e)
        return (
            None,
            None,
            f"⚠️ Error during processing: {error_msg}",
            f"❌ Error: {error_msg}",
        )


def create_demo_interface():
    """Create the Gradio interface."""

    # Custom CSS for better styling
    custom_css = """
    .title-text {
        text-align: center;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3399ff;
    }
    """

    with gr.Blocks(title="🏠 Room Reconstruction Demo") as demo:
        # Header
        gr.Markdown(
            """
        # 🏠 Room Reconstruction from Photos
        
        **Transform photos of a room into 2D floor plans and interactive 3D models!**
        
        This demo uses AI-powered depth estimation to analyze room photographs and generate:
        - 📐 2D Floor Plan with approximate dimensions
        - 🎮 Interactive 3D Model visualization
        - 📏 Room measurements (width, depth, area)
        """
        )

        # Instructions
        with gr.Accordion("📖 How to Use (Click to expand)", open=False):
            gr.Markdown(
                """
            ### Getting Best Results
            
            1. **Upload 4-5 photos** of the same room from different angles
            2. **Take photos from corners** - aim to capture walls and floor
            3. **Good lighting** helps the AI understand depth better
            4. **Avoid clutter** if possible - cleaner rooms give better results
            5. **Set room width** - if you know the actual width, enter it for better scale
            
            ### Photo Tips
            - Stand in corners and photograph towards the center
            - Include the floor and at least 2 walls in each shot
            - Avoid extreme wide-angle lens distortion
            - Natural lighting works best
            
            ### Limitations
            - Measurements are **approximate estimates**
            - Works best with rectangular rooms
            - May struggle with very cluttered or dark rooms
            - Not suitable for professional/legal measurements
            """
            )

        with gr.Row():
            # Left column: Image uploads
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Upload Room Photos (2-5 images)")

                with gr.Row():
                    img1 = gr.Image(label="Photo 1", type="numpy", height=150)
                    img2 = gr.Image(label="Photo 2", type="numpy", height=150)

                with gr.Row():
                    img3 = gr.Image(
                        label="Photo 3 (optional)", type="numpy", height=150
                    )
                    img4 = gr.Image(
                        label="Photo 4 (optional)", type="numpy", height=150
                    )

                with gr.Row():
                    img5 = gr.Image(
                        label="Photo 5 (optional)", type="numpy", height=150
                    )

                    with gr.Column():
                        room_width = gr.Slider(
                            minimum=2.0,
                            maximum=10.0,
                            value=4.0,
                            step=0.5,
                            label="Assumed Room Width (meters)",
                            info="Estimate the actual width for better scaling",
                        )

                        process_btn = gr.Button(
                            "🚀 Generate Floor Plan & 3D Model",
                            variant="primary",
                            size="lg",
                        )

                status_text = gr.Textbox(label="Status", interactive=False)

            # Right column: Results
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")

                with gr.Tabs():
                    with gr.TabItem("📐 Floor Plan"):
                        floor_plan_output = gr.Image(
                            label="2D Floor Plan", type="pil", height=400
                        )

                    with gr.TabItem("🎮 3D Model"):
                        model_3d_output = gr.Plot(label="Interactive 3D Model")

                    with gr.TabItem("📏 Measurements"):
                        measurements_output = gr.Markdown(
                            value="*Upload images and click 'Generate' to see measurements*"
                        )

        # Connect the button
        process_btn.click(
            fn=process_images,
            inputs=[img1, img2, img3, img4, img5, room_width],
            outputs=[
                floor_plan_output,
                model_3d_output,
                measurements_output,
                status_text,
            ],
        )

        # Footer
        gr.Markdown(
            """
        ---
        
        ### 🔧 Technical Details
        
        This demo uses:
        - **DPT (Dense Prediction Transformer)** for monocular depth estimation
        - **Open3D** for 3D point cloud processing
        - **Plotly** for interactive 3D visualization
        - **Gradio** for the web interface
        
        **Disclaimer:** This is a proof-of-concept demonstration. Measurements are approximate 
        and should not be used for construction, legal, or professional purposes.
        
        ---
        *Built for home renovation industry proof-of-concept*
        """
        )

    return demo


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("🏠 Room Reconstruction Demo")
    print("=" * 60)
    print("\nStarting web interface...")
    print("The AI model will be loaded when you first process images.")
    print("\n")

    demo = create_demo_interface()

    # Launch the demo
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
