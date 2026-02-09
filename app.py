"""Room Reconstruction Demo application with two-pass calibration flow."""

import os
import sys
from typing import List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_COMPLIANCE_PROFILE
from modules.room_reconstructor import RoomReconstructor

_reconstructor = None


def get_reconstructor() -> RoomReconstructor:
    """Get singleton reconstructor instance."""
    global _reconstructor
    if _reconstructor is None:
        print("\n" + "=" * 60)
        print("🏠 Room Reconstruction Demo - Initializing...")
        print("=" * 60 + "\n")
        _reconstructor = RoomReconstructor()
    return _reconstructor



def _resolve_uploaded_path(file_obj) -> str:
    """Extract a filesystem path from a Gradio uploaded file object."""
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return str(file_obj.name)
    if isinstance(file_obj, dict):
        path = file_obj.get("path") or file_obj.get("name")
        return str(path) if path else ""
    return ""


def _collect_images(images: List[object]) -> List[np.ndarray]:
    out = []
    for img in images:
        if img is None:
            continue

        arr = None
        if isinstance(img, (str, os.PathLike)) or hasattr(img, "name"):
            img_path = _resolve_uploaded_path(img)
            if not img_path or not os.path.exists(img_path):
                continue
            with Image.open(img_path) as pil_img:
                arr = np.array(pil_img.convert("RGB"))
        elif isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            arr = img

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]

        out.append(arr)
    return out



def _format_measurements(result: dict) -> str:
    """Render measurements + compliance summary markdown."""
    m = result.get("measurements", {})
    c = result.get("compliance", {})
    status = result.get("status", "UNKNOWN")
    diagnostic_mode = bool(result.get("diagnostic_mode", False))

    quality = result.get("quality_gate", {})
    quality_lines = []
    if quality:
        quality_lines.append(f"- Quality gate pass: `{quality.get('passed', False)}`")
        failures = quality.get("failures", [])
        if failures:
            quality_lines.append("- Capture issues:")
            for f in failures:
                quality_lines.append(f"  - {f}")

    calibration = result.get("calibration", {})
    calibration_line = ""
    if calibration:
        calibration_line = (
            f"- Calibration uncertainty: {calibration.get('uncertainty_mm', 0):.2f} mm"
        )

    diagnostic_line = ""
    if diagnostic_mode:
        diagnostic_line = (
            "- Diagnostic mode: `ON` (registration gate relaxed for testing; not standards-compliant output)"
        )

    accuracy = result.get("accuracy_metrics", {})
    accuracy_lines = []
    if accuracy:
        accuracy_lines.append(
            f"- Predicted critical error: {accuracy.get('predicted_critical_error_mm', 0):.2f} mm"
        )
        accuracy_lines.append(
            f"- Predicted overall error: {accuracy.get('predicted_overall_error_mm', 0):.2f} mm"
        )

    return f"""
### 📏 Measurements

| Dimension | Metric | Imperial |
|-----------|--------|----------|
| **Width** | {m.get('width_m', 0):.3f} m | {m.get('width_ft', 0):.3f} ft |
| **Depth** | {m.get('depth_m', 0):.3f} m | {m.get('depth_ft', 0):.3f} ft |
| **Area** | {m.get('area_sqm', 0):.3f} m² | {m.get('area_sqft', 0):.3f} sq ft |
| **Perimeter** | {m.get('perimeter_m', 0):.3f} m | {(m.get('perimeter_m', 0) * 3.28084):.3f} ft |

### ✅ Compliance
- Profile: `{c.get('profile', 'n/a')}`
- Status: `{status}`

{calibration_line}
{diagnostic_line}
{chr(10).join(accuracy_lines)}
{chr(10).join(quality_lines)}
"""



def run_pass1(
    uploaded_files,
    compliance_profile,
    accurate_mode,
    diagnostic_mode,
    progress=gr.Progress(),
):
    """Pass 1: process images and generate provisional outputs."""
    images = _collect_images(list(uploaded_files or []))

    if len(images) < 2:
        context = {
            "status": "FAIL",
            "error": "Upload at least 2 images.",
            "quality_failures": [],
            "accurate_mode": bool(accurate_mode),
            "diagnostic_mode": bool(diagnostic_mode),
        }
        return (
            None,
            None,
            "⚠️ Upload at least 2 images.",
            "❌ Error",
            "",
            [],
            None,
            None,
            context,
        )

    reconstructor = get_reconstructor()

    def progress_cb(p, msg):
        progress(min(max(p, 0.0), 1.0), desc=msg)

    result = reconstructor.reconstruct_from_arrays(
        images,
        progress_callback=progress_cb,
        compliance_profile=compliance_profile,
        accurate_mode=bool(accurate_mode),
        diagnostic_mode=bool(diagnostic_mode),
    )

    if not result.get("success", False):
        msg = result.get("error", "Unknown error")
        quality = result.get("quality_gate", {})
        failures = quality.get("failures", [])
        failure_text = "\n".join([f"- {f}" for f in failures]) if failures else ""
        status = result.get("status", "FAIL")
        details = f"### ❌ {result.get('status', 'FAIL')}\n{msg}\n{failure_text}"
        context = {
            "status": status,
            "error": msg,
            "quality_failures": failures,
            "accurate_mode": bool(accurate_mode),
            "diagnostic_mode": bool(diagnostic_mode),
        }
        return (
            None,
            None,
            details,
            f"❌ {result.get('status', 'FAIL')}",
            "",
            [],
            None,
            None,
            context,
        )

    floor_plan_img = Image.open(result["outputs"]["floor_plan_image"])
    plotly_fig = result["figures"]["plotly_3d"]
    status = result.get("status", "OK")

    if status == "NEEDS_CALIBRATION":
        info = (
            "### Pass 1 Complete\n"
            "Click two points on the provisional floor plan that correspond to a known real-world distance, "
            "then enter that distance and run Pass 2."
        )
        context = {
            "status": status,
            "error": "",
            "quality_failures": result.get("quality_gate", {}).get("failures", []),
            "accurate_mode": bool(accurate_mode),
            "diagnostic_mode": bool(diagnostic_mode),
        }
        return (
            floor_plan_img,
            plotly_fig,
            info,
            "🟡 NEEDS_CALIBRATION",
            result.get("session_id", ""),
            [],
            None,
            None,
            context,
        )

    measurements = _format_measurements(result)
    context = {
        "status": status,
        "error": "",
        "quality_failures": result.get("quality_gate", {}).get("failures", []),
        "accurate_mode": bool(accurate_mode),
        "diagnostic_mode": bool(diagnostic_mode),
    }
    return (
        floor_plan_img,
        plotly_fig,
        measurements,
        f"✅ {status}",
        "",
        [],
        result["outputs"].get("floor_plan_dxf"),
        result["outputs"].get("qa_report_json"),
        context,
    )



def add_calibration_point(evt: gr.SelectData, points_state: List[Tuple[float, float]]):
    """Capture two click points from the floor plan image."""
    points = list(points_state or [])

    x = y = None
    if isinstance(evt.index, (tuple, list)) and len(evt.index) >= 2:
        x, y = float(evt.index[0]), float(evt.index[1])
    else:
        x = float(getattr(evt, "x", 0.0))
        y = float(getattr(evt, "y", 0.0))

    if len(points) >= 2:
        points = []
    points.append((x, y))

    if len(points) == 1:
        text = f"Picked point 1: ({x:.1f}, {y:.1f}). Pick point 2."
    else:
        text = (
            f"Picked point 2: ({x:.1f}, {y:.1f}). "
            "Enter known distance and click 'Pass 2: Calibrate & Finalize'."
        )

    return points, text



def _build_pass2_blocked_message(pass1_context: dict) -> str:
    """Explain why pass-2 calibration is unavailable."""
    if not isinstance(pass1_context, dict):
        return "⚠️ Run Pass 1 first."

    status = pass1_context.get("status", "")
    failures = pass1_context.get("quality_failures", []) or []
    accurate_mode = bool(pass1_context.get("accurate_mode", True))

    if status == "NON_COMPLIANT_QUICK_MODE" or not accurate_mode:
        return (
            "### ⚠️ Pass 2 Blocked\n"
            "Pass 1 was run in quick mode. Enable **Accurate mode** and rerun Pass 1 until "
            "status is `NEEDS_CALIBRATION`."
        )

    if status == "INSUFFICIENT_CAPTURE":
        lines = "\n".join([f"- {f}" for f in failures]) if failures else "- Capture quality gate failed."
        return (
            "### ⚠️ Pass 2 Blocked\n"
            "Pass 1 did not reach `NEEDS_CALIBRATION` because capture quality failed:\n"
            f"{lines}\n\n"
            "Add more overlapping photos with stronger camera translation, then rerun Pass 1."
        )

    if status and status != "NEEDS_CALIBRATION":
        return (
            "### ⚠️ Pass 2 Blocked\n"
            f"Pass 1 status is `{status}`. Rerun Pass 1 and continue only when status is `NEEDS_CALIBRATION`."
        )

    return "⚠️ Run Pass 1 first."


def run_pass2(
    session_id,
    known_distance,
    known_unit,
    points_state,
    pass1_context,
    progress=gr.Progress(),
):
    """Pass 2: finalize with known-distance calibration."""
    if not session_id:
        blocked_msg = _build_pass2_blocked_message(pass1_context)
        return gr.update(), gr.update(), blocked_msg, "❌ Error", gr.update(), gr.update()

    points = points_state or []
    if len(points) != 2:
        return (
            gr.update(),
            gr.update(),
            "⚠️ Click exactly two calibration points.",
            "❌ Error",
            gr.update(),
            gr.update(),
        )

    if known_distance is None or known_distance <= 0:
        return (
            gr.update(),
            gr.update(),
            "⚠️ Enter a positive known distance.",
            "❌ Error",
            gr.update(),
            gr.update(),
        )

    reconstructor = get_reconstructor()

    def progress_cb(p, msg):
        progress(min(max(p, 0.0), 1.0), desc=msg)

    result = reconstructor.finalize_with_calibration(
        session_id,
        {
            "point1_px": points[0],
            "point2_px": points[1],
            "known_distance": float(known_distance),
            "known_distance_unit": known_unit,
        },
        progress_callback=progress_cb,
    )

    if not result.get("success", False):
        msg = result.get("error", "Calibration failed")
        if "Calibration session not found or expired" in msg:
            msg = (
                f"{msg}\n\n"
                "Run Pass 1 again (without restarting the app) to create a fresh calibration session."
            )
        return (
            gr.update(),
            gr.update(),
            f"### ❌ Calibration Failed\n{msg}",
            "❌ FAIL",
            gr.update(),
            gr.update(),
        )

    floor_plan_img = Image.open(result["outputs"]["floor_plan_image"])
    plotly_fig = result["figures"]["plotly_3d"]
    measurements = _format_measurements(result)

    return (
        floor_plan_img,
        plotly_fig,
        measurements,
        f"✅ {result.get('status', 'PASS')}",
        result["outputs"].get("floor_plan_dxf"),
        result["outputs"].get("qa_report_json"),
    )



def create_demo_interface():
    """Create Gradio UI."""
    with gr.Blocks(title="Room Reconstruction - Accurate 2D Floor Plan") as demo:
        gr.Markdown(
            """
# 🏠 Reconstruction-Grade 2D Floor Plan (Single-Room)

This interface supports a **two-pass calibrated workflow**:
1. **Pass 1**: Build provisional geometry and quality-check capture.
2. **Pass 2**: Click two known points, enter measured distance, finalize calibrated outputs.

Outputs include **PNG + DXF + QA JSON** when compliant.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Upload Photos (8-12 recommended for accurate mode)")
                photo_files = gr.Files(
                    label="Room Photos (bulk upload supported)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )

                compliance_profile = gr.Dropdown(
                    choices=["us_residential_v1", "commercial_boma_v1", "global_ipms_v1"],
                    value=DEFAULT_COMPLIANCE_PROFILE,
                    label="Compliance Profile",
                )
                accurate_mode = gr.Checkbox(
                    value=True,
                    label="Accurate mode (fail-closed quality gates)",
                )
                diagnostic_mode = gr.Checkbox(
                    value=False,
                    label="Diagnostic mode (relax registration gate only, testing)",
                )

                pass1_btn = gr.Button("Pass 1: Analyze & Build Provisional Plan", variant="primary")

                gr.Markdown("### 📐 Pass 2 Calibration")
                known_distance = gr.Number(label="Known Distance", value=1.0, precision=3)
                known_unit = gr.Dropdown(
                    choices=["m", "ft", "in"],
                    value="m",
                    label="Distance Unit",
                )
                calibration_text = gr.Markdown("Click two points on the floor plan image after Pass 1.")
                pass2_btn = gr.Button("Pass 2: Calibrate & Finalize", variant="secondary")

                status_text = gr.Textbox(label="Status", interactive=False)
                session_id = gr.Textbox(label="Session ID", interactive=False)
                point_state = gr.State([])
                pass1_context_state = gr.State(
                    {
                        "status": "IDLE",
                        "error": "",
                        "quality_failures": [],
                        "accurate_mode": True,
                        "diagnostic_mode": False,
                    }
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Outputs")
                floor_plan_output = gr.Image(label="2D Floor Plan", type="pil", height=420)
                model_3d_output = gr.Plot(label="3D Reconstruction")
                measurements_output = gr.Markdown(value="Run Pass 1 to begin.")
                dxf_output = gr.File(label="DXF Output")
                qa_output = gr.File(label="QA Report JSON")

        pass1_btn.click(
            fn=run_pass1,
            inputs=[
                photo_files,
                compliance_profile,
                accurate_mode,
                diagnostic_mode,
            ],
            outputs=[
                floor_plan_output,
                model_3d_output,
                measurements_output,
                status_text,
                session_id,
                point_state,
                dxf_output,
                qa_output,
                pass1_context_state,
            ],
        )

        floor_plan_output.select(
            fn=add_calibration_point,
            inputs=[point_state],
            outputs=[point_state, calibration_text],
        )

        pass2_btn.click(
            fn=run_pass2,
            inputs=[session_id, known_distance, known_unit, point_state, pass1_context_state],
            outputs=[
                floor_plan_output,
                model_3d_output,
                measurements_output,
                status_text,
                dxf_output,
                qa_output,
            ],
        )

        gr.Markdown(
            """
---
**Notes**
- Accurate mode requires sufficient capture quality and successful SfM registration.
- If quality checks fail, add more sharp images with better overlap.
- Calibration requires two clicked points matching a real measured span.
- Diagnostic mode relaxes only registration ratio to help debug capture; final status is `DIAGNOSTIC_ONLY`.
"""
        )

    return demo



def main():
    print("\n" + "=" * 60)
    print("🏠 Room Reconstruction Demo")
    print("=" * 60)
    print("\nStarting web interface...\n")

    demo = create_demo_interface()
    demo.launch(server_name="0.0.0.0", server_port=7870, share=False, show_error=True)


if __name__ == "__main__":
    main()
