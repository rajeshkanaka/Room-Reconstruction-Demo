# Room Reconstruction Demo

A standards-oriented single-room reconstruction system that converts photos into calibrated 2D floor plans and 3D geometry with QA outputs.

## 1) What This Project Is

This repository implements a two-pass pipeline for room reconstruction:

- Pass 1: Build provisional geometry and run capture quality gates.
- Pass 2: Apply known-distance calibration, generate final deliverables, and evaluate compliance status.

Primary target profile:

- `us_residential_v1` (ANSI/Fannie aligned reporting profile for software-generated floor plans).

Important scope statement:

- This project is aligned to profile-driven reporting and QA metrics.
- It does **not** claim blanket ANSI certification for all room-dimension practices.

## 2) Competition-Grade Positioning

This README is written for technical due diligence and industry review:

- Explicit compliance profile thresholds and pass/fail behavior.
- Fail-closed accurate mode for insufficient capture or bad calibration.
- Deliverable contract: PNG + DXF + QA JSON.
- Transparent status model and troubleshooting guidance.

## 3) Core Deliverables

After successful finalization (Pass 2), the system produces:

- Calibrated floor plan PNG.
- CAD-compatible DXF.
- QA report JSON with capture, calibration, SfM, and tolerance metrics.
- Interactive 3D HTML viewer.
- Point cloud PLY.

Typical output files in `/outputs`:

- `floor_plan_<timestamp>.png`
- `floor_plan_<timestamp>.dxf`
- `qa_report_<timestamp>.json`
- `room_3d_<timestamp>.html`
- `room_pointcloud_<timestamp>.ply`

DXF layer set:

- `A-WALL-EXT`
- `A-WALL-INT`
- `A-OPENING`
- `A-DIMS`
- `A-ANNO`

## 4) Workflow Overview

### Pass 1: Analyze and Build Provisional Plan

Pass 1 performs:

- SfM and geometric reconstruction.
- Capture quality gate evaluation.
- Provisional floor plan generation.
- Session creation for calibration.

Pass 1 status outcomes:

- `NEEDS_CALIBRATION`: quality gate passed; proceed to Pass 2.
- `INSUFFICIENT_CAPTURE`: fail-closed; improve capture and rerun.
- `NON_COMPLIANT_QUICK_MODE`: quick mode output only; not standards-compliant final output.

### Pass 2: Calibrate and Finalize

Pass 2 requires:

- Exactly two clicked points on the provisional plan.
- A known real-world distance and unit (`m`, `ft`, `in`).

Pass 2 performs:

- Scale factor computation from clicked segment and known distance.
- Calibration uncertainty check against profile threshold.
- Global geometry scaling.
- Final output generation (PNG + DXF + QA JSON).
- Compliance status computation.

Pass 2 final statuses:

- `PASS`: quality + calibration + profile tolerances passed.
- `FAIL`: calibration or tolerance checks failed.
- `DIAGNOSTIC_ONLY`: diagnostic mode was enabled; output is testing-only.

## 5) Compliance Profiles and Thresholds

| Profile | Intended context | Min images | Min registration ratio | Calibration uncertainty max | Critical tolerance | Overall tolerance |
|---|---|---:|---:|---:|---:|---:|
| `us_residential_v1` | ANSI/Fannie aligned residential reporting | 6 | 70% | 8 mm | 15 mm | 30 mm |
| `commercial_boma_v1` | Future commercial profile | 8 | 75% | 6 mm | 12 mm | 25 mm |
| `global_ipms_v1` | Future global profile | 8 | 75% | 6 mm | 12 mm | 25 mm |

Notes:

- `commercial_boma_v1` and `global_ipms_v1` are present as profile frameworks; method scope remains single-room v1.
- Accurate mode uses fail-closed behavior when quality/calibration gates fail.

## 6) Accuracy and QA Semantics

QA report accuracy metrics are computed from:

- Calibration uncertainty.
- Mean reprojection error from SfM.

Current implementation uses proxy predicted metrics:

- `predicted_critical_error_mm`
- `predicted_overall_error_mm`

These are compared against profile thresholds for pass/fail logic. Ground-truth benchmark datasets are a recommended next step for production-grade validation.

## 7) Capture Protocol (Recommended)

For robust indoor reconstruction and stable calibration:

- Capture 8-12 photos (minimum profile threshold still applies).
- Maintain strong overlap between adjacent photos.
- Move camera position between shots (translation, not only rotation).
- Keep images sharp; avoid motion blur.
- Include floor-wall intersections in multiple views.
- Use consistent focal setting if possible.
- Measure one reliable physical span for Pass 2 calibration.

Calibration click guidance:

- Pick two clear structural endpoints on a long segment.
- Avoid ambiguous textured regions.
- Use the same segment for real-world tape measurement.

## 8) Installation

### Prerequisites

- Python 3.9+
- `uv` (recommended) or `pip`
- macOS/Linux/Windows
- Optional GPU improves performance

### Install with uv

```bash
git clone https://github.com/rajeshkanaka/Room-Reconstruction-Demo.git
cd Room-Reconstruction-Demo

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run behavior:

- The depth model is downloaded from Hugging Face on first execution.
- Initial startup may take several minutes depending on network and hardware.

## 9) Run the Application

### Web UI

```bash
uv run python app.py
```

Open:

- `http://127.0.0.1:7870`

UI highlights:

- Bulk multi-image upload is supported.
- Pass 2 failures retain Pass 1 preview (no blanking of visuals).

### CLI

Pass 1 only:

```bash
uv run python run_cli.py sample_images/*.jpeg --compliance-profile us_residential_v1
```

Quick mode (non-compliant fast run):

```bash
uv run python run_cli.py sample_images/*.jpeg --quick-mode
```

Diagnostic mode (testing only):

```bash
uv run python run_cli.py sample_images/*.jpeg --diagnostic-mode
```

Pass 2 from CLI requires calibration points JSON:

```json
{
  "point1_px": [100.0, 220.0],
  "point2_px": [820.0, 220.0]
}
```

Finalize:

```bash
uv run python run_cli.py sample_images/*.jpeg \
  --compliance-profile us_residential_v1 \
  --known-distance 12.5 \
  --known-distance-unit ft \
  --calibration-points-json calibration_points.json
```

## 10) Status Model Reference

| Status | Meaning | What to do next |
|---|---|---|
| `NEEDS_CALIBRATION` | Pass 1 succeeded and session is active | Run Pass 2 with known distance |
| `INSUFFICIENT_CAPTURE` | Quality gate failed | Retake or add better photos |
| `NON_COMPLIANT_QUICK_MODE` | Quick mode output only | Use accurate mode for standards path |
| `PASS` | Finalized output passed gates/tolerances | Use deliverables |
| `FAIL` | Finalization failed | Fix calibration/capture and rerun |
| `DIAGNOSTIC_ONLY` | Diagnostic mode was enabled | Do not treat as standards-compliant |

## 11) Project Architecture

Primary modules:

- `modules/room_reconstructor.py`: orchestration and two-pass lifecycle.
- `modules/sfm_processor.py`: pycolmap/COLMAP SfM + optional dense MVS hooks.
- `modules/dense_reconstructor.py`: TSDF fusion and dense geometry assembly.
- `modules/floor_plan_generator.py`: floor plane normalization, wall extraction, 2D geometry.
- `modules/calibration.py`: scale factor and uncertainty computation.
- `modules/quality_gate.py`: image-count, blur, overlap, registration gating.
- `modules/qa_report.py`: compliance metrics and JSON report generation.
- `modules/dxf_exporter.py`: calibrated DXF export.
- `app.py`: Gradio UI, pass-1/pass-2 UX.
- `run_cli.py`: CLI workflow.

For deeper engineering detail:

- See `/ARCHITECTURE.md`.

Repository layout:

```text
Room-Reconstruction-Demo/
|- app.py
|- run_cli.py
|- config.py
|- requirements.txt
|- ARCHITECTURE.md
|- README.md
|- modules/
|  |- room_reconstructor.py
|  |- sfm_processor.py
|  |- dense_reconstructor.py
|  |- floor_plan_generator.py
|  |- calibration.py
|  |- quality_gate.py
|  |- qa_report.py
|  |- dxf_exporter.py
|  |- depth_estimator.py
|  |- visualizer_3d.py
|- outputs/
|- colmap_workspace/
|- sample_images/
|- tests/
```

Technology stack:

| Component | Primary libraries |
|---|---|
| Depth inference | `torch`, `transformers`, `timm`, `accelerate` |
| Photogrammetry | `pycolmap` |
| 3D geometry and fusion | `open3d`, `numpy`, `scipy` |
| Image processing | `opencv-python`, `Pillow`, `scikit-image` |
| Floor plan and reporting | `matplotlib`, `ezdxf`, built-in `json` |
| Visualization and UI | `plotly`, `gradio` |

## 12) Configuration Essentials

Main configuration file:

- `/config.py`

Key controls:

- `DEFAULT_COMPLIANCE_PROFILE`
- `ACCURATE_MODE_DEFAULT`
- `DIAGNOSTIC_MIN_REGISTRATION_RATIO`
- `SFM_*` tuning options
- `MVS_MAX_IMAGE_SIZE`
- `ASSUMED_ROOM_WIDTH_METERS` (quick mode only)

## 13) Troubleshooting

### "Calibration uncertainty too high (...)"

Cause:

- Click points are noisy/ambiguous or chosen on weak-support region.

Fix:

- Re-run Pass 1 and click clearer, farther structural points.
- Use exact measured distance for the same segment.
- Prefer `us_residential_v1` unless commercial threshold is explicitly required.

### "Calibration session not found or expired"

Cause:

- App restarted or session invalidated between passes.

Fix:

- Run Pass 1 again, then immediately run Pass 2.

### SfM logs show repeated "No good initial image pair" or partial registrations

Cause:

- Low overlap, low parallax, blur, or weak texture.

Fix:

- Retake with stronger translation and overlap.
- Increase photo count and viewpoint diversity.

### "PatchMatch requires CUDA but COLMAP was not compiled with it"

Cause:

- Dense PatchMatch path requires CUDA-enabled COLMAP build.

Impact:

- Dense MVS stage is skipped; pipeline still proceeds via available geometry paths.

### Open3D warning about clamped PLY colors

Cause:

- Color value normalization at write time.

Impact:

- Typically benign for geometry; mostly a color-range warning.

## 14) Verification

Run sanity checks:

```bash
uv run python -m py_compile app.py modules/sfm_processor.py
uv run python -m unittest discover -s tests -v
```

## 15) Current Limitations

- Scope is single-room v1.
- Final accuracy is capture-dependent.
- QA metrics are proxy predictions, not full ground-truth benchmark certification.
- Commercial/global profiles exist as structured placeholders and tighter threshold presets.

## 16) Roadmap

- Ground-truth dataset benchmarking and acceptance dashboards.
- Richer CAD semantics (openings, symbols, annotation standards).
- Automated calibration aid overlays and click-quality guidance.
- Expanded profile implementations beyond v1 room scope.

## 17) License and Acknowledgments

No separate license file is currently included in this repository. Add a project license before external redistribution.

Acknowledgments:

- COLMAP and pycolmap communities.
- Open3D contributors.
- PyTorch and Hugging Face ecosystems.
