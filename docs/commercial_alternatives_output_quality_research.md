# Commercial Alternatives Research: Output-Quality Upgrade Paths

Date: 2026-02-09  
Project: `Room-Reconstruction-Demo`  
Goal: Identify commercial APIs/tools that can significantly improve floor-plan quality and measurement reliability with minimal disruption to current architecture.

## 1) Current Pipeline and Practical Plug-In Points

Current core pipeline already has strong seams for integration:
- Geometry orchestration: `modules/room_reconstructor.py` (`_run_geometry_pipeline`, `reconstruct_from_arrays`, `finalize_with_calibration`)
- SfM/MVS backend: `modules/sfm_processor.py` (pycolmap/COLMAP)
- Standards and quality controls: `modules/quality_gate.py`, `modules/compliance_profile.py`
- Deliverables: `modules/floor_plan_generator.py`, `modules/dxf_exporter.py`, `modules/qa_report.py`

This means commercial integrations can be done with low blast radius if we normalize outputs into the existing internal schema (`points`, `colors`, registration stats, reprojection stats, metadata).

## 2) Evaluation Criteria

Scoring is directional (1 to 5, higher is better), based on official product docs and fit to your target:
- Output-quality uplift potential (geometry, registration robustness, floor-plan fidelity)
- Plug-in ease into current codebase
- Fit to your standards target (US residential reporting + explicit QA)
- Workflow compatibility with photos-only single-room capture

## 3) Ranked Options (Most Practical First)

| Option | Type | Quality Uplift | Plug-In Ease | Photos-Only Fit | Key Output Improvements | Overall Fit |
|---|---|---:|---:|---:|---|---:|
| **Agisoft Metashape Pro** | Commercial photogrammetry engine (local + Python API) | 5 | 4 | 4 | Better alignment, depth maps/dense cloud, scale bars/control points, automation reports | **High** |
| **RealityCapture (Epic)** | Commercial photogrammetry engine (CLI) | 5 | 4 | 4 | Strong image alignment + dense reconstruction, CLI automation, distance constraints | **High** |
| **PIX4Dengine Cloud API** | Commercial cloud processing API | 4 | 3 | 3 | Cloud automation, structured project lifecycle, downloadable reconstruction artifacts | **Medium-High** |
| **CubiCasa Integrate/Conversion API** | Floor-plan-as-a-service API | 3 (2D) | 5 | 4 | Fast 2D outputs, structured files (SVG/JSON/PNG/PDF), optional CAD add-ons | **Medium** |
| **Matterport Platform + Add-ons** | Capture platform + APIs/add-ons | 4 | 2 | 2 | MatterPak points/mesh, CAD and schematic floor plans, vendor ecosystem | **Medium-Low** |
| **HOVER API** | Property measurement API | 2 to 3 | 3 | 3 | Material/takeoff outputs + multi-format exports; useful for estimating workflows | **Low-Medium** |
| **iGUIDE ecosystem** | Hardware + cloud platform | 4 | 1 | 1 | High measurement confidence with controlled capture workflow + CAD outputs | **Low for current setup** |

## 4) What Exactly Improves in Output Quality

### A. Metashape Pro (Best quality/ease balance)

What improves:
- Registration robustness in hard indoor scenes (repetitive textures, low parallax edge cases).
- Denser and cleaner point clouds for wall extraction and polygon closure.
- Better scale handling via scale bars/control points.
- Built-in processing reports useful for QA evidence.

Why it is easy to plug in:
- Add a backend adapter that writes temporary images, runs Metashape via Python API, reads dense cloud + camera stats.
- Keep your existing `quality_gate`, calibration, DXF, and QA report modules unchanged.

Minimal-change integration path:
1. Add `modules/commercial_backends/metashape_backend.py`.
2. Return normalized dict compatible with current `sfm_result` shape.
3. In `RoomReconstructor._run_geometry_pipeline`, choose backend by config flag.

Expected impact in your outputs:
- Fewer "No reconstruction produced" cases.
- Better wall continuity in provisional and final floor plans.
- More stable calibrated dimensions after Pass 2.

### B. RealityCapture (Best for automation + speed)

What improves:
- Industrial-grade alignment and reconstruction pipeline with strong CLI automation.
- Supports distance constraints/control workflows (`defineDistance`) in command set.
- Good fit for scripted batch processing and repeatable runs.

Why it is easy to plug in:
- Keep app/UI and post-processing unchanged.
- Swap only SfM/MVS producer behind `_run_geometry_pipeline`.

Minimal-change integration path:
1. Add `modules/commercial_backends/realitycapture_backend.py` that shells out to RC CLI.
2. Export dense cloud/mesh, parse to `numpy` arrays.
3. Map RC stats into existing QA schema fields.

Expected impact in your outputs:
- Higher registration success on challenging sets.
- More complete dense geometry before 2D extraction.

### C. PIX4Dengine Cloud API (Strong managed-cloud option)

What improves:
- Managed cloud reconstruction with explicit API lifecycle (create project, upload, start processing, download outputs).
- Reduces local environment fragility and dependency issues.

Why integration is moderate (not trivial):
- Async job orchestration + polling + storage handling required.
- Cloud round-trip changes UX timing and error handling.

Minimal-change integration path:
1. Add `modules/commercial_backends/pix4d_cloud_backend.py`.
2. Convert current image list to API upload workflow.
3. Download point cloud/mesh and feed existing floor-plan/calibration/export stack.

Expected impact in your outputs:
- More consistent backend processing across machines.
- Potentially better geometry consistency than current open-source-only path.

### D. CubiCasa API (Fastest 2D improvement with least engineering effort)

What improves:
- Rapid, structured 2D floor-plan outputs and metadata.
- Optional CAD and extra deliverables through add-on flow.

Why it is easy to plug in:
- API-based, no heavy local 3D stack required.
- Can run as alternative provider for 2D artifacts while keeping your QA packaging layer.

Critical limitation:
- It is not a full replacement for reconstruction-grade 3D geometry.
- For strict +/-15 mm critical tolerances, this should be treated as a productivity path, not guaranteed compliance path.

### E. Matterport / HOVER / iGUIDE (Workflow shift options)

These can improve business-ready deliverables, but require larger capture/workflow changes:
- Matterport: strong ecosystem, MatterPak and CAD/floor-plan add-ons, but capture/process model differs from your photos-only pipeline.
- HOVER: useful commercial outputs and integrations; vendor-stated accuracy is typically percentage-based and may not meet your millimeter target.
- iGUIDE: strong measurement confidence with dedicated capture hardware; not drop-in for arbitrary photos.

## 5) Recommended Strategy for Your Current Setup

### Recommendation 1 (Primary): **Commercial geometry backend swap**

Use **Metashape Pro** or **RealityCapture** as a pluggable geometry backend while keeping your current:
- UI (Pass 1/Pass 2)
- Calibration logic
- Quality gate semantics
- DXF/PNG/QA report generation

Why this is the best dual-purpose upgrade:
- Large quality gain with small architecture change.
- Preserves your standards-driven QA and compliance framing.
- Keeps ownership of outputs and tolerance checks in your code.

### Recommendation 2 (Secondary): **CubiCasa as fast parallel provider**

Add CubiCasa as an optional "fast plan" backend for:
- quick-turn plan generation,
- operator preview,
- fallback when SfM is unstable.

Keep your reconstruction-grade path as the authoritative final for strict jobs.

## 6) Integration Blueprint (Low-Change)

### Step 1: Add backend interface

Create `modules/commercial_backends/base.py`:
- `run(images: list[np.ndarray], session_meta: dict) -> dict`
- Return normalized keys:
  - `success`
  - `dense_points`, `dense_colors`
  - `num_registered`, `num_images`
  - `reprojection_error_px`
  - `source_backend`

### Step 2: Route in one place

Modify only `RoomReconstructor._run_geometry_pipeline` to select backend:
- `colmap` (current)
- `metashape`
- `realitycapture`
- `pix4d_cloud`
- `cubicasa_2d` (special path)

### Step 3: Preserve existing downstream stack

Do **not** rewrite:
- `quality_gate.py`
- `calibration.py`
- `floor_plan_generator.py` output contracts
- `dxf_exporter.py`
- `qa_report.py`

Only enrich QA report with `source_backend` and backend-native metrics.

### Step 4: Benchmark gate before production switch

Run your curated dataset and require all:
- Critical spans MAE <= 15 mm
- Overall spans P95 <= 30 mm
- Area error <= 2.5%
- Pass-to-success conversion >= 90% on qualified captures

## 7) Commercial Choice by Business Objective

| Objective | Best choice | Why |
|---|---|---|
| Max geometric accuracy with minimal rewrite | **Metashape Pro** | Strong photogrammetry + Python automation + scale/control workflows |
| Fast industrial CLI integration | **RealityCapture** | Mature command pipeline and batch-friendly operations |
| Reduce local infra burden | **PIX4Dengine Cloud API** | Cloud API lifecycle and managed processing |
| Fastest operator-friendly 2D output | **CubiCasa API** | Structured floor-plan outputs with minimal engineering |
| Enterprise ecosystem / digital twin workflows | Matterport | Strong platform and add-ons but larger workflow shift |

## 8) Risks and Constraints to Decide Early

- Licensing and per-project costs can exceed open-source stack quickly.
- Cloud APIs introduce data-governance and latency constraints.
- Some vendors require capture discipline/hardware beyond ad-hoc phone photos.
- Vendor-stated accuracy is often context-specific; must validate on your own benchmark set.

## 9) Source Links (Official/Product Docs)

- RealityCapture all commands (official help): https://rshelp.capturingreality.com/en-US/appbasics/allcommands.htm
- RealityCapture pricing/licensing page: https://www.capturingreality.com/pricing
- Agisoft Metashape User Manual PDF (v2.2.2): https://www.agisoft.com/pdf/metashape-pro_2_2_en.pdf
- Agisoft licensing page: https://www.agisoft.com/buy/online-store/
- PIX4Dengine API docs: https://developer.pix4d.com/engine/api/endpoint/project
- PIX4D support docs index: https://support.pix4d.com/hc/en-us
- CubiCasa Integrate API docs: https://docs.cubi.casa/integrate/
- CubiCasa Conversion API docs: https://docs.cubi.casa/conversion/
- Matterport APIs overview: https://docs.matterport.com/developer-tools/apis
- Matterport API reference: https://api.matterport.com
- Matterport add-ons (MatterPak/CAD/floor plans): https://matterport.com/add-ons
- Matterport measured model claim context: https://matterport.com/blog/why-measurements-in-matterport-are-actually-accurate
- HOVER API docs: https://docs.hover.to/reference/getting-started-with-your-api
- HOVER integrations/API overview: https://hover.to/integrations/api/
- iGUIDE CAD floor plan details: https://youriguide.com/cad-floor-plans/
- iGUIDE platform/accuracy messaging: https://goiguide.com/

## 10) Bottom Line

If your priority is **significant quality improvement without major architecture change**, the strongest path is:
1. **Metashape Pro or RealityCapture as pluggable geometry backend** (primary upgrade)
2. **CubiCasa API as optional fast 2D parallel path** (productivity upgrade)

This combination gives both immediate UX/business gains and a realistic path toward reconstruction-grade outputs under your existing standards-first framework.
