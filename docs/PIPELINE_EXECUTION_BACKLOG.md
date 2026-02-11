# 2D Floor Plan Quality Backlog

## Status Snapshot
- T1: Completed
- T2: Completed
- T3: Completed
- T4: Completed
- T5: Completed
- T6: Completed
- T7: Completed
- T8: Completed
- T9: Completed

## T1 - Plane-First Wall Extraction
- Goal: Extract walls from 3D vertical planes first, then project to 2D.
- Scope:
  - Add vertical wall-plane detection from fused point cloud.
  - Filter planes by verticality, height span, and minimum segment length.
  - Use these segments as primary input to room topology.
  - Keep depth-edge wall extraction only as fallback/augment.
- Acceptance criteria:
  - On synthetic room point cloud, detect >=4 wall segments.
  - On sample run, wall extraction should not collapse into disconnected micro-segments.
  - Pipeline remains backward-compatible if Open3D is unavailable.
- Tests:
  - Unit: synthetic vertical-plane extraction test.
  - Integration: end-to-end run with sample images generates closed room plan.

## T2 - Wall Graph Optimization
- Goal: Convert raw segments into a consistent wall graph before room extraction.
- Scope:
  - Endpoint snapping with graph constraints.
  - Loop closure constraints.
  - Soft Manhattan regularization and outlier segment pruning.
- Acceptance criteria:
  - Typical plans produce 1 closed primary loop.
  - Opposite walls in rectangular rooms remain consistent within tolerance.
- Tests:
  - Synthetic rectangular and L-shaped cases.
  - Regression tests against skewed/misaligned outputs.

## T3 - Multi-Cue Scale Solver
- Goal: Reduce absolute dimension error from single assumed-width prior.
- Scope:
  - Fuse cues: camera baseline statistics, known architectural priors, optional user anchor.
  - Replace single-axis calibration with weighted solver + confidence.
- Acceptance criteria:
  - Median absolute width/depth error improves on benchmark set.
  - Reported scale confidence available in output payload.
- Tests:
  - Deterministic synthetic scale tests.
  - Dataset-level evaluation script.

## T4 - Multi-View Opening Fusion
- Goal: Improve door/window placement consistency.
- Scope:
  - Triangulate/aggregate opening detections across views.
  - Attach openings to walls using geometric consistency + visibility checks.
- Acceptance criteria:
  - Fewer duplicate/misattached openings on sample set.
  - Door/window counts stable across reruns.
- Tests:
  - Opening dedupe and wall-assignment tests.
  - End-to-end regression with annotated sample scenes.

## T5 - Quality Scoring and Output Gating
- Goal: Prevent low-confidence geometry from being presented as final.
- Scope:
  - Compute confidence from geometry consistency, closure, and scale quality.
  - Add output modes: `high_confidence`, `approximate`, `needs_more_images`.
- Acceptance criteria:
  - Low-confidence scenes include explicit warnings and actionable guidance.
  - High-confidence scenes continue normal export flow.
- Tests:
  - Mode selection tests.
  - UI/API contract tests for confidence fields.

## T6 - Wall-Graph Closure Hardening
- Goal: Improve loop closure before room polygon extraction.
- Scope:
  - Endpoint clustering using order-independent snapping.
  - Near-perpendicular corner snapping to line intersections.
  - Tightened long-jump prevention in boundary ordering.
- Acceptance criteria:
  - Fragmented corner gaps close in synthetic tests.
  - Reduced false topology fragmentation on sample room.
- Tests:
  - Perpendicular junction closure regression.
  - Noisy polygon regularization tests.

## T7 - Multi-View Consistency Gate
- Goal: Remove wall candidates that are not corroborated across views.
- Scope:
  - Per-segment support scoring across depth-derived observations.
  - Filter unstable segments before topology solve.
  - Record support diagnostics in quality flags.
- Acceptance criteria:
  - One-off diagonal segments are dropped in synthetic tests.
  - Supported wall count remains sufficient for room extraction.
- Tests:
  - Multi-view support filtering unit test.
  - Quality warnings when support is weak.

## T8 - Confidence-Aware Export Behavior
- Goal: Surface confidence level in every exported artifact.
- Scope:
  - Attach quality metadata to floor-plan model.
  - Add quality banners in PNG/SVG/DXF outputs.
  - Enforce export policy (`normal_export` vs approximate/needs-more-images).
- Acceptance criteria:
  - Approximate plans include explicit warning banners.
  - High-confidence plans keep normal export behavior.
- Tests:
  - Rendering tests for quality banners across SVG/DXF/PNG.
  - Quality policy gating tests.

## T9 - Sample Room Regression Pack
- Goal: Freeze the 9-image sample room as a repeatable benchmark gate.
- Scope:
  - Baseline fixture with acceptance thresholds.
  - Live optional regression test for `sample_images/*.jpeg`.
  - Assertions for wall count, closure score, fallback usage, and quality mode/policy consistency.
- Acceptance criteria:
  - Running the live benchmark yields a machine-readable report in `outputs/t9_sample_room_report.json`.
  - Guardrails prevent fallback geometry from being marked industry-ready/high-confidence.
- Tests:
  - Guardrail logic unit test.
  - Live benchmark test behind `RUN_SAMPLE_REGRESSION=1`.
