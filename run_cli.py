#!/usr/bin/env python3
"""Room Reconstruction CLI with accurate-mode calibration workflow."""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



def _expand_images(patterns):
    image_paths = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        elif os.path.isfile(pattern):
            image_paths.append(pattern)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in image_paths if os.path.splitext(p.lower())[1] in valid_ext]



def _load_calibration_points(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "point1_px" not in payload or "point2_px" not in payload:
        raise ValueError("Calibration JSON must contain point1_px and point2_px")

    return payload["point1_px"], payload["point2_px"]



def print_result_summary(result):
    status = result.get("status", "UNKNOWN")
    print("\n" + "=" * 60)
    print(f"Status: {status}")
    print("=" * 60)

    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        compliance = result.get("compliance", {})
        if compliance:
            print(f"Compliance profile: {compliance.get('profile', 'n/a')}")
            print(f"Compliance status: {compliance.get('status', status)}")
        if result.get("diagnostic_mode", False):
            print("Diagnostic mode: ON (registration gate relaxed for testing)")

        quality = result.get("quality_gate", {})
        if quality:
            print("\nQuality gate:")
            print(f"  Passed: {quality.get('passed', False)}")
            failures = quality.get("failures", [])
            if failures:
                print("  Failures:")
                for f in failures:
                    print(f"    - {f}")
        print("")
        return

    compliance = result.get("compliance", {})
    print(f"Compliance profile: {compliance.get('profile', 'n/a')}")
    print(f"Compliance status: {compliance.get('status', status)}")
    if result.get("diagnostic_mode", False):
        print("Diagnostic mode: ON (registration gate relaxed for testing)")

    m = result.get("measurements", {})
    if m:
        print("\nMeasurements:")
        print(f"  Width:  {m.get('width_m', 0):.3f} m ({m.get('width_ft', 0):.3f} ft)")
        print(f"  Depth:  {m.get('depth_m', 0):.3f} m ({m.get('depth_ft', 0):.3f} ft)")
        print(f"  Area:   {m.get('area_sqm', 0):.3f} m² ({m.get('area_sqft', 0):.3f} sq ft)")

    quality = result.get("quality_gate", {})
    if quality:
        print("\nQuality gate:")
        print(f"  Passed: {quality.get('passed', False)}")
        failures = quality.get("failures", [])
        if failures:
            print("  Failures:")
            for f in failures:
                print(f"    - {f}")

    outputs = result.get("outputs", {})
    if outputs:
        print("\nOutput files:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")

    if status == "NEEDS_CALIBRATION":
        print("\nCalibration required:")
        print(f"  Session ID: {result.get('session_id', '')}")
        print("  Provide --known-distance and --calibration-points-json to finalize.")

    print("")



def main():
    parser = argparse.ArgumentParser(
        description="Room Reconstruction CLI (accurate-mode with calibration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("images", nargs="+", help="Image paths (glob patterns supported)")
    parser.add_argument(
        "--compliance-profile",
        default="us_residential_v1",
        choices=["us_residential_v1", "commercial_boma_v1", "global_ipms_v1"],
        help="Compliance profile",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Disable accurate-mode quality gates and calibration workflow",
    )
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="Relax only SfM registration ratio gate for testing (final status becomes DIAGNOSTIC_ONLY)",
    )
    parser.add_argument(
        "--known-distance",
        type=float,
        default=None,
        help="Known real-world distance for pass-2 calibration",
    )
    parser.add_argument(
        "--known-distance-unit",
        default="m",
        choices=["m", "ft", "in"],
        help="Unit for known distance",
    )
    parser.add_argument(
        "--calibration-points-json",
        type=str,
        default=None,
        help="JSON file containing point1_px and point2_px from provisional floor plan",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open Open3D point cloud visualization on success",
    )

    args = parser.parse_args()

    image_paths = _expand_images(args.images)
    if not image_paths:
        print("❌ No valid image files found.")
        return 1

    print("\n" + "=" * 60)
    print("🏠 Room Reconstruction CLI")
    print("=" * 60)
    print(f"Images: {len(image_paths)}")
    for i, p in enumerate(image_paths, 1):
        print(f"  {i}. {os.path.basename(p)}")

    # Delay heavy imports until after argument parsing/validation.
    from modules.room_reconstructor import RoomReconstructor

    reconstructor = RoomReconstructor()

    accurate_mode = not args.quick_mode
    result = reconstructor.reconstruct(
        image_paths,
        compliance_profile=args.compliance_profile,
        accurate_mode=accurate_mode,
        diagnostic_mode=bool(args.diagnostic_mode),
    )

    if (
        accurate_mode
        and result.get("success")
        and result.get("status") == "NEEDS_CALIBRATION"
        and args.known_distance is not None
        and args.calibration_points_json
    ):
        p1, p2 = _load_calibration_points(args.calibration_points_json)
        result = reconstructor.finalize_with_calibration(
            result["session_id"],
            {
                "point1_px": p1,
                "point2_px": p2,
                "known_distance": args.known_distance,
                "known_distance_unit": args.known_distance_unit,
            },
        )

    print_result_summary(result)

    if args.visualize and result.get("success") and result.get("status") != "NEEDS_CALIBRATION":
        try:
            points = reconstructor.last_result.get("data", {}).get("points")
            colors = reconstructor.last_result.get("data", {}).get("colors")
            if points is not None:
                reconstructor.visualizer.visualize_open3d(points, colors)
        except Exception as exc:
            print(f"Visualization skipped: {exc}")

    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
