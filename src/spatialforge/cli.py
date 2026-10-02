"""Command-line interface: python -m spatialforge validate-lidar <folder>."""

from __future__ import annotations

import argparse

from spatialforge.lidar import LidarValidationResult, Status, validate_lidar_capture


def format_report(r: LidarValidationResult) -> str:
    m = r.metadata
    lines = [f"LiDAR capture: {r.capture_path}"]
    if "note" in m:
        lines.append(m["note"])

    lines += ["", "Files / folders:"]
    for name, ok in r.present.items():
        lines.append(f"  [{'x' if ok else ' '}] {name}")

    lines += [
        "",
        "Frame counts:",
        f"  depth frames:      {r.depth_frame_count}",
        f"  confidence frames: {r.confidence_frame_count}",
        f"  pose rows:         {r.pose_frame_count}",
    ]
    for key, label in (
        ("depth_confidence_aligned", "depth/confidence names aligned"),
        ("pose_depth_aligned", "pose IDs match depth frames"),
    ):
        if key in m:
            lines.append(f"  {label}: {'yes' if m[key] else 'NO'}")
    if r.duplicate_pose_frames:
        lines.append(f"  duplicate pose frame IDs: {len(r.duplicate_pose_frames)}")
    for what, ids in r.missing_frames.items():
        shown = ", ".join(str(i) for i in ids[:10]) + (" ..." if len(ids) > 10 else "")
        lines.append(f"  missing: {len(ids)} {what} ({shown})")

    if "depth_resolution" in m:
        lines += ["", "Depth image:", f"  resolution: {m['depth_resolution']}", f"  dtype: {m['depth_dtype']}"]
        s = m.get("depth_stats")
        if s:
            lines += [
                f"  valid (non-zero) depth over {s['sampled_frames']} sampled frames:",
                f"    min {s['min']}   median {s['median']:.0f}   max {s['max']}",
                f"    unit: {s['unit']}",
            ]
    if "confidence_resolution" in m:
        hist = ", ".join(f"{k}: {v}" for k, v in m["confidence_histogram"].items())
        lines += [
            "",
            "Confidence image:",
            f"  resolution: {m['confidence_resolution']}",
            f"  dtype: {m['confidence_dtype']}",
            f"  level histogram (pixels, sampled frames): {hist}",
        ]

    sensor = []
    if "intrinsics" in m:
        i = m["intrinsics"]
        sensor.append(f"  camera matrix: fx={i['fx']} fy={i['fy']} cx={i['cx']} cy={i['cy']}")
    if "odometry_fx_range" in m:
        lo, hi = m["odometry_fx_range"]
        sensor.append(f"  per-frame fx in odometry.csv: {lo} .. {hi}")
    if "imu_sample_count" in m:
        sensor.append(f"  IMU samples: {m['imu_sample_count']}")
    if "rgb_size_mb" in m:
        sensor.append(f"  rgb.mp4 size: {m['rgb_size_mb']} MB")
    if sensor:
        lines += ["", "Sensor metadata:"] + sensor

    if r.warnings:
        lines += ["", f"Warnings ({len(r.warnings)}):"] + [f"  - {w}" for w in r.warnings]
    if r.errors:
        lines += ["", f"Errors ({len(r.errors)}):"] + [f"  - {e}" for e in r.errors]
    lines += ["", f"RESULT: {r.status.value}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spatialforge")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate-lidar", help="validate one LiDAR capture folder")
    p.add_argument("path", help="path to the capture folder")
    args = parser.parse_args(argv)

    result = validate_lidar_capture(args.path)
    print(format_report(result))
    return 1 if result.status is Status.INVALID else 0
