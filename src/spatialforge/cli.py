"""Command-line interface: python -m spatialforge validate-lidar <folder>."""

from __future__ import annotations

import argparse
import sys

from spatialforge.lidar import LidarValidationResult, Status, validate_lidar_capture
from spatialforge.lidar.reconstruction import (
    ReconstructionError,
    ReconstructionOptions,
    ReconstructionResult,
    reconstruct_lidar,
    sanity_flags,
)


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


def _xyz(v) -> str:
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def _intr(i) -> str:
    return f"fx={i.fx:.2f} fy={i.fy:.2f} cx={i.cx:.2f} cy={i.cy:.2f}"


def format_reconstruction_report(r: ReconstructionResult, max_range_m: float) -> str:
    sx, sy = r.scale_factors
    cloud_ext = r.cloud_max - r.cloud_min
    traj_ext = r.trajectory_max - r.trajectory_min
    lines = [
        f"Capture: {r.capture_path.parent.name}  ({r.capture_path})",
        f"Frames: {r.frames_used} used of {r.frames_available} available",
        f"Depth scale: {r.depth_scale} m per raw unit (assumed; see README)",
        f"Pose convention: {r.pose_convention}",
        "",
        f"Intrinsics (frame {r.example_frame}):",
        f"  source {r.source_size[0]}x{r.source_size[1]}: {_intr(r.example_source_intrinsics)}",
        f"  scale factors: sx={sx:.5f} sy={sy:.5f}",
        f"  depth  {r.depth_size[0]}x{r.depth_size[1]}: {_intr(r.example_depth_intrinsics)}",
        f"  depth image centre: ({r.depth_size[0] / 2:.1f}, {r.depth_size[1] / 2:.1f})",
        "",
        "Points:",
        f"  candidate pixels:        {r.candidate_points}",
        f"  removed (depth/range):   {r.removed_by_depth}",
        f"  removed (confidence):    {r.removed_by_confidence}",
        f"  after filtering:         {r.points_after_filter}",
        f"  after voxel downsample:  {r.points_after_downsample}",
        "",
        "Extents in native capture frame (metres):",
        f"  camera path min {_xyz(r.trajectory_min)} max {_xyz(r.trajectory_max)} extent {_xyz(traj_ext)}",
        f"  point cloud min {_xyz(r.cloud_min)} max {_xyz(r.cloud_max)} extent {_xyz(cloud_ext)}",
    ]
    flags = sanity_flags(r, max_range_m)
    lines += ["", "Sanity check: " + ("OK, extents look like a plausible indoor scan" if not flags else "SUSPICIOUS")]
    lines += [f"  - {f}" for f in flags]
    if r.warnings:
        lines += ["", f"Capture warnings ({len(r.warnings)}):"] + [f"  - {w}" for w in r.warnings]
    lines += ["", f"Runtime: {r.runtime_s:.1f} s", f"PLY written: {r.output_path}"]
    return "\n".join(lines)


def _size_arg(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1920x1440")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spatialforge")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate-lidar", help="validate one LiDAR capture folder")
    p.add_argument("path", help="path to the capture folder")

    r = sub.add_parser("reconstruct-lidar", help="build a metric PLY point cloud from a LiDAR capture")
    r.add_argument("path", help="path to the capture folder")
    r.add_argument("--output", required=True, help="PLY file to write")
    r.add_argument("--frame-step", type=int, default=30, help="use every Nth frame (default 30)")
    r.add_argument("--max-frames", type=int, default=60, help="cap on frames used, evenly spaced (default 60)")
    r.add_argument("--min-confidence", type=int, default=2, help="minimum confidence level 0-2 (default 2)")
    r.add_argument("--voxel-size", type=float, default=0.02, help="voxel size in metres (default 0.02)")
    r.add_argument("--min-range", type=float, default=0.1, help="minimum depth in metres (default 0.1)")
    r.add_argument("--max-range", type=float, default=5.0, help="maximum depth in metres (default 5.0)")
    r.add_argument("--source-size", type=_size_arg, help="override the image size the intrinsics refer to")
    args = parser.parse_args(argv)

    if args.command == "validate-lidar":
        result = validate_lidar_capture(args.path)
        print(format_report(result))
        return 1 if result.status is Status.INVALID else 0

    options = ReconstructionOptions(
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        voxel_size=args.voxel_size,
        min_range_m=args.min_range,
        max_range_m=args.max_range,
        source_size=args.source_size,
    )
    try:
        rec = reconstruct_lidar(args.path, args.output, options)
    except ReconstructionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_reconstruction_report(rec, options.max_range_m))
    return 0
