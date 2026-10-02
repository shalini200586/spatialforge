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


def format_drift_report(rep: dict, output_dir) -> str:
    seq, loop = rep["sequential_edges"], rep["loop_closures"]
    b, a, pc = rep["before"], rep["after"], rep["pose_correction"]
    lines = [
        f"Capture: {rep['capture']}   frames used: {rep['frames_used']} of {rep['frames_available']}",
        "",
        f"Sequential ICP: {seq['attempted']} attempted, {seq['accepted']} accepted, {seq['rejected']} rejected",
        f"  rejection reasons: {seq['rejection_reasons']}",
    ]
    if seq["accepted"]:
        lines.append(
            f"  accepted: mean fitness {seq['accepted_mean_fitness']:.3f}, "
            f"inlier RMSE mean {seq['accepted_mean_inlier_rmse_m']:.4f} m / median {seq['accepted_median_inlier_rmse_m']:.4f} m"
        )
    lines.append(
        f"Loop closures: {loop['candidates']} candidates, {loop['accepted']} accepted, {loop['rejected']} rejected"
    )
    for kind, res in rep["constraint_residual_before_after"].items():
        rb, ra = res["before_mean_translation_m_and_rotation_deg"], res["after_mean_translation_m_and_rotation_deg"]
        lines.append(
            f"  {kind} constraint residual (mean): before {rb[0]:.4f} m / {rb[1]:.2f} deg -> after {ra[0]:.4f} m / {ra[1]:.2f} deg"
        )
    lines += [
        "",
        f"{'metric':36s}{'OFF (before)':>14s}{'ON (after)':>14s}",
    ]
    for key in ("overlap", "neighbour_residual_median_m", "neighbour_inlier_fraction", "floor_band_thickness_m",
                "floor_peak_share", "wall_slab_cells_per_1000_points", "footprint_x_m", "footprint_z_m"):
        lines.append(f"{key:36s}{b[key]:>14.4f}{a[key]:>14.4f}")
    lines += [
        "",
        f"Pose correction: translation mean {pc['mean_translation_m']:.3f} m / max {pc['max_translation_m']:.3f} m, "
        f"rotation mean {pc['mean_rotation_deg']:.2f} deg / max {pc['max_rotation_deg']:.2f} deg",
        f"Roll/pitch change: {pc['max_roll_pitch_change_deg_before_gravity_constraint']:.2f} deg raw optimiser output, "
        f"{pc['max_roll_pitch_change_deg']:.2f} deg after gravity constraint",
        "",
        "CORRECTION ACCEPTED" if rep["accepted"] else f"CORRECTION REJECTED: {rep['fallback_reason']}",
        f"Poses to use: {rep['production_poses']}",
        f"Runtime: {rep['runtime_s']:.1f} s",
        f"Artifacts: {output_dir}",
    ]
    return "\n".join(lines)


def format_planes_report(rep: dict, output_dir) -> str:
    f = rep["floor"]
    lines = [
        f"Capture: {rep['capture']}   frames: {rep['frames_used']} of {rep['frames_available']}   points: {rep['point_count']}",
        f"Pose source: {rep['pose_source']}" + (f"  (drift correction rejected: {rep['drift_fallback_reason']})" if rep["drift_fallback_reason"] else ""),
        f"Camera path Y range: {rep['camera_y_range_m'][0]:.2f} .. {rep['camera_y_range_m'][1]:.2f} m",
        "",
    ]
    if f["observed"]:
        lines += [
            f"FLOOR observed: height y = {f['height_m']:.3f} m, tilt {f['tilt_deg']:.2f} deg",
            f"  plane [a,b,c,d] = {[round(v, 4) for v in f['plane']]}",
            f"  inliers {f['inlier_count']} ({f['inlier_ratio']:.1%} of points), residual median {f['residual_median_m']:.4f} m, "
            f"p90 {f['residual_p90_m']:.4f} m, robust sigma {f['sigma_m']:.4f} m",
            f"  solid area {f['solid_area_m2']:.1f} m2 ({f['coverage_ratio']:.0%} of scanned footprint), "
            f"extent {f['x_extent_m']:.1f} x {f['z_extent_m']:.1f} m",
        ]
    else:
        lines.append(f"FLOOR not observed: {f['reject_reason']}")
    levels = rep["ceiling_levels"]
    if not levels:
        lines.append(f"CEILING not observed: {rep['ceiling']['reject_reason']}")
    for k, lvl in enumerate(levels, 1):
        h = lvl["height"]
        lines += [
            f"CEILING level {k}{' (largest)' if k == 1 else ''}: y = {lvl['height_m']:.3f} m, tilt {lvl['tilt_deg']:.2f} deg",
            f"  inliers {lvl['inlier_count']} ({lvl['inlier_ratio']:.1%} of points), residual median {lvl['residual_median_m']:.4f} m, "
            f"p90 {lvl['residual_p90_m']:.4f} m, robust sigma {lvl['sigma_m']:.4f} m",
            f"  solid area {lvl['solid_area_m2']:.1f} m2 ({lvl['coverage_ratio']:.0%} of scanned footprint), "
            f"extent {lvl['x_extent_m']:.1f} x {lvl['z_extent_m']:.1f} m",
            f"  CEILING HEIGHT {h['value_m']:.3f} m   interval [{h['confidence_interval_m'][0]:.3f}, {h['confidence_interval_m'][1]:.3f}] m   "
            f"confidence {h['confidence']:.2f}",
        ]
    if rep["warnings"]:
        lines += ["", f"Warnings ({len(rep['warnings'])}):"] + [f"  - {w}" for w in rep["warnings"]]
    lines += ["", f"Runtime: {rep['runtime_s']:.1f} s", f"Artifacts: {output_dir}"]
    return "\n".join(lines)


def format_walls_report(rep: dict, output_dir) -> str:
    lines = [
        f"Capture: {rep['capture']}   frames: {rep['frames_used']} of {rep['frames_available']}   "
        f"points: {rep['point_count']} ({rep['zone_point_count']} in the wall height band)",
        f"Pose source: {rep['pose_source']}   floor y = {rep['floor_y_m']:.3f} m"
        + (f"   ceiling levels: {rep['ceiling_levels_m']} m" if rep["ceiling_levels_m"] else ""),
        f"Dominant directions: {[round(d, 1) for d in rep['dominant_directions_deg']]} deg "
        f"(strength {rep['manhattan_strength']:.2f})",
        "",
        f"Candidate lines {rep['candidate_lines']}, duplicate lines merged {rep['merged_duplicate_lines']}, "
        f"duplicate segments suppressed {rep['suppressed_duplicate_segments']}, "
        f"rejected candidates {rep['rejected_candidates']}",
        f"ACCEPTED: {rep['accepted_walls']} walls, {rep['accepted_segments']} observed segments",
        "",
        "  id        length  observed  orient  snap  vspan  inliers  resid p50/p90   pos.unc  evidence  gaps(m)",
    ]
    for w in rep["walls"]:
        gaps = ",".join(f"{g['length_m']:.2f}" for g in w["gaps"]) or "-"
        lines.append(
            f"  {w['id']}  {w['length_m']:6.2f}  {w['observed_length_m']:7.2f}  {w['orientation_deg']:6.1f}  "
            f"{'yes' if w['snapped'] else 'no ':>4}  {w['vertical_span_m']:5.2f}  {w['inlier_count']:7d}  "
            f"{w['residual_median_m']:.3f}/{w['residual_p90_m']:.3f}   {w['position_uncertainty_m']:.3f}   "
            f"{w['evidence']:<8}  {gaps}"
        )
    if rep["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {x}" for x in rep["warnings"]]
    lines += ["", f"Runtime: {rep['runtime_s']:.1f} s", f"Artifacts: {output_dir}"]
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

    d = sub.add_parser(
        "correct-lidar-drift",
        help="drift-correction ablation: baseline vs corrected poses, with metrics, PLYs and images",
    )
    d.add_argument("path", help="path to the capture folder")
    d.add_argument("--output-dir", required=True, help="folder for before/after PLY, PNG and drift_report.json")
    d.add_argument("--frame-step", type=int, default=1, help="use every Nth frame (default 1)")
    d.add_argument("--max-frames", type=int, default=400,
                   help="cap on frames used, evenly spaced (default 400; ICP needs overlapping neighbours)")
    d.add_argument("--min-confidence", type=int, default=2, help="minimum confidence level 0-2 (default 2)")
    d.add_argument("--voxel-size", type=float, default=0.02, help="output voxel size in metres (default 0.02)")
    d.add_argument("--reg-voxel", type=float, default=0.05, help="registration voxel size in metres (default 0.05)")
    d.add_argument("--no-loop-closure", action="store_true", help="disable loop-closure constraints")
    d.add_argument("--source-size", type=_size_arg, help="override the image size the intrinsics refer to")

    h = sub.add_parser(
        "analyze-horizontal-planes",
        help="detect floor and ceiling planes and measure ceiling height with a confidence interval",
    )
    h.add_argument("path", help="path to the capture folder")
    h.add_argument("--output-dir", required=True, help="folder for horizontal_planes.json, plots and inlier PLYs")
    h.add_argument("--frame-step", type=int, default=1, help="use every Nth frame (default 1)")
    h.add_argument("--max-frames", type=int, default=400, help="cap on frames used, evenly spaced (default 400)")
    h.add_argument("--min-confidence", type=int, default=2, help="minimum confidence level 0-2 (default 2)")
    h.add_argument("--voxel-size", type=float, default=0.02, help="voxel size in metres (default 0.02)")
    h.add_argument("--max-plane-tilt", type=float, default=5.0, help="max tilt from horizontal in degrees (default 5)")
    h.add_argument("--min-ceiling-height", type=float, default=2.0, help="ceiling search range, metres above floor")
    h.add_argument("--max-ceiling-height", type=float, default=4.5, help="ceiling search range, metres above floor")
    h.add_argument("--source-size", type=_size_arg, help="override the image size the intrinsics refer to")

    w = sub.add_parser(
        "analyze-walls",
        help="extract structural vertical walls as metric 2D segments (no rooms or openings yet)",
    )
    w.add_argument("path", help="path to the capture folder")
    w.add_argument("--output-dir", required=True, help="folder for walls.json, walls_topdown.png, wall_inliers.ply")
    w.add_argument("--frame-step", type=int, default=1, help="use every Nth frame (default 1)")
    w.add_argument("--max-frames", type=int, default=400, help="cap on frames used, evenly spaced (default 400)")
    w.add_argument("--min-confidence", type=int, default=2, help="minimum confidence level 0-2 (default 2)")
    w.add_argument("--voxel-size", type=float, default=0.02, help="voxel size in metres (default 0.02)")
    w.add_argument("--max-wall-tilt", type=float, default=5.0, help="max deviation from vertical in degrees (default 5)")
    w.add_argument("--source-size", type=_size_arg, help="override the image size the intrinsics refer to")
    args = parser.parse_args(argv)

    if args.command == "validate-lidar":
        result = validate_lidar_capture(args.path)
        print(format_report(result))
        return 1 if result.status is Status.INVALID else 0

    if args.command == "analyze-walls":
        from spatialforge.lidar.walls import WallOptions  # walls_run imports Open3D (via the drift stage)
        from spatialforge.lidar.walls_run import run_wall_analysis

        options = ReconstructionOptions(
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            voxel_size=args.voxel_size,
            source_size=args.source_size,
        )
        try:
            report, _ = run_wall_analysis(args.path, args.output_dir, options, WallOptions(max_tilt_deg=args.max_wall_tilt))
        except ReconstructionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_walls_report(report, args.output_dir))
        return 0

    if args.command == "analyze-horizontal-planes":
        from spatialforge.lidar.planes import PlaneOptions  # run_plane_analysis imports Open3D (via drift)
        from spatialforge.lidar.planes_run import run_plane_analysis

        options = ReconstructionOptions(
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            voxel_size=args.voxel_size,
            source_size=args.source_size,
        )
        plane_options = PlaneOptions(
            max_tilt_deg=args.max_plane_tilt,
            min_ceiling_height_m=args.min_ceiling_height,
            max_ceiling_height_m=args.max_ceiling_height,
        )
        try:
            report, _ = run_plane_analysis(args.path, args.output_dir, options, plane_options)
        except ReconstructionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_planes_report(report, args.output_dir))
        return 0

    if args.command == "correct-lidar-drift":
        from spatialforge.lidar.drift import DriftOptions  # imports Open3D, so only loaded here
        from spatialforge.lidar.drift_run import run_drift_ablation

        options = ReconstructionOptions(
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            voxel_size=args.voxel_size,
            source_size=args.source_size,
        )
        drift_options = DriftOptions(
            reg_voxel=args.reg_voxel,
            normal_radius=3 * args.reg_voxel,
            enable_loop_closure=not args.no_loop_closure,
        )
        try:
            report = run_drift_ablation(args.path, args.output_dir, options, drift_options)
        except ReconstructionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_drift_report(report, args.output_dir))
        return 0

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
