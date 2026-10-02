"""Drift-correction ablation: baseline poses vs corrected poses, with metrics and artifacts."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from spatialforge.lidar import drift, geometry
from spatialforge.lidar.reconstruction import (
    CaptureContext,
    ReconstructionError,
    ReconstructionOptions,
    ReconstructionResult,
    build_world_cloud,
    load_frame_points,
    pose_rotation,
    prepare_capture,
    write_cloud_ply,
)
from spatialforge.lidar.visualize import render_topdown, shared_bounds

METHOD = (
    "odometry initial guess; point-to-plane ICP between consecutive sampled frames; conservative "
    "loop closures; Open3D pose-graph optimisation with odometry anchors; roll/pitch kept from the "
    "supplied poses; automatic fallback to the original poses when metrics do not support the correction"
)

REASON_CATEGORY = {
    "fitness": "low ICP fitness",
    "inlier": "high inlier RMSE",
    "correction": "correction larger than allowed",
    "too": "too few points",
}


def _round(value, digits=5):
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _edge_stats(edges: list[drift.EdgeRecord]) -> dict:
    accepted = [e for e in edges if e.accepted]
    reasons = Counter(REASON_CATEGORY.get(e.reason.split(" ")[0], e.reason) for e in edges if not e.accepted)
    out = {
        "attempted": len(edges),
        "accepted": len(accepted),
        "rejected": len(edges) - len(accepted),
        "rejection_reasons": dict(sorted(reasons.items())),
    }
    if accepted:
        rmse = [e.inlier_rmse for e in accepted]
        fit = [e.fitness for e in accepted]
        out.update(
            accepted_mean_fitness=float(np.mean(fit)),
            accepted_median_fitness=float(np.median(fit)),
            accepted_mean_inlier_rmse_m=float(np.mean(rmse)),
            accepted_median_inlier_rmse_m=float(np.median(rmse)),
            accepted_median_translation_correction_m=float(np.median([e.translation_correction_m for e in accepted])),
            accepted_median_rotation_correction_deg=float(np.median([e.rotation_correction_deg for e in accepted])),
        )
    return out


def _write_trajectory(path: Path, frames: list[int], poses: list[np.ndarray]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x", "y", "z"] + [f"r{i}{j}" for i in range(3) for j in range(3)])
        for fid, T in zip(frames, poses):
            w.writerow([fid, *(f"{v:.6f}" for v in T[:3, 3]), *(f"{v:.8f}" for v in T[:3, :3].ravel())])


@dataclass
class DriftComputation:
    """Everything the drift stage produces, before any files are written."""

    ctx: CaptureContext
    base: ReconstructionResult
    frames: list[int]
    original: list[np.ndarray]
    correction: drift.CorrectionResult
    before_cloud: np.ndarray
    after_cloud: np.ndarray
    before: dict
    after: dict
    accepted: bool
    fallback_reason: str | None
    changes: dict

    @property
    def pose_source(self) -> str:
        return "corrected" if self.accepted else "original"

    @property
    def production_cloud(self) -> np.ndarray:
        return self.after_cloud if self.accepted else self.before_cloud

    @property
    def production_poses(self) -> list[np.ndarray]:
        return self.correction.corrected_poses if self.accepted else self.original


def compute_drift_ablation(
    capture: str | Path,
    output_path: Path,
    recon_opts: ReconstructionOptions,
    dopts: drift.DriftOptions,
    rules: drift.AcceptanceRules,
) -> DriftComputation:
    """Ticket 3 logic: baseline vs corrected poses on the same frames, accepted or rejected by the rules."""
    ctx, base = prepare_capture(capture, output_path, recon_opts)
    frames = ctx.frames
    if len(frames) < 3:
        raise ReconstructionError("need at least 3 sampled frames for drift correction; lower --frame-step")

    # One pass over the frames: registration cloud + metric points. Raw clouds are not kept.
    reg_clouds, metric_points = [], []
    for fid in frames:
        cam = load_frame_points(ctx, fid)
        reg_clouds.append(drift.make_registration_cloud(cam, dopts))
        metric_points.append(geometry.voxel_downsample(cam.astype(np.float32), dopts.metric_voxel))
    original = [drift.pose_matrix(pose_rotation(ctx, f), ctx.poses[f]["t"]) for f in frames]

    correction = drift.estimate_corrected_poses(frames, original, reg_clouds, dopts)
    del reg_clouds

    corrected_by_frame = dict(zip(frames, correction.corrected_poses))
    before_cloud = build_world_cloud(ctx, base)
    after_cloud = build_world_cloud(ctx, None, corrected_by_frame)
    before = drift.compute_metrics(metric_points, original, before_cloud)
    after = drift.compute_metrics(metric_points, correction.corrected_poses, after_cloud)
    accepted, fallback_reason, changes = drift.judge_correction(before, after, correction, rules)
    return DriftComputation(
        ctx, base, frames, original, correction, before_cloud, after_cloud, before, after,
        accepted, fallback_reason, changes,
    )


def run_drift_ablation(
    capture: str | Path,
    output_dir: str | Path,
    recon_opts: ReconstructionOptions,
    drift_opts: drift.DriftOptions | None = None,
    rules: drift.AcceptanceRules | None = None,
) -> dict:
    """Run baseline and corrected reconstructions on the same frames; write artifacts; return the report."""
    dopts = drift_opts or drift.DriftOptions()
    rules = rules or drift.AcceptanceRules()
    out = Path(output_dir)
    started = time.perf_counter()

    comp = compute_drift_ablation(capture, out / "before.ply", recon_opts, dopts, rules)
    ctx, base, frames, original, correction = comp.ctx, comp.base, comp.frames, comp.original, comp.correction
    before_cloud, after_cloud, before, after = comp.before_cloud, comp.after_cloud, comp.before, comp.after
    accepted, fallback_reason, changes = comp.accepted, comp.fallback_reason, comp.changes

    # Artifacts. after.ply is always the corrected candidate, so the ablation is inspectable even
    # when it is rejected; `production_poses` in the report says which poses should actually be used.
    write_cloud_ply(out / "before.ply", before_cloud)
    write_cloud_ply(out / "after.ply", after_cloud)
    _write_trajectory(out / "trajectory_before.csv", frames, original)
    _write_trajectory(out / "trajectory_after.csv", frames, correction.corrected_poses)
    bounds = shared_bounds([before_cloud, after_cloud])
    render_topdown(before_cloud, bounds, "Drift correction OFF (supplied poses)", out / "before_topdown.png",
                   np.array([T[:3, 3] for T in original]))
    after_title = "Drift correction ON" + ("" if accepted else " (candidate, REJECTED: original poses retained)")
    render_topdown(after_cloud, bounds, after_title, out / "after_topdown.png",
                   np.array([T[:3, 3] for T in correction.corrected_poses]))

    seq = [e for e in correction.edges if e.kind == "sequential"]
    loops = [e for e in correction.edges if e.kind == "loop"]
    loop_stats = _edge_stats(loops)
    report = {
        "capture": ctx.root.parent.name,
        "capture_path": str(ctx.root),
        "frames_available": base.frames_available,
        "frames_used": len(frames),
        "method": METHOD,
        "parameters": {
            "frame_step": recon_opts.frame_step,
            "max_frames": recon_opts.max_frames,
            "min_confidence": recon_opts.min_confidence,
            "output_voxel_m": recon_opts.voxel_size,
            "drift_options": asdict(dopts),
            "acceptance_rules": asdict(rules),
        },
        "sequential_edges": _edge_stats(seq),
        "loop_closures": {
            "candidates": correction.loop_candidates,
            "accepted": loop_stats["accepted"],
            "rejected": loop_stats["rejected"],
            "rejection_reasons": loop_stats["rejection_reasons"],
            "edges": [e.to_dict() for e in loops],
        },
        "constraint_residual_before_after": {
            kind: {
                "before_mean_translation_m_and_rotation_deg": correction.residual_before[kind],
                "after_mean_translation_m_and_rotation_deg": correction.residual_after[kind],
            }
            for kind in correction.residual_before
        },
        "before": before,
        "after": after,
        "relative_improvement": changes,
        "pose_correction": {
            "mean_translation_m": correction.mean_translation_m,
            "max_translation_m": correction.max_translation_m,
            "mean_rotation_deg": correction.mean_rotation_deg,
            "max_rotation_deg": correction.max_rotation_deg,
            "max_roll_pitch_change_deg_before_gravity_constraint": correction.max_raw_tilt_change_deg,
            "max_roll_pitch_change_deg": correction.max_tilt_change_deg,
        },
        "accepted": accepted,
        "fallback_reason": fallback_reason,
        "production_poses": "corrected" if accepted else "original",
        "runtime_s": time.perf_counter() - started,
    }
    report = _round(report)
    with open(out / "drift_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report
