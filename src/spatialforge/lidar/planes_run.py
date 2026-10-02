"""Horizontal-plane analysis of a capture using the production poses from the drift stage."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from spatialforge.lidar import drift
from spatialforge.lidar.drift_run import compute_drift_ablation
from spatialforge.lidar.planes import PlaneAnalysis, PlaneOptions, analyze_horizontal_planes
from spatialforge.lidar.reconstruction import ReconstructionOptions, write_cloud_ply
from spatialforge.lidar.visualize import render_side_projection, render_vertical_profile


def run_plane_analysis(
    capture: str | Path,
    output_dir: str | Path,
    recon_opts: ReconstructionOptions,
    plane_opts: PlaneOptions | None = None,
    drift_opts: drift.DriftOptions | None = None,
    rules: drift.AcceptanceRules | None = None,
) -> tuple[dict, PlaneAnalysis]:
    """Detect floor/ceiling on the production cloud (corrected poses only if Ticket 3 accepted them)."""
    started = time.perf_counter()
    out = Path(output_dir)
    comp = compute_drift_ablation(
        capture, out / "production.ply", recon_opts,
        drift_opts or drift.DriftOptions(), rules or drift.AcceptanceRules(),
    )
    cloud = comp.production_cloud
    camera_positions = np.array([T[:3, 3] for T in comp.production_poses])
    analysis = analyze_horizontal_planes(cloud, camera_positions, plane_opts)

    out.mkdir(parents=True, exist_ok=True)
    if analysis.floor.observed:
        write_cloud_ply(out / "floor_inliers.ply", analysis.floor.inliers.astype(np.float32))
    if analysis.ceiling_levels:
        merged = np.vstack([lvl.inliers for lvl in analysis.ceiling_levels]).astype(np.float32)
        write_cloud_ply(out / "ceiling_inliers.ply", merged)  # all accepted ceiling levels
    render_vertical_profile(analysis, out / "vertical_profile.png")
    render_side_projection(cloud, analysis, out / "side_projection.png")

    report = {
        "capture": comp.ctx.root.parent.name,
        "pose_source": comp.pose_source,
        "drift_fallback_reason": comp.fallback_reason,
        "frames_used": len(comp.frames),
        "frames_available": comp.base.frames_available,
        "runtime_s": time.perf_counter() - started,
        **analysis.to_dict(),
    }
    report["runtime_s"] = round(report["runtime_s"], 2)
    with open(out / "horizontal_planes.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report, analysis
