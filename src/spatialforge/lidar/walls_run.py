"""Structural wall analysis of a capture, on the production cloud and Ticket 4's floor/ceiling planes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from spatialforge.lidar import drift
from spatialforge.lidar.drift_run import compute_drift_ablation
from spatialforge.lidar.planes import PlaneOptions, analyze_horizontal_planes
from spatialforge.lidar.reconstruction import ReconstructionError, ReconstructionOptions, write_cloud_ply
from spatialforge.lidar.visualize import render_walls_topdown, shared_bounds
from spatialforge.lidar.walls import WallAnalysis, WallOptions, extract_walls


def compute_walls(
    capture: str | Path,
    output_dir: str | Path,
    recon_opts: ReconstructionOptions,
    wall_opts: WallOptions | None = None,
    plane_opts: PlaneOptions | None = None,
):
    """Production poses (Ticket 3) -> floor/ceiling planes (Ticket 4) -> structural walls (Ticket 5).

    Returns (drift computation, plane analysis, wall analysis). Shared by the wall and room commands.
    """
    comp = compute_drift_ablation(
        capture, Path(output_dir) / "production.ply", recon_opts, drift.DriftOptions(), drift.AcceptanceRules()
    )
    cloud = comp.production_cloud
    cameras = np.array([T[:3, 3] for T in comp.production_poses])
    planes = analyze_horizontal_planes(cloud, cameras, plane_opts)
    if not planes.floor.observed:
        raise ReconstructionError(f"floor plane not found ({planes.floor.reject_reason}); cannot analyse walls")
    walls = extract_walls(cloud, planes.floor.fit, [lvl.fit for lvl in planes.ceiling_levels], wall_opts)
    return comp, planes, walls


def run_wall_analysis(
    capture: str | Path,
    output_dir: str | Path,
    recon_opts: ReconstructionOptions,
    wall_opts: WallOptions | None = None,
    plane_opts: PlaneOptions | None = None,
) -> tuple[dict, WallAnalysis]:
    """Production poses (Ticket 3) -> floor/ceiling planes (Ticket 4) -> structural walls."""
    started = time.perf_counter()
    out = Path(output_dir)
    comp, planes, walls = compute_walls(capture, out, recon_opts, wall_opts, plane_opts)
    cloud = comp.production_cloud

    out.mkdir(parents=True, exist_ok=True)
    if walls.walls:
        write_cloud_ply(out / "wall_inliers.ply", np.vstack([w.inliers for w in walls.walls]).astype(np.float32))
    render_walls_topdown(cloud, walls, shared_bounds([cloud]), out / "walls_topdown.png")

    full = walls.to_dict()
    candidates = {"rejected": full.pop("rejected"), "options": full["options"]}
    report = {
        "capture": comp.ctx.root.parent.name,
        "pose_source": comp.pose_source,
        "drift_fallback_reason": comp.fallback_reason,
        "frames_used": len(comp.frames),
        "frames_available": comp.base.frames_available,
        "floor_y_m": round(float(planes.floor.height_m), 4),
        "ceiling_levels_m": [round(float(h["value_m"]), 3) for h in planes.ceiling_level_heights],
        **full,
        "runtime_s": round(time.perf_counter() - started, 2),
    }
    with open(out / "walls.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(out / "wall_candidates.json", "w", encoding="utf-8") as fh:
        json.dump(candidates, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return report, walls
