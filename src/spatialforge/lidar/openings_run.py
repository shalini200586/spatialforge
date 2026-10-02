"""Opening analysis of a capture: walls (T5) + rooms (T6) + floor/ceilings (T4) + 3D points (T2)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from spatialforge.lidar.openings import (
    OpeningOptions,
    OpeningResult,
    candidates_to_dict,
    detect_openings,
    openings_to_dict,
)
from spatialforge.lidar.planes import PlaneOptions
from spatialforge.lidar.reconstruction import ReconstructionOptions, build_world_cloud
from spatialforge.lidar.rooms import RoomOptions, WallInput, build_topology
from spatialforge.lidar.rooms_run import ceiling_levels_from_planes
from spatialforge.lidar.visualize import render_opening_profile, render_openings_topdown, shared_bounds
from spatialforge.lidar.walls import WallOptions
from spatialforge.lidar.walls_run import compute_walls


def frame_subset_clouds(comp, opts: OpeningOptions) -> list[np.ndarray]:
    """Deterministic frame subsets: consecutive blocks of frames are dealt out round-robin, so every subset sees
    the building at several different times, and a different set of views than the others."""
    frames = comp.frames
    poses = dict(zip(frames, comp.production_poses))
    clouds = []
    for k in range(opts.subset_count):
        sub = [f for i, f in enumerate(frames) if (i // opts.subset_block_frames) % opts.subset_count == k]
        if not sub:
            continue
        ctx = dataclasses.replace(comp.ctx, frames=sub)
        clouds.append(build_world_cloud(ctx, None, {f: poses[f] for f in sub}))
    return clouds


def run_opening_analysis(
    capture: str | Path,
    output_dir: str | Path,
    recon_opts: ReconstructionOptions,
    opening_opts: OpeningOptions | None = None,
    room_opts: RoomOptions | None = None,
    wall_opts: WallOptions | None = None,
    plane_opts: PlaneOptions | None = None,
) -> tuple[dict, OpeningResult]:
    started = time.perf_counter()
    out = Path(output_dir)
    opts = opening_opts or OpeningOptions()
    comp, planes, walls = compute_walls(capture, out, recon_opts, wall_opts, plane_opts)
    wall_inputs = [WallInput.from_dict(w) for w in walls.to_dict()["walls"]]
    topo = build_topology(wall_inputs, room_opts, ceiling_levels_from_planes(planes))
    subsets = frame_subset_clouds(comp, opts)
    lowest_ceiling = min((float(h["value_m"]) for h in planes.ceiling_level_heights), default=None)
    cloud = comp.production_cloud
    res = detect_openings(cloud, planes.floor.fit, wall_inputs, topo, subsets, opts, lowest_ceiling)

    out.mkdir(parents=True, exist_ok=True)
    render_openings_topdown(cloud, topo, wall_inputs, res, shared_bounds([cloud]), out / "openings_topdown.png")
    by_cand = {o.candidate.id: o for o in res.openings}
    outcome = {r["candidate_id"]: f"rejected ({r['stage']}): {r['reason']}" for r in res.rejected}
    for c in res.candidates:
        grid = res.grids.get(c.id)
        if grid is None:
            continue
        o = by_cand.get(c.id)
        label = f"{o.status}: {o.id}" if o else outcome.get(c.id, "rejected")
        render_opening_profile(grid, c, o, label[:150], out / "opening_profiles" / f"{o.id if o else c.id}.png")

    report = {
        "capture": comp.ctx.root.parent.name,
        "pose_source": comp.pose_source,
        "frames_used": len(comp.frames),
        "frame_subsets": len(subsets),
        "rooms_available": len(topo.rooms),
        "walls_considered": len(wall_inputs),
        **openings_to_dict(res),
        "runtime_s": round(time.perf_counter() - started, 2),
    }
    with open(out / "openings.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(out / "opening_candidates.json", "w", encoding="utf-8") as fh:
        json.dump(candidates_to_dict(res), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report, res
