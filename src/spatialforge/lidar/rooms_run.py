"""Room topology of a capture: Ticket 5 walls (and Ticket 4 ceilings) -> rooms, areas, ceilings."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from spatialforge.lidar.planes import PlaneOptions
from spatialforge.lidar.reconstruction import ReconstructionOptions
from spatialforge.lidar.rooms import (
    RoomOptions,
    TopologyResult,
    WallInput,
    build_topology,
    graph_to_dict,
    rooms_to_geojson,
    topology_to_dict,
)
from spatialforge.lidar.visualize import render_rooms_topdown, shared_bounds
from spatialforge.lidar.walls import WallOptions
from spatialforge.lidar.walls_run import compute_walls


def ceiling_levels_from_planes(planes) -> list[dict]:
    """Ticket 4 ceiling levels as {height, interval, X-Z inlier points} for spatial room association."""
    return [
        {"height_m": float(h["value_m"]), "interval_m": [float(v) for v in h["confidence_interval_m"]],
         "points_xz": lvl.inliers[:, [0, 2]]}
        for lvl, h in zip(planes.ceiling_levels, planes.ceiling_level_heights)
    ]


def run_room_analysis(
    capture: str | Path,
    output_dir: str | Path,
    recon_opts: ReconstructionOptions,
    room_opts: RoomOptions | None = None,
    wall_opts: WallOptions | None = None,
    plane_opts: PlaneOptions | None = None,
) -> tuple[dict, TopologyResult]:
    started = time.perf_counter()
    out = Path(output_dir)
    comp, planes, walls = compute_walls(capture, out, recon_opts, wall_opts, plane_opts)
    wall_inputs = [WallInput.from_dict(w) for w in walls.to_dict()["walls"]]
    topo = build_topology(wall_inputs, room_opts, ceiling_levels_from_planes(planes))

    out.mkdir(parents=True, exist_ok=True)
    cloud = comp.production_cloud
    render_rooms_topdown(cloud, topo, wall_inputs, shared_bounds([cloud]), out / "rooms_topdown.png")

    report = {
        "capture": comp.ctx.root.parent.name,
        "pose_source": comp.pose_source,
        "frames_used": len(comp.frames),
        "floor_y_m": round(float(planes.floor.height_m), 4),
        "ceiling_levels_m": [round(float(h["value_m"]), 3) for h in planes.ceiling_level_heights],
        **topology_to_dict(topo),
        "runtime_s": round(time.perf_counter() - started, 2),
    }
    with open(out / "rooms.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(out / "wall_graph.json", "w", encoding="utf-8") as fh:
        json.dump(graph_to_dict(topo), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(out / "room_polygons.geojson", "w", encoding="utf-8") as fh:
        json.dump(rooms_to_geojson(topo), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report, topo
