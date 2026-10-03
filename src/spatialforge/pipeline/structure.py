"""Shared structural backend: floor/ceiling -> walls -> rooms -> openings, from any MetricScene.

This is the code that used to live inside the LiDAR orchestration. The algorithms are unchanged; they now read a
MetricScene (gravity-aligned metric points, +Y up) instead of LiDAR capture files. A tier may pass its own option
objects (for example looser tolerances for a noisier cloud); the defaults are the LiDAR ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from spatialforge.pipeline.lidar_adapter import StageOutputs, StageRecord
from spatialforge.pipeline.scene import MetricScene
from spatialforge.pipeline.stages import StageTimer


@dataclass
class StructureConfig:
    plane_opts: object | None = None  # lidar.planes.PlaneOptions
    wall_opts: object | None = None  # lidar.walls.WallOptions
    room_opts: object | None = None  # lidar.rooms.RoomOptions
    opening_opts: object | None = None  # lidar.openings.OpeningOptions


def run_structure_stages(scene: MetricScene, out: StageOutputs, config: StructureConfig | None = None) -> StageOutputs:
    """Fill `out` (floor, ceilings, walls, topology, openings) from the scene. Failures become warnings, not exceptions."""
    from spatialforge.lidar.openings import OpeningOptions, detect_openings
    from spatialforge.lidar.planes import analyze_horizontal_planes
    from spatialforge.lidar.rooms import WallInput, build_topology
    from spatialforge.lidar.rooms_run import ceiling_levels_from_planes
    from spatialforge.lidar.walls import extract_walls

    config = config or StructureConfig()
    stages, stage_warnings = out.stages, out.stage_warnings
    cloud, cameras = scene.points_xyz_m, scene.camera_positions

    # floor + ceilings
    planes = None
    try:
        with StageTimer(stages, "floor_and_ceiling_planes") as rec:
            planes = analyze_horizontal_planes(cloud, cameras, config.plane_opts)
            if not planes.floor.observed:
                rec.warnings.append(f"Floor plane not found ({planes.floor.reject_reason}); walls, rooms and openings "
                                    "cannot be derived.")
            rec.warnings += [w for w in planes.warnings]
    except Exception:
        pass  # recorded by the timer; the pipeline continues without planes
    if planes is not None and planes.floor.observed:
        out.floor_y_m = float(planes.floor.height_m)
        out.ceiling_levels = [{"height_m": float(h["value_m"]), "interval_m": [float(x) for x in h["confidence_interval_m"]],
                               "confidence": float(h["confidence"])}
                              for h in planes.ceiling_level_heights]
    stage_warnings += stages[-1].warnings
    if stages[-1].status == "failed":
        stage_warnings.append(f"Floor/ceiling stage failed: {stages[-1].error}")

    wall_inputs, wall_analysis, topo = [], None, None
    if out.floor_y_m is None:
        for name in ("structural_walls", "room_topology", "openings"):
            stages.append(StageRecord(name, "skipped", warnings=["skipped: no floor plane"]))
        return out

    # walls
    try:
        with StageTimer(stages, "structural_walls") as rec:
            wall_analysis = extract_walls(cloud, planes.floor.fit, [lvl.fit for lvl in planes.ceiling_levels], config.wall_opts)
            rec.warnings += list(wall_analysis.warnings)
            rec.details = {"walls": len(wall_analysis.walls), "rejected_candidates": len(wall_analysis.rejected)}
        out.wall_dicts = wall_analysis.to_dict()["walls"]
        wall_inputs = [WallInput.from_dict(w) for w in out.wall_dicts]
        out.debug["wall_analysis"] = wall_analysis
    except Exception:
        stage_warnings.append(f"Wall stage failed: {stages[-1].error}")
    stage_warnings += [w for w in stages[-1].warnings]

    # rooms
    if wall_inputs:
        try:
            with StageTimer(stages, "room_topology") as rec:
                topo = build_topology(wall_inputs, config.room_opts, ceiling_levels_from_planes(planes))
                rec.warnings += list(topo.warnings)
                rec.details = {"rooms": len(topo.rooms), "candidate_faces": topo.candidate_faces,
                               "rejected_faces": len(topo.rejected_faces)}
            out.topology = topo
        except Exception:
            stage_warnings.append(f"Room stage failed: {stages[-1].error}")
    else:
        stages.append(StageRecord("room_topology", "skipped", warnings=["skipped: no walls"]))

    # openings
    if wall_inputs:
        try:
            with StageTimer(stages, "openings") as rec:
                oopts = config.opening_opts or OpeningOptions()
                lowest = min((float(h["value_m"]) for h in planes.ceiling_level_heights), default=None)
                subsets = scene.frame_subsets()
                res = detect_openings(cloud, planes.floor.fit, wall_inputs, topo, subsets, oopts, lowest)
                rec.warnings += list(res.warnings)
                rec.details = {"candidates": len(res.candidates), "accepted": len(res.accepted),
                               "low_confidence": len(res.low_confidence), "rejected": len(res.rejected),
                               "frame_subsets": len(subsets)}
            out.openings = res
        except Exception:
            stage_warnings.append(f"Opening stage failed: {stages[-1].error}")
    else:
        stages.append(StageRecord("openings", "skipped", warnings=["skipped: no walls"]))
    out.debug["wall_inputs"] = wall_inputs
    return out
