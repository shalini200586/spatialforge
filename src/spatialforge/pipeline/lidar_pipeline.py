"""One-command LiDAR pipeline: capture -> property.json + plan.png + run_report.json (+ diagnostics/).

Orchestration only: it calls the existing validated stages (capture validation, metric reconstruction with the
drift fallback decision, floor/ceiling planes, walls, room topology, openings) and packages their results. It does
not change any algorithm.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spatialforge.pipeline.lidar_adapter import StageOutputs, StageRecord, assemble_property
from spatialforge.pipeline.models import ModelError
from spatialforge.pipeline.render import render_plan
from spatialforge.pipeline.serialization import property_to_dict, write_json


class PipelineFailure(Exception):
    """The run cannot produce a property (invalid capture, no metric reconstruction)."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message


@dataclass
class PipelineOptions:
    frame_step: int = 1
    max_frames: int = 400
    min_confidence: int = 2
    voxel_size: float = 0.02


@dataclass
class PipelineResult:
    status: str  # success | partial | failure
    exit_code: int
    output_dir: Path
    property: dict | None
    run_report: dict
    files: dict = field(default_factory=dict)


# ---------- the heavy stages ----------


class _Timer:
    def __init__(self, out_stages: list[StageRecord], name: str):
        self.rec = StageRecord(name, "ok")
        out_stages.append(self.rec)

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self.rec

    def __exit__(self, exc_type, exc, tb):
        self.rec.seconds = time.perf_counter() - self.t0
        if exc is not None:
            self.rec.status, self.rec.error = "failed", f"{type(exc).__name__}: {exc}"
        elif self.rec.warnings and self.rec.status == "ok":
            self.rec.status = "warning"
        return False  # never swallow here; callers decide whether a failure is fatal


def run_lidar_stages(capture: str | Path, opts: PipelineOptions) -> StageOutputs:
    """Run the real LiDAR stages. Raises PipelineFailure only for essential stages."""
    # imports are local: Open3D (used by the drift stage) is slow to import and unneeded for validation errors
    from spatialforge.lidar.drift import AcceptanceRules, DriftOptions
    from spatialforge.lidar.drift_run import compute_drift_ablation
    from spatialforge.lidar.geometry import DEPTH_SCALE_M_PER_UNIT
    from spatialforge.lidar.openings import OpeningOptions, detect_openings
    from spatialforge.lidar.openings_run import frame_subset_clouds
    from spatialforge.lidar.planes import analyze_horizontal_planes
    from spatialforge.lidar.reconstruction import ReconstructionError, ReconstructionOptions
    from spatialforge.lidar.rooms import WallInput, build_topology
    from spatialforge.lidar.rooms_run import ceiling_levels_from_planes
    from spatialforge.lidar.validator import resolve_capture_root, validate_lidar_capture
    from spatialforge.lidar.walls import extract_walls

    capture = Path(capture)
    stages: list[StageRecord] = []
    stage_warnings: list[str] = []

    # 1. capture validation (essential)
    try:
        with _Timer(stages, "capture_validation") as rec:
            v = validate_lidar_capture(capture)
            rec.warnings = list(v.warnings)
            rec.details = {"status": v.status.value, "depth_frames": v.depth_frame_count, "pose_rows": v.pose_frame_count}
            if v.errors:
                raise PipelineFailure("capture_validation", "; ".join(v.errors))
    except PipelineFailure:
        raise
    except Exception as exc:
        raise PipelineFailure("capture_validation", f"{type(exc).__name__}: {exc}") from exc
    root = resolve_capture_root(capture)

    # 2. metric reconstruction + drift decision (essential)
    recon = ReconstructionOptions(frame_step=opts.frame_step, max_frames=opts.max_frames,
                                  min_confidence=opts.min_confidence, voxel_size=opts.voxel_size)
    try:
        with _Timer(stages, "metric_reconstruction_and_pose_check") as rec:
            comp = compute_drift_ablation(capture, root / "unused.ply", recon, DriftOptions(), AcceptanceRules())
            rec.details = {"frames_used": len(comp.frames), "pose_source": comp.pose_source,
                           "points": int(len(comp.production_cloud))}
            if not comp.accepted:
                rec.details["pose_refinement_rejected"] = comp.fallback_reason
    except ReconstructionError as exc:
        raise PipelineFailure("metric_reconstruction_and_pose_check", str(exc)) from exc
    except Exception as exc:
        raise PipelineFailure("metric_reconstruction_and_pose_check", f"{type(exc).__name__}: {exc}") from exc

    cloud = comp.production_cloud
    cameras = np.array([T[:3, 3] for T in comp.production_poses])
    meta = v.metadata
    device = {
        "model": None,  # not recorded in the capture files
        "depth_resolution": meta.get("depth_resolution"), "depth_dtype": meta.get("depth_dtype"),
        "rgb_resolution": f"{comp.base.source_size[0]}x{comp.base.source_size[1]}",
        "camera_matrix": meta.get("intrinsics"), "imu_samples": meta.get("imu_sample_count"),
    }
    out = StageOutputs(
        capture_path=str(capture), capture_name=capture.name, device=device,
        validation_warnings=list(v.warnings),
        frames={"used": len(comp.frames), "available": comp.base.frames_available, "frame_step": opts.frame_step,
                "max_frames": opts.max_frames},
        pose_source=comp.pose_source,
        drift={"correction_accepted": bool(comp.accepted), "fallback_reason": comp.fallback_reason,
               "mean_translation_m": comp.correction.mean_translation_m,
               "max_translation_m": comp.correction.max_translation_m,
               "max_rotation_deg": comp.correction.max_rotation_deg},
        floor_y_m=None, ceiling_levels=[], wall_dicts=[], topology=None, openings=None,
        stage_warnings=stage_warnings, stages=stages,
        parameters={"min_confidence": opts.min_confidence, "voxel_size_m": opts.voxel_size},
        debug={"cloud": cloud},
    )

    # 3. floor + ceilings
    planes = None
    try:
        with _Timer(stages, "floor_and_ceiling_planes") as rec:
            planes = analyze_horizontal_planes(cloud, cameras)
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

    # 4. walls
    try:
        with _Timer(stages, "structural_walls") as rec:
            wall_analysis = extract_walls(cloud, planes.floor.fit, [lvl.fit for lvl in planes.ceiling_levels])
            rec.warnings += list(wall_analysis.warnings)
            rec.details = {"walls": len(wall_analysis.walls), "rejected_candidates": len(wall_analysis.rejected)}
        out.wall_dicts = wall_analysis.to_dict()["walls"]
        wall_inputs = [WallInput.from_dict(w) for w in out.wall_dicts]
        out.debug["wall_analysis"] = wall_analysis
    except Exception:
        stage_warnings.append(f"Wall stage failed: {stages[-1].error}")
    stage_warnings += [w for w in stages[-1].warnings]

    # 5. rooms
    if wall_inputs:
        try:
            with _Timer(stages, "room_topology") as rec:
                topo = build_topology(wall_inputs, None, ceiling_levels_from_planes(planes))
                rec.warnings += list(topo.warnings)
                rec.details = {"rooms": len(topo.rooms), "candidate_faces": topo.candidate_faces,
                               "rejected_faces": len(topo.rejected_faces)}
            out.topology = topo
        except Exception:
            stage_warnings.append(f"Room stage failed: {stages[-1].error}")
    else:
        stages.append(StageRecord("room_topology", "skipped", warnings=["skipped: no walls"]))

    # 6. openings
    if wall_inputs:
        try:
            with _Timer(stages, "openings") as rec:
                oopts = OpeningOptions()
                lowest = min((float(h["value_m"]) for h in planes.ceiling_level_heights), default=None)
                subsets = frame_subset_clouds(comp, oopts)
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
    out.parameters["depth_scale_m_per_unit"] = DEPTH_SCALE_M_PER_UNIT
    return out


# ---------- packaging ----------


def _write_diagnostics(out_dir: Path, so: StageOutputs) -> list[str]:
    """Small debug artifacts (never the point clouds). Returns problems as warnings; never raises."""
    problems = []
    diag = out_dir / "diagnostics"
    try:
        from spatialforge.lidar.openings import candidates_to_dict
        from spatialforge.lidar.visualize import (
            render_openings_topdown, render_rooms_topdown, render_walls_topdown, shared_bounds,
        )

        diag.mkdir(parents=True, exist_ok=True)
        cloud = so.debug.get("cloud")
        summary = {"frames": so.frames, "pose_refinement": so.drift, "stages": {s.name: s.details for s in so.stages}}
        wa, wi, topo, ores = (so.debug.get(k) for k in ("wall_analysis", "wall_inputs", "topology", "openings"))
        topo = so.topology
        ores = so.openings
        if cloud is not None and len(cloud) and wa is not None:
            bounds = shared_bounds([cloud])
            render_walls_topdown(cloud, wa, bounds, diag / "walls_topdown.png")
            summary["rejected_wall_candidates"] = len(wa.rejected)
            if topo is not None:
                render_rooms_topdown(cloud, topo, wi, bounds, diag / "rooms_topdown.png")
            if ores is not None:
                render_openings_topdown(cloud, topo, wi, ores, bounds, diag / "openings_topdown.png")
                write_json(diag / "opening_candidates.json", candidates_to_dict(ores))
        write_json(diag / "stage_summaries.json", summary)
    except Exception as exc:  # diagnostics are optional
        problems.append(f"Diagnostics could not be fully written: {type(exc).__name__}: {exc}")
    return problems


def _counts(prop_dict: dict, so: StageOutputs) -> dict:
    walls = prop_dict["walls"]
    return {
        "rooms": len(prop_dict["rooms"]), "walls": len(walls),
        "walls_strong_or_moderate": sum(1 for w in walls if w["evidence_quality"] in ("strong", "moderate")),
        "openings": len(prop_dict["openings"]), "unverified_openings": len(prop_dict["unverified_openings"]),
        "ceiling_levels": len(so.ceiling_levels),
        "rooms_with_ceiling": sum(1 for r in prop_dict["rooms"] if r["ceiling"]["observed"]),
    }


def process_capture(capture: str | Path, output: str | Path, options: PipelineOptions | None = None,
                    stages_fn=None, tier: str = "lidar") -> PipelineResult:
    """Run the pipeline and write property.json, plan.png, run_report.json and diagnostics/.

    `stages_fn(capture, options) -> StageOutputs` is injectable so tests can supply stage outputs.
    """
    options = options or PipelineOptions()
    stages_fn = stages_fn or run_lidar_stages
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    report: dict = {"tier": tier, "capture": str(capture), "output": str(out_dir),
                    "started_at": started.isoformat(timespec="seconds"), "errors": [], "warnings": []}

    def finish(status: str, code: int, prop_dict=None, so: StageOutputs | None = None, files=None) -> PipelineResult:
        ended = datetime.now(timezone.utc)
        report.update(status=status, ended_at=ended.isoformat(timespec="seconds"),
                      runtime_s=round(time.perf_counter() - t0, 2))
        if so is not None:
            report["stages"] = [{"name": s.name, "status": s.status, "seconds": round(s.seconds, 3),
                                 "warnings": s.warnings, "error": s.error, "details": s.details} for s in so.stages]
            report["production_pose_source"] = so.pose_source
            report["fallback_decisions"] = [{
                "stage": "pose_refinement",
                "decision": "kept the supplied (original) poses" if not so.drift.get("correction_accepted") else "used corrected poses",
                "reason": so.drift.get("fallback_reason") or "metrics supported the correction"}]
        if prop_dict is not None:
            report["counts"] = _counts(prop_dict, so)
            report["warnings"] = prop_dict["warnings"]
        report["outputs"] = {k: str(v) for k, v in (files or {}).items()}
        write_json(out_dir / "run_report.json", report)
        return PipelineResult(status, code, out_dir, prop_dict, report, files or {})

    try:
        so = stages_fn(capture, options)
    except PipelineFailure as exc:
        report["errors"].append({"stage": exc.stage, "message": exc.message})
        return finish("failure", 1)
    except Exception as exc:  # unexpected: still a clean failure with the reason recorded
        report["errors"].append({"stage": "pipeline", "message": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc(limit=4)})
        return finish("failure", 1)

    try:
        prop = assemble_property(so)
        t_render = time.perf_counter()
        render_info = render_plan(prop, out_dir / "plan.png", f"SpatialForge floor plan — {so.capture_name}")
        prop.timing["render_plan_s"] = round(time.perf_counter() - t_render, 3)
        prop_dict = property_to_dict(prop)
    except (ModelError, ValueError) as exc:
        report["errors"].append({"stage": "assembly", "message": f"{type(exc).__name__}: {exc}"})
        return finish("failure", 1, so=so)

    prop_dict["timing"]["total_s"] = round(time.perf_counter() - t0, 3)
    write_json(out_dir / "property.json", prop_dict)
    report["diagnostic_warnings"] = _write_diagnostics(out_dir, so)
    files = {"property.json": out_dir / "property.json", "plan.png": out_dir / "plan.png",
             "run_report.json": out_dir / "run_report.json"}
    report["plan"] = {"rooms_drawn": render_info.rooms_drawn, "walls_drawn": render_info.walls_drawn,
                      "openings_drawn": render_info.openings_drawn}
    failed = [s.name for s in so.stages if s.status == "failed"]
    status = "success" if (prop_dict["property"]["status"] == "complete" and not failed) else "partial"
    return finish(status, 0, prop_dict, so, files)
