"""One-command LiDAR pipeline: capture -> property.json + plan.png + run_report.json (+ diagnostics/).

Orchestration only: it calls the existing validated stages (capture validation, metric reconstruction with the
drift fallback decision, floor/ceiling planes, walls, room topology, openings) and packages their results. It does
not change any algorithm.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spatialforge.pipeline import logs
from spatialforge.pipeline.lidar_adapter import StageOutputs, StageRecord, assemble_property
from spatialforge.pipeline.messages import ErrorRecord, classify_all, classify_error
from spatialforge.pipeline.models import ModelError
from spatialforge.pipeline.provenance import runtime_provenance, scrub
from spatialforge.pipeline.report import build_report, exit_code
from spatialforge.pipeline.stage_model import fold_stages, standard_name
from spatialforge.pipeline.render import render_plan
from spatialforge.pipeline.scene import MetricScene
from spatialforge.pipeline.serialization import property_to_dict, write_json
from spatialforge.pipeline.stages import PipelineFailure, StageTimer
from spatialforge.pipeline.structure import run_structure_stages


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


_Timer = StageTimer  # historical name
LOG = logging.getLogger("spatialforge.pipeline")


def run_lidar_stages(capture: str | Path, opts: PipelineOptions) -> StageOutputs:
    """Run the real LiDAR stages. Raises PipelineFailure only for essential stages."""
    # imports are local: Open3D (used by the drift stage) is slow to import and unneeded for validation errors
    from spatialforge.lidar.drift import AcceptanceRules, DriftOptions
    from spatialforge.lidar.drift_run import compute_drift_ablation
    from spatialforge.lidar.geometry import DEPTH_SCALE_M_PER_UNIT
    from spatialforge.lidar.openings import OpeningOptions
    from spatialforge.lidar.openings_run import frame_subset_clouds
    from spatialforge.lidar.reconstruction import ReconstructionError, ReconstructionOptions
    from spatialforge.lidar.validator import resolve_capture_root, validate_lidar_capture

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

    # 3-6. floor/ceiling, walls, rooms, openings: the shared structural backend, fed by a MetricScene
    oopts = OpeningOptions()
    scene = MetricScene(
        cloud, "lidar", geometry_quality="strong", scale_quality="sensor",
        camera_poses=np.asarray(comp.production_poses), frame_refs=[str(f) for f in comp.frames],
        subsets_fn=lambda: frame_subset_clouds(comp, oopts),
        metadata={"pose_source": comp.pose_source},
    )
    run_structure_stages(scene, out)
    out.parameters["depth_scale_m_per_unit"] = DEPTH_SCALE_M_PER_UNIT
    from spatialforge.damage.evidence import lidar_evidence_fn

    out.evidence_fn = lidar_evidence_fn(root, comp, out)
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
        "damage_regions": len(prop_dict["damage_regions"]), "concealed_damage_flags": len(prop_dict["concealed_damage_flags"]),
        "scope_line_items": len(prop_dict["scope_line_items"]),
    }


def _run_damage(prop, so: StageOutputs, out_dir: Path) -> bool:
    """Visible-damage analysis on the tier's RGB evidence. Never fatal: a failure is a warning and an empty damage layer.
    Returns True if the analysis ran."""
    logs.announce("damage_analysis")
    rec = StageRecord("damage_analysis", "ok")
    t0 = time.perf_counter()
    ran = False
    try:
        from spatialforge.damage.analyze import analyze_damage, apply_to_property

        images = so.evidence_fn()
        if not images:
            rec.status = "skipped"
            rec.warnings = ["no RGB evidence with usable geometry was available for damage analysis"]
        else:
            outcome = analyze_damage(prop, images, out_dir / "diagnostics" / "damage")
            apply_to_property(prop, outcome, len(images))
            rec.warnings = list(outcome.warnings)
            rec.details = {"images": len(images), "detections": len(outcome.regions),
                           "with_metric_extent": outcome.report["detections_with_metric_extent"],
                           "concealed_flags": len(outcome.flags), "scope_items": len(outcome.items)}
            rec.status = "warning" if rec.warnings else "ok"
            ran = True
    except Exception as exc:
        LOG.exception("damage analysis failed")
        rec.status, rec.error = "failed", f"{type(exc).__name__}: {exc}"
        prop.warnings.append(f"Damage analysis failed and was skipped ({type(exc).__name__}: {exc}); the damage layer is empty.")
    rec.seconds = time.perf_counter() - t0
    so.stages.append(rec)
    return ran


def _no_geometry_error(prop, so: StageOutputs) -> ErrorRecord:
    """Why there is nothing to report: the most specific reason the stages can give."""
    extra = so.debug.get("run_report_extra", {})
    rooms = extra.get("photos", {}).get("rooms") if isinstance(extra.get("photos"), dict) else None
    if rooms and all(r.get("status") == "unreconstructed" for r in rooms):
        return ErrorRecord("SFM_FAILED", "frontend", "No room folder could be reconstructed: structure-from-motion failed for every room.",
                           "Photograph each room with more overlap, include corners, and keep the originals.", False)
    if extra.get("metric_scale_available") is False:
        return ErrorRecord("METRIC_SCALE_UNAVAILABLE", "metric_scale",
                           "No metric scale could be established, so no dimensions can be reported (SfM alone is scale-ambiguous).",
                           "Check that the metric-depth weights are installed (python scripts/download_models.py) and that the footage "
                           "gives a trajectory the depth model agrees with.", False)
    return ErrorRecord("NO_GEOMETRY", "structure", "No walls or rooms could be recovered, so there is no trustworthy property geometry.",
                       "The capture may not show enough structure (no floor visible, too short, too fast).", False)


def _prune_diagnostics(diag: Path, full: bool) -> None:
    """Normal runs keep the output small: point clouds and keyframe images are only kept with --diagnostics."""
    if full or not diag.is_dir():
        return
    for d in ("keyframes", "keyframes_dense"):
        shutil.rmtree(diag / d, ignore_errors=True)
    for ply in diag.rglob("*.ply"):
        try:
            ply.unlink()
        except OSError:
            pass


def _list_files(out_dir: Path) -> list[str]:
    names = [n for n in ("property.json", "plan.png", "run_report.json") if (out_dir / n).exists()]
    if (out_dir / "diagnostics").is_dir():
        names.append("diagnostics/")
    return names


def process_capture(capture: str | Path, output: str | Path, options: PipelineOptions | None = None,
                    stages_fn=None, tier: str = "lidar", damage: bool = True, *, diagnostics: bool = False,
                    display_output: str | Path | None = None, tier_requested: str | None = None, detection: str | None = None,
                    debug: bool = False) -> PipelineResult:
    """Run one tier and write property.json, plan.png, run_report.json and diagnostics/ into `output`.

    `stages_fn(capture, options) -> StageOutputs` is injectable so tests can supply stage outputs. `display_output` is the
    path recorded in the report when `output` is a staging folder. Status: success | partial | failure (see pipeline.report).
    """
    options = options or PipelineOptions()
    stages_fn = stages_fn or run_lidar_stages
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    config = {"tier": tier, "damage": damage, "diagnostics": diagnostics, "debug": debug}

    def finish(status: str, raw_stages: list, errors: list[ErrorRecord], prop=None, prop_dict=None, so: StageOutputs | None = None,
               extras: dict | None = None, counts=None, plan=None, diag_warnings=None) -> PipelineResult:
        decisions = None
        if so is not None:
            decisions = so.decisions if so.decisions is not None else [{
                "stage": "pose_refinement",
                "decision": "kept the supplied (original) poses" if not so.drift.get("correction_accepted") else "used corrected poses",
                "reason": so.drift.get("fallback_reason") or "metrics supported the correction"}] if tier == "lidar" else []
        ex = dict(extras or {})
        if so is not None:
            ex.setdefault("production_pose_source", so.pose_source)
            ex.update(so.debug.get("run_report_extra", {}))
        if plan is not None:
            ex["plan"] = plan
        warns = None
        if prop is not None:
            warns = list(prop.warning_records) if prop.warning_records else [w.to_dict() for w in classify_all(prop.warnings)]
        report = build_report(
            tier=tier, tier_requested=tier_requested, detection=detection, status=status, capture=capture,
            display_output=display_output or out_dir, started=started.isoformat(timespec="seconds"),
            ended=datetime.now(timezone.utc).isoformat(timespec="seconds"), duration_s=time.perf_counter() - t0,
            raw_stages=raw_stages, property_warning_records=warns, errors=errors, counts=counts, decisions=decisions, config=config,
            files=_list_files(out_dir) + ([] if (out_dir / "run_report.json").exists() else ["run_report.json"]), extras=ex,
            diagnostic_warnings=diag_warnings)
        report = scrub(report)
        write_json(out_dir / "run_report.json", report)
        files = {n.rstrip("/"): out_dir / n.rstrip("/") for n in _list_files(out_dir)}
        return PipelineResult(status, exit_code(status), out_dir, prop_dict, report, files)

    if hasattr(options, "output_dir"):
        options.output_dir = out_dir  # a tier that writes its own diagnostics needs to know where
    try:
        so = stages_fn(capture, options)
    except PipelineFailure as exc:
        std = standard_name(exc.stage) or "frontend"
        err = classify_error(exc.message, std, exc.stage)
        if exc.code:
            err.code, err.hint = exc.code, exc.hint or err.hint
        _prune_diagnostics(out_dir / "diagnostics", diagnostics)
        return finish("failure", exc.stages or [], [err], extras=dict(exc.extra))
    except Exception as exc:  # unexpected: still a clean failure with the reason recorded
        LOG.exception("unexpected error")
        err = ErrorRecord("UNEXPECTED_ERROR", "frontend", f"{type(exc).__name__}: {exc}",
                          "Run again with --debug for the traceback; details are in diagnostics/run.log.", False, "pipeline")
        extras = {"traceback": traceback.format_exc(limit=6)} if debug else {}
        return finish("failure", [], [err], extras=extras)

    try:
        prop = so.prebuilt_property if so.prebuilt_property is not None else assemble_property(so)
    except (ModelError, ValueError) as exc:
        err = ErrorRecord("ASSEMBLY_FAILED", "geometry", f"The property could not be assembled: {exc}", None, False, "assembly")
        _prune_diagnostics(out_dir / "diagnostics", diagnostics)
        return finish("failure", so.stages, [err], so=so)

    if not prop.rooms and not prop.walls:
        # nothing trustworthy to report: this is a FAILURE, not a "partial" with an empty property
        prop.warning_records = [w.to_dict() for w in classify_all(prop.warnings)]
        _prune_diagnostics(out_dir / "diagnostics", diagnostics)
        return finish("failure", so.stages, [_no_geometry_error(prop, so)], prop=prop, so=so)

    damage_ran = False
    if damage and so.evidence_fn is not None:
        damage_ran = _run_damage(prop, so, out_dir)
    elif not damage:
        so.stages.append(StageRecord("damage_analysis", "skipped", 0.0, [], None, {"reason": "disabled (--no-damage)"}))

    prop.provenance["runtime"] = runtime_provenance(tier, damage_ran)
    prop.provenance["fallback_decisions"] = so.decisions if so.decisions is not None else (
        [{"stage": "pose_refinement", "decision": "kept the supplied (original) poses" if not so.drift.get("correction_accepted")
          else "used corrected poses", "reason": so.drift.get("fallback_reason") or "metrics supported the correction"}]
        if tier == "lidar" else [])
    if detection:
        prop.provenance.setdefault("pipeline", {})["tier_detection"] = detection
    prop.warning_records = [w.to_dict() for w in classify_all(prop.warnings)]

    try:
        logs.announce("rendering")
        t_render = time.perf_counter()
        render_info = render_plan(prop, out_dir / "plan.png", f"SpatialForge floor plan — {so.capture_name}")
        so.stages.append(StageRecord("rendering", "ok", time.perf_counter() - t_render))
        logs.announce("serialization")
        t_ser = time.perf_counter()
        std_seconds = {s["stage"]: s["duration_seconds"] for s in fold_stages(so.stages, lambda std, m: [])}
        prop.timing = {"stages": std_seconds, "total_duration_seconds": round(time.perf_counter() - t0, 3)}
        prop_dict = scrub(property_to_dict(prop))
    except (ModelError, ValueError) as exc:
        err = ErrorRecord("SERIALIZATION_FAILED", "serialization", f"The property failed validation: {exc}", None, False, "serialization")
        return finish("failure", so.stages, [err], so=so)

    # write the property last: its presence means the run produced a valid, serialised property
    diag_warnings = _write_diagnostics(out_dir, so)
    _prune_diagnostics(out_dir / "diagnostics", diagnostics)
    write_json(out_dir / "property.json", prop_dict)
    so.stages.append(StageRecord("serialization", "ok", time.perf_counter() - t_ser))
    plan = {"rooms_drawn": render_info.rooms_drawn, "walls_drawn": render_info.walls_drawn, "openings_drawn": render_info.openings_drawn}
    failed = [s.name for s in so.stages if s.status == "failed" and s.name != "damage_analysis"]  # an optional layer cannot downgrade the run
    status = "success" if (prop_dict["property"]["status"] == "complete" and not failed) else "partial"
    return finish(status, so.stages, [], prop=prop, prop_dict=prop_dict, so=so, counts=_counts(prop_dict, so), plan=plan,
                  diag_warnings=diag_warnings)
