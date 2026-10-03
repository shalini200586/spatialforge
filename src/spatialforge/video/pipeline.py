"""Tier 2 front end: walkthrough video -> metric, gravity-aligned MetricScene -> shared structural backend.

    video -> validation -> deterministic keyframes -> PyCOLMAP sparse SfM (arbitrary scale)
          -> metric monocular depth on keyframes -> robust metric scale (depth vs SfM) -> scaled trajectory
          -> fused metric cloud -> gravity from geometry -> MetricScene -> floor/walls/rooms/openings -> Property

Metres exist only if the scale estimate is available. If it is not, the run returns an honest partial property with no
geometry; it never converts SfM units to metres by assumption.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spatialforge import __version__
from spatialforge.pipeline.lidar_adapter import MeasureContext, StageOutputs, StageRecord
from spatialforge.pipeline.scene import MetricScene
from spatialforge.pipeline.serialization import write_json
from spatialforge.pipeline.stages import PipelineFailure, StageTimer
from spatialforge.pipeline.structure import StructureConfig, run_structure_stages
from spatialforge.video.depth import DepthBackend, DepthUnavailable
from spatialforge.video.diagnostics import render_trajectory_png, write_binary_ply
from spatialforge.video.fusion import FusionFrame, FusionOptions, fuse_frames
from spatialforge.video.gravity import GravityOptions, estimate_gravity, rotation_to_y_up
from spatialforge.video.keyframes import KeyframeOptions, KeyframeResult, extract_keyframes
from spatialforge.video.scale import (
    ScaleEstimate, ScaleOptions, estimate_metric_scale, frame_alignment_factor, frame_scale, metric_camera_to_world,
)
from spatialforge.video.sfm import SfmOptions, SfmRun, cam_to_world, run_sfm
from spatialforge.video.uncertainty import propagate
from spatialforge.video.validator import validate_video

RETRY_MAX_ELAPSED_S = 240.0  # only try the denser-keyframe retry while the run is still comfortably inside its budget


@dataclass
class VideoOptions:
    keyframes: KeyframeOptions = field(default_factory=KeyframeOptions)
    sfm: SfmOptions = field(default_factory=SfmOptions)
    scale: ScaleOptions = field(default_factory=ScaleOptions)
    fusion: FusionOptions = field(default_factory=FusionOptions)
    gravity: GravityOptions = field(default_factory=GravityOptions)
    max_depth_frames: int = 64  # keyframes that get a metric depth map (evenly spaced over the registered ones)
    retry_denser_keyframes: bool = True
    output_dir: Path | None = None  # set by the orchestration; diagnostics are written below it
    # injectable seams (tests): defaults are the real implementations
    keyframes_fn: object = None
    sfm_fn: object = None
    depth_backend: DepthBackend | None = None
    depth_backend_factory: object = None


# ---------- helpers ----------


def depth_intrinsics(model: str, params: list[float], image_size: tuple[int, int], depth_shape: tuple[int, int]):
    """(fx, fy, cx, cy, k1) at depth-map resolution from a COLMAP camera."""
    W, H = image_size
    h, w = depth_shape
    sx, sy = w / W, h / H
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[:3]
        k1 = params[3] if model != "SIMPLE_PINHOLE" and len(params) > 3 else 0.0
        return f * sx, f * sy, cx * sx, cy * sy, float(k1)
    if model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = params[:4]
        return fx * sx, fy * sy, cx * sx, cy * sy, 0.0
    raise ValueError(f"unsupported camera model {model}")


def evenly_spaced(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    return [items[i] for i in np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))]


def _rec(stages, name):
    return StageTimer(stages, name)


def _write_trajectory(path: Path, names: list[str], kres: KeyframeResult, run: SfmRun, T_world: dict | None = None) -> None:
    """Per registered keyframe: video frame, time, camera centre in SfM units and (when known) in metres, +Y up."""
    by_file = {k.file: k for k in kres.keyframes}
    rows = []
    for n in names:
        p = run.poses[n]
        C = -p.rotation_cw.T @ p.translation_cw
        k = by_file.get(n)
        rows.append({"name": n, "frame_index": None if k is None else k.frame_index, "t_s": None if k is None else round(k.t, 3),
                     "position_sfm_units": [round(float(v), 5) for v in C],
                     "position_m": None if T_world is None else [round(float(v), 4) for v in T_world[n][:3, 3]]})
    write_json(path, {"note": "camera centres of the registered keyframes; position_m is gravity-aligned metres (null when "
                              "metric scale is unavailable)", "cameras": rows})


def _frontend_dict(val, kres: KeyframeResult, attempts: list[dict]) -> dict:
    return {"video": val.info.__dict__, "validation_warnings": val.warnings, "keyframes": kres.to_dict(),
            "keyframe_strategy": {"candidate_sampling": "fixed temporal rate, one decode pass", "blur_filter":
                                  "Laplacian variance below a fraction of the local median", "rapid_motion_filter":
                                  "median LK flow between consecutive candidates", "selection": "median LK parallax since the "
                                  "last keyframe, minimum temporal gap, maximum gap", "thinning": "drop the keyframe whose "
                                  "removal leaves the smallest gap", "randomness": "none"},
            "sfm_attempts": attempts}


# ---------- the stages ----------


def _pycolmap_version() -> str | None:
    try:
        import pycolmap

        return str(pycolmap.__version__)
    except ImportError:
        return None


def run_video_stages(video: str | Path, opts: VideoOptions) -> StageOutputs:
    video = Path(video)
    stages: list[StageRecord] = []
    warnings: list[str] = []
    out_dir = Path(opts.output_dir) if opts.output_dir else Path(tempfile.mkdtemp(prefix="sf_video_"))
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="sf_sfm_"))
    extra: dict = {}

    def fail(stage, message):
        raise PipelineFailure(stage, message, stages=stages, extra={"video": extra, "metric_scale_available": False})

    try:
        # 1. validation (essential)
        with _rec(stages, "video_validation") as rec:
            val = validate_video(video)
            rec.warnings = list(val.warnings)
            rec.details = {k: v for k, v in val.info.__dict__.items() if k != "path"}
        if not val.ok:
            fail("video_validation", "; ".join(val.errors))

        # 2. keyframes (essential)
        kfn = opts.keyframes_fn or extract_keyframes
        kdir = diag / "keyframes"
        if kdir.exists():
            shutil.rmtree(kdir)
        try:
            with _rec(stages, "keyframe_extraction") as rec:
                kres = kfn(video, kdir, opts.keyframes)
                rec.warnings = list(kres.warnings)
                rec.details = {"keyframes": len(kres.keyframes), "candidates": kres.candidates,
                               "rejected_blur": kres.rejected_blur, "rejected_rapid_motion": kres.rejected_motion,
                               "thinned": kres.thinned, **kres.video_facts}
        except Exception as exc:
            fail("keyframe_extraction", f"{type(exc).__name__}: {exc}")
        if len(kres.keyframes) < opts.sfm.min_registered_images:
            extra.update(_frontend_dict(val, kres, []))
            write_json(diag / "video_frontend.json", extra)
            fail("keyframe_extraction", f"only {len(kres.keyframes)} usable keyframes (need at least "
                                        f"{opts.sfm.min_registered_images}); the video may be too static, blurred or short")

        # 3. SfM (essential)
        sfm_fn = opts.sfm_fn or run_sfm
        attempts: list[dict] = []
        try:
            with _rec(stages, "sfm_reconstruction") as rec:
                run = sfm_fn(kdir, workspace / "sfm", opts.sfm)
                attempts.append({"keyframes": len(kres.keyframes), **run.result.to_dict()})
                gaps = np.diff([k.t for k in kres.keyframes]) if len(kres.keyframes) > 1 else np.array([0.0])
                elapsed = sum(s.seconds for s in stages)
                if (opts.retry_denser_keyframes and run.result.tracking_quality in ("weak", "failure")
                        and float(np.median(gaps)) > 0.3 and elapsed < RETRY_MAX_ELAPSED_S):
                    rec.warnings.append("tracking was weak: retried once with denser keyframes")
                    dense = KeyframeOptions(**{**opts.keyframes.__dict__,
                                               "candidate_hz": opts.keyframes.candidate_hz * 1.5,
                                               "min_gap_s": opts.keyframes.min_gap_s / 2,
                                               "min_parallax": opts.keyframes.min_parallax * 0.75,
                                               "target_max": int(opts.keyframes.target_max * 1.5),
                                               "max_candidates": int(opts.keyframes.max_candidates * 1.5)})
                    ddir = diag / "keyframes_dense"
                    try:
                        kres2 = kfn(video, ddir, dense)
                        run2 = sfm_fn(ddir, workspace / "sfm_dense", opts.sfm)
                        attempts.append({"keyframes": len(kres2.keyframes), "denser_retry": True, **run2.result.to_dict()})
                        if run2.result.registered_images > run.result.registered_images:
                            run, kres = run2, kres2
                            shutil.rmtree(kdir, ignore_errors=True)
                            kdir = ddir
                            rec.warnings.append("the denser-keyframe attempt registered more images and was kept")
                        else:
                            shutil.rmtree(ddir, ignore_errors=True)
                    except Exception as exc:  # the first result stands
                        shutil.rmtree(ddir, ignore_errors=True)
                        rec.warnings.append(f"the denser-keyframe retry failed ({type(exc).__name__}: {exc}); kept the first attempt")
                s = run.result
                rec.details = {"registered": s.registered_images, "keyframes": s.keyframes_attempted,
                               "sparse_points": s.sparse_points, "reprojection_px": s.mean_reprojection_error_px,
                               "tracking_quality": s.tracking_quality, "models": s.model_sizes[:6], **s.seconds}
        except PipelineFailure:
            raise
        except Exception as exc:
            extra.update(_frontend_dict(val, kres, attempts))
            fail("sfm_reconstruction", f"{type(exc).__name__}: {exc}")
        sfm = run.result
        extra.update(_frontend_dict(val, kres, attempts))
        extra["pycolmap_version"] = _pycolmap_version()
        extra["tracking"] = {"quality": sfm.tracking_quality, "reasons": sfm.quality_reasons}
        write_json(diag / "video_frontend.json", extra)
        cam = sfm.camera
        names = sorted(run.poses)
        cams_sfm = np.array([cam_to_world(run.poses[n].rotation_cw, run.poses[n].translation_cw)[:3, 3] for n in names]).reshape(-1, 3)
        _write_trajectory(diag / "trajectory.json", names, kres, run)
        if sfm.tracking_quality == "failure":
            render_trajectory_png(diag / "trajectory.png", cams_sfm, run.points_xyz, False,
                                  f"SfM failed: {sfm.registered_images}/{sfm.keyframes_attempted} registered (arbitrary scale)")
            fail("sfm_reconstruction", "no coherent reconstruction: " + "; ".join(sfm.quality_reasons))
        if sfm.tracking_quality == "weak":
            warnings.append(f"SfM tracking quality is WEAK: {sfm.registered_images}/{sfm.keyframes_attempted} keyframes "
                            "registered in the largest connected model; the trajectory covers only part of the walkthrough.")
        if len(sfm.model_sizes) > 1:
            warnings.append(f"SfM produced {len(sfm.model_sizes)} disconnected models (sizes {sfm.model_sizes[:5]}); only the "
                            "largest is used, so the rest of the video is not reconstructed.")
        warnings.append("Camera intrinsics were estimated by COLMAP (SIMPLE_RADIAL, shared by all keyframes); no focal-length "
                        "metadata was used.")

        # 4. metric depth + scale
        backend = opts.depth_backend
        scale_est: ScaleEstimate | None = None
        depth_maps: dict[str, np.ndarray] = {}
        backend_info: dict = {}
        with _rec(stages, "metric_depth_and_scale") as rec:
            try:
                if backend is None:
                    if opts.depth_backend_factory is not None:
                        backend = opts.depth_backend_factory()
                    else:
                        from spatialforge.video.depth import DepthAnythingMetric

                        backend = DepthAnythingMetric()
                backend_info = backend.describe()
                if not backend.metric:
                    raise DepthUnavailable("the depth backend does not produce metric depth")
            except DepthUnavailable as exc:
                backend = None
                rec.warnings.append(f"metric depth unavailable: {exc}")
            if backend is not None:
                import cv2

                sel = evenly_spaced(names, opts.max_depth_frames)
                try:
                    frame_scales = []
                    for n in sel:
                        img = cv2.cvtColor(cv2.imread(str(kdir / n)), cv2.COLOR_BGR2RGB)
                        depth = backend.predict(img)
                        depth_maps[n] = depth
                        xy, xyz = run.observations.get(n, (np.zeros((0, 2)), np.zeros((0, 3))))
                        pose = run.poses[n]
                        zs = (xyz @ pose.rotation_cw.T + pose.translation_cw)[:, 2] if len(xyz) else np.zeros(0)
                        frame_scales.append(frame_scale(n, xy, zs, depth, (cam.width, cam.height), opts.scale))
                    scale_est = estimate_metric_scale(frame_scales, opts.scale)
                    rec.details = {"depth_frames": len(sel), "scale_available": scale_est.available,
                                   "scale_quality": scale_est.quality, "metres_per_sfm_unit": scale_est.scale,
                                   "correspondences": scale_est.correspondences}
                except Exception as exc:  # e.g. out of memory: degrade to "no metric scale", never to invented metres
                    scale_est, depth_maps = None, {}
                    rec.warnings.append(f"metric depth inference failed ({type(exc).__name__}: {exc})")
        scale_report = scale_est.to_dict() if scale_est is not None else {
            "metric_scale_available": False, "scale_quality": "unavailable", "metres_per_sfm_unit": None, "median_scale": None,
            "frames_used": 0, "frames_tried": 0, "correspondences": 0, "frame_spread_rel": None, "standard_error_rel": None,
            "sigma_rel": None, "reasons": ["no metric depth backend is available"], "per_frame": []}
        scale_report["depth_backend"] = backend_info
        write_json(diag / "scale_report.json", scale_report)
        extra["metric_scale_available"] = bool(scale_est and scale_est.available)
        extra["metric_scale"] = {k: v for k, v in scale_report.items() if k != "per_frame"}

        provenance = _provenance(video, val, kres, run, scale_report, backend_info)
        base = dict(
            capture_path=str(video), capture_name=video.stem, tier="video", validation_warnings=[], pose_source="sfm_scaled",
            drift={}, floor_y_m=None, ceiling_levels=[], wall_dicts=[], topology=None, openings=None,
            stage_warnings=warnings, stages=stages, provenance_extra=provenance,
            device={"model": None, "video": {k: v for k, v in val.info.__dict__.items() if k != "path"}},
            capture_extra={"video": {"keyframes": len(kres.keyframes), "registered_images": sfm.registered_images,
                                     "tracking_quality": sfm.tracking_quality,
                                     "metric_scale_available": bool(scale_est and scale_est.available)}},
            frames={"keyframes": len(kres.keyframes), "registered": sfm.registered_images,
                    "depth_frames": len(depth_maps)},
            debug={"run_report_extra": {"video": extra, "metric_scale_available": extra["metric_scale_available"],
                                        "tracking_quality": sfm.tracking_quality}},
        )
        decisions = [{"stage": "tracking_quality", "decision": f"tracking {sfm.tracking_quality.upper()}",
                      "reason": "; ".join(sfm.quality_reasons)}]

        if scale_est is None or not scale_est.available:
            reason = "; ".join(scale_est.reasons) if scale_est is not None else "no metric depth backend is available"
            msg = ("Metric scale is NOT available, so no dimensions are reported: SfM alone is scale-ambiguous and SfM "
                   f"units are not converted to metres by assumption ({reason}).")
            warnings.append(msg)
            decisions.append({"stage": "metric_scale", "decision": "no metric scale; geometry withheld", "reason": reason})
            for name in ("gravity_alignment", "depth_fusion", "structural_walls", "room_topology", "openings"):
                stages.append(StageRecord(name, "skipped", warnings=["skipped: no metric scale"]))
            render_trajectory_png(diag / "trajectory.png", cams_sfm, run.points_xyz, False,
                                  "Camera path in SfM units (metric scale unavailable)")
            write_binary_ply(diag / "sparse_sfm.ply", run.points_xyz, comment="SfM units, arbitrary scale, arbitrary orientation")
            return StageOutputs(**base, decisions=decisions, status_blockers=[msg],
                                parameters={"max_depth_frames": opts.max_depth_frames})

        # 5. scaled trajectory + fusion in the SfM world frame (metric, orientation still arbitrary)
        s = scale_est.scale
        decisions.append({"stage": "metric_scale", "decision": f"{scale_est.quality} metric scale, {s:.4f} m per SfM unit",
                          "reason": "; ".join(scale_est.reasons[:2])})
        T_wc = {n: metric_camera_to_world(run.poses[n].rotation_cw, run.poses[n].translation_cw, s) for n in names}
        fframes, excluded = [], []
        fs_by_name = {f.name: f for f in scale_est.frames}
        for n, depth in depth_maps.items():
            factor = frame_alignment_factor(fs_by_name[n], s)
            if factor is None:
                excluded.append(n)
                continue
            fx, fy, cx, cy, k1 = depth_intrinsics(cam.model, cam.params, (cam.width, cam.height), depth.shape)
            fframes.append(FusionFrame(n, depth * factor, fx, fy, cx, cy, T_wc[n], k1))
        if excluded:
            warnings.append(f"{len(excluded)} depth keyframes were excluded from fusion: their depth scale disagreed with "
                            "the SfM geometry by more than 2.5x.")
        decisions.append({"stage": "depth_alignment", "decision": "each frame's depth is rescaled to the SfM geometry "
                          "at the global metric scale", "reason": "single-frame metric depth varies by tens of percent "
                          "between frames; the global scale is the only absolute quantity taken from the depth model"})
        with _rec(stages, "depth_fusion") as rec:
            fused = fuse_frames(fframes, opts.fusion)
            rec.warnings = list(fused.warnings)
            rec.details = fused.stats()
        if len(fused.points) < 1000:
            warnings.append("The fused metric cloud has too few points for structural analysis.")
            for name in ("gravity_alignment", "structural_walls", "room_topology", "openings"):
                stages.append(StageRecord(name, "skipped", warnings=["skipped: metric cloud too sparse"]))
            return StageOutputs(**base, decisions=decisions, status_blockers=["Fused metric cloud too sparse."],
                                parameters={"max_depth_frames": opts.max_depth_frames})

        # 6. gravity
        cams_m = np.array([T_wc[n][:3, 3] for n in names])
        up_hints = np.array([-T_wc[n][:3, 1] for n in names])  # image-up in world coordinates: a weak hint only
        with _rec(stages, "gravity_alignment") as rec:
            grav = estimate_gravity(fused.points, cams_m, up_hints, opts.gravity)
            rec.details = grav.to_dict()
            rec.warnings = list(grav.notes)
        R_up = rotation_to_y_up(grav.up)
        cloud = fused.points @ R_up.T
        T_up = {n: np.block([[R_up @ T_wc[n][:3, :3], (R_up @ T_wc[n][:3, 3])[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
                for n in names}
        # shift so the camera path starts near the origin (cosmetic; keeps coordinates small)
        shift = np.median(np.array([T_up[n][:3, 3] for n in names]), axis=0) * np.array([1.0, 0.0, 1.0])
        cloud = cloud - shift
        for n in names:
            T_up[n][:3, 3] -= shift
        extra["gravity"] = grav.to_dict()
        extra["world_from_sfm"] = {"note": "x_world = rotation @ (metres_per_sfm_unit * x_sfm) + translation",
                                   "metres_per_sfm_unit": float(s), "rotation": R_up.tolist(),
                                   "translation": (-shift).tolist()}
        decisions.append({"stage": "gravity", "decision": f"{grav.quality} gravity estimate (confidence {grav.confidence:.2f})",
                          "reason": grav.method + "; sign: " + grav.sign_basis})

        unc = propagate(scale_est.sigma_rel, sfm.mean_reprojection_error_px, sfm.registered_ratio,
                        fused.consistency_removed_fraction)
        cap = {"strong": None, "moderate": "moderate", "weak": "weak"}[scale_est.quality]
        ctx = MeasureContext("video", unc.total_rel_sigma, unc.k, cap, unc.to_dict())
        extra["uncertainty"] = unc.to_dict()
        warnings.append(f"Video tier: metres come from a monocular metric-depth model calibrated against SfM; dimensions "
                        f"carry a relative 1-sigma of about {unc.total_rel_sigma:.0%} (scale {unc.scale_rel_sigma:.0%}, "
                        f"SfM {unc.sfm_rel_sigma:.0%}, depth {unc.depth_rel_sigma:.0%}) and are less certain than LiDAR.")
        blockers = []
        if sfm.tracking_quality == "weak":
            blockers.append("Tracking is weak, so the property is reported as partial.")
        if scale_est.quality == "weak":
            blockers.append("Metric scale is weak (large frame-to-frame disagreement), so the property is reported as partial.")
        if grav.quality == "weak":
            blockers.append("Gravity (vertical) confidence is low, so floor/ceiling/wall geometry may be tilted.")

        # diagnostics: trajectory + clouds in the gravity-aligned metric frame
        _write_trajectory(diag / "trajectory.json", names, kres, run, T_up)
        sparse_m = (run.points_xyz * s) @ R_up.T - shift
        write_binary_ply(diag / "sparse_sfm.ply", sparse_m, comment="metres, gravity-aligned (+Y up)")
        write_binary_ply(diag / "metric_cloud.ply", cloud, comment="metres, gravity-aligned (+Y up), fused from keyframe depth")
        render_trajectory_png(diag / "trajectory.png", np.array([T_up[n][:3, 3] for n in names]), sparse_m, True,
                              f"Camera path and sparse SfM points, metres ({sfm.registered_images} cameras)")

        # 7. shared structural backend
        geometry_quality = {"strong": 2, "moderate": 1, "weak": 0}
        gq = min(geometry_quality[sfm.tracking_quality], geometry_quality[grav.quality], geometry_quality[scale_est.quality])
        scene = MetricScene(cloud, "video", ("weak", "moderate", "strong")[gq], scale_est.quality,
                            camera_poses=np.array([T_up[n] for n in names]), frame_refs=names,
                            warnings=list(warnings), metadata={"metres_per_sfm_unit": s})
        subs = _subset_fn(fframes, opts.fusion, R_up, shift)
        scene.subsets_fn = subs
        out = StageOutputs(**base, decisions=decisions, uncertainty=ctx, status_blockers=blockers,
                           parameters={"max_depth_frames": opts.max_depth_frames, "voxel_size_m": opts.fusion.voxel_m})
        out.debug["cloud"] = cloud
        out.debug["run_report_extra"]["video"] = extra
        run_structure_stages(scene, out, StructureConfig())
        return out
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _subset_fn(frames: list[FusionFrame], fopts: FusionOptions, R_up: np.ndarray, shift: np.ndarray, count: int = 3, block: int = 3):
    """Independent view subsets (round-robin blocks of frames) for the opening-stability check."""
    def make() -> list[np.ndarray]:
        out = []
        for k in range(count):
            sub = [f for i, f in enumerate(frames) if (i // block) % count == k]
            if not sub:
                continue
            r = fuse_frames(sub, FusionOptions(**{**fopts.__dict__, "min_views": 1}))
            out.append(r.points @ R_up.T - shift)
        return out
    return make


def _provenance(video, val, kres, run: SfmRun, scale_report: dict, backend_info: dict) -> dict:
    s = run.result
    return {
        "pipeline": {"spatialforge_version": __version__, "schema_version": "1.0", "tier": "video"},
        "video": {k: v for k, v in val.info.__dict__.items() if k != "path"},
        "keyframe_strategy": {"count": len(kres.keyframes), "candidates": kres.candidates, "rejected_blur": kres.rejected_blur,
                              "rejected_rapid_motion": kres.rejected_motion, "thinned": kres.thinned,
                              "method": "temporal spacing + Laplacian sharpness + LK parallax, deterministic"},
        "sfm": {"backend": "pycolmap", "version": _pycolmap_version(), "matching": "sequential", "registered_images":
                s.registered_images, "keyframes": s.keyframes_attempted, "registered_ratio": round(s.registered_ratio, 4),
                "sparse_points": s.sparse_points, "mean_reprojection_error_px": s.mean_reprojection_error_px,
                "tracking_quality": s.tracking_quality, "camera": None if s.camera is None else s.camera.__dict__,
                "unregistered": s.unregistered[:50], "disconnected_models": s.model_sizes[:8]},
        "metric_depth": backend_info,
        "metric_scale": {k: v for k, v in scale_report.items() if k not in ("per_frame", "depth_backend")},
        "coordinate_frame": "gravity-aligned from scene geometry (+Y up), arbitrary yaw, metres; not georeferenced",
        "assumptions": [
            "Metric scale is estimated, not measured: it is the median agreement between a monocular metric-depth model "
            "and the SfM geometry, and its spread is carried into every interval.",
            "SfM and depth are pinhole + one radial term; the camera is shared by all keyframes.",
            "Walls are vertical and floors horizontal once gravity is estimated from the point cloud.",
            "Intervals are engineering uncertainty ranges, not calibrated confidence intervals.",
        ],
    }
