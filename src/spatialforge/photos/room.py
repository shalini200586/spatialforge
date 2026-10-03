"""One room folder -> a LOCAL metric reconstruction (gravity-aligned, +Y up, metres, arbitrary yaw and origin).

    registered photos (SfM, arbitrary scale) -> metric depth per photo (cached) -> robust metric scale of THIS room
    -> scaled poses -> fused metric cloud -> gravity -> MetricScene -> shared structural stages -> local canonical Property

Nothing is invented: if SfM fails, or no metric scale can be established, the room is reported as unreconstructed /
unscaled with no geometry. Each room has its own scale estimate; other rooms' scales are never borrowed here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from spatialforge.pipeline.lidar_adapter import MeasureContext, StageOutputs, StageRecord, assemble_property
from spatialforge.pipeline.models import Property
from spatialforge.pipeline.scene import MetricScene
from spatialforge.pipeline.stages import StageTimer
from spatialforge.pipeline.structure import StructureConfig, run_structure_stages
from spatialforge.video.fusion import FusionFrame, FusionOptions, fuse_frames
from spatialforge.video.gravity import GravityEstimate, GravityOptions, estimate_gravity, rotation_to_y_up
from spatialforge.video.pipeline import depth_intrinsics
from spatialforge.video.scale import (
    ScaleEstimate, ScaleOptions, estimate_metric_scale, frame_alignment_factor, frame_scale, metric_camera_to_world,
)
from spatialforge.video.sfm import SfmRun
from spatialforge.video.uncertainty import UncertaintyModel, propagate

RANK = {"failure": 0, "unavailable": 0, "weak": 1, "moderate": 2, "strong": 3, "sensor": 3}
NAME = {0: "failure", 1: "weak", 2: "moderate", 3: "strong"}
GRAVITY_SIGMA = {"strong": 0.01, "moderate": 0.03, "weak": 0.06}  # relative length error from a tilted vertical axis (heuristic)


@dataclass
class PhotoRoomOptions:
    # a room has 2-8 photos: few frames, so fewer are demanded; "moderate" scale needs at least 3 supporting images
    scale: ScaleOptions = field(default_factory=lambda: ScaleOptions(
        min_frames=2, min_total_correspondences=100, min_correspondences=10, min_frames_for_moderate=3))
    # With 2-8 photos, different views mostly see different surfaces: a multi-view voxel check would delete most of the cloud
    # (measured: 80-96% removed on a synthetic room). Each photo's depth is aligned to the SfM geometry instead, and the missing
    # cross-check is paid for in the uncertainty model below (a larger single-view depth term).
    fusion: FusionOptions = field(default_factory=lambda: FusionOptions(pixel_stride=2, min_views=1))
    uncertainty: UncertaintyModel = field(default_factory=lambda: UncertaintyModel(depth_base=0.05))
    gravity: GravityOptions = field(default_factory=GravityOptions)
    structure: StructureConfig = field(default_factory=StructureConfig)


@dataclass
class PhotoRoomResult:
    index: int
    canonical_id: str  # room_001 ... deterministic from the sorted folder names
    source_label: str  # the folder name; never interpreted
    folder: str
    image_names: list[str]
    status: str  # reconstructed | unreconstructed (SfM failed) | unscaled (no metric scale)
    quality: str  # strong | moderate | weak | failure
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sfm: dict | None = None
    scale: dict | None = None
    gravity: dict | None = None
    uncertainty: dict | None = None
    rel_sigma: float | None = None
    prop: Property | None = None  # local canonical property (local frame); rooms/walls/openings with local ids
    primary_room_id: str | None = None  # local id of the main polygon
    ambiguity: list[str] = field(default_factory=list)
    poses: dict = field(default_factory=dict)  # image name -> 4x4 camera-to-world, local metric frame
    cloud: np.ndarray | None = None
    sparse: np.ndarray | None = None
    stages: list[StageRecord] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def usable(self) -> bool:
        return self.status == "reconstructed" and self.prop is not None

    def report(self, timing: bool = True) -> dict:
        """Plain-dict summary. `timing=False` leaves out the volatile numbers (used for the deterministic property.json)."""
        p = self.prop
        return {
            "canonical_id": self.canonical_id, "source_label": self.source_label, "folder": self.folder, "images": self.image_names,
            "status": self.status, "quality": self.quality, "reasons": self.reasons, "warnings": self.warnings,
            "sfm": self.sfm, "scale": None if self.scale is None else {k: v for k, v in self.scale.items() if k != "per_frame"},
            "gravity": self.gravity, "uncertainty": self.uncertainty, "primary_local_room": self.primary_room_id,
            "ambiguity": self.ambiguity,
            "local_counts": None if p is None else {"rooms": len(p.rooms), "walls": len(p.walls), "openings": len(p.openings),
                                                     "unverified_openings": len(p.unverified_openings)},
            "local_area_m2": None if p is None or not p.rooms else round(max(r.area.value for r in p.rooms), 3),
            "stages": [{"name": s.name, "status": s.status, "warnings": s.warnings,
                        **({"seconds": round(s.seconds, 3)} if timing else {})} for s in self.stages],
            **({"runtime_s": round(self.seconds, 2)} if timing else {}),
        }


def failed_result(room_idx, room, names, status, quality, reasons, sfm=None, scale=None, t0=None, stages=None, warnings=None):
    return PhotoRoomResult(room_idx, room.canonical_id, room.source_label, room.folder, names, status, quality, list(reasons),
                           list(warnings or []), sfm=sfm, scale=scale, stages=stages or [],
                           seconds=0.0 if t0 is None else time.perf_counter() - t0)


def reconstruct_room(room, index: int, run: SfmRun, depth_for, opts: PhotoRoomOptions | None = None) -> PhotoRoomResult:
    """`room`: RoomFolder; `run`: this room's SfM; `depth_for(name) -> (h, w) metric depth in metres` (cached by the caller)."""
    opts = opts or PhotoRoomOptions()
    t0 = time.perf_counter()
    sfm = run.result
    names = sorted(set(run.poses) | set(sfm.unregistered))
    sfm_d = sfm.to_dict()
    stages: list[StageRecord] = []
    quality = sfm.tracking_quality
    reasons = list(sfm.quality_reasons)
    if sfm.registered_images < 3 and quality == "strong":
        quality = "moderate"
        reasons.append("only two photos are registered: not enough views for a strong reconstruction")
    sfm_d["tracking_quality"] = quality
    if quality == "failure":
        return failed_result(index, room, names, "unreconstructed", "failure", ["SfM failed: " + "; ".join(reasons)], sfm_d, None, t0, stages)

    # --- metric depth (cached by the caller) and this room's own scale ---
    reg = sorted(run.poses)
    with StageTimer(stages, "metric_depth_and_scale") as rec:
        depth_maps, frame_scales = {}, []
        for n in reg:
            d = depth_for(n)
            depth_maps[n] = d
            cam = run.cameras[n]
            xy, xyz = run.observations.get(n, (np.zeros((0, 2)), np.zeros((0, 3))))
            pose = run.poses[n]
            zs = (xyz @ pose.rotation_cw.T + pose.translation_cw)[:, 2] if len(xyz) else np.zeros(0)
            frame_scales.append(frame_scale(n, xy, zs, d, (cam.width, cam.height), opts.scale))
        est: ScaleEstimate = estimate_metric_scale(frame_scales, opts.scale)
        rec.details = {"images": len(reg), "scale_available": est.available, "scale_quality": est.quality,
                       "metres_per_sfm_unit": est.scale, "correspondences": est.correspondences}
        rec.warnings = [] if est.available else ["no metric scale: " + "; ".join(est.reasons)]
    scale_d = est.to_dict()
    if not est.available:
        msg = ("Metric scale could not be established for this room, so no metric dimensions are reported: "
               + "; ".join(est.reasons))
        return failed_result(index, room, names, "unscaled", "failure", reasons + [msg], sfm_d, scale_d, t0, stages, [msg])

    s = float(est.scale)
    warnings: list[str] = []
    T_wc = {n: metric_camera_to_world(run.poses[n].rotation_cw, run.poses[n].translation_cw, s) for n in reg}
    frames, excluded = [], []
    fsc = {f.name: f for f in est.frames}
    for n, d in depth_maps.items():
        factor = frame_alignment_factor(fsc[n], s)
        if factor is None:
            excluded.append(n)
            continue
        cam = run.cameras[n]
        fx, fy, cx, cy, k1 = depth_intrinsics(cam.model, cam.params, (cam.width, cam.height), d.shape)
        frames.append(FusionFrame(n, d * factor, fx, fy, cx, cy, T_wc[n], k1))
    if excluded:
        warnings.append(f"{len(excluded)} photo(s) were excluded from fusion: their depth disagreed with the SfM geometry by more than 2.5x.")
    fopts = opts.fusion
    if len(frames) < 3 and fopts.min_views > 1:
        fopts = FusionOptions(**{**fopts.__dict__, "min_views": 1})
        warnings.append("fewer than three usable photos: the multi-view consistency check was skipped")
    with StageTimer(stages, "depth_fusion") as rec:
        fused = fuse_frames(frames, fopts)
        rec.warnings, rec.details = list(fused.warnings), fused.stats()
    if len(fused.points) < 1000:
        msg = "The fused metric cloud has too few points for structural analysis."
        return failed_result(index, room, names, "unscaled", "failure", reasons + [msg], sfm_d, scale_d, t0, stages, warnings + [msg])

    # --- gravity (photos are EXIF-corrected, so image-up is a usable hint) ---
    cams_m = np.array([T_wc[n][:3, 3] for n in reg])
    up_hints = np.array([-T_wc[n][:3, 1] for n in reg])
    with StageTimer(stages, "gravity_alignment") as rec:
        grav: GravityEstimate = estimate_gravity(fused.points, cams_m, up_hints, opts.gravity)
        rec.details, rec.warnings = grav.to_dict(), list(grav.notes)
    R_up = rotation_to_y_up(grav.up)
    cloud = fused.points @ R_up.T
    T_up = {n: np.block([[R_up @ T_wc[n][:3, :3], (R_up @ T_wc[n][:3, 3])[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for n in reg}
    shift = np.median(np.array([T_up[n][:3, 3] for n in reg]), axis=0) * np.array([1.0, 0.0, 1.0])
    cloud = cloud - shift
    for n in reg:
        T_up[n][:3, 3] -= shift

    # --- uncertainty for this room (stitching terms are added later, at property level) ---
    unc = propagate(est.sigma_rel, sfm.mean_reprojection_error_px, sfm.registered_ratio, fused.consistency_removed_fraction,
                    opts.uncertainty)
    g_sig = GRAVITY_SIGMA[grav.quality]
    rel = float(np.hypot(unc.total_rel_sigma, g_sig))
    unc_d = {**unc.to_dict(), "gravity_relative_sigma": round(g_sig, 4), "room_relative_sigma": round(rel, 4)}
    cap = {"strong": None, "moderate": "moderate", "weak": "weak"}[est.quality]
    ctx = MeasureContext("photos", rel, unc.k, cap, unc_d)

    # --- shared structural backend ---
    scene = MetricScene(cloud, "photos", NAME[min(RANK[quality], RANK[grav.quality], RANK[est.quality])], est.quality,
                        camera_poses=np.array([T_up[n] for n in reg]), frame_refs=reg, warnings=list(warnings),
                        metadata={"metres_per_sfm_unit": s})
    out = StageOutputs(
        capture_path=room.folder, capture_name=room.source_label, tier="photos", validation_warnings=[], pose_source="sfm_scaled",
        drift={}, floor_y_m=None, ceiling_levels=[], wall_dicts=[], topology=None, openings=None, stage_warnings=list(warnings),
        stages=stages, device={"model": None}, frames={"photos": len(room.images), "registered": len(reg)}, uncertainty=ctx,
        provenance_extra={"pipeline": {"tier": "photos", "stage": "room_local"}},
        parameters={"fusion_voxel_m": fopts.voxel_m})
    out.debug["cloud"] = cloud
    run_structure_stages(scene, out, opts.structure)
    prop = assemble_property(out)

    # --- one main polygon per folder ---
    primary, ambiguity = None, []
    if prop.rooms:
        order = sorted(prop.rooms, key=lambda r: (-(3 if r.topology_quality == "strong" else 2 if r.topology_quality == "moderate" else 1),
                                                   -r.area.value, r.id))
        primary = order[0].id
        if len(order) > 1:
            ambiguity.append(f"{len(order)} local room polygons were found; {primary} ({order[0].area.value:.1f} m2) is the "
                             "primary, the others are ambiguous and are not used for placement: "
                             + ", ".join(f"{r.id} ({r.area.value:.1f} m2)" for r in order[1:]))
    q = NAME[min(RANK[quality], RANK[est.quality], RANK[grav.quality])]
    reasons += [f"metric scale {est.quality} ({s:.4f} m per SfM unit, sigma {est.sigma_rel:.0%})", f"gravity {grav.quality}"]
    return PhotoRoomResult(
        index, room.canonical_id, room.source_label, room.folder, names, "reconstructed", q, reasons, prop.warnings + warnings,
        sfm=sfm_d, scale=scale_d, gravity=grav.to_dict(), uncertainty=unc_d, rel_sigma=rel, prop=prop, primary_room_id=primary,
        ambiguity=ambiguity, poses=T_up, cloud=cloud, sparse=(run.points_xyz * s) @ R_up.T - shift, stages=stages,
        seconds=time.perf_counter() - t0)
