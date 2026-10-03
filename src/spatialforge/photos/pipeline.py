"""Tier 1 orchestration: room photo folders -> ONE placed property.

    validate -> one COLMAP database (all images, exhaustive matching) -> per room: SfM, metric depth, own metric scale,
    local metric scene, shared structural stages -> cross-room evidence (verified matches) -> pair reconstructions ->
    stitch constraints (visual, doorway) -> robust global placement with loop and overlap checks -> merge -> Property.

Nothing is placed without evidence, nothing is scaled without a metric-depth scale estimate, nothing is guessed.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spatialforge import __version__
from spatialforge.photos.diagnostics import render_stitching_png
from spatialforge.photos.merge import (
    PlacedRoom, dedupe_openings, dedupe_walls, footprint_measurement, local_geom, scale_offsets, transform_room, union_footprint,
)
from spatialforge.photos.room import RANK, PhotoRoomOptions, PhotoRoomResult, failed_result, reconstruct_room
from spatialforge.photos.sfm import ColmapPhotoBackend, PhotoSfmOptions
from spatialforge.photos.stitch import (
    ComponentPlacement, RoomStitchConstraint, StitchOptions, adjacent_edges, apply, build_doorway_constraint,
    build_visual_constraint, components, compose, cross_room_evidence, invert, match_doorways, overlap_report,
    place_component, visual_pair_transform,
)
from spatialforge.photos.validator import validate_photo_dir
from spatialforge.pipeline.lidar_adapter import StageOutputs, StageRecord
from spatialforge.pipeline.models import Footprint, ModelError, Property
from spatialforge.pipeline.render import render_plan
from spatialforge.pipeline.serialization import write_json
from spatialforge.pipeline.stages import PipelineFailure, StageTimer
from spatialforge.video.depth import DepthUnavailable
from spatialforge.video.diagnostics import write_binary_ply

SCALE_INCONSISTENT = 0.20  # a room whose metric scale differs from the property consensus by more than this is flagged


@dataclass
class PhotoOptions:
    room: PhotoRoomOptions = field(default_factory=PhotoRoomOptions)
    stitch: StitchOptions = field(default_factory=StitchOptions)
    sfm: PhotoSfmOptions = field(default_factory=PhotoSfmOptions)
    allow_doorway_only: bool = True  # weak placements from a unique doorway match when there is no image evidence
    output_dir: Path | None = None  # set by the orchestration
    # injectable seams (tests): defaults are the real implementations
    sfm_backend: object = None
    depth_backend: object = None
    depth_backend_factory: object = None


def _rank(res: PhotoRoomResult) -> tuple:
    p = res.prop
    strong = sum(1 for w in p.walls if w.evidence_quality == "strong")
    area = max((r.area.value for r in p.rooms if r.id == res.primary_room_id), default=0.0)
    return (RANK.get(res.quality, 0), strong, round(area, 3))


def _placement_sigma(rooms: list[str], cons: list[RoomStitchConstraint], anchor: str) -> dict[str, float]:
    """Placement uncertainty of every room relative to the anchor, accumulated along the best constraints (m)."""
    sig = {anchor: 0.0}
    order = sorted(cons, key=lambda c: (c.translation_uncertainty_m, c.source_room_id, c.target_room_id))
    changed = True
    while changed:
        changed = False
        for c in order:
            a, b = c.source_room_id, c.target_room_id
            for x, y in ((a, b), (b, a)):
                if x in sig and y not in sig:
                    dist = math.hypot(c.translation_x_m, c.translation_z_m)
                    sig[y] = math.sqrt(sig[x] ** 2 + c.translation_uncertainty_m ** 2 +
                                       (math.radians(c.rotation_uncertainty_deg) * dist) ** 2)
                    changed = True
    return sig


def _plan_png(prop: Property, path: Path, title: str) -> None:
    try:
        render_plan(prop, path, title)
    except Exception:  # a diagnostic picture must never break the run
        pass


def run_photo_stages(capture: str | Path, opts: PhotoOptions) -> StageOutputs:
    t_all = time.perf_counter()
    capture = Path(capture)
    stages: list[StageRecord] = []
    warnings: list[str] = []
    decisions: list[dict] = []
    out_dir = Path(opts.output_dir) if opts.output_dir else Path(tempfile.mkdtemp(prefix="sf_photos_"))
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="sf_photo_sfm_"))
    stitch = opts.stitch
    try:
        # ---- 1. validation (essential) ----
        with StageTimer(stages, "photo_validation") as rec:
            val = validate_photo_dir(capture)
            rec.warnings = val.all_warnings()
            rec.details = {"folders": len(val.rooms), "images_per_folder": {r.source_label: len(r.images) for r in val.rooms}}
        write_json(diag / "photo_frontend.json", {"validation": val.to_dict()})
        if not val.ok:
            raise PipelineFailure("photo_validation", "; ".join(val.all_errors()), stages=stages,
                                  extra={"photos": {"validation": val.to_dict()}, "metric_scale_available": False})
        warnings += [f"input: {w}" for w in val.all_warnings()]

        # ---- 2. features + exhaustive matching, once for the whole property (essential) ----
        backend = opts.sfm_backend or ColmapPhotoBackend(workspace, opts.sfm)
        try:
            with StageTimer(stages, "features_and_exhaustive_matching") as rec:
                records = backend.prepare(val.rooms)
                rec.details = {"images": len(records), **getattr(backend, "seconds", {})}
        except Exception as exc:
            raise PipelineFailure("features_and_exhaustive_matching", f"{type(exc).__name__}: {exc}", stages=stages,
                                  extra={"photos": {"validation": val.to_dict()}}) from exc
        pair_stats = backend.pair_stats()
        room_of = {n: r.room_id for n, r in records.items()}

        # ---- 3. metric depth backend (cached per run) ----
        depth_backend, depth_err = None, None
        try:
            if opts.depth_backend is not None:
                depth_backend = opts.depth_backend
            elif opts.depth_backend_factory is not None:
                depth_backend = opts.depth_backend_factory()
            else:
                from spatialforge.video.depth import DepthAnythingMetric

                depth_backend = DepthAnythingMetric()
            if not depth_backend.metric:
                raise DepthUnavailable("the depth backend does not produce metric depth")
        except DepthUnavailable as exc:
            depth_backend, depth_err = None, str(exc)
            warnings.append(f"Metric depth is unavailable ({exc}); no room can be given metric dimensions.")
        depth_cache: dict = {}

        def depth_for(name: str):
            if depth_backend is None:
                raise DepthUnavailable(depth_err or "no metric depth backend")
            if name not in depth_cache:
                loader = getattr(backend, "load_rgb", None)
                if loader is not None:
                    rgb = loader(name)
                else:
                    import cv2

                    rgb = cv2.cvtColor(cv2.imread(records[name].work_path), cv2.COLOR_BGR2RGB)
                depth_cache[name] = depth_backend.predict(rgb)
            return depth_cache[name]

        # ---- 4. one local metric reconstruction per room folder ----
        results: list[PhotoRoomResult] = []
        runs: dict = {}
        for k, room in enumerate(val.rooms):
            names = sorted(n for n, r in records.items() if r.room_id == room.canonical_id)
            t0 = time.perf_counter()
            try:
                with StageTimer(stages, f"{room.canonical_id}_reconstruction") as rec:
                    run = backend.map_subset(names, room.canonical_id)
                    runs[room.canonical_id] = run
                    try:
                        res = reconstruct_room(room, k, run, depth_for, opts.room)
                    except DepthUnavailable as exc:
                        msg = f"No metric depth backend: {exc}. No metric dimensions are reported for this room."
                        res = failed_result(k, room, names, "unscaled", "failure", [msg], run.result.to_dict(), None, t0, [], [msg])
                    rec.details = {"status": res.status, "quality": res.quality, "registered": run.result.registered_images,
                                   "images": len(names)}
                    rec.warnings = [w for w in res.warnings if "Metric scale" in w or "no metric" in w.lower()][:2]
            except Exception as exc:
                msg = f"Reconstruction failed: {type(exc).__name__}: {exc}"
                res = failed_result(k, room, names, "unreconstructed", "failure", [msg], None, None, t0, [], [msg])
            results.append(res)
        by_id = {r.canonical_id: r for r in results}
        usable = [r for r in results if r.usable]

        # ---- 5. cross-room evidence, pair reconstructions, constraints ----
        geoms = {r.canonical_id: local_geom(r.canonical_id, r.prop, r.primary_room_id, _rank(r)) for r in usable}
        constraints: list[RoomStitchConstraint] = []
        pair_records: list[dict] = []
        scale_obs: dict[tuple[str, str], float] = {}
        with StageTimer(stages, "cross_room_stitching") as rec:
            evidences = cross_room_evidence(pair_stats, room_of, stitch)
            for ev in evidences:
                a, b = ev.room_a, ev.room_b
                rec_d = {"evidence": ev.to_dict(), "stitch": {"status": "not_attempted", "reason": ""}}
                ua, ub = by_id[a].usable, by_id[b].usable
                if not ev.accepted:
                    rec_d["stitch"]["reason"] = "cross-room evidence below thresholds: " + ev.reason
                elif not (ua and ub):
                    rec_d["stitch"]["reason"] = "a room of this pair has no usable local reconstruction"
                else:
                    try:
                        joint = backend.map_subset(sorted(by_id[a].poses) + sorted(by_id[b].poses), f"pair_{a}_{b}")
                        tf, why = visual_pair_transform(by_id[a].poses, by_id[b].poses, joint.poses, stitch)
                    except Exception as exc:
                        tf, why = None, f"joint reconstruction failed: {type(exc).__name__}: {exc}"
                    if tf is None:
                        rec_d["stitch"] = {"status": "rejected", "reason": why}
                    else:
                        c = build_visual_constraint(ev, tf, geoms[a], geoms[b], stitch)
                        constraints.append(c)
                        if tf.log_scale_a_over_b is not None:
                            scale_obs[(a, b)] = tf.log_scale_a_over_b
                        rec_d["stitch"] = {"status": "constraint", "evidence_type": c.evidence_type, "quality": c.quality,
                                           "transform": tf.to_dict()}
                pair_records.append(rec_d)
            if opts.allow_doorway_only:  # weak, only to join components that have no image evidence, and only if unique
                ids = sorted(g for g in geoms)
                comp_of = {r: i for i, comp in enumerate(components(ids, constraints)) for r in comp}
                for i, a in enumerate(ids):
                    for b in ids[i + 1:]:
                        if comp_of[a] == comp_of[b] or not (geoms[a].openings and geoms[b].openings):
                            continue
                        c, why = build_doorway_constraint(geoms[a], geoms[b], stitch)
                        rd = {"evidence": None, "stitch": {"status": "constraint" if c else "rejected", "evidence_type": "doorway",
                                                            "reason": why, "rooms": [a, b]}}
                        if c is not None:
                            constraints.append(c)
                            old, new = comp_of[b], comp_of[a]
                            comp_of = {r: (new if v == old else v) for r, v in comp_of.items()}
                            rd["stitch"].update(quality=c.quality)
                        pair_records.append(rd)
            rec.details = {"room_pairs": len(evidences), "accepted_evidence": sum(1 for e in evidences if e.accepted),
                           "constraints": len(constraints)}

        # ---- 6. global placement ----
        placements: list[ComponentPlacement] = []
        remaining = sorted(r.canonical_id for r in usable)
        with StageTimer(stages, "global_placement") as rec:
            while remaining:
                comp = components(remaining, constraints)[0]
                pl = place_component(comp, constraints, geoms, stitch)
                placements.append(pl)
                remaining = [r for r in remaining if r not in pl.rooms]
            rec.details = {"components": len(placements), "constraints_rejected": sum(len(p.rejected) for p in placements)}
        main = max(placements, key=lambda p: (len(p.rooms), tuple(geoms[p.anchor].rank), ), default=None)
        placed_ids = [] if main is None else sorted(main.rooms)
        all_constraints = constraints
        rejected = [c for c in all_constraints if c.status == "rejected"]
        overlaps = [] if main is None else overlap_report(placed_ids, main.poses, geoms, stitch)
        loops = [] if main is None else main.loops

        # ---- 7. merge into the canonical property ----
        prop, adjacency, scale_report = _build_property(
            capture, val, backend, depth_backend, results, by_id, main, geoms, scale_obs, overlaps, loops, rejected,
            pair_records, pair_stats, placements, stages, warnings, runs, t_all)
        decisions.append({"stage": "stitching", "decision": f"{len(placed_ids)} of {len(results)} room folders placed in one property",
                          "reason": "; ".join(main.actions) if main and main.actions else "no placement conflicts"})

        # ---- 8. diagnostics (never fatal) ----
        problems = _write_diagnostics(diag, val, results, usable, geoms, constraints, pair_records, placements, main, overlaps,
                                      loops, adjacency, scale_report, stitch)
        warnings += problems
        extra = {"photos": {
            "folders": len(results), "rooms": [r.report() for r in results], "placed_rooms": placed_ids,
            "unplaced_rooms": prop.provenance["photos"]["unplaced_rooms"], "constraints": [c.to_dict() for c in constraints],
            "loops": loops, "overlaps": overlaps, "scale_consistency": scale_report, "adjacency": adjacency},
            "metric_scale_available": any(r.usable for r in results), "tracking_quality": "per-room"}
        out = StageOutputs(
            capture_path=str(capture), capture_name=capture.name, tier="photos", validation_warnings=[], pose_source="sfm_scaled",
            drift={}, floor_y_m=None, ceiling_levels=[], wall_dicts=[], topology=None, openings=None, stage_warnings=[],
            stages=stages, device={"model": None}, frames={"folders": len(results), "images": len(records)},
            decisions=decisions, prebuilt_property=prop, parameters={"stitch": stitch.__dict__},
            debug={"run_report_extra": extra})
        return out
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ---------------- property assembly ----------------


def _build_property(capture, val, backend, depth_backend, results, by_id, main, geoms, scale_obs, overlaps, loops, rejected,
                    pair_records, pair_stats, placements, stages, warnings, runs, t_all):
    placed_ids = [] if main is None else sorted(main.rooms)
    k = 2.0
    # --- cross-room scale consistency (comparable numbers: each room's scale as seen in the joint models) ---
    pairs = [(a, b, v) for (a, b), v in scale_obs.items() if a in placed_ids and b in placed_ids]
    offsets = scale_offsets(pairs)
    scale_report = {"method": "signed ln(s_A/s_B) of the two rooms' metric scales seen in each pair's joint reconstruction, "
                              "solved by least squares (offsets sum to zero)",
                    "rooms": {}, "pairs": [{"room_a": a, "room_b": b, "ln_scale_ratio": round(v, 4)} for a, b, v in pairs],
                    "scale_inconsistent": False, "regularisation_applied": False}
    for rid in placed_ids:
        s = by_id[rid].scale or {}
        u = offsets.get(rid)
        dev = None if u is None else abs(math.exp(u) - 1.0)
        scale_report["rooms"][rid] = {"metres_per_sfm_unit_room_model": s.get("metres_per_sfm_unit"), "quality": s.get("scale_quality"),
                                      "sigma_rel": s.get("sigma_rel"), "log_offset_vs_property_median": None if u is None else round(u, 4),
                                      "deviation_vs_property_median": None if dev is None else round(dev, 4)}
    inconsistent = [r for r, v in scale_report["rooms"].items() if v["deviation_vs_property_median"] is not None
                    and v["deviation_vs_property_median"] > SCALE_INCONSISTENT]
    scale_report["scale_inconsistent"] = bool(inconsistent)
    if inconsistent:
        warnings.append("scale_inconsistent: the metric scales of " + ", ".join(inconsistent) + f" differ from the property median by more than "
                        f"{SCALE_INCONSISTENT:.0%}; they are NOT forced equal and their uncertainty is widened.")
    # --- extra relative uncertainty per placed room (scale inconsistency, loop residual, overlap conflict) ---
    sigma_pos = {} if main is None else _placement_sigma(placed_ids, main.constraints, main.anchor)
    loop_rel = 0.0
    if loops:
        loop_rel = min(0.1, max(l["translation_residual_m"] for l in loops) / 5.0)
    placed: list[PlacedRoom] = []
    for rid in placed_ids:
        res = by_id[rid]
        u = offsets.get(rid)
        scale_term = abs(u) if (u is not None and abs(math.exp(u) - 1.0) > SCALE_INCONSISTENT) else 0.0
        ov = max([o["fraction_of_smaller"] for o in overlaps if rid in (o["room_a"], o["room_b"])], default=0.0)
        overlap_term = 0.2 * ov if ov > 0.02 else 0.0
        extra = float(np.sqrt(scale_term ** 2 + loop_rel ** 2 + overlap_term ** 2))
        placed.append(PlacedRoom(rid, res.source_label, res.prop, res.primary_room_id, main.poses[rid], extra,
                                 sigma_pos.get(rid, 0.0), res.quality))
        res.uncertainty = {**(res.uncertainty or {}), "scale_inconsistency_term": round(scale_term, 4),
                           "loop_term": round(loop_rel, 4), "overlap_term": round(overlap_term, 4),
                           "placement_sigma_m": round(sigma_pos.get(rid, 0.0), 4), "extra_relative_sigma": round(extra, 4)}

    rooms, walls, openings, unverified = [], [], [], []
    for pr in placed:
        r, w, o, u, _ = transform_room(pr, "photos", k)
        if r is not None:
            rooms.append(r)
        walls += w
        openings += o
        unverified += u
    room_ids = {r.id for r in rooms}
    # a wall between two rooms is reconstructed from both sides: keep one
    walls, merged_walls = dedupe_walls(rooms, walls, openings, unverified)

    # --- adjacency and doorway connections between placed rooms ---
    adjacency, merged_into = [], {}
    poly = {r.id: np.array([[p.x, p.z] for p in r.polygon]) for r in rooms}
    rooms_by_id = {r.id: r for r in rooms}
    ops_by_id = {o.id: o for o in openings}
    rooms_sorted = sorted(room_ids)
    con_between = {tuple(sorted((c.source_room_id, c.target_room_id))): c for c in (main.constraints if main else [])}
    for i, a in enumerate(rooms_sorted):
        for b in rooms_sorted[i + 1:]:
            pose_ba = compose(invert(main.poses[a]), main.poses[b])
            doors = match_doorways(geoms[a], geoms[b], pose_ba, StitchOptions())
            geometric = adjacent_edges(poly[a], poly[b])
            c = con_between.get((a, b))
            if not (doors or geometric or c):
                continue
            entry = {"rooms": [a, b], "kind": None, "quality": None, "via_openings": [], "constraint_evidence": None if c is None else c.evidence_type}
            if doors:
                m = doors[0]
                oa, ob = f"{a}_{m.opening_a}", f"{b}_{m.opening_b}"
                keep, drop = ops_by_id.get(oa), ops_by_id.get(ob)
                if keep is not None and drop is not None:
                    if drop.existence_quality == "moderate" and keep.existence_quality != "moderate":
                        keep, drop = drop, keep
                    keep.room_ids, keep.connects, keep.connected_room_ids = [a, b], [a, b], [a, b]
                    for rm in (rooms_by_id[a], rooms_by_id[b]):
                        if drop.id in rm.opening_ids:
                            rm.opening_ids.remove(drop.id)
                        if drop.id in rm.opens_to_unmodelled:
                            rm.opens_to_unmodelled.remove(drop.id)
                        if keep.id not in rm.opening_ids:
                            rm.opening_ids.append(keep.id)
                        if keep.id in rm.opens_to_unmodelled:
                            rm.opens_to_unmodelled.remove(keep.id)
                    for w in walls:
                        if drop.id in w.opening_ids:
                            w.opening_ids.remove(drop.id)
                    openings = [o for o in openings if o.id != drop.id]
                    ops_by_id.pop(drop.id, None)
                    merged_into[drop.id] = keep.id
                    rooms_by_id[a].connected_room_ids.append(b)
                    rooms_by_id[b].connected_room_ids.append(a)
                    entry.update(kind="verified_connection", via_openings=[keep.id],
                                 quality="strong" if (c is not None and c.quality == "strong") else "moderate",
                                 note=f"the same doorway is seen from both rooms ({m.opening_a} / {m.opening_b}, {m.distance_m:.2f} m apart after placement)")
            if entry["kind"] is None:
                if geometric:
                    rooms_by_id[a].adjacent_room_ids.append(b)
                    rooms_by_id[b].adjacent_room_ids.append(a)
                    entry.update(kind="geometric_adjacency", quality=("moderate" if c is not None and c.quality != "weak" else "weak"))
                else:
                    entry.update(kind="probable_adjacency_visual_only" if c is not None and "visual" in c.evidence_type else "probable_adjacency_doorway_only",
                                 quality="weak" if c is None or c.quality == "weak" else "moderate",
                                 note="placed next to each other by stitch evidence, but the room outlines are not geometrically adjacent")
            else:
                for x, y in ((a, b), (b, a)):
                    if y not in rooms_by_id[x].adjacent_room_ids:
                        rooms_by_id[x].adjacent_room_ids.append(y)
            adjacency.append(entry)
    openings, merged_ops = dedupe_openings(rooms, walls, openings, unverified)
    merged_into.update(merged_ops)
    for rm in rooms:
        rm.adjacent_room_ids = sorted(set(rm.adjacent_room_ids))
        rm.connected_room_ids = sorted(set(rm.connected_room_ids))

    # --- footprint of the placed rooms only ---
    footprint = None
    if rooms:
        fp = union_footprint([poly[r] for r in rooms_sorted])
        if fp is not None:
            pts, area = fp
            perim = sum(math.hypot(pts[i].x - pts[(i + 1) % len(pts)].x, pts[i].z - pts[(i + 1) % len(pts)].z) for i in range(len(pts)))
            mean_rel = float(np.mean([by_id[r].rel_sigma or 0.15 for r in rooms_sorted]))
            stitch_sigma = float(np.sqrt(np.mean([sigma_pos.get(r, 0.0) ** 2 for r in rooms_sorted])))
            try:
                footprint = Footprint(pts, footprint_measurement(area, perim, mean_rel, stitch_sigma, "photos", k),
                                      "placed_rooms_union", False,
                                      "Outline of the union of the successfully placed rooms. Rooms that could not be placed are "
                                      "excluded, so this is NOT a verified property footprint.")
            except ModelError:
                footprint = None
    # --- unplaced rooms ---
    unplaced = []
    placed_set = set(placed_ids)
    for r in results:
        if r.canonical_id in placed_set:
            continue
        if r.status != "reconstructed":
            why = "; ".join(r.reasons[:2]) or r.status
        else:
            rej = [c for c in rejected if r.canonical_id in (c.source_room_id, c.target_room_id)]
            why = ("no stitch evidence connects it to the placed rooms" if not rej
                   else "its stitch constraint was rejected: " + "; ".join(c.notes[-1] for c in rej))
        unplaced.append({"canonical_id": r.canonical_id, "source_label": r.source_label, "status": r.status if r.status != "reconstructed" else "unplaced",
                         "quality": r.quality, "reason": why})
    for u in unplaced:
        warnings.append(f"{u['canonical_id']} ({u['source_label']}) is not part of the property plan: {u['reason']}.")
    for r in results:
        for w in r.ambiguity:
            warnings.append(f"{r.canonical_id}: {w}")
    for c in rejected:
        warnings.append(f"Stitch constraint {c.source_room_id}->{c.target_room_id} was rejected: {c.notes[-1] if c.notes else 'inconsistent'}")
    weak_edges = [c for c in (main.constraints if main else []) if c.quality == "weak"]
    if weak_edges:
        warnings.append(f"{len(weak_edges)} room placement(s) rest on weak evidence ("
                        + ", ".join(f"{c.source_room_id}->{c.target_room_id} {c.evidence_type}" for c in weak_edges) + ").")
    if overlaps and max(o["fraction_of_smaller"] for o in overlaps) > 0.05:
        worst = max(overlaps, key=lambda o: o["fraction_of_smaller"])
        warnings.append(f"Placed rooms {worst['room_a']} and {worst['room_b']} overlap by {worst['fraction_of_smaller']:.0%} of the smaller room.")
    for rid in placed_ids:
        if by_id[rid].quality in ("weak",):
            warnings.append(f"{rid}: local reconstruction quality is weak ({'; '.join(by_id[rid].reasons[-2:])}).")
    rels = [by_id[r].rel_sigma for r in placed_ids if by_id[r].rel_sigma]
    if rels:
        warnings.append(f"Photo tier: metric scale is estimated from monocular imagery per room; dimensions carry a relative 1-sigma of "
                        f"about {min(rels):.0%}-{max(rels):.0%} (plus stitching terms) and are less certain than video or LiDAR.")
    if not rooms:
        warnings.append("No closed rooms were recovered; only structural wall segments are reported.")
    warnings.append("Opening recall has not been verified: openings in walls that were not detected, or in unclosed regions, are not reported.")
    warnings.append("No physical ground truth was supplied: dimensions are not benchmarked, and intervals are diagnostic uncertainty "
                    "ranges, not calibrated confidence intervals.")
    warnings.append("Damage detection is not part of this version: damage, concealed_damage_flags and scope_line_items are empty.")

    # --- status: complete only if the evidence supports a coherent property ---
    all_placed = len(placed_ids) == len(results)
    coherent = (all_placed and len(rooms) == len(results) and not rejected and not unplaced and not inconsistent
                and all(by_id[r].quality in ("strong", "moderate") for r in placed_ids)
                and all(c.quality != "weak" for c in (main.constraints if main else []))
                and (not overlaps or max(o["fraction_of_smaller"] for o in overlaps) <= 0.05) and footprint is not None)
    status = "complete" if coherent else "partial"

    imgs = {r.canonical_id: len(r.image_names) for r in results}
    meta_counts = {"images": sum(len(r.images) for r in val.rooms), "with_exif": sum(1 for r in val.rooms for i in r.images if i.has_exif),
                   "with_35mm_focal": sum(1 for r in val.rooms for i in r.images if i.focal_35mm), "devices": sorted(
                       {f"{i.make or '?'} {i.model or '?'}" for r in val.rooms for i in r.images if i.has_exif})}
    provenance = {
        "pipeline": {"spatialforge_version": __version__, "schema_version": "1.0", "tier": "photos"},
        "stages": [{"name": s.name, "status": s.status} for s in stages],
        "photos": {
            "folders": len(results), "images_per_folder": {f"{r.canonical_id} ({r.source_label})": imgs[r.canonical_id] for r in results},
            "metadata_availability": meta_counts,
            "sfm": backend.describe() if hasattr(backend, "describe") else {},
            "metric_depth": depth_backend.describe() if depth_backend is not None else None,
            "rooms": [r.report(timing=False) for r in results], "scale_consistency": scale_report,
            "stitching": {"constraints": [c.to_dict() for c in (main.constraints if main else [])] + [c.to_dict() for c in rejected],
                          "pair_evidence": [{k2: v for k2, v in p["evidence"].items() if k2 != "image_pairs"} | {"stitch": p["stitch"]}
                                            for p in pair_records if p["evidence"] is not None],
                          "components": [{"rooms": sorted(p.rooms), "anchor": p.anchor} for p in placements],
                          "loops": loops, "overlaps": overlaps, "actions": [] if main is None else main.actions},
            "placed_rooms": placed_ids, "unplaced_rooms": unplaced, "adjacency": adjacency, "merged_duplicate_openings": merged_into,
            "merged_duplicate_walls": merged_walls,
            "source_labels": {r.canonical_id: r.source_label for r in results},
        },
        "coordinate_frame": "global plan frame of the anchor room (the strongest reconstructed room): gravity-aligned, metres, "
                            "arbitrary yaw; not georeferenced",
        "assumptions": [
            "Metric scale is estimated per room from monocular metric depth against SfM, never assumed; no typical room size, door "
            "width or ceiling height is used anywhere.",
            "Rooms are placed only from cross-room image matches verified by geometry and/or a unique doorway match.",
            "Room folder names are kept as source labels and never interpreted (no room-type inference).",
            "Intervals are engineering uncertainty ranges, not calibrated confidence intervals.",
        ],
    }
    prop = Property(
        capture={"tier": "photos", "source": {"name": Path(capture).name, "path": str(capture)}, "device": {"model": None},
                 "photos": {"folders": len(results), "images_per_folder": imgs, "metadata": meta_counts}},
        status=status, rooms=rooms, walls=walls, openings=openings, unverified_openings=unverified, footprint=footprint,
        warnings=warnings, provenance=provenance,
        timing={"stages": {s.name: round(s.seconds, 3) for s in stages}, "stages_total_s": round(sum(s.seconds for s in stages), 3)})
    prop.validate()
    return prop, adjacency, scale_report


# ---------------- diagnostics ----------------


def _write_diagnostics(diag: Path, val, results, usable, geoms, constraints, pair_records, placements, main, overlaps, loops,
                       adjacency, scale_report, stitch) -> list[str]:
    problems: list[str] = []
    try:
        write_json(diag / "photo_frontend.json", {"validation": val.to_dict(), "rooms": [r.report() for r in results],
                                                  "scale_consistency": scale_report})
        for r in results:
            d = diag / "rooms" / r.canonical_id
            write_json(d / "reconstruction.json", r.report())
            if r.sparse is not None and len(r.sparse):
                write_binary_ply(d / "sparse_sfm.ply", r.sparse, comment="metres, room-local gravity-aligned frame")
            if r.cloud is not None and len(r.cloud):
                write_binary_ply(d / "metric_cloud.ply", r.cloud, comment="metres, room-local gravity-aligned frame")
            if r.prop is not None:
                _plan_png(r.prop, d / "local_plan.png", f"{r.canonical_id} ({r.source_label}) — local reconstruction")
        s = diag / "stitching"
        ids = sorted(g for g in geoms)
        write_json(s / "room_graph.json", {
            "nodes": [{"id": r.canonical_id, "source_label": r.source_label, "status": r.status, "quality": r.quality} for r in results],
            "edges": [{"source": c.source_room_id, "target": c.target_room_id, "evidence_type": c.evidence_type, "quality": c.quality,
                       "status": c.status} for c in constraints],
            "components": [{"rooms": sorted(p.rooms), "anchor": p.anchor} for p in placements]})
        write_json(s / "stitch_constraints.json", {"constraints": [c.to_dict() for c in constraints], "pair_records": pair_records,
                                                   "actions": [] if main is None else main.actions})
        write_json(s / "overlap_report.json", {"overlaps": overlaps, "loops": loops, "adjacency": adjacency,
                                               "thresholds": {"max_fraction_of_smaller": stitch.overlap_max_fraction}})

        def draw(poses, name, title):
            rooms_d, edges = [], []
            for rid in sorted(poses):
                g, res = geoms[rid], next(x for x in results if x.canonical_id == rid)
                p = poses[rid]
                walls = [(apply(p, np.array([[sg.start.x, sg.start.z]]))[0], apply(p, np.array([[sg.end.x, sg.end.z]]))[0])
                         for w in res.prop.walls if w.evidence_quality != "weak" for sg in w.segments]
                cams = apply(p, np.array([[T[0, 3], T[2, 3]] for T in res.poses.values()]))
                rooms_d.append({"id": rid, "polygon": None if g.polygon is None else apply(p, g.polygon), "walls": walls, "cameras": cams})
            for c in constraints:
                if c.source_room_id in poses and c.target_room_id in poses:
                    edges.append({"a": c.source_room_id, "b": c.target_room_id, "quality": c.quality, "status": c.status})
            unplaced = [r.canonical_id for r in results if r.canonical_id not in poses]
            render_stitching_png(s / name, rooms_d, edges, title, [("Not placed: " + ", ".join(unplaced)) if unplaced else "All reconstructed rooms placed."])

        if main is not None:
            draw(main.initial_poses, "stitching_initial.png", "Stitching: initial placement (constraints composed from the anchor)")
            draw(main.poses, "stitching_final.png", "Stitching: final placement (robust least squares, loop/overlap checks)")
        else:
            for name in ("stitching_initial.png", "stitching_final.png"):
                render_stitching_png(s / name, [], [], "Stitching: no room could be placed", [])
    except Exception as exc:  # diagnostics are optional
        problems.append(f"Diagnostics could not be fully written: {type(exc).__name__}: {exc}")
    return problems
