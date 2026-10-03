"""Place each room's local reconstruction into ONE canonical Property.

Canonical ids are deterministic from the sorted folder order: room_001, room_002, ... Walls and openings are prefixed with
their room (room_001_wall_003). Only the primary polygon of a folder becomes a Room; other local polygons are recorded as
ambiguity. Rooms that could not be placed are not drawn or counted: they are listed in provenance and warnings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatialforge.pipeline.lidar_adapter import MeasureContext
from spatialforge.pipeline.models import UNMODELLED, Ceiling, Measurement, Opening, Point2D, Property, Room, Segment, Wall
from spatialforge.photos.stitch import OpeningGeom, RoomGeom, apply


@dataclass
class PlacedRoom:
    canonical_id: str
    source_label: str
    prop: Property  # local property (local frame)
    primary_local_id: str | None
    pose: tuple  # local -> global (theta, tx, tz)
    extra_rel_sigma: float = 0.0  # stitching / scale-consistency / loop / overlap terms, relative
    position_sigma_m: float = 0.0  # placement uncertainty of this room relative to the anchor
    quality: str = "moderate"


def local_geom(canonical_id: str, prop: Property, primary_local_id: str | None, rank: tuple) -> RoomGeom:
    room = next((r for r in prop.rooms if r.id == primary_local_id), None)
    poly = None if room is None else np.array([[p.x, p.z] for p in room.polygon])
    ops = []
    for o in prop.openings:
        if o.type in ("door", "opening"):
            lw = None if o.width.low is None else o.width.low
            ops.append(OpeningGeom(o.id, o.wall_id, np.array([o.left_jamb.x, o.left_jamb.z]), np.array([o.right_jamb.x, o.right_jamb.z]),
                                   o.width.value, o.existence_quality, lw, o.width.high))
    pts = []
    for w in prop.walls:
        if w.evidence_quality in ("strong", "moderate"):
            pts += [[w.start.x, w.start.z], [w.end.x, w.end.z]]
    return RoomGeom(canonical_id, poly, ops, rank, np.array(pts) if pts else None)


def _pt(pose, p: Point2D) -> Point2D:
    q = apply(pose, np.array([[p.x, p.z]]))[0]
    return Point2D(float(q[0]), float(q[1]))


def _widen(ctx: MeasureContext, m: Measurement | None) -> Measurement | None:
    if m is None:
        return None
    iv = None if m.low is None else (m.low, m.high)
    return ctx.measure(m.value, m.unit, iv, m.confidence, m.quality)


def _rotate_orientation(m: Measurement, theta_deg: float) -> Measurement:
    v = (m.value + theta_deg) % 180.0
    shift = v - m.value
    return Measurement(v, m.unit, None if m.low is None else m.low + shift, None if m.high is None else m.high + shift,
                       m.confidence, m.quality, m.source_tier)


def transform_room(pr: PlacedRoom, tier: str = "photos", k: float = 2.0) -> tuple[Room | None, list[Wall], list[Opening], list[Opening], dict]:
    """The local geometry of one folder, placed globally with canonical ids and (optionally) widened intervals."""
    prop, pose = pr.prop, pr.pose
    ctx = MeasureContext(tier, pr.extra_rel_sigma, k) if pr.extra_rel_sigma > 0 else MeasureContext(tier, 0.0, k)
    theta_deg = math.degrees(pose[0])
    primary = next((r for r in prop.rooms if r.id == pr.primary_local_id), None)
    keep_walls = []
    for w in prop.walls:
        in_primary = primary is not None and w.id in primary.wall_ids
        if in_primary or w.evidence_quality in ("strong", "moderate"):
            keep_walls.append(w)
    wmap = {w.id: f"{pr.canonical_id}_{w.id}" for w in keep_walls}
    omap = {o.id: f"{pr.canonical_id}_{o.id}" for o in prop.openings + prop.unverified_openings if o.wall_id in wmap}
    pos_extra = pr.position_sigma_m

    def pos_unc(m):
        if pos_extra <= 0:
            return _widen(ctx, m)
        base = _widen(ctx, m)
        lo = base.value - math.hypot(base.value - (base.low if base.low is not None else base.value), k * pos_extra)
        hi = base.value + math.hypot((base.high if base.high is not None else base.value) - base.value, k * pos_extra)
        return Measurement(base.value, base.unit, lo, hi, base.confidence, base.quality, base.source_tier)

    walls = []
    for w in keep_walls:
        walls.append(Wall(
            id=wmap[w.id], start=_pt(pose, w.start), end=_pt(pose, w.end), length=_widen(ctx, w.length),
            observed_length=_widen(ctx, w.observed_length), orientation=_rotate_orientation(w.orientation, theta_deg),
            evidence_quality=w.evidence_quality, position_uncertainty=pos_unc(w.position_uncertainty),
            segments=[Segment(_pt(pose, s.start), _pt(pose, s.end), _widen(ctx, s.length)) for s in w.segments],
            room_ids=[pr.canonical_id] if primary is not None and w.id in primary.wall_ids else [],
            opening_ids=[omap[i] for i in w.opening_ids if i in omap]))

    def op(o: Opening) -> Opening:
        return Opening(
            id=omap[o.id], wall_id=wmap[o.wall_id], type=o.type, width=_widen(ctx, o.width), position=_pt(pose, o.position),
            left_jamb=_pt(pose, o.left_jamb), right_jamb=_pt(pose, o.right_jamb), existence_quality=o.existence_quality,
            type_quality=o.type_quality, observability=o.observability, height=_widen(ctx, o.height),
            sill_height=_widen(ctx, o.sill_height), room_ids=[pr.canonical_id] if primary is not None else [],
            connects=[UNMODELLED], connected_room_ids=[])

    openings = [op(o) for o in prop.openings if o.id in omap]
    unverified = [op(o) for o in prop.unverified_openings if o.id in omap]
    room = None
    if primary is not None:
        room = Room(
            id=pr.canonical_id, polygon=[_pt(pose, p) for p in primary.polygon], area=_widen(ctx, primary.area),
            perimeter=_widen(ctx, primary.perimeter), wall_ids=[wmap[i] for i in primary.wall_ids if i in wmap],
            wall_lengths=[_widen(ctx, m) for m in primary.wall_lengths], ceiling=Ceiling(
                primary.ceiling.observed, _widen(ctx, primary.ceiling.height), primary.ceiling.ambiguous, primary.ceiling.coverage),
            topology_quality=primary.topology_quality, length=_widen(ctx, primary.length), width=_widen(ctx, primary.width),
            dimension_method=primary.dimension_method,
            opening_ids=[omap[i] for i in primary.opening_ids if i in omap],
            opens_to_unmodelled=[omap[i] for i in primary.opens_to_unmodelled if i in omap])
        # an opening that leads out of this room into space nobody modelled stays "unmodelled" until a connection is found
        room.opening_ids = [o.id for o in openings]
        room.opens_to_unmodelled = [o.id for o in openings]
    return room, walls, openings, unverified, {"walls": wmap, "openings": omap}


# ---------------- cross-room scale consistency ----------------


def scale_offsets(pairs: list[tuple[str, str, float]]) -> dict[str, float]:
    """Least-squares log-scale offset per room from pairwise ln(s_A / s_B) observations (gauge: offsets sum to zero)."""
    rooms = sorted({r for a, b, _ in pairs for r in (a, b)})
    if not rooms:
        return {}
    idx = {r: i for i, r in enumerate(rooms)}
    A = np.zeros((len(pairs) + 1, len(rooms)))
    y = np.zeros(len(pairs) + 1)
    for k, (a, b, v) in enumerate(pairs):
        A[k, idx[a]], A[k, idx[b]], y[k] = 1.0, -1.0, v
    A[-1, :] = 1.0  # sum of offsets = 0
    u, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {r: float(u[idx[r]]) for r in rooms}


# ---------------- footprint ----------------


def union_footprint(polys: list[np.ndarray], cell: float = 0.05, simplify_m: float = 0.08) -> tuple[list[Point2D], float] | None:
    """Outline and area of the union of room polygons (rasterised; deterministic). None if there is nothing to outline."""
    import cv2

    from spatialforge.photos.stitch import raster_polygon

    if not polys:
        return None
    allp = np.vstack(polys)
    pad = 0.2
    bounds = (allp[:, 0].min() - pad, allp[:, 1].min() - pad, allp[:, 0].max() + pad, allp[:, 1].max() + pad)
    mask = np.zeros_like(raster_polygon(polys[0], bounds, cell))
    for p in polys:
        mask |= raster_polygon(p, bounds, cell)
    area = float(mask.sum()) * cell * cell
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, simplify_m / cell, True).reshape(-1, 2).astype(float)
    pts = [Point2D(bounds[0] + (x + 0.5) * cell, bounds[1] + (y + 0.5) * cell) for x, y in c]
    return pts, area


def footprint_measurement(area: float, perimeter: float, room_rel_sigma: float, stitch_sigma_m: float, tier: str, k: float = 2.0) -> Measurement:
    """Area interval: relative length error (twice, for an area) in quadrature with a boundary band of the stitch error."""
    h = math.hypot(k * 2.0 * room_rel_sigma * area, 0.5 * perimeter * k * stitch_sigma_m)
    return Measurement(area, "m2", max(0.0, area - h), area + h, None, "weak", tier)


# ---------------- duplicates across rooms ----------------


def _wall_vec(w: Wall) -> tuple[np.ndarray, np.ndarray, float]:
    a, b = np.array([w.start.x, w.start.z]), np.array([w.end.x, w.end.z])
    L = float(np.linalg.norm(b - a))
    return a, (b - a) / max(L, 1e-9), L


def _src_room(wall_id: str) -> str:
    return wall_id.split("_wall_")[0]


def dedupe_walls(rooms: list[Room], walls: list[Wall], openings: list[Opening], unverified: list[Opening],
                 max_angle_deg: float = 8.0, max_distance_m: float = 0.35, min_overlap_fraction: float = 0.4) -> tuple[list[Wall], dict]:
    """A wall between two rooms is reconstructed from both sides. Parallel walls of DIFFERENT rooms that lie within
    `max_distance_m` and overlap are the same wall: keep the stronger one, point every reference at it."""
    order = {"strong": 0, "moderate": 1, "weak": 2}
    rank = lambda w: (order[w.evidence_quality], 0 if w.room_ids else 1, -w.length.value, w.id)  # noqa: E731
    keep = sorted(walls, key=rank)
    removed: dict[str, str] = {}
    kept: list[Wall] = []
    for w in keep:
        a, d, L = _wall_vec(w)
        merged = False
        for k in kept:
            if _src_room(k.id) == _src_room(w.id):
                continue
            ka, kd, kL = _wall_vec(k)
            ang = math.degrees(math.acos(min(1.0, abs(float(d @ kd)))))
            if ang > max_angle_deg:
                continue
            n = np.array([-kd[1], kd[0]])
            mid = a + d * L / 2
            if abs(float((mid - ka) @ n)) > max_distance_m:
                continue
            t0, t1 = sorted([float((a - ka) @ kd), float((a + d * L - ka) @ kd)])
            overlap = min(kL, t1) - max(0.0, t0)
            if overlap >= min_overlap_fraction * min(L, kL):
                removed[w.id] = k.id
                merged = True
                for rid in w.room_ids:
                    if rid not in k.room_ids:
                        k.room_ids.append(rid)
                for oid in w.opening_ids:
                    if oid not in k.opening_ids:
                        k.opening_ids.append(oid)
                break
        if not merged:
            kept.append(w)
    for r in rooms:
        new = []
        for wid in r.wall_ids:
            wid = removed.get(wid, wid)
            if wid not in new:
                new.append(wid)
        r.wall_ids = new
    for o in openings + unverified:
        o.wall_id = removed.get(o.wall_id, o.wall_id)
    kept_ids = {w.id for w in kept}
    return [w for w in walls if w.id in kept_ids], removed


def dedupe_openings(rooms: list[Room], walls: list[Wall], openings: list[Opening], unverified: list[Opening],
                    max_distance_m: float = 0.5, max_angle_deg: float = 15.0, max_width_ratio: float = 1.35) -> tuple[list[Opening], dict]:
    """The same doorway seen from two rooms (or through another room's doorway) is one opening: keep the better observed one."""
    order = {"strong": 0, "moderate": 1, "weak": 2}
    rank = lambda o: (order[o.existence_quality], order[o.observability], o.id)  # noqa: E731
    kept: list[Opening] = []
    removed: dict[str, str] = {}
    for o in sorted(openings, key=rank):
        a = np.array([o.left_jamb.x, o.left_jamb.z])
        b = np.array([o.right_jamb.x, o.right_jamb.z])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        mid = (a + b) / 2
        dup = None
        for k in kept:
            ka, kb = np.array([k.left_jamb.x, k.left_jamb.z]), np.array([k.right_jamb.x, k.right_jamb.z])
            kd = (kb - ka) / max(np.linalg.norm(kb - ka), 1e-9)
            ratio = max(o.width.value, k.width.value) / max(min(o.width.value, k.width.value), 1e-9)
            if (np.linalg.norm(mid - (ka + kb) / 2) <= max_distance_m and ratio <= max_width_ratio
                    and math.degrees(math.acos(min(1.0, abs(float(d @ kd))))) <= max_angle_deg):
                dup = k
                break
        if dup is None:
            kept.append(o)
        else:
            removed[o.id] = dup.id
    for r in rooms:
        for lst in (r.opening_ids, r.opens_to_unmodelled):
            for i, oid in enumerate(list(lst)):
                if oid in removed:
                    lst[i] = removed[oid]
            seen: list[str] = []
            for oid in lst:
                if oid not in seen:
                    seen.append(oid)
            lst[:] = seen
    for w in walls:
        seen = []
        for oid in w.opening_ids:
            oid = removed.get(oid, oid)
            if oid not in seen:
                seen.append(oid)
        w.opening_ids = seen
    kept_ids = {o.id for o in kept}
    return [o for o in openings if o.id in kept_ids], removed
