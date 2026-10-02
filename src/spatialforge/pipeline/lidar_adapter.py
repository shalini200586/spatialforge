"""Map LiDAR stage outputs onto the canonical (sensor-independent) property model.

Policy: never fabricate. A stage that provides no numeric confidence gives `confidence = None` plus its quality
label; intervals are the stage's own uncertainty ranges; missing optional values stay null; anything incomplete is
said so in `warnings` and in `property.status = "partial"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spatialforge import __version__
from spatialforge.pipeline.models import (
    UNMODELLED, Ceiling, Footprint, Measurement, ModelError, Opening, Point2D, Property, Room, Segment, Wall,
)


@dataclass
class StageRecord:
    name: str
    status: str  # ok | warning | failed | skipped
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    details: dict = field(default_factory=dict)


@dataclass
class StageOutputs:
    """Plain results of the LiDAR stages (what the adapter needs; heavy objects go in `debug`)."""

    capture_path: str
    capture_name: str
    device: dict
    validation_warnings: list[str]
    frames: dict  # used, available, step, max
    pose_source: str  # original | corrected
    drift: dict  # accepted, fallback_reason, ...
    floor_y_m: float | None
    ceiling_levels: list[dict]  # height_m, interval_m, confidence
    wall_dicts: list[dict]  # Ticket 5 wall records
    topology: object | None  # rooms.TopologyResult
    openings: object | None  # openings.OpeningResult
    stage_warnings: list[str]
    stages: list[StageRecord]
    parameters: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)  # heavy objects for diagnostics only


def _m(value, unit, interval=None, confidence=None, quality=None) -> Measurement:
    low = high = None
    if interval is not None:
        low, high = min(interval[0], value), max(interval[1], value)  # an interval always contains its own value
    return Measurement(float(value), unit, low, high, confidence, quality, "lidar")


def _pt(p) -> Point2D:
    return Point2D(float(p[0]), float(p[1]))


def _walls(out: StageOutputs) -> list[Wall]:
    walls = []
    for d in out.wall_dicts:
        q = d["evidence"]
        o, ou = d["orientation_deg"], d["orientation_uncertainty_deg"]
        walls.append(Wall(
            id=d["id"], start=_pt(d["start"]), end=_pt(d["end"]),
            length=_m(d["length_m"], "m", quality=q), observed_length=_m(d["observed_length_m"], "m", quality=q),
            orientation=_m(o, "deg", (o - ou, o + ou), quality=q), evidence_quality=q,
            position_uncertainty=_m(d["position_uncertainty_m"], "m"),
            segments=[Segment(_pt(s["start"]), _pt(s["end"]), _m(s["length_m"], "m", quality=q)) for s in d["segments"]],
        ))
    return walls


def assemble_property(out: StageOutputs) -> Property:
    warnings: list[str] = [f"capture: {w}" for w in out.validation_warnings] + list(out.stage_warnings)
    walls = _walls(out)
    topo, opening_result = out.topology, out.openings
    conf_by_height = {round(l["height_m"], 3): l.get("confidence") for l in out.ceiling_levels}

    rooms: list[Room] = []
    if topo is not None:
        for r in topo.rooms:
            try:
                rooms.append(_room(r, conf_by_height, opening_result))
            except ModelError as exc:
                warnings.append(f"{r.id} was dropped from the output: {exc}")
    room_ids = {r.id for r in rooms}

    # openings: accepted ones are product geometry; low-confidence ones are listed separately and not drawn
    id_map: dict[str, str] = {}
    openings: list[Opening] = []
    unverified: list[Opening] = []
    if opening_result is not None:
        for i, o in enumerate(opening_result.accepted, 1):
            id_map[o.id] = f"opening_{i:03d}"
        for o in opening_result.accepted:
            openings.append(_opening(o, id_map[o.id], room_ids))
        for i, o in enumerate(opening_result.low_confidence, 1):
            unverified.append(_opening(o, f"unverified_{i:03d}", room_ids))
    for room in rooms:
        conn = opening_result.connectivity.get(room.id, {}) if opening_result is not None else {}
        room.connected_room_ids = [x for x in conn.get("connected_room_ids", []) if x in room_ids]
        room.opening_ids = [id_map[x] for x in conn.get("opening_ids", []) if x in id_map]
        room.opens_to_unmodelled = [id_map[x] for x in conn.get("opens_to_unmodelled", []) if x in id_map]
    for w in walls:
        w.room_ids = [r.id for r in rooms if w.id in r.wall_ids]
        w.opening_ids = [o.id for o in openings if o.wall_id == w.id]

    # honest completeness assessment
    structural = [w for w in walls if w.evidence_quality in ("strong", "moderate")]
    loose = [w for w in structural if not w.room_ids]
    if not rooms:
        warnings.append("No closed rooms were recovered; only structural wall segments are reported.")
    if loose and rooms:
        warnings.append(f"Incomplete room topology: {len(loose)} of {len(structural)} structural walls are not part of "
                        "any recovered room; unclosed wall regions remain.")
    elif loose:
        warnings.append(f"{len(loose)} structural walls do not form any closed room.")
    if rooms:
        no_ceiling = [r.id for r in rooms if not r.ceiling.observed]
        if no_ceiling:
            warnings.append(f"Ceiling height was not observed for {len(no_ceiling)} of {len(rooms)} rooms.")
    warnings.append("Opening recall has not been verified: openings in walls that were not detected, or in unclosed "
                    "regions, are not reported.")
    if unverified:
        warnings.append(f"{len(unverified)} low-confidence opening candidate(s) are excluded from the plan and listed "
                        "under unverified_openings.")
    warnings.append("No physical ground truth was supplied: dimensions are not benchmarked, and intervals are "
                    "diagnostic uncertainty ranges, not calibrated confidence intervals.")
    warnings.append("Damage detection is not part of this version: damage, concealed_damage_flags and "
                    "scope_line_items are empty.")
    failed = [s.name for s in out.stages if s.status == "failed"]
    status = "complete" if (rooms and not loose and not failed) else "partial"
    if failed:
        warnings.append("Stage(s) failed and were skipped: " + ", ".join(failed))

    footprint = None
    if topo is not None and topo.outer_boundary is not None:
        ob = topo.outer_boundary
        try:
            footprint = Footprint(
                [Point2D(p["x"], p["z"]) for p in ob["polygon"]], _m(ob["area_m2"], "m2", quality="weak"),
                "wall_graph_outer_boundary", False,
                "Outer face of the connected wall graph: an envelope of the recovered walls, NOT a verified property "
                "footprint.")
        except ModelError:
            footprint = None

    prop = Property(
        capture={"tier": "lidar", "source": {"name": out.capture_name, "path": out.capture_path}, "device": out.device},
        status=status, rooms=rooms, walls=walls, openings=openings, unverified_openings=unverified, footprint=footprint,
        warnings=warnings,
        provenance=_provenance(out),
        timing={"stages": {s.name: round(s.seconds, 3) for s in out.stages},
                "stages_total_s": round(sum(s.seconds for s in out.stages), 3)},
    )
    prop.validate()
    return prop


def _room(r, conf_by_height: dict, opening_result) -> Room:
    quality = r.topology_quality
    lens = [_m(v, "m", iv, quality=quality) for v, iv in zip(r.wall_lengths_m, r.wall_length_intervals_m)]
    per_lo = sum(min(a, v) for (a, _), v in zip(r.wall_length_intervals_m, r.wall_lengths_m))
    per_hi = sum(max(b, v) for (_, b), v in zip(r.wall_length_intervals_m, r.wall_lengths_m))
    c = r.ceiling
    if c.get("ceiling_observed"):
        h = c["ceiling_height_m"]
        ceiling = Ceiling(True, _m(h, "m", c.get("ceiling_height_interval_m"), conf_by_height.get(round(h, 3))),
                          False, c.get("coverage"))
    else:
        ceiling = Ceiling(False, None, bool(c.get("ambiguous")), None)
    return Room(
        id=r.id, polygon=[_pt(p) for p in r.polygon],
        area=_m(r.area_m2, "m2", r.area_interval_m2, quality=quality),
        perimeter=_m(r.perimeter_m, "m", (per_lo, per_hi), quality=quality),  # sum of per-edge extremes: conservative
        wall_ids=list(r.wall_ids), wall_lengths=lens, ceiling=ceiling, topology_quality=quality,
        length=None if r.length_m is None else _m(r.length_m, "m", r.length_interval_m, quality=quality),
        width=None if r.width_m is None else _m(r.width_m, "m", r.width_interval_m, quality=quality),
        dimension_method=r.dimension_method, adjacent_room_ids=[a["room_id"] for a in r.adjacent],
    )


def _opening(o, new_id: str, room_ids: set) -> Opening:
    lj, rj = o.left_jamb, o.right_jamb
    rooms = [r for r in o.room_ids if r in room_ids]
    connects = [c for c in o.connects if c == UNMODELLED or c in room_ids]
    return Opening(
        id=new_id, wall_id=o.candidate.wall_id, type=o.type,
        width=_m(o.width_m, "m", o.width_interval_m, quality=o.width_quality),
        position=Point2D((lj["x"] + rj["x"]) / 2, (lj["z"] + rj["z"]) / 2),
        left_jamb=Point2D(lj["x"], lj["z"]), right_jamb=Point2D(rj["x"], rj["z"]),
        existence_quality=o.existence_quality, type_quality=o.type_quality, observability=o.observability,
        height=None if o.height_m is None else _m(o.height_m, "m"),
        sill_height=None if o.sill_height_m is None else _m(o.sill_height_m, "m"),
        room_ids=rooms, connects=connects,
        connected_room_ids=rooms if len(rooms) == 2 and o.status == "accepted" else [],
    )


def _provenance(out: StageOutputs) -> dict:
    return {
        "pipeline": {"spatialforge_version": __version__, "schema_version": "1.0", "tier": "lidar"},
        "stages": [{"name": s.name, "status": s.status} for s in out.stages],
        "production_pose_source": out.pose_source,
        "pose_refinement": out.drift,
        "depth_scale_m_per_raw_unit": 0.001,
        "depth_scale_basis": "assumed 0.001 m per raw depth unit, supported by multi-frame consistency and indoor "
                             "scale, not proven by the data",
        "pose_convention": "camera-to-world, world +Y vertical",
        "coordinate_frame": "capture-native world frame, top-down X-Z, metres; not reoriented or georeferenced",
        "frames": out.frames,
        "parameters": out.parameters,
        "floor_height_y_m": out.floor_y_m,
        "ceiling_levels_m": [round(l["height_m"], 3) for l in out.ceiling_levels],
        "assumptions": [
            "Depth values are millimetres (0.001 m per unit).",
            "Camera intrinsics refer to the RGB image and are scaled proportionally to the depth image.",
            "Walls are vertical and floors horizontal in the capture's native frame (+Y is vertical).",
            "Room polygons close only where wall evidence connects; nothing is extended beyond 0.5 m to close a corner.",
            "Openings are structural gaps verified in 3D; door leaves, hardware and glazing are not detected.",
        ],
    }
