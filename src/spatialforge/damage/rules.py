"""Concealed-damage RISK flags: explicit, rule-based, never a claim that hidden damage exists.

Each rule looks at VISIBLE detections whose position on a surface is known metrically and says "this pattern is commonly
associated with a hidden problem, so inspect". A flag always names its rule id and the exact condition that fired, lists the
visible detections that triggered it, and carries `claims_presence = False`. Rules that need a metric position are simply
not evaluated for detections without one (reported as such in the damage report), so they can never fire on a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from spatialforge.damage.geometry import Surface
from spatialforge.pipeline.models import DamageRegion

NOTE = ("Rule-based risk flag derived from visible evidence only. It does NOT assert that hidden damage is present; "
        "it identifies where an inspection is warranted.")
QUALITY_RANK = {"weak": 1, "moderate": 2, "strong": 3}


@dataclass
class RuleOptions:
    junction_m: float = 0.30  # "near the floor-wall / ceiling-wall junction"
    cluster_m: float = 1.5  # stains on one wall this close together
    opening_corner_m: float = 0.30
    long_crack_m: float = 1.0
    colocated_m: float = 0.30


RULES = {
    "CDR-001": ("stain_near_floor_wall_junction", "moisture"),
    "CDR-002": ("stain_near_ceiling_junction", "moisture"),
    "CDR-003": ("stain_cluster_on_wall", "moisture"),
    "CDR-004": ("crack_at_opening_corner", "structural_movement"),
    "CDR-005": ("long_crack", "structural_movement"),
    "CDR-006": ("crack_with_colocated_stain", "moisture"),
}


def _box(d: DamageRegion):
    e = d.surface_extent
    return e["u_min"], e["u_max"], e["v_min"], e["v_max"]


def _gap(a0, a1, b0, b1) -> float:
    return max(0.0, max(a0, b0) - min(a1, b1))


def _quality(ds: list[DamageRegion]) -> str:
    return {1: "weak", 2: "moderate", 3: "moderate"}[min(QUALITY_RANK[d.quality] for d in ds)]  # a risk flag is never "strong"


def evaluate_rules(regions: list[DamageRegion], surfaces: list[Surface], opts: RuleOptions | None = None) -> tuple[list[dict], dict]:
    """(flags, evaluation) where `evaluation` records, per rule, what was evaluated, triggered and skipped (and why)."""
    opts = opts or RuleOptions()
    by_surface = {s.id: s for s in surfaces}
    usable = [d for d in regions if d.metric_reliable and d.surface_extent and d.surface_id in by_surface]
    skipped = [d.id for d in regions if d not in usable]
    flags: list[dict] = []
    ev = {rid: {"rule": name, "risk": risk, "regions_evaluated": 0, "triggered": 0} for rid, (name, risk) in RULES.items()}

    def add(rid, explanation, ds, surface_id, room_id):
        name, risk = RULES[rid]
        flags.append({"flag_id": "", "rule_id": rid, "rule_name": name, "risk": risk, "explanation": explanation,
                      "damage_ids": sorted(d.id for d in ds), "surface_id": surface_id, "room_id": room_id,
                      "quality": _quality(ds), "claims_presence": False, "note": NOTE})
        ev[rid]["triggered"] += 1

    stains = [d for d in usable if d.damage_class == "stain"]
    cracks = [d for d in usable if d.damage_class == "crack"]
    for d in stains:
        s = by_surface[d.surface_id]
        u0, u1, v0, v1 = _box(d)
        if s.kind == "wall":
            ev["CDR-001"]["regions_evaluated"] += 1
            if v0 <= opts.junction_m:
                add("CDR-001", f"A stain on wall {d.surface_id} starts {max(v0, 0):.2f} m above the floor (rule: lowest edge <= "
                    f"{opts.junction_m:.2f} m). Staining at the floor-wall junction is commonly associated with moisture rising "
                    "from, or leaking into, the wall base, flooring or plumbing behind it.", [d], d.surface_id, d.room_id)
            if s.ceiling_height is not None:
                ev["CDR-002"]["regions_evaluated"] += 1
                if s.ceiling_height - v1 <= opts.junction_m:
                    add("CDR-002", f"A stain on wall {d.surface_id} reaches {s.ceiling_height - v1:.2f} m below the ceiling (rule: "
                        f"top edge within {opts.junction_m:.2f} m of the ceiling). Staining at the wall-ceiling junction is commonly "
                        "associated with a leak from the roof, an upper floor or services above.", [d], d.surface_id, d.room_id)
        elif s.kind == "ceiling":
            ev["CDR-002"]["regions_evaluated"] += 1
            add("CDR-002", f"A stain on the ceiling of {d.room_id} (rule: any stain on a ceiling surface). Ceiling staining is "
                "commonly associated with a leak from the roof, an upper floor or services above.", [d], d.surface_id, d.room_id)
    # clusters of stains along one wall
    for sid in sorted({d.surface_id for d in stains if by_surface[d.surface_id].kind == "wall"}):
        group = sorted((d for d in stains if d.surface_id == sid), key=lambda d: _box(d)[0])
        for d in group:
            ev["CDR-003"]["regions_evaluated"] += 1
        hit = set()
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                ca, cb = (_box(a)[0] + _box(a)[1]) / 2, (_box(b)[0] + _box(b)[1]) / 2
                if abs(ca - cb) <= opts.cluster_m:
                    hit.update([a.id, b.id])
        if hit:
            ds = [d for d in group if d.id in hit]
            add("CDR-003", f"{len(ds)} stains lie within {opts.cluster_m:.1f} m of each other along wall {sid} (rule: >= 2 stains "
                "with centres <= 1.5 m apart on one wall). Repeated staining along a wall is commonly associated with a "
                "concealed leak or condensation path in or behind it.", ds, sid, ds[0].room_id)
    for d in cracks:
        s = by_surface[d.surface_id]
        u0, u1, v0, v1 = _box(d)
        ev["CDR-005"]["regions_evaluated"] += 1
        if d.length is not None and d.length.value >= opts.long_crack_m:
            add("CDR-005", f"A crack on {d.surface_id} is {d.length.value:.2f} m long (rule: length >= {opts.long_crack_m:.1f} m). "
                "Long cracks are commonly associated with settlement or structural movement that may extend behind the finish.",
                [d], d.surface_id, d.room_id)
        if s.kind == "wall":
            for (j0, j1, hgt) in s.openings:
                if hgt is None:
                    continue
                ev["CDR-004"]["regions_evaluated"] += 1
                near = min(math.hypot(_gap(u0, u1, ju, ju), _gap(v0, v1, hgt, hgt)) for ju in (j0, j1))
                if near <= opts.opening_corner_m:
                    add("CDR-004", f"A crack on wall {d.surface_id} comes within {near:.2f} m of the head corner of an opening "
                        f"(rule: <= {opts.opening_corner_m:.2f} m). Cracks radiating from door and window corners are commonly "
                        "associated with stress concentration or lintel movement.", [d], d.surface_id, d.room_id)
                    break
    for c in cracks:
        for st in stains:
            if c.surface_id != st.surface_id:
                continue
            ev["CDR-006"]["regions_evaluated"] += 1
            cu0, cu1, cv0, cv1 = _box(c)
            su0, su1, sv0, sv1 = _box(st)
            if math.hypot(_gap(cu0, cu1, su0, su1), _gap(cv0, cv1, sv0, sv1)) <= opts.colocated_m:
                add("CDR-006", f"A crack and a stain on {c.surface_id} lie within {opts.colocated_m:.2f} m of each other (rule: "
                    "co-located crack and stain). A stained crack is commonly associated with water entering through the crack "
                    "from behind the surface.", [c, st], c.surface_id, c.room_id)
    flags.sort(key=lambda f: (f["rule_id"], f["surface_id"] or "", f["damage_ids"]))
    for i, f in enumerate(flags, 1):
        f["flag_id"] = f"concealed_{i:03d}"
    ev_out = {"rules": ev, "regions_without_metric_position_not_evaluated": skipped,
              "note": "rules that need a metric position on a surface are not evaluated for these regions"}
    return flags, ev_out
