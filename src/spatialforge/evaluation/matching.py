"""Correspondence between ground truth and a predicted property. Deliberately conservative.

IDs win: a ground-truth item with `predicted_id` is matched to exactly that predicted item. Everything else is matched greedily
by the smallest difference, and only inside a tolerance; an item with no acceptable partner stays unmatched (a miss or a
phantom), never forced into a pairing that would flatter the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from spatialforge.evaluation.models import GateConfig, GtOpening, GtRoom, GtWall

GENERIC = "opening"


@dataclass
class Pair:
    gt: object
    pred: object | None
    how: str  # "id" | "geometry" | "unmatched"


def _greedy(gt_items, pred_items, diff, tolerance_ok):
    """Greedy one-to-one assignment by ascending difference. Returns (pairs [(gt_index, pred_index)], leftovers are unmatched)."""
    cand = sorted((diff(g, p), gi, pi) for gi, g in enumerate(gt_items) for pi, p in enumerate(pred_items) if tolerance_ok(g, p))
    used_g, used_p, out = set(), set(), []
    for _, gi, pi in cand:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gi, pi))
    return out


def match_rooms(gt_rooms: list[GtRoom], pred_rooms: list[dict], cfg: GateConfig) -> list[Pair]:
    by_id = {r["id"]: r for r in pred_rooms}
    pairs: dict[int, Pair] = {}
    taken: set[str] = set()
    for i, g in enumerate(gt_rooms):
        if g.predicted_id and g.predicted_id in by_id and g.predicted_id not in taken:
            pairs[i] = Pair(g, by_id[g.predicted_id], "id")
            taken.add(g.predicted_id)
    rest_g = [(i, g) for i, g in enumerate(gt_rooms) if i not in pairs]
    rest_p = [r for r in pred_rooms if r["id"] not in taken]

    def area(r):
        return (r.get("area") or {}).get("value")

    def ok(gi, p):
        g, a = gi[1], area(p)
        return g.floor_area_m2 and a is not None and abs(a - g.floor_area_m2) / g.floor_area_m2 <= cfg.area_match_tolerance

    for gi, pi in _greedy(rest_g, rest_p, lambda g, p: abs(area(p) - g[1].floor_area_m2) / g[1].floor_area_m2, ok):
        pairs[rest_g[gi][0]] = Pair(rest_g[gi][1], rest_p[pi], "geometry")
    return [pairs.get(i, Pair(g, None, "unmatched")) for i, g in enumerate(gt_rooms)]


def predicted_walls(room: dict) -> list[dict]:
    """The room's walls as [{id, value, measurement}], from the parallel wall_ids / wall_lengths lists."""
    return [{"id": wid, "value": m["value"], "measurement": m} for wid, m in zip(room.get("wall_ids", []), room.get("wall_lengths", []))
            if m and m.get("value") is not None]


def match_walls(gt_walls: list[GtWall], pred_walls: list[dict], cfg: GateConfig) -> list[Pair]:
    by_id = {w["id"]: w for w in pred_walls}
    pairs: dict[int, Pair] = {}
    taken: set[str] = set()
    for i, g in enumerate(gt_walls):
        if g.predicted_id and g.predicted_id in by_id and g.predicted_id not in taken:
            pairs[i] = Pair(g, by_id[g.predicted_id], "id")
            taken.add(g.predicted_id)
    rest_g = [(i, g) for i, g in enumerate(gt_walls) if i not in pairs]
    rest_p = [w for w in pred_walls if w["id"] not in taken]

    def ok(gi, p):
        return gi[1].length_m > 0 and abs(p["value"] - gi[1].length_m) / gi[1].length_m <= cfg.wall_match_tolerance

    for gi, pi in _greedy(rest_g, rest_p, lambda g, p: abs(p["value"] - g[1].length_m), ok):
        pairs[rest_g[gi][0]] = Pair(rest_g[gi][1], rest_p[pi], "geometry")
    return [pairs.get(i, Pair(g, None, "unmatched")) for i, g in enumerate(gt_walls)]


def _type_compatible(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a == b or GENERIC in (a, b)


def match_openings(gt_openings: list[GtOpening], pred_openings: list[dict], room_map: dict[str, str | None], cfg: GateConfig
                   ) -> tuple[list[Pair], list[dict]]:
    """Returns (one Pair per ground-truth opening, the predicted openings nobody matched = phantoms)."""
    by_id = {o["id"]: o for o in pred_openings}
    pairs: dict[int, Pair] = {}
    taken: set[str] = set()
    for i, g in enumerate(gt_openings):
        if g.predicted_id and g.predicted_id in by_id and g.predicted_id not in taken:
            pairs[i] = Pair(g, by_id[g.predicted_id], "id")
            taken.add(g.predicted_id)
    rest_g = [(i, g) for i, g in enumerate(gt_openings) if i not in pairs]
    rest_p = [o for o in pred_openings if o["id"] not in taken]

    def width(o):
        return (o.get("width") or {}).get("value")

    def ok(gi, p):
        g, w = gi[1], width(p)
        if w is None or abs(w - g.width_m) > cfg.opening_match_max_width_diff_m or not _type_compatible(g.type, p.get("type")):
            return False
        mapped = room_map.get(g.room_id) if g.room_id else None
        return not (mapped and p.get("room_ids") and mapped not in p["room_ids"])  # a room we know must agree when the opening knows its rooms

    for gi, pi in _greedy(rest_g, rest_p, lambda g, p: abs(width(p) - g[1].width_m), ok):
        pairs[rest_g[gi][0]] = Pair(rest_g[gi][1], rest_p[pi], "geometry")
    matched_ids = {p.pred["id"] for p in pairs.values() if p.pred}
    return [pairs.get(i, Pair(g, None, "unmatched")) for i, g in enumerate(gt_openings)], [o for o in pred_openings if o["id"] not in matched_ids]


def match_rooms_between(a_rooms: list[dict], b_rooms: list[dict], tolerance: float) -> list[tuple[dict, dict]]:
    """Repeat captures: pair rooms of capture A with rooms of capture B by floor area (greedy, inside the tolerance)."""
    def area(r):
        return (r.get("area") or {}).get("value")

    ra = [r for r in a_rooms if area(r)]
    rb = [r for r in b_rooms if area(r)]
    pairs = _greedy(ra, rb, lambda a, b: abs(area(a) - area(b)) / area(a), lambda a, b: abs(area(a) - area(b)) / area(a) <= tolerance)
    return [(ra[i], rb[j]) for i, j in pairs]


def match_walls_between(a: list[dict], b: list[dict], tolerance: float) -> list[tuple[dict, dict]]:
    pairs = _greedy(a, b, lambda x, y: abs(x["value"] - y["value"]), lambda x, y: x["value"] > 0 and abs(x["value"] - y["value"]) / x["value"] <= tolerance)
    return [(a[i], b[j]) for i, j in pairs]
