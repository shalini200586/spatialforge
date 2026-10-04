"""Evaluate one property.json against ground truth (and two captures against each other), and apply the assignment gates.

Rules that keep this honest:
  * a gate with no ground truth behind it is NOT EVALUATED, never PASS;
  * ground truth that exists but has no matching prediction (a missed opening, a room with no ceiling height) is a FAIL;
  * a threshold that was never supplied (`null` in the gate config) leaves its gate NOT EVALUATED and only reports the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from spatialforge.evaluation import matching as M
from spatialforge.evaluation.metrics import abs_error, aggregate, covered, interval_stats, pct_error, spread, summarize
from spatialforge.evaluation.models import (FAIL, NOT_EVALUATED, NOT_PHYSICAL_BANNER, PASS, GateConfig, GroundTruth)

RESULT_SCHEMA = "spatialforge-evaluation/1"


def _gate(name: str, status: str, detail: str, value=None, threshold=None) -> dict:
    return {"gate": name, "status": status, "value": value, "threshold": threshold, "detail": detail}


def _r(x, n=4):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- geometry helpers


def _polygon_area(poly: list[dict]) -> float:
    pts = [(p["x"], p["z"]) for p in poly]
    return abs(sum(x1 * z2 - x2 * z1 for (x1, z1), (x2, z2) in zip(pts, pts[1:] + pts[:1]))) / 2.0


def _inside(poly: list[tuple[float, float]], x: float, z: float) -> bool:
    inside = False
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1) + x1:
            inside = not inside
    return inside


def room_overlap_count(rooms: list[dict], min_area_m2: float = 0.1, cell: float = 0.05) -> int:
    """Number of room pairs whose polygons overlap by more than `min_area_m2` (raster estimate on a 5 cm grid; deterministic)."""
    polys = [(r["id"], [(p["x"], p["z"]) for p in r["polygon"]]) for r in rooms if len(r.get("polygon") or []) >= 3]
    n = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            a, b = polys[i][1], polys[j][1]
            x0, x1 = max(min(p[0] for p in a), min(p[0] for p in b)), min(max(p[0] for p in a), max(p[0] for p in b))
            z0, z1 = max(min(p[1] for p in a), min(p[1] for p in b)), min(max(p[1] for p in a), max(p[1] for p in b))
            if x0 >= x1 or z0 >= z1:
                continue
            hits, x = 0, x0 + cell / 2
            while x < x1:
                z = z0 + cell / 2
                while z < z1:
                    hits += _inside(a, x, z) and _inside(b, x, z)
                    z += cell
                x += cell
            n += (hits * cell * cell) > min_area_m2
    return n


def _predicted_pairs(rooms: list[dict]) -> set[frozenset]:
    out = set()
    for r in rooms:
        for other in list(r.get("adjacent_room_ids", [])) + list(r.get("connected_room_ids", [])):
            if other and other != "unmodelled" and other != r["id"]:
                out.add(frozenset((r["id"], other)))
    return out


def _runtime(report: dict | None) -> dict:
    if not report:
        return {"total_duration_seconds": None, "stages": {}}
    return {"total_duration_seconds": report.get("total_duration_seconds"),
            "stages": {s["stage"]: s["duration_seconds"] for s in report.get("stages", []) if s.get("status") != "skipped"}}


# ---------------------------------------------------------------- evaluate


def evaluate(prop: dict, gt: GroundTruth, gates: GateConfig | None = None, run_report: dict | None = None, tier: str | None = None) -> dict:
    gates = gates or GateConfig()
    tier = tier or (prop.get("capture") or {}).get("tier")
    rooms = prop.get("rooms", [])
    pred_openings = prop.get("openings", [])
    room_pairs = M.match_rooms(gt.rooms, rooms, gates)
    room_map = {p.gt.id: (p.pred["id"] if p.pred else None) for p in room_pairs}
    pred_to_gt = {p.pred["id"]: p.gt.id for p in room_pairs if p.pred}

    # walls
    wall_items, wall_missing, ceil_items, area_items = [], [], [], []
    for p in room_pairs:
        g, pr = p.gt, p.pred
        if g.walls:
            walls = M.predicted_walls(pr) if pr else []
            for wp in M.match_walls(g.walls, walls, gates):
                if wp.pred is None:
                    wall_missing.append(f"{g.id}/{wp.gt.id}")
                    continue
                ae, pe = abs_error(wp.pred["value"], wp.gt.length_m), pct_error(wp.pred["value"], wp.gt.length_m)
                wall_items.append({"room": g.id, "gt_wall": wp.gt.id, "pred_wall": wp.pred["id"], "matched_by": wp.how,
                                   "truth_m": wp.gt.length_m, "predicted_m": _r(wp.pred["value"]), "abs_error_m": _r(ae),
                                   "pct_error": _r(pe, 3), "covered": covered(wp.gt.length_m, wp.pred["measurement"]),
                                   "_m": wp.pred["measurement"]})
        if g.ceiling_height_m is not None:
            h = ((pr or {}).get("ceiling") or {}).get("height")
            if pr and h and h.get("value") is not None:
                ceil_items.append({"room": g.id, "truth_m": g.ceiling_height_m, "predicted_m": _r(h["value"]),
                                   "abs_error_m": _r(abs_error(h["value"], g.ceiling_height_m)), "covered": covered(g.ceiling_height_m, h),
                                   "_m": h})
            else:
                ceil_items.append({"room": g.id, "truth_m": g.ceiling_height_m, "predicted_m": None, "abs_error_m": None, "covered": None,
                                   "note": "no predicted ceiling height" if pr else "room not matched", "_m": None})
        if g.floor_area_m2 is not None:
            a = (pr or {}).get("area")
            if pr and a and a.get("value") is not None:
                area_items.append({"room": g.id, "truth_m2": g.floor_area_m2, "predicted_m2": _r(a["value"]),
                                   "abs_error_m2": _r(abs_error(a["value"], g.floor_area_m2)), "pct_error": _r(pct_error(a["value"], g.floor_area_m2), 3),
                                   "covered": covered(g.floor_area_m2, a), "_m": a})
            else:
                area_items.append({"room": g.id, "truth_m2": g.floor_area_m2, "predicted_m2": None, "abs_error_m2": None, "pct_error": None,
                                   "covered": None, "note": "room not matched" if not pr else "no predicted area", "_m": None})

    # openings
    open_pairs, phantoms = M.match_openings(gt.openings, pred_openings, room_map, gates)
    matched_open = []
    for op in open_pairs:
        if op.pred and (op.pred.get("width") or {}).get("value") is not None:
            w = op.pred["width"]
            matched_open.append({"gt_opening": op.gt.id, "pred_opening": op.pred["id"], "matched_by": op.how, "truth_m": op.gt.width_m,
                                 "predicted_m": _r(w["value"]), "abs_error_m": _r(abs_error(w["value"], op.gt.width_m)),
                                 "within_target": abs_error(w["value"], op.gt.width_m) <= gates.opening_width_abs_m,
                                 "covered": covered(op.gt.width_m, w), "_m": w})
    missed = [op.gt.id for op in open_pairs if op.pred is None]
    n_gt_open = len(gt.openings)
    recall = (len(matched_open) / n_gt_open) if n_gt_open else None

    # property level
    foot = ((prop.get("property") or {}).get("footprint") or {}).get("area")
    footprint = None
    if gt.footprint_area_m2 is not None:
        footprint = {"truth_m2": gt.footprint_area_m2, "predicted_m2": _r(foot["value"]) if foot else None,
                     "abs_error_m2": _r(abs_error(foot["value"], gt.footprint_area_m2)) if foot else None,
                     "pct_error": _r(pct_error(foot["value"], gt.footprint_area_m2), 3) if foot else None,
                     "covered": covered(gt.footprint_area_m2, foot), "_m": foot}
    overlap = room_overlap_count(rooms)
    adjacency = None
    if gt.adjacency:
        want = {frozenset(p) for p in gt.adjacency}
        got = {frozenset(pred_to_gt[x] for x in pair) for pair in _predicted_pairs(rooms) if all(x in pred_to_gt for x in pair)}
        tp = len(want & got)
        adjacency = {"truth_pairs": len(want), "predicted_pairs_among_matched_rooms": len(got), "correct": tp,
                     "recall": _r(tp / len(want)), "precision": _r(tp / len(got)) if got else None,
                     "missing": sorted(sorted(p) for p in want - got), "extra": sorted(sorted(p) for p in got - want)}

    # confidence (interval) calibration
    pairs = ([(i["truth_m"], i["_m"]) for i in wall_items] + [(i["truth_m"], i["_m"]) for i in ceil_items if i["_m"]]
             + [(i["truth_m2"], i["_m"]) for i in area_items if i["_m"]] + [(i["truth_m"], i["_m"]) for i in matched_open]
             + ([(footprint["truth_m2"], footprint["_m"])] if footprint and footprint["_m"] else []))
    conf = interval_stats(pairs)

    wall_pcts = [i["pct_error"] for i in wall_items if i["pct_error"] is not None]
    result = {
        "schema": RESULT_SCHEMA, "banner": None if gt.physical_measurement else NOT_PHYSICAL_BANNER,
        "property_id": gt.property_id, "tier": tier, "status": (run_report or {}).get("status"),
        "ground_truth": {"physical_measurement": gt.physical_measurement, "measured_with": gt.measured_with,
                         "measurement_uncertainty_m": gt.measurement_uncertainty_m, "rooms": len(gt.rooms), "openings": n_gt_open},
        "rooms": {"matched": [{"gt": p.gt.id, "pred": p.pred["id"] if p.pred else None, "matched_by": p.how} for p in room_pairs],
                  "unmatched_gt": [p.gt.id for p in room_pairs if p.pred is None],
                  "unmatched_pred": [r["id"] for r in rooms if r["id"] not in pred_to_gt]},
        "walls": {"items": wall_items, "unmatched_gt": wall_missing, "abs_error_m": summarize([i["abs_error_m"] for i in wall_items]),
                  "pct_error": summarize(wall_pcts)},
        "openings": {"matched": matched_open, "missed": missed, "phantom": [o["id"] for o in phantoms],
                     "unverified_not_counted": len(prop.get("unverified_openings", [])),
                     "counts": {"ground_truth": n_gt_open, "matched": len(matched_open), "missed": len(missed), "phantom": len(phantoms)},
                     "recall": _r(recall), "width_abs_error_m": summarize([i["abs_error_m"] for i in matched_open]),
                     "within_target": {"target_m": gates.opening_width_abs_m, "n": sum(i["within_target"] for i in matched_open),
                                       "of": len(matched_open),
                                       "fraction": _r(sum(i["within_target"] for i in matched_open) / len(matched_open)) if matched_open else None}},
        "ceiling": {"items": ceil_items, "abs_error_m": summarize([i["abs_error_m"] for i in ceil_items if i["abs_error_m"] is not None])},
        "area": {"items": area_items, "abs_error_m2": summarize([i["abs_error_m2"] for i in area_items if i["abs_error_m2"] is not None]),
                 "pct_error": summarize([i["pct_error"] for i in area_items if i["pct_error"] is not None])},
        "property": {"footprint": footprint, "room_overlap_count": overlap, "adjacency": adjacency},
        "confidence": conf, "runtime": _runtime(run_report),
    }
    result["gates"] = _apply_gates(result, gates, run_report, tier, wall_pcts)
    result["metrics"] = flatten_metrics(result)
    for section in (result["walls"]["items"], result["openings"]["matched"], result["ceiling"]["items"], result["area"]["items"]):
        for i in section:
            i.pop("_m", None)
    if footprint:
        footprint.pop("_m", None)
    return result


def _apply_gates(res: dict, g: GateConfig, report: dict | None, tier: str | None, wall_pcts: list[float]) -> list[dict]:
    out = []
    o = res["openings"]
    # opening width
    if not res["ground_truth"]["openings"]:
        out.append(_gate("opening_width", NOT_EVALUATED, "no ground-truth openings", threshold=f"<= {g.opening_width_abs_m} m on >= {g.opening_width_min_fraction:.0%}"))
    elif not o["within_target"]["of"]:
        out.append(_gate("opening_width", FAIL, "ground-truth openings exist but none was matched to a detected opening",
                         threshold=f"<= {g.opening_width_abs_m} m on >= {g.opening_width_min_fraction:.0%}"))
    else:
        f = o["within_target"]["fraction"]
        out.append(_gate("opening_width", PASS if f >= g.opening_width_min_fraction else FAIL,
                         f"{o['within_target']['n']} of {o['within_target']['of']} matched openings within {g.opening_width_abs_m} m",
                         f, f"<= {g.opening_width_abs_m} m on >= {g.opening_width_min_fraction:.0%} of matched openings"))
    # opening detection (counts always reported; a gate only when thresholds were supplied)
    c = o["counts"]
    detail = f"{c['matched']} matched, {c['missed']} missed, {c['phantom']} phantom of {c['ground_truth']} ground-truth openings"
    if not c["ground_truth"]:
        out.append(_gate("opening_detection", NOT_EVALUATED, "no ground-truth openings"))
    elif g.opening_detection_min_recall is None and g.opening_detection_max_phantoms is None:
        out.append(_gate("opening_detection", NOT_EVALUATED, detail + "; no recall/phantom threshold was supplied", o["recall"]))
    else:
        ok = ((g.opening_detection_min_recall is None or (o["recall"] or 0) >= g.opening_detection_min_recall)
              and (g.opening_detection_max_phantoms is None or c["phantom"] <= g.opening_detection_max_phantoms))
        out.append(_gate("opening_detection", PASS if ok else FAIL, detail, o["recall"],
                         f"recall >= {g.opening_detection_min_recall}, phantoms <= {g.opening_detection_max_phantoms}"))
    # ceiling
    items = res["ceiling"]["items"]
    if not items:
        out.append(_gate("ceiling_height", NOT_EVALUATED, "no ground-truth ceiling heights", threshold=f"<= {g.ceiling_abs_m} m per room"))
    else:
        bad = [i["room"] for i in items if i["abs_error_m"] is None or i["abs_error_m"] > g.ceiling_abs_m]
        worst = max((i["abs_error_m"] for i in items if i["abs_error_m"] is not None), default=None)
        out.append(_gate("ceiling_height", FAIL if bad else PASS,
                         (f"failing or missing in: {', '.join(bad)}" if bad else f"all {len(items)} rooms within {g.ceiling_abs_m} m"),
                         worst, f"<= {g.ceiling_abs_m} m per room"))
    # tier wall gates
    for gate, gtier, thr in (("photo_wall_length", "photos", g.photo_wall_pct), ("video_wall_length", "video", g.video_wall_pct)):
        if tier != gtier:
            out.append(_gate(gate, NOT_EVALUATED, f"applies to the {gtier} tier (this run: {tier})", threshold=f"{g.wall_aggregate} <= {thr}%"))
        elif not res["walls"]["items"] and not res["walls"]["unmatched_gt"]:
            out.append(_gate(gate, NOT_EVALUATED, "no ground-truth wall lengths", threshold=f"{g.wall_aggregate} <= {thr}%"))
        elif not wall_pcts:
            out.append(_gate(gate, FAIL, "ground-truth walls exist but none was matched", threshold=f"{g.wall_aggregate} <= {thr}%"))
        else:
            v = aggregate(wall_pcts, g.wall_aggregate)
            miss = len(res["walls"]["unmatched_gt"])
            out.append(_gate(gate, PASS if v <= thr and not miss else FAIL,
                             f"{g.wall_aggregate} error {v:.2f}% over {len(wall_pcts)} matched walls (max {max(wall_pcts):.2f}%)"
                             + (f"; {miss} ground-truth wall(s) unmatched" if miss else ""), _r(v, 3), f"{g.wall_aggregate} <= {thr}%"))
    # photo footprint and stitching
    fp = res["property"]["footprint"]
    if tier != "photos":
        out.append(_gate("photo_footprint", NOT_EVALUATED, f"applies to the photos tier (this run: {tier})", threshold=f"<= {g.photo_footprint_pct}%"))
        out.append(_gate("photo_stitching", NOT_EVALUATED, f"applies to the photos tier (this run: {tier})"))
    else:
        if fp is None:
            out.append(_gate("photo_footprint", NOT_EVALUATED, "no ground-truth footprint area", threshold=f"<= {g.photo_footprint_pct}%"))
        elif fp["pct_error"] is None:
            out.append(_gate("photo_footprint", FAIL, "no predicted footprint", threshold=f"<= {g.photo_footprint_pct}%"))
        else:
            out.append(_gate("photo_footprint", PASS if fp["pct_error"] <= g.photo_footprint_pct else FAIL,
                             f"footprint area error {fp['pct_error']:.2f}%", fp["pct_error"], f"<= {g.photo_footprint_pct}%"))
        unm = res["rooms"]["unmatched_gt"]
        if len(res["rooms"]["matched"]) < 2:
            out.append(_gate("photo_stitching", NOT_EVALUATED, "needs a ground-truth property of two or more rooms"))
        else:
            ok = not unm and res["property"]["room_overlap_count"] == 0
            out.append(_gate("photo_stitching", PASS if ok else FAIL,
                             f"{len(res['rooms']['matched']) - len(unm)} of {len(res['rooms']['matched'])} rooms placed; "
                             f"{res['property']['room_overlap_count']} overlapping room pair(s)"
                             + (f"; not placed: {', '.join(unm)}" if unm else "")))
    # drift accountability (an evidence gate, not a measurement against truth)
    if tier != "lidar":
        out.append(_gate("drift_accountability", NOT_EVALUATED, "applies to the LiDAR tier"))
    else:
        pose = (report or {}).get("production_pose_source")
        decided = any(d.get("stage") == "pose_refinement" for d in (report or {}).get("fallback_decisions", []))
        out.append(_gate("drift_accountability", PASS if pose and decided else FAIL if report else NOT_EVALUATED,
                         f"production poses: {pose}; pose-refinement decision recorded: {decided}" if report else "no run_report.json supplied"))
    # confidence calibration
    cc = res["confidence"]
    cov = cc["coverage_rate"]
    detail = (f"{cc['covered']} of {cc['with_interval']} intervals contain the truth ({cov:.0%}); {cc['without_interval']} measurement(s) carried no interval; "
              f"mean interval width {cc['mean_interval_width']}") if cov is not None else "no measurement with an interval could be tested"
    if g.confidence_min_coverage is None or cov is None:
        out.append(_gate("confidence_calibration", NOT_EVALUATED, detail + ("; no coverage threshold was supplied" if cov is not None else ""), _r(cov)))
    else:
        out.append(_gate("confidence_calibration", PASS if cov >= g.confidence_min_coverage else FAIL, detail, _r(cov), f"coverage >= {g.confidence_min_coverage}"))
    # repeatability needs a second capture
    for name in ("repeat_ceiling", "repeat_wall"):
        out.append(_gate(name, NOT_EVALUATED, "needs a second capture of the same property (evaluate --repeat)"))
    return out


def flatten_metrics(res: dict) -> dict:
    """The numbers a before/after comparison tracks, as one flat mapping (None where not measurable)."""
    w, o, c, a, p = res["walls"], res["openings"], res["ceiling"], res["area"], res["property"]
    rt = res["runtime"]
    return {
        "walls.matched": len(w["items"]), "walls.unmatched_gt": len(w["unmatched_gt"]),
        "walls.mean_abs_error_m": w["abs_error_m"]["mean"], "walls.max_abs_error_m": w["abs_error_m"]["max"],
        "walls.mean_pct_error": w["pct_error"]["mean"], "walls.max_pct_error": w["pct_error"]["max"],
        "openings.matched": o["counts"]["matched"], "openings.missed": o["counts"]["missed"], "openings.phantom": o["counts"]["phantom"],
        "openings.width_mean_abs_error_m": o["width_abs_error_m"]["mean"], "openings.width_within_target_fraction": o["within_target"]["fraction"],
        "ceiling.mean_abs_error_m": c["abs_error_m"]["mean"], "ceiling.max_abs_error_m": c["abs_error_m"]["max"],
        "area.mean_pct_error": a["pct_error"]["mean"], "area.max_pct_error": a["pct_error"]["max"],
        "footprint.pct_error": (p["footprint"] or {}).get("pct_error"), "property.room_overlap_count": p["room_overlap_count"],
        "adjacency.recall": (p["adjacency"] or {}).get("recall"), "adjacency.precision": (p["adjacency"] or {}).get("precision"),
        "confidence.coverage_rate": res["confidence"]["coverage_rate"], "confidence.mean_interval_width": res["confidence"]["mean_interval_width"],
        "runtime.total_s": rt["total_duration_seconds"],
        **{f"runtime.stage.{k}_s": v for k, v in sorted(rt["stages"].items())},
    }


# ---------------------------------------------------------------- repeatability


def evaluate_repeat(prop_a: dict, prop_b: dict, gates: GateConfig | None = None) -> dict:
    """Two captures of the same room/property against each other (no ground truth needed)."""
    gates = gates or GateConfig()
    room_pairs = M.match_rooms_between(prop_a.get("rooms", []), prop_b.get("rooms", []), gates.area_match_tolerance)
    walls, ceilings, areas = [], [], []
    for ra, rb in room_pairs:
        for wa, wb in M.match_walls_between(M.predicted_walls(ra), M.predicted_walls(rb), gates.wall_match_tolerance):
            d = abs_error(wa["value"], wb["value"])
            mean = (wa["value"] + wb["value"]) / 2
            pct = d / mean * 100 if mean else None
            walls.append({"room_a": ra["id"], "room_b": rb["id"], "wall_a": wa["id"], "wall_b": wb["id"], "a_m": _r(wa["value"]), "b_m": _r(wb["value"]),
                          "diff_m": _r(d), "diff_pct": _r(pct, 3), "within": d <= gates.repeat_wall_abs_m or (pct is not None and pct <= gates.repeat_wall_pct)})
        ha, hb = [((r.get("ceiling") or {}).get("height") or {}).get("value") for r in (ra, rb)]
        if ha is not None and hb is not None:
            ceilings.append({"room_a": ra["id"], "room_b": rb["id"], "a_m": _r(ha), "b_m": _r(hb), "spread_m": _r(spread([ha, hb]))})
        aa, ab = ra["area"]["value"], rb["area"]["value"]
        areas.append({"room_a": ra["id"], "room_b": rb["id"], "a_m2": _r(aa), "b_m2": _r(ab), "spread_m2": _r(spread([aa, ab]))})
    gl = []
    if not walls:
        gl.append(_gate("repeat_wall", NOT_EVALUATED, "no wall could be paired between the two captures", threshold=f"<= {gates.repeat_wall_abs_m} m or <= {gates.repeat_wall_pct}%"))
    else:
        bad = [w for w in walls if not w["within"]]
        gl.append(_gate("repeat_wall", FAIL if bad else PASS, f"{len(walls) - len(bad)} of {len(walls)} paired walls agree",
                        max(w["diff_m"] for w in walls), f"<= {gates.repeat_wall_abs_m} m or <= {gates.repeat_wall_pct}%"))
    if not ceilings:
        gl.append(_gate("repeat_ceiling", NOT_EVALUATED, "no room has a ceiling height in both captures", threshold=f"spread <= {gates.repeat_ceiling_spread_m} m"))
    else:
        worst = max(c["spread_m"] for c in ceilings)
        gl.append(_gate("repeat_ceiling", PASS if worst <= gates.repeat_ceiling_spread_m else FAIL, f"largest spread {worst} m over {len(ceilings)} room(s)",
                        worst, f"spread <= {gates.repeat_ceiling_spread_m} m"))
    return {"schema": "spatialforge-repeatability/1", "rooms_paired": len(room_pairs), "rooms_a": len(prop_a.get("rooms", [])),
            "rooms_b": len(prop_b.get("rooms", [])), "walls": walls, "ceilings": ceilings, "areas": areas, "gates": gl,
            "summary": {"wall_diff_m": summarize([w["diff_m"] for w in walls]), "ceiling_spread_m": summarize([c["spread_m"] for c in ceilings]),
                        "area_spread_m2": summarize([x["spread_m2"] for x in areas])}}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
