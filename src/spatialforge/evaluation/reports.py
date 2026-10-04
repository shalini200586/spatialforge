"""Markdown and JSON rendering of evaluation results."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else str(v)
    return str(v)


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def gates_table(gates: list[dict]) -> str:
    return table(["Gate", "Status", "Value", "Threshold", "Detail"], [[g["gate"], g["status"], g["value"], g["threshold"], g["detail"]] for g in gates])


def render_markdown(res: dict) -> str:
    L = [f"# Evaluation: {res['property_id']} ({res.get('tier') or 'tier unknown'})", ""]
    if res.get("banner"):
        L += [f"> **{res['banner']}.** The reference values are not tape/laser measurements, so nothing below establishes dimensional accuracy.", ""]
    gt = res["ground_truth"]
    L += [f"Ground truth: {gt['rooms']} room(s), {gt['openings']} opening(s); measured with {gt['measured_with'] or 'not stated'}; "
          f"measurement uncertainty {_fmt(gt['measurement_uncertainty_m'])} m. Run status: {res.get('status') or 'not supplied'}.", "", "## Gates", "",
          gates_table(res["gates"]), ""]
    r = res["rooms"]
    L += ["## Room matching", "", table(["Ground truth", "Predicted", "Matched by"], [[m["gt"], m["pred"], m["matched_by"]] for m in r["matched"]]), ""]
    if r["unmatched_pred"]:
        L += [f"Predicted rooms with no ground truth: {', '.join(r['unmatched_pred'])}.", ""]
    w = res["walls"]
    L += ["## Walls", ""]
    if w["items"] or w["unmatched_gt"]:
        L += [table(["Room", "Wall", "Truth m", "Predicted m", "Abs err m", "Err %", "Truth in interval"],
                    [[i["room"], i["gt_wall"], i["truth_m"], i["predicted_m"], i["abs_error_m"], i["pct_error"], i["covered"]] for i in w["items"]]), ""]
        L += [f"Mean absolute error {_fmt(w['abs_error_m']['mean'])} m, mean {_fmt(w['pct_error']['mean'])}%, max {_fmt(w['pct_error']['max'])}%. "
              f"Ground-truth walls with no prediction: {', '.join(w['unmatched_gt']) or 'none'}.", ""]
    else:
        L += ["No ground-truth wall lengths.", ""]
    o = res["openings"]
    c = o["counts"]
    L += ["## Openings", "", f"Ground truth {c['ground_truth']}; matched {c['matched']}, missed {c['missed']}, phantom {c['phantom']} "
          f"(unverified candidates are not counted: {o['unverified_not_counted']}).", ""]
    if o["matched"]:
        L += [table(["Opening", "Truth m", "Predicted m", "Abs err m", f"Within {o['within_target']['target_m']} m", "Truth in interval"],
                    [[i["gt_opening"], i["truth_m"], i["predicted_m"], i["abs_error_m"], i["within_target"], i["covered"]] for i in o["matched"]]), ""]
    if o["missed"] or o["phantom"]:
        L += [f"Missed: {', '.join(o['missed']) or 'none'}. Phantom: {', '.join(o['phantom']) or 'none'}.", ""]
    L += ["## Ceiling heights and areas", ""]
    ce = res["ceiling"]["items"]
    L += [table(["Room", "Truth m", "Predicted m", "Abs err m", "Truth in interval"], [[i["room"], i["truth_m"], i["predicted_m"], i["abs_error_m"], i["covered"]] for i in ce]) if ce else "No ground-truth ceiling heights.", ""]
    ar = res["area"]["items"]
    L += [table(["Room", "Truth m2", "Predicted m2", "Abs err m2", "Err %", "Truth in interval"],
                [[i["room"], i["truth_m2"], i["predicted_m2"], i["abs_error_m2"], i["pct_error"], i["covered"]] for i in ar]) if ar else "No ground-truth floor areas.", ""]
    p = res["property"]
    L += ["## Property", ""]
    if p["footprint"]:
        f = p["footprint"]
        L += [f"Footprint area: truth {_fmt(f['truth_m2'])} m2, predicted {_fmt(f['predicted_m2'])} m2, error {_fmt(f['pct_error'])}%."]
    L += [f"Overlapping room pairs: {p['room_overlap_count']}."]
    if p["adjacency"]:
        a = p["adjacency"]
        L += [f"Adjacency: {a['correct']} of {a['truth_pairs']} true pairs found (recall {_fmt(a['recall'])}, precision {_fmt(a['precision'])}); "
              f"missing {a['missing']}, extra {a['extra']}."]
    cf = res["confidence"]
    L += ["", "## Interval coverage", "", f"{cf['covered']} of {cf['with_interval']} reported intervals contain the ground truth "
          f"(coverage {_fmt(cf['coverage_rate'])}); {cf['without_interval']} of {cf['measurements']} measurements carried no interval; "
          f"mean interval width {_fmt(cf['mean_interval_width'])}. Intervals are engineering estimates, not calibrated probabilities.", "",
          "## Runtime", "", f"Total {_fmt(res['runtime']['total_duration_seconds'])} s."]
    if res["runtime"]["stages"]:
        L += ["", table(["Stage", "Seconds"], [[k, v] for k, v in res["runtime"]["stages"].items()])]
    return "\n".join(L) + "\n"


def render_repeat_markdown(rep: dict) -> str:
    L = ["# Repeatability: capture A vs capture B", "", f"Rooms paired {rep['rooms_paired']} (A has {rep['rooms_a']}, B has {rep['rooms_b']}).", "",
         gates_table(rep["gates"]), ""]
    if rep["walls"]:
        L += ["## Walls", "", table(["Room A", "Wall A", "Wall B", "A m", "B m", "Diff m", "Diff %", "Within"],
                                    [[w["room_a"], w["wall_a"], w["wall_b"], w["a_m"], w["b_m"], w["diff_m"], w["diff_pct"], w["within"]] for w in rep["walls"]]), ""]
    if rep["ceilings"]:
        L += ["## Ceiling", "", table(["Room A", "A m", "B m", "Spread m"], [[c["room_a"], c["a_m"], c["b_m"], c["spread_m"]] for c in rep["ceilings"]]), ""]
    if rep["areas"]:
        L += ["## Area", "", table(["Room A", "A m2", "B m2", "Spread m2"], [[a["room_a"], a["a_m2"], a["b_m2"], a["spread_m2"]] for a in rep["areas"]]), ""]
    return "\n".join(L) + "\n"


def write_reports(result: dict, out_dir: str | Path, stem: str = "evaluation", markdown: str | None = None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / f"{stem}.json", out / f"{stem}.md"
    j.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    m.write_text(markdown if markdown is not None else render_markdown(result), encoding="utf-8")
    return [j, m]
