"""Head-to-head against an incumbent (a consumer scanning app) on the same rooms, dimension by dimension.

The importer reads `benchmark/incumbent_template.csv`-shaped files: room_id, dimension_id, ground_truth_m, spatialforge_m,
incumbent_m, notes. Rows with a blank number are placeholders and are counted as pending, never as a win or a tie. The incumbent's
name and version belong in the notes of the real export; nothing here names one.
"""

from __future__ import annotations

import csv
from pathlib import Path

from spatialforge.evaluation.reports import table

COLUMNS = ["room_id", "dimension_id", "ground_truth_m", "spatialforge_m", "incumbent_m", "notes"]


def _num(x: str | None) -> float | None:
    if x is None or not x.strip():
        return None
    return float(x)


def read_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        rows = []
        for n, r in enumerate(reader, start=2):
            try:
                rows.append({"room_id": r["room_id"].strip(), "dimension_id": r["dimension_id"].strip(), "ground_truth_m": _num(r["ground_truth_m"]),
                             "spatialforge_m": _num(r["spatialforge_m"]), "incumbent_m": _num(r["incumbent_m"]), "notes": (r["notes"] or "").strip()})
            except ValueError as exc:
                raise ValueError(f"{path} line {n}: {exc}") from exc
    return rows


def compare(rows: list[dict], tie_tolerance_m: float = 0.005) -> dict:
    done, pending = [], []
    for r in rows:
        if None in (r["ground_truth_m"], r["spatialforge_m"], r["incumbent_m"]):
            pending.append(r)
            continue
        es, ei = abs(r["spatialforge_m"] - r["ground_truth_m"]), abs(r["incumbent_m"] - r["ground_truth_m"])
        winner = "tie" if abs(es - ei) <= tie_tolerance_m else "spatialforge" if es < ei else "incumbent"
        done.append({**r, "spatialforge_error_m": round(es, 4), "incumbent_error_m": round(ei, 4), "winner": winner})
    wins = sum(d["winner"] == "spatialforge" for d in done)
    ties = sum(d["winner"] == "tie" for d in done)
    rooms = sorted({d["room_id"] for d in done})
    notes = []
    if not done:
        notes.append("No complete row: nothing has been compared yet (placeholders only).")
    elif len(rooms) != 2:
        notes.append(f"The comparison covers {len(rooms)} room(s); the assignment asks for two.")
    return {"schema": "spatialforge-incumbent-comparison/1", "tie_tolerance_m": tie_tolerance_m, "rows": done, "pending_rows": len(pending),
            "rooms": rooms, "counts": {"compared": len(done), "spatialforge_wins": wins, "ties": ties, "incumbent_wins": len(done) - wins - ties},
            "beat_or_tie_pct": round((wins + ties) / len(done) * 100, 1) if done else None, "notes": notes,
            "mean_error_m": {"spatialforge": round(sum(d["spatialforge_error_m"] for d in done) / len(done), 4) if done else None,
                             "incumbent": round(sum(d["incumbent_error_m"] for d in done) / len(done), 4) if done else None}}


def render_markdown(res: dict) -> str:
    c = res["counts"]
    L = ["# Incumbent comparison", "",
         f"{c['compared']} dimension(s) compared across {len(res['rooms'])} room(s); {res['pending_rows']} placeholder row(s) still pending. "
         f"A tie means the two errors differ by at most {res['tie_tolerance_m']} m.", ""]
    if res["beat_or_tie_pct"] is not None:
        L += [f"**SpatialForge beats or ties the incumbent on {res['beat_or_tie_pct']}% of dimensions** "
              f"({c['spatialforge_wins']} wins, {c['ties']} ties, {c['incumbent_wins']} losses). "
              f"Mean absolute error: SpatialForge {res['mean_error_m']['spatialforge']} m, incumbent {res['mean_error_m']['incumbent']} m.", ""]
    L += [f"> {n}" for n in res["notes"]] + ([""] if res["notes"] else [])
    if res["rows"]:
        L += [table(["Room", "Dimension", "Truth m", "SpatialForge m", "Incumbent m", "SpatialForge err", "Incumbent err", "Winner", "Notes"],
                    [[d["room_id"], d["dimension_id"], d["ground_truth_m"], d["spatialforge_m"], d["incumbent_m"], d["spatialforge_error_m"],
                      d["incumbent_error_m"], d["winner"], d["notes"]] for d in res["rows"]]), ""]
    return "\n".join(L) + "\n"
