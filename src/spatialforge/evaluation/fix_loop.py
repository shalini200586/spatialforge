"""Fix-loop support: freeze a baseline, ship a fix, freeze again, compare before/after.

    evaluation_runs/before/<name>/{manifest.json, evaluation.json, property.json}
    evaluation_runs/after/<name>/...

The manifest records what is needed to reproduce the run: git SHA, capture, tier, command, ground-truth file, config, the
property output and the metrics. Nothing here invents a result: it only reads evaluations that `evaluate` produced.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from spatialforge.evaluation.benchmark import evaluate, load_json
from spatialforge.evaluation.models import load_gate_config, load_ground_truth
from spatialforge.evaluation.reports import table, write_reports
from spatialforge.pipeline.provenance import redact_text

SLOTS = ("before", "after")


def git_sha(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def freeze_baseline(run_dir: str | Path, ground_truth: str | Path, name: str, slot: str = "before", root: str | Path = "evaluation_runs",
                    command: str | None = None, gates_path: str | Path | None = None, capture: str | None = None,
                    repo: str | Path | None = None) -> Path:
    """Evaluate `run_dir` (a `process` output folder) against `ground_truth` and store the frozen record under root/slot/name."""
    if slot not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}")
    run_dir = Path(run_dir)
    prop_path, report_path = run_dir / "property.json", run_dir / "run_report.json"
    if not prop_path.is_file():
        raise FileNotFoundError(f"{prop_path} not found: a failed run has nothing to freeze")
    gt, gates = load_ground_truth(ground_truth), load_gate_config(gates_path)
    report = load_json(report_path) if report_path.is_file() else None
    prop = load_json(prop_path)
    result = evaluate(prop, gt, gates, report)
    dest = Path(root) / slot / name
    if dest.exists():
        raise FileExistsError(f"{dest} already exists: a frozen baseline is never overwritten (choose another --name)")
    dest.mkdir(parents=True)
    shutil.copy2(prop_path, dest / "property.json")
    write_reports(result, dest)
    manifest = {
        "schema": "spatialforge-frozen-run/1", "slot": slot, "name": name, "git_sha": git_sha(repo),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture": redact_text(capture or (report or {}).get("input", {}).get("path") or "not recorded"),
        "tier": result["tier"], "command": redact_text(command or "not recorded"),
        "ground_truth_file": Path(ground_truth).name, "ground_truth_physical_measurement": gt.physical_measurement,
        "config": {"gates": gates.to_dict(), "process": (report or {}).get("config")},
        "property_output": "property.json", "metrics": result["metrics"], "gates": {g["gate"]: g["status"] for g in result["gates"]},
        "spatialforge_version": (report or {}).get("spatialforge_version"),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def _change(b, a):
    if b is None or a is None:
        return None, None
    d = a - b
    return round(d, 6), (round(d / abs(b) * 100, 2) if b else None)


def compare_runs(before_dir: str | Path, after_dir: str | Path) -> dict:
    b, a = load_json(Path(before_dir) / "evaluation.json"), load_json(Path(after_dir) / "evaluation.json")
    mb, ma = load_json(Path(before_dir) / "manifest.json"), load_json(Path(after_dir) / "manifest.json")
    rows = []
    for k in sorted(set(b["metrics"]) | set(a["metrics"])):
        bv, av = b["metrics"].get(k), a["metrics"].get(k)
        ab, pc = _change(bv, av)
        rows.append({"metric": k, "before": bv, "after": av, "abs_change": ab, "pct_change": pc})
    gb, ga = {g["gate"]: g for g in b["gates"]}, {g["gate"]: g for g in a["gates"]}
    gates = [{"gate": k, "before": gb.get(k, {}).get("status"), "after": ga.get(k, {}).get("status"),
              "value_before": gb.get(k, {}).get("value"), "value_after": ga.get(k, {}).get("value")} for k in sorted(set(gb) | set(ga))]
    return {"schema": "spatialforge-run-comparison/1", "before": {"git_sha": mb["git_sha"], "name": mb["name"]},
            "after": {"git_sha": ma["git_sha"], "name": ma["name"]}, "same_ground_truth": mb["ground_truth_file"] == ma["ground_truth_file"],
            "banner": b.get("banner") or a.get("banner"), "metrics": rows, "gates": gates}


def render_comparison(cmp: dict) -> str:
    L = ["# Before / after", "", f"Before: {cmp['before']['name']} at `{cmp['before']['git_sha'][:12]}`. After: {cmp['after']['name']} at `{cmp['after']['git_sha'][:12]}`."]
    if cmp.get("banner"):
        L += ["", f"> **{cmp['banner']}**"]
    if not cmp["same_ground_truth"]:
        L += ["", "> WARNING: the two runs used different ground-truth files; the comparison is not like for like."]
    L += ["", "## Gates", "", table(["Gate", "Before", "After", "Value before", "Value after"],
                                   [[g["gate"], g["before"], g["after"], g["value_before"], g["value_after"]] for g in cmp["gates"]]), "",
          "## Metrics", "", table(["Metric", "Before", "After", "Absolute change", "Percent change"],
                                  [[m["metric"], m["before"], m["after"], m["abs_change"], m["pct_change"]] for m in cmp["metrics"]])]
    return "\n".join(L) + "\n"
