"""Command-line handlers for `evaluate` and `compare-incumbent` (wired into spatialforge.cli)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from spatialforge.evaluation import incumbent
from spatialforge.evaluation.benchmark import evaluate, evaluate_repeat, load_json
from spatialforge.evaluation.models import FAIL, load_gate_config, load_ground_truth
from spatialforge.evaluation.reports import render_markdown, render_repeat_markdown, write_reports


def _locate(path: str) -> tuple[Path, Path | None]:
    """A result folder or a property.json path -> (property.json, run_report.json if it sits next to it)."""
    p = Path(path)
    prop = p / "property.json" if p.is_dir() else p
    if not prop.is_file():
        raise FileNotFoundError(f"property.json not found at {prop} (a failed run writes none)")
    report = prop.with_name("run_report.json")
    return prop, report if report.is_file() else None


def run_evaluate(args) -> int:
    try:
        gt = load_ground_truth(args.ground_truth)
        gates = load_gate_config(args.gates)
        prop_path, report_path = _locate(args.property)
        report = load_json(args.run_report) if args.run_report else (load_json(report_path) if report_path else None)
        result = evaluate(load_json(prop_path), gt, gates, report, tier=args.tier)
        parts = [render_markdown(result)]
        results = {"evaluation": result}
        if args.repeat:
            other, _ = _locate(args.repeat)
            rep = evaluate_repeat(load_json(prop_path), load_json(other), gates)
            results["repeatability"] = rep
            parts.append(render_repeat_markdown(rep))
            # the repeat gates belong in the headline table too
            for g in result["gates"]:
                for rg in rep["gates"]:
                    if g["gate"] == rg["gate"]:
                        g.update(rg)
            parts[0] = render_markdown(result)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR [EVALUATION_INPUT] {exc}", file=sys.stderr)
        return 1
    text = "\n".join(parts)
    if args.output:
        write_reports(result, args.output, markdown=text)
        if "repeatability" in results:
            Path(args.output, "repeatability.json").write_text(json.dumps(results["repeatability"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {Path(args.output) / 'evaluation.json'} and evaluation.md")
    print(text)
    return 1 if args.strict and any(g["status"] == FAIL for g in result["gates"]) else 0


def run_compare_incumbent(args) -> int:
    try:
        gates = load_gate_config(args.gates)
        res = incumbent.compare(incumbent.read_rows(args.csv), gates.tie_tolerance_m)
    except (OSError, ValueError) as exc:
        print(f"ERROR [INCUMBENT_INPUT] {exc}", file=sys.stderr)
        return 1
    text = incumbent.render_markdown(res)
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "incumbent_comparison.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "incumbent_comparison.md").write_text(text, encoding="utf-8")
    print(text)
    return 0
