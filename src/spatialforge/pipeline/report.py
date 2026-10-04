"""run_report.json: one schema for every tier.

    schema, spatialforge_version, tier (+ how it was chosen), status, exit_code, input, output, started_at, ended_at,
    total_duration_seconds, stages (the fixed twelve), substages, warnings, errors, counts, fallback_decisions, config,
    determinism, and (tier-specific detail blocks such as `video` / `photos`, kept out of property.json).

Status: success | partial | failure. Exit code: 0 for success and partial, 1 for failure (2 is reserved for command-line usage).
"""

from __future__ import annotations

from pathlib import Path

from spatialforge import __version__
from spatialforge.pipeline.messages import ErrorRecord, WarningRecord, classify_all
from spatialforge.pipeline.provenance import DETERMINISM, redact_text
from spatialforge.pipeline.stage_model import STANDARD_STAGES, fold_stages, substages

REPORT_SCHEMA = "spatialforge.run_report"
STATUSES = ("success", "partial", "failure")


def exit_code(status: str) -> int:
    return 1 if status == "failure" else 0


def _warnings_of():
    def for_stage(std: str, messages: list[str]) -> list[WarningRecord]:
        recs = classify_all(messages)
        for r in recs:
            r.stage = std if r.code == "GENERAL_WARNING" else r.stage
        return recs

    return for_stage


def dedupe_records(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        key = (r["code"], r["message"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def build_report(*, tier: str, tier_requested: str | None, detection: str | None, status: str, capture, display_output, started: str,
                 ended: str, duration_s: float, raw_stages: list, property_warning_records: list[dict] | None, errors: list[ErrorRecord],
                 counts: dict | None, decisions: list | None, config: dict, files: list[str], extras: dict | None = None,
                 incomplete: bool = False, diagnostic_warnings: list[str] | None = None) -> dict:
    stages = fold_stages(raw_stages, _warnings_of())
    # a failure is attributed to its stage: the stage entry carries the error, the status says it failed
    for e in errors:
        entry = next((s for s in stages if s["stage"] == e.stage), None)
        if entry is not None:
            entry["status"] = "failed"
            entry["error"] = {"code": e.code, "message": e.message}
    stage_warnings = [w for s in stages for w in s["warnings"]]
    warnings = dedupe_records(list(property_warning_records or []) + stage_warnings)  # a failed run has no property: keep its stage warnings
    rep = {
        "schema": REPORT_SCHEMA, "report_version": 1, "spatialforge_version": __version__,
        "tier": tier, "tier_requested": tier_requested or tier, "tier_detection": detection,
        "status": status, "exit_code": exit_code(status),
        "input": {"name": Path(str(capture)).name, "path": redact_text(str(capture))},
        "output": {"directory": redact_text(str(display_output)), "files": sorted(files), "incomplete": incomplete},
        "started_at": started, "ended_at": ended, "total_duration_seconds": round(duration_s, 3),
        "stages": stages, "substages": substages(raw_stages),
        "warnings": warnings, "errors": [e.to_dict() for e in errors],
        "counts": counts, "fallback_decisions": decisions or [], "config": config, "determinism": DETERMINISM,
        "diagnostic_warnings": diagnostic_warnings or [],
    }
    for k, v in (extras or {}).items():
        rep.setdefault(k, v)
    return rep


def check_report_shape(rep: dict) -> list[str]:
    """Problems with the structure of a run report (used by tests and as a self-check): empty list means well-formed."""
    problems = []
    if rep.get("schema") != REPORT_SCHEMA:
        problems.append("schema")
    if rep.get("status") not in STATUSES:
        problems.append("status")
    if [s.get("stage") for s in rep.get("stages", [])] != list(STANDARD_STAGES):
        problems.append("stages are not the standard list")
    for s in rep.get("stages", []):
        if set(s) < {"stage", "status", "duration_seconds", "warnings", "error"}:
            problems.append(f"stage entry {s.get('stage')} lacks a field")
        if s.get("status") not in ("ok", "warning", "failed", "skipped"):
            problems.append(f"stage {s.get('stage')} has status {s.get('status')}")
    for w in rep.get("warnings", []):
        if set(w) < {"code", "stage", "severity", "message", "recoverable"}:
            problems.append("a warning is not structured")
    for e in rep.get("errors", []):
        if not {"code", "stage", "message"} <= set(e):
            problems.append("an error is not structured")
    if "total_duration_seconds" not in rep:
        problems.append("total_duration_seconds")
    return problems
