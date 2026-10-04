"""One stage vocabulary and one timing schema for every tier.

Each tier's code keeps its own detailed stage records (`capture_validation`, `sfm_reconstruction`, `cross_room_stitching`, ...).
This module folds them into the fixed list below, in this order, so run_report.json always has the same twelve entries:

    {"stage": "metric_scale", "status": "ok|warning|failed|skipped", "duration_seconds": 1.23, "warnings": [...], "error": null}

A stage a tier does not use is `skipped`. The detailed records are kept under `substages` for debugging.
"""

from __future__ import annotations

import re

STANDARD_STAGES = ("validation", "frontend", "metric_scale", "geometry", "gravity", "structure", "rooms", "openings",
                   "stitching", "damage", "rendering", "serialization")

# raw stage name (after stripping a "room_001/" prefix) -> standard stage
_RAW_TO_STANDARD = {
    # lidar
    "capture_validation": "validation", "metric_reconstruction_and_pose_check": "frontend",
    "floor_and_ceiling_planes": "geometry", "structural_walls": "structure", "room_topology": "rooms", "openings": "openings",
    # video
    "video_validation": "validation", "keyframe_extraction": "frontend", "sfm_reconstruction": "frontend",
    "metric_depth_and_scale": "metric_scale", "depth_fusion": "geometry", "gravity_alignment": "gravity",
    # photos
    "photo_validation": "validation", "features_and_exhaustive_matching": "frontend",
    "cross_room_stitching": "stitching", "global_placement": "stitching",
    # shared
    "damage_analysis": "damage", "rendering": "rendering", "serialization": "serialization",
}
_PHOTO_ROOM = re.compile(r"^room_\d+_reconstruction$")  # the per-room SfM mapping (its sub-stages are listed separately)

_SEVERITY = {"skipped": 0, "ok": 1, "warning": 2, "failed": 3}


def standard_name(raw: str) -> str | None:
    base = raw.split("/")[-1]
    if _PHOTO_ROOM.match(base):
        return "frontend"
    return _RAW_TO_STANDARD.get(base)


def fold_stages(raw_records: list, warning_records_for) -> list[dict]:
    """The twelve standard entries from a tier's raw StageRecords.

    `warning_records_for(stage, messages)` turns a record's readable messages into structured warnings.
    """
    out = {s: {"stage": s, "status": "skipped", "duration_seconds": 0.0, "warnings": [], "error": None, "substages": []}
           for s in STANDARD_STAGES}
    for rec in raw_records:
        std = standard_name(rec.name)
        if std is None:
            continue
        e = out[std]
        e["duration_seconds"] += float(rec.seconds or 0.0)
        e["substages"].append(rec.name)
        status = rec.status if rec.status in _SEVERITY else "ok"
        if e["status"] == "skipped" or _SEVERITY[status] > _SEVERITY[e["status"]]:
            if not (status == "skipped" and e["status"] != "skipped"):
                e["status"] = status
        if rec.warnings:
            e["warnings"] += warning_records_for(std, list(rec.warnings))
        if rec.error and e["error"] is None:
            e["error"] = {"message": rec.error, "raw_stage": rec.name}
    for e in out.values():
        e["duration_seconds"] = round(e["duration_seconds"], 3)
        e["warnings"] = [w.to_dict() if hasattr(w, "to_dict") else w for w in e["warnings"]]
    return [out[s] for s in STANDARD_STAGES]


def substages(raw_records: list) -> list[dict]:
    return [{"name": r.name, "stage": standard_name(r.name), "status": r.status, "duration_seconds": round(float(r.seconds or 0.0), 3),
             "details": r.details} for r in raw_records]
