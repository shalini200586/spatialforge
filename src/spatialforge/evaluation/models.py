"""Ground truth, gate configuration and result vocabulary."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT EVALUATED"
GT_SCHEMA = "spatialforge-ground-truth/1"
SCHEMA_FILE = Path(__file__).resolve().parents[3] / "benchmark" / "ground_truth.schema.json"
NOT_PHYSICAL_BANNER = "NOT PHYSICAL GROUND TRUTH BENCHMARK"


@dataclass
class GateConfig:
    """Assignment thresholds and matching tolerances. The defaults equal benchmark/gates.json; override with that file format."""

    opening_width_abs_m: float = 0.02
    opening_width_min_fraction: float = 0.85
    ceiling_abs_m: float = 0.015
    repeat_ceiling_spread_m: float = 0.01
    repeat_wall_abs_m: float = 0.01
    repeat_wall_pct: float = 0.5
    photo_wall_pct: float = 8.0
    video_wall_pct: float = 3.0
    photo_footprint_pct: float = 8.0
    wall_aggregate: str = "mean"  # how a wall gate summarises the matched walls: mean | median | max
    opening_detection_min_recall: float | None = None  # no assignment number was supplied: reported, gate not evaluated
    opening_detection_max_phantoms: int | None = None
    confidence_min_coverage: float | None = None
    area_match_tolerance: float = 0.25  # relative area difference allowed when matching rooms without ids
    wall_match_tolerance: float = 0.25
    opening_match_max_width_diff_m: float = 0.5
    tie_tolerance_m: float = 0.005  # incumbent comparison: errors closer than this are a tie

    def to_dict(self) -> dict:
        return asdict(self)


def load_gate_config(path: str | Path | None = None) -> GateConfig:
    if path is None:
        return GateConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(GateConfig)}
    unknown = [k for k in raw if k not in known and not k.startswith("_")]
    if unknown:
        raise ValueError(f"unknown gate setting(s) in {path}: {', '.join(unknown)}")
    return GateConfig(**{k: v for k, v in raw.items() if k in known})


@dataclass
class GtWall:
    id: str
    length_m: float
    predicted_id: str | None = None


@dataclass
class GtRoom:
    id: str
    predicted_id: str | None = None
    floor_area_m2: float | None = None
    ceiling_height_m: float | None = None
    walls: list[GtWall] | None = None


@dataclass
class GtOpening:
    id: str
    width_m: float
    type: str | None = None
    predicted_id: str | None = None
    room_id: str | None = None
    connects: list[str] | None = None


@dataclass
class GroundTruth:
    property_id: str
    rooms: list[GtRoom]
    openings: list[GtOpening]
    adjacency: list[tuple[str, str]]
    damage: list[dict]
    footprint_area_m2: float | None
    physical_measurement: bool
    measured_with: str | None
    measurement_uncertainty_m: float | None
    repeat_group: str | None
    capture_id: str | None
    raw: dict


def validate_ground_truth(raw: dict) -> list[str]:
    from spatialforge.pipeline.schema import validate_against_schema

    problems = validate_against_schema(raw, json.loads(SCHEMA_FILE.read_text(encoding="utf-8")))
    if problems:
        return problems
    ids = [r["id"] for r in raw["rooms"]]
    if len(ids) != len(set(ids)):
        problems.append("rooms: duplicate room ids")
    for pair in raw.get("adjacency", []):
        problems += [f"adjacency: unknown room {r!r}" for r in pair if r not in ids]
    for o in raw.get("openings", []):
        if o.get("room_id") not in (None, *ids):
            problems.append(f"openings: {o['id']} refers to unknown room {o['room_id']!r}")
    return problems


def parse_ground_truth(raw: dict) -> GroundTruth:
    problems = validate_ground_truth(raw)
    if problems:
        raise ValueError("invalid ground truth: " + "; ".join(problems[:8]))
    rooms = [GtRoom(r["id"], r.get("predicted_id"), r.get("floor_area_m2"), r.get("ceiling_height_m"),
                    [GtWall(w["id"], w["length_m"], w.get("predicted_id")) for w in r["walls"]] if "walls" in r else None)
             for r in raw["rooms"]]
    openings = [GtOpening(o["id"], o["width_m"], o.get("type"), o.get("predicted_id"), o.get("room_id"), o.get("connects"))
                for o in raw.get("openings", [])]
    return GroundTruth(raw["property_id"], rooms, openings, [tuple(p[:2]) for p in raw.get("adjacency", [])], raw.get("damage", []),
                       raw.get("footprint_area_m2"), bool(raw.get("physical_measurement", False)), raw.get("measured_with"),
                       raw.get("measurement_uncertainty_m"), raw.get("repeat_group"), raw.get("capture_id"), raw)


def load_ground_truth(path: str | Path) -> GroundTruth:
    return parse_ground_truth(json.loads(Path(path).read_text(encoding="utf-8")))
