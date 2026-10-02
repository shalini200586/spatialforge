"""Sensor-independent canonical property model (schema v1.0).

Case-study materials supplied no separate schema file; SpatialForge defines schema v1.0. It is NOT an official
evaluator schema. Nothing here knows about LiDAR: other tiers (video, photos) are meant to fill the same model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SCHEMA_VERSION = "1.0"
QUALITIES = ("strong", "moderate", "weak")
OPENING_TYPES = ("door", "window", "opening")
STATUSES = ("complete", "partial")
UNMODELLED = "unmodelled"


class ModelError(ValueError):
    """The canonical model would be invalid."""


def _finite(x, name: str) -> float:
    if x is None or isinstance(x, bool) or not math.isfinite(float(x)):
        raise ModelError(f"{name} must be a finite number, got {x!r}")
    return float(x)


@dataclass
class Measurement:
    """A value with unit, optional interval, optional confidence and a quality label.

    `confidence` is a heuristic score in [0, 1], NOT a probability, and is None unless a stage really produced one.
    `quality` is the stage's own qualitative rating. The interval is the stage's uncertainty range, never invented.
    """

    value: float
    unit: str
    low: float | None = None
    high: float | None = None
    confidence: float | None = None
    quality: str | None = None
    source_tier: str = "lidar"

    def __post_init__(self):
        self.value = _finite(self.value, "measurement value")
        if not self.unit:
            raise ModelError("measurement needs a unit")
        if (self.low is None) != (self.high is None):
            raise ModelError("interval needs both low and high")
        if self.low is not None:
            self.low, self.high = _finite(self.low, "interval low"), _finite(self.high, "interval high")
            if self.low > self.high:
                raise ModelError(f"interval low {self.low} is above high {self.high}")
            if self.low > self.value + 1e-9:
                raise ModelError(f"interval low {self.low} is above the value {self.value}")
            if self.high < self.value - 1e-9:
                raise ModelError(f"interval high {self.high} is below the value {self.value}")
        if self.confidence is not None:
            self.confidence = _finite(self.confidence, "confidence")
            if not 0.0 <= self.confidence <= 1.0:
                raise ModelError(f"confidence {self.confidence} is outside [0, 1]")
        if self.quality is not None and self.quality not in QUALITIES:
            raise ModelError(f"quality must be one of {QUALITIES}, got {self.quality!r}")

    def to_dict(self) -> dict:
        return {
            "value": self.value, "unit": self.unit,
            "interval": None if self.low is None else {"low": self.low, "high": self.high},
            "confidence": self.confidence, "quality": self.quality, "source_tier": self.source_tier,
        }


@dataclass
class Point2D:
    x: float
    z: float

    def __post_init__(self):
        self.x, self.z = _finite(self.x, "x"), _finite(self.z, "z")

    def to_dict(self) -> dict:
        return {"x": self.x, "z": self.z}


def polygon_area(points: list[Point2D]) -> float:
    n = len(points)
    return 0.5 * sum(points[i].x * points[(i + 1) % n].z - points[(i + 1) % n].x * points[i].z for i in range(n))


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def validate_polygon(points: list[Point2D], what: str = "polygon") -> None:
    if len(points) < 3:
        raise ModelError(f"{what} needs at least 3 vertices")
    n = len(points)
    xy = [(p.x, p.z) for p in points]
    for i in range(n):  # self-intersection first: a symmetric bow-tie also has zero net area, but this is the real cause
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_cross(xy[i], xy[(i + 1) % n], xy[j], xy[(j + 1) % n]):
                raise ModelError(f"{what} is self-intersecting")
    if abs(polygon_area(points)) < 1e-9:
        raise ModelError(f"{what} has zero area")


@dataclass
class Segment:
    start: Point2D
    end: Point2D
    length: Measurement

    def to_dict(self) -> dict:
        return {"start": self.start.to_dict(), "end": self.end.to_dict(), "length": self.length.to_dict()}


@dataclass
class Wall:
    id: str
    start: Point2D
    end: Point2D
    length: Measurement  # extent from first to last observed segment (gaps included)
    observed_length: Measurement  # sum of observed segments
    orientation: Measurement  # degrees, [0, 180)
    evidence_quality: str  # strong | moderate | weak
    position_uncertainty: Measurement  # across the wall, metres (diagnostic, not calibrated)
    segments: list[Segment] = field(default_factory=list)
    room_ids: list[str] = field(default_factory=list)
    opening_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.evidence_quality not in QUALITIES:
            raise ModelError(f"wall evidence_quality must be one of {QUALITIES}")
        if self.length.value < 0 or self.observed_length.value < 0:
            raise ModelError("wall length cannot be negative")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "start": self.start.to_dict(), "end": self.end.to_dict(),
            "length": self.length.to_dict(), "observed_length": self.observed_length.to_dict(),
            "orientation": self.orientation.to_dict(), "evidence_quality": self.evidence_quality,
            "position_uncertainty": self.position_uncertainty.to_dict(),
            "segments": [s.to_dict() for s in self.segments], "room_ids": self.room_ids, "opening_ids": self.opening_ids,
        }


@dataclass
class Ceiling:
    observed: bool
    height: Measurement | None = None
    ambiguous: bool = False
    coverage: float | None = None

    def __post_init__(self):
        if self.observed and self.height is None:
            raise ModelError("an observed ceiling needs a height")
        if not self.observed and self.height is not None:
            raise ModelError("an unobserved ceiling cannot carry a height")

    def to_dict(self) -> dict:
        return {"observed": self.observed, "ambiguous": self.ambiguous, "coverage": self.coverage,
                "height": None if self.height is None else self.height.to_dict()}


@dataclass
class Room:
    id: str
    polygon: list[Point2D]
    area: Measurement
    perimeter: Measurement
    wall_ids: list[str]
    wall_lengths: list[Measurement]
    ceiling: Ceiling
    topology_quality: str
    length: Measurement | None = None
    width: Measurement | None = None
    dimension_method: str | None = None
    adjacent_room_ids: list[str] = field(default_factory=list)
    connected_room_ids: list[str] = field(default_factory=list)
    opening_ids: list[str] = field(default_factory=list)
    opens_to_unmodelled: list[str] = field(default_factory=list)

    def __post_init__(self):
        validate_polygon(self.polygon, f"room {self.id} polygon")
        if self.area.value < 0:
            raise ModelError(f"room {self.id} area cannot be negative")
        if self.perimeter.value < 0:
            raise ModelError(f"room {self.id} perimeter cannot be negative")
        if self.topology_quality not in QUALITIES:
            raise ModelError(f"room topology_quality must be one of {QUALITIES}")
        if (self.length is None) != (self.width is None):
            raise ModelError("room length and width are reported together or not at all")

    def to_dict(self) -> dict:
        opt = lambda m: None if m is None else m.to_dict()
        return {
            "id": self.id, "polygon": [p.to_dict() for p in self.polygon], "area": self.area.to_dict(),
            "perimeter": self.perimeter.to_dict(), "length": opt(self.length), "width": opt(self.width),
            "dimension_method": self.dimension_method, "wall_ids": self.wall_ids,
            "wall_lengths": [m.to_dict() for m in self.wall_lengths], "ceiling": self.ceiling.to_dict(),
            "adjacent_room_ids": self.adjacent_room_ids, "connected_room_ids": self.connected_room_ids,
            "opening_ids": self.opening_ids, "opens_to_unmodelled": self.opens_to_unmodelled,
            "topology_quality": self.topology_quality,
        }


@dataclass
class Opening:
    id: str
    wall_id: str
    type: str  # door | window | opening
    width: Measurement
    position: Point2D  # midpoint between the jambs
    left_jamb: Point2D
    right_jamb: Point2D
    existence_quality: str
    type_quality: str
    observability: str
    height: Measurement | None = None
    sill_height: Measurement | None = None
    room_ids: list[str] = field(default_factory=list)
    connects: list[str] = field(default_factory=list)  # room ids and/or "unmodelled"
    connected_room_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.type not in OPENING_TYPES:
            raise ModelError(f"opening type must be one of {OPENING_TYPES}")
        if self.width.value <= 0:
            raise ModelError("opening width must be positive")
        for q in (self.existence_quality, self.type_quality, self.observability):
            if q not in QUALITIES:
                raise ModelError(f"opening quality must be one of {QUALITIES}")

    def to_dict(self) -> dict:
        opt = lambda m: None if m is None else m.to_dict()
        return {
            "id": self.id, "wall_id": self.wall_id, "type": self.type, "width": self.width.to_dict(),
            "height": opt(self.height), "sill_height": opt(self.sill_height), "position": self.position.to_dict(),
            "left_jamb": self.left_jamb.to_dict(), "right_jamb": self.right_jamb.to_dict(),
            "room_ids": self.room_ids, "connects": self.connects, "connected_room_ids": self.connected_room_ids,
            "existence_quality": self.existence_quality, "type_quality": self.type_quality,
            "observability": self.observability,
        }


@dataclass
class DamageRegion:
    """Placeholder for the damage layer (not implemented yet): the list in the property stays empty."""

    id: str
    polygon: list[Point2D]
    damage_type: str
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "polygon": [p.to_dict() for p in self.polygon], "damage_type": self.damage_type,
                "confidence": self.confidence}


@dataclass
class Footprint:
    polygon: list[Point2D]
    area: Measurement
    kind: str
    complete: bool
    note: str

    def __post_init__(self):
        validate_polygon(self.polygon, "footprint polygon")

    def to_dict(self) -> dict:
        return {"polygon": [p.to_dict() for p in self.polygon], "area": self.area.to_dict(), "kind": self.kind,
                "complete": self.complete, "note": self.note}


@dataclass
class Property:
    capture: dict
    status: str  # complete | partial
    rooms: list[Room]
    walls: list[Wall]
    openings: list[Opening]
    unverified_openings: list[Opening] = field(default_factory=list)  # low confidence: kept out of the product plan
    footprint: Footprint | None = None
    damage: list[DamageRegion] = field(default_factory=list)
    concealed_damage_flags: list[dict] = field(default_factory=list)
    scope_line_items: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.status not in STATUSES:
            raise ModelError(f"status must be one of {STATUSES}")
        for kind, items in (("room", self.rooms), ("wall", self.walls), ("opening", self.openings + self.unverified_openings)):
            ids = [i.id for i in items]
            if len(ids) != len(set(ids)):
                raise ModelError(f"duplicate {kind} ids")
        wall_ids = {w.id for w in self.walls}
        room_ids = {r.id for r in self.rooms}
        opening_ids = {o.id for o in self.openings}
        for r in self.rooms:
            for w in r.wall_ids:
                if w not in wall_ids:
                    raise ModelError(f"room {r.id} references unknown wall {w}")
            for rid in r.adjacent_room_ids + r.connected_room_ids:
                if rid not in room_ids:
                    raise ModelError(f"room {r.id} references unknown room {rid}")
            for oid in r.opening_ids + r.opens_to_unmodelled:
                if oid not in opening_ids:
                    raise ModelError(f"room {r.id} references unknown opening {oid}")
        for o in self.openings + self.unverified_openings:
            if o.wall_id not in wall_ids:
                raise ModelError(f"opening {o.id} references unknown wall {o.wall_id}")
            for rid in o.room_ids + [c for c in o.connects if c != UNMODELLED]:
                if rid not in room_ids:
                    raise ModelError(f"opening {o.id} references unknown room {rid}")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "capture": self.capture,
            "property": {
                "status": self.status,
                "footprint": None if self.footprint is None else self.footprint.to_dict(),
                "room_count": len(self.rooms), "wall_count": len(self.walls), "opening_count": len(self.openings),
            },
            "rooms": [r.to_dict() for r in self.rooms],
            "walls": [w.to_dict() for w in self.walls],
            "openings": [o.to_dict() for o in self.openings],
            "unverified_openings": [o.to_dict() for o in self.unverified_openings],
            "damage": [d.to_dict() for d in self.damage],
            "concealed_damage_flags": list(self.concealed_damage_flags),
            "scope_line_items": list(self.scope_line_items),
            "warnings": list(self.warnings),
            "provenance": self.provenance,
            "timing": self.timing,
        }
