"""Canonical property model, schema and deterministic serialization."""

import json
from pathlib import Path

import pytest
from helpers_pipeline import meas, simple_opening, simple_property, simple_wall, square_room

from spatialforge.pipeline import SCHEMA_VERSION
from spatialforge.pipeline.models import (
    Ceiling, Measurement, ModelError, Opening, Point2D, Property, Room, validate_polygon,
)
from spatialforge.pipeline.schema import build_schema, validate_against_schema
from spatialforge.pipeline.serialization import dumps, property_to_dict, strip_volatile

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "schema" / "property.schema.json"


def test_measurement_serialization():
    d = Measurement(3.42, "m", 3.35, 3.49, 0.72, "moderate", "lidar").to_dict()
    assert d == {"value": 3.42, "unit": "m", "interval": {"low": 3.35, "high": 3.49}, "confidence": 0.72,
                 "quality": "moderate", "source_tier": "lidar"}
    plain = Measurement(1.0, "m").to_dict()
    assert plain["interval"] is None and plain["confidence"] is None and plain["quality"] is None


def test_room_serialization():
    d = square_room().to_dict()
    assert d["id"] == "room_001" and len(d["polygon"]) == 4 and d["area"]["value"] == 16.0
    assert d["length"]["value"] == 4.0 and d["ceiling"]["observed"] and d["ceiling"]["height"]["value"] == 2.4
    assert d["wall_ids"] == ["w1", "w2", "w3", "w4"] and d["dimension_method"] == "opposite_wall_distances"


def test_wall_serialization():
    d = simple_wall().to_dict()
    assert d["evidence_quality"] == "strong" and d["length"]["value"] == 4.0
    assert d["start"] == {"x": 0.0, "z": 0.0} and len(d["segments"]) == 1
    assert d["position_uncertainty"]["unit"] == "m" and d["orientation"]["unit"] == "deg"


def test_opening_serialization():
    d = simple_opening().to_dict()
    assert d["type"] == "door" and d["width"]["interval"] == {"low": 0.85, "high": 0.95}
    assert d["height"]["value"] == 2.05 and d["sill_height"]["value"] == 0.0 and d["connects"] == ["room_001", "unmodelled"]
    assert d["existence_quality"] == "moderate" and d["observability"] == "strong"


def test_missing_optional_values_serialize_as_null():
    room = square_room()
    room.length = room.width = room.dimension_method = None
    room.ceiling = Ceiling(False)
    op = simple_opening()
    op.height = op.sill_height = None
    d_room, d_op = room.to_dict(), op.to_dict()
    assert d_room["length"] is None and d_room["width"] is None and d_room["ceiling"]["height"] is None
    assert d_op["height"] is None and d_op["sill_height"] is None
    prop = simple_property(rooms=[room], openings=[op])
    assert property_to_dict(prop)["property"]["footprint"] is None  # no footprint supplied -> null, not invented


def test_partial_property_and_warnings():
    prop = simple_property(rooms=[], openings=[], status="partial", warnings=["No closed rooms were recovered."])
    d = property_to_dict(prop)
    assert d["property"]["status"] == "partial" and d["rooms"] == [] and d["property"]["room_count"] == 0
    assert d["warnings"] == ["No closed rooms were recovered."]
    assert d["damage"] == [] and d["concealed_damage_flags"] == [] and d["scope_line_items"] == []
    with pytest.raises(ModelError):
        simple_property(status="finished").validate()


def test_schema_version_and_output_validates_against_the_published_schema():
    d = property_to_dict(simple_property())
    assert d["schema_version"] == SCHEMA_VERSION == "1.0"
    assert validate_against_schema(d, build_schema()) == []
    assert json.loads(SCHEMA_FILE.read_text()) == build_schema()  # the committed schema file is in sync with the code
    assert "no separate schema file" in build_schema()["description"]
    broken = dict(d, schema_version="2.0")
    assert validate_against_schema(broken, build_schema())
    assert validate_against_schema({"schema_version": "1.0"}, build_schema())  # missing required keys


def test_schema_cross_check_with_jsonschema_when_available():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(property_to_dict(simple_property()), build_schema())


def test_json_output_is_deterministic_and_sorted():
    a, b = dumps(property_to_dict(simple_property())), dumps(property_to_dict(simple_property()))
    assert a == b and a.endswith("\n")
    keys = list(json.loads(a))
    assert keys == sorted(keys)
    d = property_to_dict(simple_property())
    d2 = property_to_dict(simple_property())
    d2["timing"] = {"total_s": 99}
    assert strip_volatile(d) == strip_volatile(d2)


def test_invalid_measurement_intervals_are_rejected():
    with pytest.raises(ModelError):
        Measurement(3.0, "m", 3.5, 4.0)  # low above the value
    with pytest.raises(ModelError):
        Measurement(3.0, "m", 2.0, 2.5)  # high below the value
    with pytest.raises(ModelError):
        Measurement(3.0, "m", 3.2, 2.8)  # low above high
    with pytest.raises(ModelError):
        Measurement(3.0, "m", 2.0, None)  # half an interval
    with pytest.raises(ModelError):
        Measurement(float("nan"), "m")
    with pytest.raises(ModelError):
        Measurement(1.0, "m", confidence=1.5)
    with pytest.raises(ModelError):
        Measurement(1.0, "m", quality="excellent")


def test_negative_area_is_rejected():
    r = square_room()
    with pytest.raises(ModelError):
        Room(r.id, r.polygon, meas(-1.0, "m2"), r.perimeter, r.wall_ids, r.wall_lengths, r.ceiling, "strong")
    with pytest.raises(ModelError):
        Opening("o", "w1", "door", meas(-0.9), Point2D(0, 0), Point2D(0, 0), Point2D(1, 0), "strong", "strong", "strong")


def test_invalid_polygons_are_rejected():
    with pytest.raises(ModelError):
        validate_polygon([Point2D(0, 0), Point2D(1, 0)])  # two vertices
    with pytest.raises(ModelError):
        validate_polygon([Point2D(0, 0), Point2D(1, 0), Point2D(2, 0)])  # zero area
    with pytest.raises(ModelError, match="self-intersecting"):
        validate_polygon([Point2D(0, 0), Point2D(2, 2), Point2D(2, 0), Point2D(0, 2)])  # bow-tie
    validate_polygon([Point2D(0, 0), Point2D(2, 0), Point2D(2, 2), Point2D(0, 2)])
    r = square_room()
    with pytest.raises(ModelError):
        Room(r.id, r.polygon[:2], r.area, r.perimeter, r.wall_ids, r.wall_lengths, r.ceiling, "strong")


def test_cross_references_are_checked():
    bad = simple_property()
    bad.rooms[0].wall_ids = ["w1", "w2", "missing"]
    with pytest.raises(ModelError, match="unknown wall"):
        bad.validate()
    bad = simple_property()
    bad.openings[0].room_ids = ["room_099"]
    with pytest.raises(ModelError, match="unknown room"):
        bad.validate()
    dup = simple_property()
    dup.walls.append(simple_wall("w1"))
    with pytest.raises(ModelError, match="duplicate"):
        dup.validate()
    with pytest.raises(ModelError):
        Ceiling(True, None)  # an observed ceiling needs a height
    with pytest.raises(ModelError):
        Ceiling(False, meas(2.4))  # an unobserved one cannot have one
