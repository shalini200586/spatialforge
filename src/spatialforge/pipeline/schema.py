"""JSON Schema for property.json (schema v1.0) and a minimal validator for it.

Case-study materials supplied no separate schema file; SpatialForge defines schema v1.0. This is not an official
evaluator schema. The validator supports only the JSON Schema keywords used below, so output can be checked without
adding a dependency.
"""

from __future__ import annotations

import math

from spatialforge.pipeline.models import OPENING_TYPES, QUALITIES, SCHEMA_VERSION, STATUSES

NUM = {"type": "number"}
NUM_OR_NULL = {"type": ["number", "null"]}
STR = {"type": "string"}
STRS = {"type": "array", "items": STR}
QUALITY = {"type": "string", "enum": list(QUALITIES)}


def _ref(name: str) -> dict:
    return {"$ref": f"#/$defs/{name}"}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": sorted(required if required is not None else props),
            "additionalProperties": False}


def build_schema() -> dict:
    defs = {
        "Point": _obj({"x": NUM, "z": NUM}),
        "Interval": _obj({"low": NUM, "high": NUM}),
        "Measurement": _obj({
            "value": NUM, "unit": STR, "interval": {"anyOf": [_ref("Interval"), {"type": "null"}]},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "quality": {"type": ["string", "null"], "enum": list(QUALITIES) + [None]}, "source_tier": STR,
        }),
        "Segment": _obj({"start": _ref("Point"), "end": _ref("Point"), "length": _ref("Measurement")}),
        "Wall": _obj({
            "id": STR, "start": _ref("Point"), "end": _ref("Point"), "length": _ref("Measurement"),
            "observed_length": _ref("Measurement"), "orientation": _ref("Measurement"), "evidence_quality": QUALITY,
            "position_uncertainty": _ref("Measurement"), "segments": {"type": "array", "items": _ref("Segment")},
            "room_ids": STRS, "opening_ids": STRS,
        }),
        "Ceiling": _obj({"observed": {"type": "boolean"}, "ambiguous": {"type": "boolean"}, "coverage": NUM_OR_NULL,
                         "height": {"anyOf": [_ref("Measurement"), {"type": "null"}]}}),
        "Room": _obj({
            "id": STR, "polygon": {"type": "array", "items": _ref("Point"), "minItems": 3},
            "area": _ref("Measurement"), "perimeter": _ref("Measurement"),
            "length": {"anyOf": [_ref("Measurement"), {"type": "null"}]},
            "width": {"anyOf": [_ref("Measurement"), {"type": "null"}]},
            "dimension_method": {"type": ["string", "null"]}, "wall_ids": STRS,
            "wall_lengths": {"type": "array", "items": _ref("Measurement")}, "ceiling": _ref("Ceiling"),
            "adjacent_room_ids": STRS, "connected_room_ids": STRS, "opening_ids": STRS, "opens_to_unmodelled": STRS,
            "topology_quality": QUALITY,
        }),
        "Opening": _obj({
            "id": STR, "wall_id": STR, "type": {"type": "string", "enum": list(OPENING_TYPES)},
            "width": _ref("Measurement"), "height": {"anyOf": [_ref("Measurement"), {"type": "null"}]},
            "sill_height": {"anyOf": [_ref("Measurement"), {"type": "null"}]}, "position": _ref("Point"),
            "left_jamb": _ref("Point"), "right_jamb": _ref("Point"), "room_ids": STRS, "connects": STRS,
            "connected_room_ids": STRS, "existence_quality": QUALITY, "type_quality": QUALITY, "observability": QUALITY,
        }),
        "Footprint": _obj({"polygon": {"type": "array", "items": _ref("Point"), "minItems": 3},
                           "area": _ref("Measurement"), "kind": STR, "complete": {"type": "boolean"}, "note": STR}),
    }
    top = _obj({
        "schema_version": {"const": SCHEMA_VERSION},
        "capture": {"type": "object", "required": ["tier", "source"]},
        "property": _obj({
            "status": {"type": "string", "enum": list(STATUSES)},
            "footprint": {"anyOf": [_ref("Footprint"), {"type": "null"}]},
            "room_count": {"type": "integer", "minimum": 0}, "wall_count": {"type": "integer", "minimum": 0},
            "opening_count": {"type": "integer", "minimum": 0},
        }),
        "rooms": {"type": "array", "items": _ref("Room")},
        "walls": {"type": "array", "items": _ref("Wall")},
        "openings": {"type": "array", "items": _ref("Opening")},
        "unverified_openings": {"type": "array", "items": _ref("Opening")},
        "damage": {"type": "array"}, "concealed_damage_flags": {"type": "array"}, "scope_line_items": {"type": "array"},
        "warnings": STRS, "provenance": {"type": "object"}, "timing": {"type": "object"},
    })
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "spatialforge/property.schema.json",
            "title": f"SpatialForge property schema v{SCHEMA_VERSION}",
            "description": "Defined by SpatialForge; the case-study materials supplied no separate schema file. "
                           "Not an official evaluator schema.",
            "$defs": defs, **top}


# ---------- minimal validator (only the keywords used above) ----------

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
}


def validate_against_schema(instance, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    """Return a list of violation messages (empty = valid)."""
    root = root or schema
    errors: list[str] = []
    if "$ref" in schema:
        return validate_against_schema(instance, root["$defs"][schema["$ref"].split("/")[-1]], root, path)
    if "anyOf" in schema:
        if not any(not validate_against_schema(instance, s, root, path) for s in schema["anyOf"]):
            errors.append(f"{path}: matches none of the allowed forms")
        return errors
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {instance!r}")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](instance) for t in types):
            return [f"{path}: expected type {types}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} above maximum {schema['maximum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required key {key!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{path}: unexpected key {key!r}")
        for key, sub in props.items():
            if key in instance:
                errors += validate_against_schema(instance[key], sub, root, f"{path}.{key}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors += validate_against_schema(item, schema["items"], root, f"{path}[{i}]")
    return errors
