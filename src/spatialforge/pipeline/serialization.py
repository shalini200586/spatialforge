"""Deterministic JSON output for the canonical property."""

from __future__ import annotations

import json
import math
from pathlib import Path

from spatialforge.pipeline.models import Property
from spatialforge.pipeline.schema import build_schema, validate_against_schema

# Everything inside these top-level keys may legitimately differ between identical runs.
VOLATILE_KEYS = ("timing",)


def _clean(v, nd: int = 4):
    if isinstance(v, dict):
        return {str(k): _clean(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x, nd) for x in v]
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite number in output")
        r = round(v, nd)
        return 0.0 if r == 0 else r  # no "-0.0"
    return v


def property_to_dict(prop: Property) -> dict:
    """Validated, rounded plain-dict form. Raises if it breaks the model rules or the published schema."""
    d = _clean(prop.to_dict())
    problems = validate_against_schema(d, build_schema())
    if problems:
        raise ValueError("property does not match schema v1.0: " + "; ".join(problems[:5]))
    return d


def dumps(d: dict) -> str:
    """Sorted keys, fixed indentation, trailing newline: identical input gives identical bytes."""
    return json.dumps(d, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_json(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(d), encoding="utf-8", newline="\n")


def strip_volatile(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in VOLATILE_KEYS}
