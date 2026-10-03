"""Damage class vocabulary: every label that enters the system is normalised to one of the canonical classes."""

from __future__ import annotations

import re

from spatialforge.pipeline.models import DAMAGE_CLASSES

_SYNONYMS = {
    "crack": {"crack", "cracks", "cracking", "crack like", "crack like damage", "hairline crack", "fissure", "fissures", "split",
              "fracture", "fractured", "wall crack", "ceiling crack"},
    "stain": {"stain", "stains", "stained", "discoloration", "discolouration", "discolored", "discoloured", "discolor",
              "water stain", "moisture stain", "damp", "damp patch", "mould", "mold", "mildew", "water mark", "watermark",
              "tide mark", "blotch"},
}
_LOOKUP = {label: cls for cls, labels in _SYNONYMS.items() for label in labels}


def _clean(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/]+", " ", str(label).strip().lower()))


def normalize_damage_class(label: str) -> str | None:
    """Canonical class for a free-text label ("Hairline-Crack" -> "crack"), or None if it is not a known visible-damage class."""
    c = _clean(label)
    if c in DAMAGE_CLASSES:
        return c
    return _LOOKUP.get(c) or _LOOKUP.get(c.rstrip("s"))


def require_damage_class(label: str) -> str:
    cls = normalize_damage_class(label)
    if cls is None:
        raise ValueError(f"unknown damage class {label!r}; known classes: {', '.join(DAMAGE_CLASSES)}")
    return cls
