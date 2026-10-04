"""Runtime provenance (versions, platform) and path redaction.

Nothing personal goes into the outputs: the user's home directory is replaced by `~` everywhere in the written JSON, and
only library VERSIONS are recorded (read from package metadata without importing heavy libraries).
"""

from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path

from spatialforge import __version__

DAMAGE_METHOD = {"name": "deterministic-opencv", "version": "1",
                 "description": "crack (dark thin ridges) and stain (chroma deviation) detectors; no learned model"}
LIBRARIES = ("numpy", "opencv-python-headless", "pycolmap", "open3d", "torch", "transformers", "Pillow")

DETERMINISM = {
    "deterministic": "LiDAR reconstruction, drift stage, planes, walls, rooms, openings, scale estimation, gravity, fusion, "
                     "stitching, damage analysis, serialization and rendering are deterministic for identical inputs.",
    "not_bit_reproducible": ["PyCOLMAP structure-from-motion (video and photo tiers) is multi-threaded: registered image sets "
                             "and poses can differ slightly between runs, and so can everything built on them.",
                             "Video seeking can land a few frames apart across decoder versions."],
    "excluded_from_comparison": ["timing", "run_report.json timestamps and durations"],
}


def library_versions() -> dict:
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            continue
    return out


def runtime_provenance(tier: str, damage_enabled: bool) -> dict:
    return {
        "spatialforge_version": __version__, "tier": tier,
        "python": platform.python_version(), "python_implementation": platform.python_implementation(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "libraries": library_versions(),
        "damage_method": DAMAGE_METHOD if damage_enabled else None,
        "determinism": DETERMINISM,
    }


def _home_variants() -> list[str]:
    home = str(Path.home())
    return sorted({home, home.replace("\\", "/"), home.lower(), home.replace("\\", "/").lower()}, key=len, reverse=True)


def redact_text(s: str) -> str:
    """Replace the user's home directory in a string by `~` (case-insensitive, either slash style)."""
    low = s.lower()
    for h in _home_variants():
        hl = h.lower()
        while hl and hl in low:
            i = low.index(hl)
            s = s[:i] + "~" + s[i + len(h):]
            low = s.lower()
    return s


def scrub(obj):
    """Recursively redact home-directory paths in every string of a JSON-like structure."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj
