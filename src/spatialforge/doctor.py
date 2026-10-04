"""`python -m spatialforge doctor`: is this machine ready to run each tier? Needs no capture data and never downloads anything."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from spatialforge import __version__
from spatialforge import config as cfgmod

# (import name, distribution name, what it is for)
PACKAGES = [
    ("numpy", "numpy", "core arithmetic"),
    ("PIL", "Pillow", "16-bit PNG depth decoding, plan rendering"),
    ("open3d", "open3d", "ICP drift correction, point clouds (LiDAR)"),
    ("cv2", "opencv-python-headless", "video decoding, SfM keyframes, damage analysis"),
    ("pycolmap", "pycolmap", "structure-from-motion (video, photos)"),
    ("torch", "torch", "metric depth model runtime (video, photos)"),
    ("transformers", "transformers", "metric depth model (video, photos)"),
    ("huggingface_hub", "huggingface_hub", "model download script only"),
]
TIER_NEEDS = {
    "LiDAR": ["numpy", "PIL", "open3d"],
    "Video": ["numpy", "PIL", "cv2", "pycolmap", "torch", "transformers"],
    "Photos": ["numpy", "PIL", "cv2", "pycolmap", "torch", "transformers"],
    "Damage": ["numpy", "PIL", "cv2"],
}
TIER_INSTALL = {"LiDAR": "pip install -e .", "Video": "pip install -e .[video]", "Photos": "pip install -e .[video]", "Damage": "pip install -e .[damage]"}


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".sf_doctor_"):
            pass
        return True
    except OSError:
        return False


def run_doctor() -> dict:
    # find_spec: report presence without importing heavy libraries (torch alone takes seconds)
    pkgs = {}
    for mod, dist, why in PACKAGES:
        present = importlib.util.find_spec(mod) is not None
        pkgs[mod] = {"distribution": dist, "present": present, "version": _version(dist) if present else None, "used_for": why}
    weights = cfgmod.depth_weights_present()
    models = cfgmod.models_dir()
    py_ok = sys.version_info[:2] == (3, 11)
    tiers = {}
    for tier, needs in TIER_NEEDS.items():
        missing = [n for n in needs if not pkgs[n]["present"]]
        reasons = [f"missing {', '.join(missing)} ({TIER_INSTALL[tier]})"] if missing else []
        if tier in ("Video", "Photos") and not weights:
            reasons.append(f"metric-depth weights not found: {cfgmod.SETUP_COMMAND}")
        tiers[tier] = {"ready": not reasons, "reasons": reasons}
    return {
        "spatialforge_version": __version__,
        "python": platform.python_version(), "python_ok": py_ok, "python_note": None if py_ok else "Python 3.11 is the tested version",
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})", "packages": pkgs,
        "models": {"directory": str(models), "from_environment": cfgmod.MODELS_ENV in os.environ, "depth_weights_present": weights,
                   "repository": cfgmod.DEPTH_MODEL_REPO, "setup_command": cfgmod.SETUP_COMMAND},
        "write_access": {"current_directory": _writable(Path.cwd()), "models_directory": _writable(models) if models.exists() or models.parent.exists() else None},
        "tiers": tiers,
    }


def format_doctor(d: dict) -> str:
    from spatialforge.pipeline.provenance import redact_text

    L = [f"SpatialForge {d['spatialforge_version']}", f"Python  {d['python']}" + ("" if d["python_ok"] else f"   [{d['python_note']}]"), f"OS      {d['os']}", "",
         "Packages:"]
    for mod, p in d["packages"].items():
        L.append(f"  [{'ok' if p['present'] else '--'}] {p['distribution']:<24} {p['version'] or 'not installed':<14} {p['used_for']}")
    m = d["models"]
    L += ["", f"Metric-depth model: {m['repository']}", f"  folder: {redact_text(m['directory'])}" + ("  (from SPATIALFORGE_MODELS)" if m["from_environment"] else "  (default)"),
          f"  weights: {'present' if m['depth_weights_present'] else 'NOT FOUND  ->  ' + m['setup_command']}", "",
          f"Write access: current directory {'ok' if d['write_access']['current_directory'] else 'NO'}; models folder "
          f"{'ok' if d['write_access']['models_directory'] else 'not checked' if d['write_access']['models_directory'] is None else 'NO'}", "", "Tiers:"]
    for tier, t in d["tiers"].items():
        L.append(f"  {tier:<7} {'READY' if t['ready'] else 'NOT READY'}" + ("" if t["ready"] else "   " + "; ".join(t["reasons"])))
    return "\n".join(L)
