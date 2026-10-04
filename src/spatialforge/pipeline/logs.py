"""Console progress and the detailed run log.

Console: one concise line per step, `[3/9] Walls`, on stderr via the `spatialforge.progress` logger. Everything else (library
chatter, per-room details) goes to `diagnostics/run.log`. Native library noise is switched off where the library allows it
(COLMAP's glog output, Hugging Face progress bars).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable

from spatialforge.pipeline import stages

PROGRESS = logging.getLogger("spatialforge.progress")
LOG = logging.getLogger("spatialforge")
_MARK = "_spatialforge_handler"

# raw stage name -> console label
_LABELS = {
    "capture_validation": "Validate input", "video_validation": "Validate input", "photo_validation": "Validate input",
    "metric_reconstruction_and_pose_check": "Reconstruct metric geometry", "keyframe_extraction": "Select keyframes",
    "sfm_reconstruction": "Structure-from-motion", "metric_depth_and_scale": "Metric scale (depth model)",
    "depth_fusion": "Fuse depth", "gravity_alignment": "Gravity alignment", "floor_and_ceiling_planes": "Floor and ceiling",
    "structural_walls": "Walls", "room_topology": "Rooms", "openings": "Openings",
    "features_and_exhaustive_matching": "Features and matching", "cross_room_stitching": "Cross-room evidence",
    "global_placement": "Place rooms", "damage_analysis": "Damage analysis", "rendering": "Render plan",
    "serialization": "Write results",
}
_PLANS = {
    "lidar": ["capture_validation", "metric_reconstruction_and_pose_check", "floor_and_ceiling_planes", "structural_walls",
              "room_topology", "openings"],
    "video": ["video_validation", "keyframe_extraction", "sfm_reconstruction", "metric_depth_and_scale", "depth_fusion",
              "gravity_alignment", "floor_and_ceiling_planes", "structural_walls", "room_topology", "openings"],
    "photos": ["photo_validation", "features_and_exhaustive_matching", "ROOMS", "cross_room_stitching", "global_placement"],
}


def plan_for(tier: str, damage: bool) -> list[str]:
    return list(_PLANS.get(tier, [])) + (["damage_analysis"] if damage else []) + ["rendering", "serialization"]


class Progress:
    """Announces each step once, numbered within the tier's plan."""

    def __init__(self, tier: str, damage: bool):
        self.plan = plan_for(tier, damage)
        self.seen: set[str] = set()

    def __call__(self, raw: str) -> None:
        key = "ROOMS" if raw.startswith("room_") and raw.endswith("_reconstruction") else raw
        if key in self.seen:
            return
        self.seen.add(key)
        label = "Reconstruct rooms" if key == "ROOMS" else _LABELS.get(raw, raw)
        if key in self.plan:
            PROGRESS.info("[%d/%d] %s", self.plan.index(key) + 1, len(self.plan), label)
        else:
            PROGRESS.info("      %s", label)
        LOG.info("stage start: %s", raw)


def quiet_libraries() -> None:
    """Keep third-party libraries from flooding the console."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for name in ("transformers", "huggingface_hub", "PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import pycolmap

        pycolmap.logging.minloglevel = 2
        pycolmap.logging.alsologtostderr = False
    except Exception:
        pass


def setup(tier: str, damage: bool, log_file: Path | None = None, debug: bool = False, console: bool = True) -> Callable[[], None]:
    """Install the console progress handler, the run-log handler and the stage listener. Returns a cleanup function."""
    quiet_libraries()
    handlers: list[tuple[logging.Logger, logging.Handler]] = []
    LOG.setLevel(logging.DEBUG)
    if console:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(message)s"))
        setattr(h, _MARK, True)
        PROGRESS.addHandler(h)
        PROGRESS.setLevel(logging.INFO)
        PROGRESS.propagate = False  # progress lines go to the console only; the run log has its own copy below
        handlers.append((PROGRESS, h))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        setattr(fh, _MARK, True)
        LOG.addHandler(fh)
        PROGRESS.addHandler(fh)
        handlers += [(LOG, fh), (PROGRESS, fh)]
    stages.set_stage_listener(Progress(tier, damage))

    def cleanup() -> None:
        stages.set_stage_listener(None)
        for logger, h in handlers:
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    return cleanup


def announce(raw: str) -> None:
    """Announce a step that is not wrapped in a StageTimer (rendering, serialization)."""
    stages.notify_stage(raw)
