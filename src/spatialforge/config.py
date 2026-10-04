"""Product configuration: tiers, the options a user can set, and where model weights live.

Only externally meaningful settings are here (tier, sampling, damage, diagnostics, overwrite, debug, model location). Algorithm
thresholds stay next to the algorithms that own them (`PlaneOptions`, `WallOptions`, `StitchOptions`, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TIERS = ("lidar", "video", "photos")
TIER_CHOICES = TIERS + ("auto",)
MODELS_ENV = "SPATIALFORGE_MODELS"

DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SETUP_COMMAND = "python scripts/download_models.py"


def models_dir() -> Path:
    """Root of the model cache (outside the repository). `%SPATIALFORGE_MODELS%` if set, else `~/.spatialforge/models`."""
    env = os.environ.get(MODELS_ENV)
    return Path(env) if env else Path.home() / ".spatialforge" / "models"


def depth_weights_present() -> bool:
    """Is the metric-depth model already in the local cache? (process never downloads: it only uses what is cached)"""
    snaps = models_dir() / "hf" / ("models--" + DEPTH_MODEL_REPO.replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any(snaps.glob("*/model.safetensors"))


def model_setup_hint() -> str:
    return (f"Download the metric-depth weights once with:  {SETUP_COMMAND}\n"
            f"  (they are cached in {models_dir()} - set the {MODELS_ENV} environment variable to use another folder)")


@dataclass
class ProcessConfig:
    """What the user asked `process` to do."""

    tier: str = "auto"  # lidar | video | photos | auto
    damage: bool = True  # visible-damage analysis on the tier's RGB evidence
    diagnostics: bool = False  # richer artifacts: point clouds, keyframes (normal runs keep the output small)
    overwrite: bool = False  # replace an existing non-empty result folder
    debug: bool = False  # keep tracebacks in the console and the run log
    # sampling
    lidar_max_frames: int = 400
    lidar_frame_step: int = 1
    video_max_keyframes: int = 120
    video_max_depth_frames: int = 64
