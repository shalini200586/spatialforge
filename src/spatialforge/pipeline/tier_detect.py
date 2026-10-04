"""Tier auto-detection for `--tier auto`. It decides only when the input is unambiguous; otherwise it says why and asks for --tier.

    LiDAR  : a directory with `odometry.csv` and a `depth/` folder (directly, or inside its single sub-folder: the capture layout).
    video  : a file with a supported video extension.
    photos : a directory whose sub-folders hold images (one folder per room), and that is NOT a LiDAR capture.

Anything else - an empty directory, a directory that looks like both, a directory holding a video, a single image, an unknown
file type - is refused with the reason. Detection looks at names only; it never decodes data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spatialforge.pipeline.messages import UserError

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif")
ASK = "Pass the tier explicitly with --tier lidar, --tier video or --tier photos."


@dataclass
class TierDecision:
    tier: str
    reason: str


def _is_lidar_dir(d: Path) -> bool:
    return (d / "odometry.csv").is_file() and (d / "depth").is_dir()


def _has_images(d: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in IMAGE_EXT for p in d.iterdir())
    except OSError:
        return False


def detect_tier(path: str | Path) -> TierDecision:
    p = Path(path)
    if not p.exists():
        raise UserError("INPUT_NOT_FOUND", f"input does not exist: {p}", "Check the path you passed.")
    if p.is_file():
        ext = p.suffix.lower()
        if ext in VIDEO_EXT:
            return TierDecision("video", f"{p.name} is a video file ({ext})")
        if ext in IMAGE_EXT:
            raise UserError("TIER_AMBIGUOUS", f"{p.name} is a single image file; the photo tier takes a directory of room folders.", ASK)
        raise UserError("TIER_AMBIGUOUS", f"cannot tell the tier of {p.name}: unsupported file type {ext or '(none)'}", ASK)
    subdirs = sorted((c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")), key=lambda c: c.name.lower())
    files = [c for c in p.iterdir() if c.is_file()]
    lidar_here = _is_lidar_dir(p)
    nested = [c for c in subdirs if _is_lidar_dir(c)] if len(subdirs) == 1 else []
    capture_dirs = {"depth", "confidence"} if lidar_here else set()  # a LiDAR capture's own image folders are not rooms
    photo_dirs = [c for c in subdirs if _has_images(c) and c.name.lower() not in capture_dirs]
    videos = [c.name for c in files if c.suffix.lower() in VIDEO_EXT]
    looks_lidar = lidar_here or bool(nested)
    looks_photos = bool(photo_dirs) and not looks_lidar
    if looks_lidar and photo_dirs and lidar_here:
        raise UserError("TIER_AMBIGUOUS", f"{p.name} looks like a LiDAR capture (odometry.csv + depth/) AND has sub-folders of images.", ASK)
    if looks_lidar:
        return TierDecision("lidar", "odometry.csv and depth/ found" + (" inside the single sub-folder" if nested and not lidar_here else ""))
    if videos and not photo_dirs:
        raise UserError("TIER_AMBIGUOUS", f"{p.name} is a directory holding a video ({videos[0]}): pass the video FILE.", ASK)
    if videos and photo_dirs:
        raise UserError("TIER_AMBIGUOUS", f"{p.name} holds both a video ({videos[0]}) and image folders.", ASK)
    if looks_photos:
        return TierDecision("photos", f"{len(photo_dirs)} sub-folder(s) of images and no LiDAR files")
    if not subdirs and not files:
        raise UserError("TIER_AMBIGUOUS", f"{p.name} is an empty directory.", ASK)
    raise UserError("TIER_AMBIGUOUS", f"cannot tell the tier of {p.name}: it is neither a LiDAR capture (odometry.csv + depth/), a "
                    "video file, nor a directory of room folders containing images.", ASK)
