"""MetricScene: the sensor-independent hand-over between a tier front end and the shared structural backend.

A front end (LiDAR, video) produces a gravity-aligned metric point cloud (+Y up, metres); floor/ceiling detection,
walls, rooms and openings consume only this. Nothing in here knows how the points were obtained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

SCALE_QUALITIES = ("sensor", "strong", "moderate", "weak")  # "sensor": metres come from a depth sensor, not an estimate
GEOMETRY_QUALITIES = ("strong", "moderate", "weak")
TIERS = ("lidar", "video", "photos")


class SceneError(ValueError):
    pass


@dataclass
class MetricScene:
    points_xyz_m: np.ndarray  # (N, 3) float, world frame, +Y vertical (up), metres
    source_tier: str
    geometry_quality: str
    scale_quality: str
    camera_poses: np.ndarray | None = None  # (M, 4, 4) camera-to-world, same frame and units as the points
    frame_refs: list[str] = field(default_factory=list)  # keyframe / frame identifiers, one per pose
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Optional, not serialised: lazily builds independent point subsets (opening-stability check).
    subsets_fn: Callable[[], list[np.ndarray]] | None = None

    def __post_init__(self):
        self.points_xyz_m = np.asarray(self.points_xyz_m, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(self.points_xyz_m).all():
            raise SceneError("points contain NaN or infinity")
        if self.source_tier not in TIERS:
            raise SceneError(f"unknown source tier {self.source_tier!r}")
        if self.geometry_quality not in GEOMETRY_QUALITIES:
            raise SceneError(f"unknown geometry quality {self.geometry_quality!r}")
        if self.scale_quality not in SCALE_QUALITIES:
            raise SceneError(f"unknown scale quality {self.scale_quality!r}")
        if self.camera_poses is not None:
            self.camera_poses = np.asarray(self.camera_poses, dtype=np.float64).reshape(-1, 4, 4)
            if self.frame_refs and len(self.frame_refs) != len(self.camera_poses):
                raise SceneError("frame_refs and camera_poses differ in length")

    @property
    def camera_positions(self) -> np.ndarray:
        if self.camera_poses is None or not len(self.camera_poses):
            return np.zeros((0, 3))
        return self.camera_poses[:, :3, 3]

    def frame_subsets(self) -> list[np.ndarray]:
        return self.subsets_fn() if self.subsets_fn is not None else []

    def summary(self) -> dict:
        p = self.points_xyz_m
        return {
            "source_tier": self.source_tier, "geometry_quality": self.geometry_quality, "scale_quality": self.scale_quality,
            "points": int(len(p)), "cameras": int(len(self.camera_positions)),
            "bounds_min_m": [round(float(v), 3) for v in p.min(axis=0)] if len(p) else None,
            "bounds_max_m": [round(float(v), 3) for v in p.max(axis=0)] if len(p) else None,
            "frame_refs": len(self.frame_refs), "warnings": list(self.warnings), "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Round-trippable .npz (float32 points, to keep the file small)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, points=self.points_xyz_m.astype(np.float32),
            poses=self.camera_poses if self.camera_poses is not None else np.zeros((0, 4, 4)),
            meta=np.array(json.dumps({
                "source_tier": self.source_tier, "geometry_quality": self.geometry_quality,
                "scale_quality": self.scale_quality, "frame_refs": self.frame_refs, "warnings": self.warnings,
                "metadata": self.metadata, "has_poses": self.camera_poses is not None}, sort_keys=True)))

    @classmethod
    def load(cls, path: Path) -> "MetricScene":
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            poses = z["poses"] if meta["has_poses"] else None
            return cls(z["points"].astype(np.float64), meta["source_tier"], meta["geometry_quality"], meta["scale_quality"],
                       poses, meta["frame_refs"], meta["warnings"], meta["metadata"])
