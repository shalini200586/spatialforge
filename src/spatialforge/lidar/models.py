"""Result objects for LiDAR capture validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Status(str, Enum):
    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID WITH WARNINGS"
    INVALID = "INVALID"


@dataclass
class LidarValidationResult:
    capture_path: Path
    status: Status = Status.VALID
    depth_frame_count: int = 0
    confidence_frame_count: int = 0
    pose_frame_count: int = 0
    # Frame IDs that are expected but absent, keyed by a short description.
    missing_frames: dict[str, list[int]] = field(default_factory=dict)
    duplicate_pose_frames: list[int] = field(default_factory=list)
    # Which expected files/folders exist: name -> bool.
    present: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Free-form facts found while inspecting (resolution, dtype, stats, ...).
    metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        if self.errors:
            self.status = Status.INVALID
        elif self.warnings:
            self.status = Status.VALID_WITH_WARNINGS
        else:
            self.status = Status.VALID
