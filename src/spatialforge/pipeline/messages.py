"""Structured warnings and errors.

Stage code still produces readable strings. This module turns each one into a record

    {"code": "VIDEO_TRACKING_WEAK", "stage": "frontend", "severity": "warning", "message": "...", "recoverable": true}

by ordered pattern rules, so every tier reports the same vocabulary. The readable `message` is always preserved. A message no
rule knows becomes `GENERAL_WARNING` (never dropped): add a rule here when a new warning is introduced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

SEVERITIES = ("info", "warning", "error")


@dataclass
class WarningRecord:
    code: str
    stage: str
    severity: str
    message: str
    recoverable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# (regex on the message, code, standard stage, severity, recoverable) - first match wins
_RULES: list[tuple[str, str, str, str, bool]] = [
    # --- input / validation ---
    (r"^input: .*no EXIF metadata", "PHOTO_NO_EXIF", "validation", "info", True),
    (r"^input: .*not in a room folder|^input: ", "INPUT_NOTE", "validation", "info", True),
    (r"^capture: ", "CAPTURE_VALIDATION_NOTE", "validation", "warning", True),
    (r"^Camera intrinsics were estimated", "INTRINSICS_ESTIMATED", "frontend", "info", True),
    # --- video / shared SfM ---
    (r"SfM tracking quality is WEAK", "VIDEO_TRACKING_WEAK", "frontend", "warning", True),
    (r"disconnected models", "SFM_DISCONNECTED_MODELS", "frontend", "warning", True),
    (r"Metric scale is NOT available|Metric depth is unavailable|Metric scale could not be established", "METRIC_SCALE_UNAVAILABLE", "metric_scale", "error", False),
    (r"Metric scale is weak|metric scale is weak", "VIDEO_WEAK_SCALE", "metric_scale", "warning", True),
    (r"Tracking is weak, so the property is reported as partial", "VIDEO_TRACKING_WEAK", "frontend", "warning", True),
    (r"Gravity \(vertical\) confidence is low", "GRAVITY_LOW_CONFIDENCE", "gravity", "warning", True),
    (r"tracking was weak: retried once|denser-keyframe attempt", "VIDEO_DENSER_RETRY", "frontend", "info", True),
    (r"axes agree to within|vertical axis differs from the camera-path normal", "GRAVITY_AXES_DISAGREE", "gravity", "warning", True),
    (r"depth keyframes were excluded from fusion|photo\(s\) were excluded from fusion", "DEPTH_FRAMES_EXCLUDED", "geometry", "warning", True),
    (r"Video tier: metres come from a monocular", "VIDEO_SCALE_ESTIMATED", "metric_scale", "info", True),
    (r"fused metric cloud has too few points", "CLOUD_TOO_SPARSE", "geometry", "error", False),
    (r"fewer than three usable photos", "PHOTO_FEW_IMAGES_FUSION", "geometry", "info", True),
    # --- photos / stitching ---
    (r"is not part of the property plan", "PHOTO_ROOM_UNPLACED", "stitching", "warning", True),
    (r"scale_inconsistent", "PHOTO_SCALE_INCONSISTENT", "stitching", "warning", True),
    (r"Stitch constraint .* was rejected", "PHOTO_STITCH_REJECTED", "stitching", "warning", True),
    (r"room placement\(s\) rest on weak evidence", "PHOTO_WEAK_STITCH", "stitching", "warning", True),
    (r"Placed rooms .* overlap", "PHOTO_ROOM_OVERLAP", "stitching", "warning", True),
    (r"Photo tier: metric scale is estimated", "PHOTO_SCALE_ESTIMATED", "metric_scale", "info", True),
    (r"local reconstruction quality is weak", "PHOTO_ROOM_WEAK", "geometry", "warning", True),
    (r"local room polygons were found", "PHOTO_ROOM_AMBIGUOUS", "rooms", "warning", True),
    # --- structure ---
    (r"Floor plane not found|Floor/ceiling stage failed|floor candidate covers only", "STRUCTURE_NO_FLOOR", "geometry", "warning", True),
    (r"Wall stage failed|Room stage failed|Opening stage failed", "STRUCTURE_STAGE_FAILED", "structure", "warning", True),
    (r"No closed rooms were recovered", "STRUCTURE_NO_ROOMS", "rooms", "warning", True),
    (r"structural walls do not form any closed room|Incomplete room topology", "STRUCTURE_INCOMPLETE_TOPOLOGY", "rooms", "warning", True),
    (r"Ceiling height was not observed", "STRUCTURE_CEILING_NOT_OBSERVED", "geometry", "info", True),
    (r"ceiling at .* local heights differ|separate ceiling levels", "STRUCTURE_CEILING_VARIATION", "geometry", "info", True),
    (r"floor tilt", "STRUCTURE_FLOOR_TILT", "geometry", "warning", True),
    (r"Opening recall has not been verified", "OPENINGS_RECALL_UNVERIFIED", "openings", "info", True),
    (r"low-confidence opening candidate", "OPENINGS_LOW_CONFIDENCE_EXCLUDED", "openings", "info", True),
    (r"was dropped from the output", "ROOM_DROPPED", "rooms", "warning", True),
    (r"Stage\(s\) failed and were skipped", "STAGE_FAILED_SKIPPED", "pipeline", "warning", True),
    # --- damage ---
    (r"^damage: .*could not be read", "DAMAGE_EVIDENCE_UNREADABLE", "damage", "warning", True),
    (r"no reliable metric extent", "DAMAGE_NO_METRIC_EXTENT", "damage", "info", True),
    (r"^damage: .*could not be associated with any surface", "DAMAGE_UNMAPPED_DROPPED", "damage", "info", True),
    (r"no RGB evidence with usable geometry", "DAMAGE_NO_EVIDENCE", "damage", "info", True),
    (r"^damage: ", "DAMAGE_NOTE", "damage", "info", True),
    (r"Damage analysis failed", "DAMAGE_FAILED", "damage", "warning", True),
    (r"Damage analysis is deterministic image processing", "DAMAGE_CANDIDATES_ONLY", "damage", "info", True),
    (r"concealed-damage RISK flag", "DAMAGE_RISK_FLAGS", "damage", "info", True),
    (r"Damage detection is not part of this version", "DAMAGE_NOT_RUN", "damage", "info", True),
    # --- general ---
    (r"No physical ground truth", "NO_GROUND_TRUTH", "pipeline", "info", True),
    (r"Diagnostics could not be", "DIAGNOSTICS_INCOMPLETE", "serialization", "warning", True),
]
_COMPILED = [(re.compile(p), c, s, v, r) for p, c, s, v, r in _RULES]


def classify_warning(message: str) -> WarningRecord:
    for rx, code, stage, severity, recoverable in _COMPILED:
        if rx.search(message):
            return WarningRecord(code, stage, severity, message, recoverable)
    return WarningRecord("GENERAL_WARNING", "pipeline", "warning", message, True)


def classify_all(messages: list[str]) -> list[WarningRecord]:
    return [classify_warning(m) for m in messages]


# ---------------- errors ----------------


@dataclass
class ErrorRecord:
    code: str
    stage: str  # standard stage name
    message: str
    hint: str | None = None
    recoverable: bool = False
    raw_stage: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class UserError(Exception):
    """An expected, user-facing failure (bad input, missing weights, output folder in the way). No traceback by default."""

    def __init__(self, code: str, message: str, hint: str | None = None, stage: str = "validation"):
        super().__init__(message)
        self.code, self.message, self.hint, self.stage = code, message, hint, stage

    def record(self) -> ErrorRecord:
        return ErrorRecord(self.code, self.stage, self.message, self.hint)


_ERROR_RULES: list[tuple[str, str, str | None]] = [
    (r"does not exist|no such file|not a directory|is not a directory", "INPUT_NOT_FOUND", "Check the path you passed."),
    (r"unsupported container|unsupported file type|HEIC|is not a supported", "INPUT_UNSUPPORTED", "Convert the input to a supported format."),
    (r"could not be opened|cannot be opened|could not be decoded|no frame could be decoded|cannot decode|truncated", "INPUT_UNREADABLE", "The file may be corrupt: re-export or re-copy it."),
    (r"too short|too small|has \d+ supported photo|photos; at most|no room sub-folders|needs at least", "INPUT_INVALID", "See the README for the expected input layout."),
    (r"model weights not found|weights not found", "MODEL_WEIGHTS_MISSING", None),
    (r"Metric scale is NOT available|metric scale", "METRIC_SCALE_UNAVAILABLE", "Metric dimensions need a metric-depth model and a trajectory that agrees with it."),
    (r"no coherent reconstruction|SfM failed|no reconstruction was produced", "SFM_FAILED", "The footage lacks texture, overlap or parallax: capture slower, with more overlap."),
    (r"depth/|odometry|capture is INVALID|capture_validation|missing", "INPUT_INVALID", "Run `python -m spatialforge validate-lidar <folder>` for details."),
]
_ERROR_COMPILED = [(re.compile(p, re.I), c, h) for p, c, h in _ERROR_RULES]


def classify_error(message: str, stage: str, raw_stage: str | None = None) -> ErrorRecord:
    for rx, code, hint in _ERROR_COMPILED:
        if rx.search(message):
            return ErrorRecord(code, stage, message, hint, False, raw_stage)
    if stage == "validation":  # anything the validators reject is a problem with the input
        return ErrorRecord("INPUT_INVALID", stage, message, "See the README for the expected input layout.", False, raw_stage)
    return ErrorRecord("PIPELINE_ERROR", stage, message, None, False, raw_stage)
