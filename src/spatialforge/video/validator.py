"""Video input validation: can this file be decoded, and is there enough of it to reconstruct anything?"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
MIN_DURATION_S = 5.0
MIN_SIDE_PX = 320
LONG_VIDEO_S = 600.0
PROBE_FRAMES = 30


@dataclass
class VideoInfo:
    path: str
    container: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_s: float = 0.0
    codec: str = ""
    probed_frames: int = 0
    seek_ok: bool = False


@dataclass
class VideoValidation:
    info: VideoInfo
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "info": self.info.__dict__}


def _fourcc(code: float) -> str:
    n = int(code)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") if n else ""


def validate_video(path: str | Path, min_duration_s: float = MIN_DURATION_S) -> VideoValidation:
    p = Path(path)
    res = VideoValidation(VideoInfo(str(p), p.suffix.lower().lstrip(".")))
    if not p.exists():
        res.errors.append(f"video file does not exist: {p}")
        return res
    if not p.is_file():
        res.errors.append(f"not a file: {p}")
        return res
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        res.errors.append(f"unsupported container {p.suffix!r} (supported: {', '.join(SUPPORTED_EXTENSIONS)})")
        return res
    if p.stat().st_size == 0:
        res.errors.append("video file is empty")
        return res

    import cv2

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            res.errors.append("the video could not be opened (unreadable or corrupt file, or an unsupported codec)")
            return res
        info = res.info
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info.codec = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))

        n = 0
        last_shape = None
        while n < PROBE_FRAMES:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            n += 1
            last_shape = frame.shape
        info.probed_frames = n
        if n == 0:
            res.errors.append("no frame could be decoded")
            return res
        if last_shape is not None:  # trust the decoded size over the container header
            info.height, info.width = int(last_shape[0]), int(last_shape[1])

        if info.fps <= 0 or info.fps > 240:
            res.warnings.append(f"frame rate {info.fps:.1f} fps looks wrong; timestamps are derived from frame index")
            info.fps = info.fps if 0 < info.fps <= 240 else 30.0
        if info.frame_count <= 0:
            res.warnings.append("frame count is not reported by the container; duration is unknown")
        else:
            info.duration_s = info.frame_count / info.fps
        # a mid-file seek: confirms the video is not truncated after the header
        if info.frame_count > 2 * PROBE_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, info.frame_count // 2)
            ok, frame = cap.read()
            info.seek_ok = bool(ok and frame is not None)
            if not info.seek_ok:
                res.warnings.append("decoding failed at the middle of the video; the file may be truncated")

        if min(info.width, info.height) < MIN_SIDE_PX:
            res.errors.append(f"resolution {info.width}x{info.height} is too small (minimum side {MIN_SIDE_PX} px)")
        if info.frame_count > 0 and info.duration_s < min_duration_s:
            res.errors.append(f"video is too short ({info.duration_s:.1f} s; minimum {min_duration_s:.0f} s)")
        if info.duration_s > LONG_VIDEO_S:
            res.warnings.append(f"long video ({info.duration_s / 60:.1f} min): keyframes are sampled sparsely")
        if 0 < info.fps < 10:
            res.warnings.append(f"low frame rate ({info.fps:.1f} fps): fast camera motion may blur or break tracking")
    finally:
        cap.release()
    return res
