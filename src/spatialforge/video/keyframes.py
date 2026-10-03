"""Deterministic keyframe selection: temporal spacing + sharpness + parallax. No randomness anywhere.

Pass 1 decodes the video once and keeps a small grayscale copy (for scoring) and a JPEG of the working-resolution
frame for every candidate (about `candidate_hz` per second). Selection then runs on that cache:

* blur filter   - a candidate is dropped when its Laplacian variance is below `blur_ratio` of the local median
                  (relative, so it adapts to resolution and scene texture);
* rapid motion  - dropped when its optical-flow displacement from the previous candidate is very large;
* spacing       - at least `min_gap_s` after the previous keyframe;
* parallax      - chosen when it has moved enough since the last keyframe (median LK flow, fraction of image
                  width), or after `max_gap_s` regardless (so a slow pan is not starved);
* thinning      - more than `target_max` keyframes are reduced by repeatedly dropping the one whose removal leaves
                  the smallest gap; fewer than `target_min` triggers a more permissive second selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class KeyframeOptions:
    candidate_hz: float = 6.0
    max_candidates: int = 1000
    work_max_side: int = 1280  # keyframes are stored (and later reconstructed) at this size
    analysis_width: int = 320
    min_gap_s: float = 0.3
    max_gap_s: float = 2.5
    min_parallax: float = 0.04  # median flow displacement / image width
    max_consecutive_motion: float = 0.25  # per-candidate displacement considered a whip-pan
    blur_ratio: float = 0.5
    min_sharpness: float = 5.0
    sharpness_window: int = 15
    target_min: int = 30
    target_max: int = 120


@dataclass
class Candidate:
    index: int  # frame index in the video
    t: float  # seconds
    sharpness: float
    gray: np.ndarray  # small grayscale analysis image
    jpeg: bytes | None = None  # working-resolution frame, encoded


@dataclass
class Keyframe:
    frame_index: int
    t: float
    sharpness: float
    parallax: float | None
    file: str | None = None

    def to_dict(self) -> dict:
        return {"frame_index": self.frame_index, "t_s": round(self.t, 3), "sharpness": round(self.sharpness, 2),
                "parallax": None if self.parallax is None else round(self.parallax, 4), "file": self.file}


@dataclass
class KeyframeResult:
    keyframes: list[Keyframe]
    candidates: int
    rejected_blur: int = 0
    rejected_motion: int = 0
    thinned: int = 0
    relaxed: bool = False
    warnings: list[str] = field(default_factory=list)
    video_facts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"count": len(self.keyframes), "candidates": self.candidates, "rejected_blur": self.rejected_blur,
                "rejected_rapid_motion": self.rejected_motion, "thinned": self.thinned, "relaxed_selection": self.relaxed,
                "warnings": self.warnings, "keyframes": [k.to_dict() for k in self.keyframes]}


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper)."""
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def flow_displacement(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """(median displacement / image width, fraction of features tracked) from sparse Lucas-Kanade flow."""
    import cv2

    pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 8:
        return 0.0, 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return 1.0, 0.0
    d = np.linalg.norm((nxt[ok] - pts[ok]).reshape(-1, 2), axis=1)
    return float(np.median(d) / prev.shape[1]), float(ok.mean())


def select_keyframes(cands: list[Candidate], opts: KeyframeOptions, relax: float = 1.0) -> KeyframeResult:
    """Pure, deterministic selection over cached candidates. `relax` < 1 lowers the spacing/parallax thresholds."""
    min_gap, min_par = opts.min_gap_s * relax, opts.min_parallax * relax
    keys: list[Keyframe] = []
    last: Candidate | None = None
    prev: Candidate | None = None
    blur = motion = 0
    sharps = [c.sharpness for c in cands]
    for i, c in enumerate(cands):
        lo = max(0, i - opts.sharpness_window // 2)
        local = float(np.median(sharps[lo:i + opts.sharpness_window // 2 + 1]))
        rapid = False
        if prev is not None:
            disp, _ = flow_displacement(prev.gray, c.gray)
            rapid = disp > opts.max_consecutive_motion
        prev_c, prev = prev, c
        if c.sharpness < max(opts.min_sharpness, opts.blur_ratio * local):
            blur += 1
            continue
        if rapid:
            motion += 1
            continue
        if last is None:
            keys.append(Keyframe(c.index, c.t, c.sharpness, None))
            last = c
            continue
        dt = c.t - last.t
        if dt < min_gap:
            continue
        par, tracked = flow_displacement(last.gray, c.gray)
        if par >= min_par or dt >= opts.max_gap_s or tracked < 0.15:
            keys.append(Keyframe(c.index, c.t, c.sharpness, par))
            last = c
    return KeyframeResult(keys, len(cands), blur, motion)


def thin_keyframes(keys: list[Keyframe], target: int) -> tuple[list[Keyframe], int]:
    """Drop interior keyframes, smallest resulting gap first (ties: lower sharpness, then earlier)."""
    keys = list(keys)
    dropped = 0
    while len(keys) > target and len(keys) > 2:
        best, best_key = None, None
        for i in range(1, len(keys) - 1):
            gap = keys[i + 1].t - keys[i - 1].t
            key = (gap, keys[i].sharpness, keys[i].t)
            if best_key is None or key < best_key:
                best, best_key = i, key
        keys.pop(best)
        dropped += 1
    return keys, dropped


def collect_candidates(video: str | Path, opts: KeyframeOptions) -> tuple[list[Candidate], dict]:
    """One sequential decode pass. Returns candidates (with encoded work frames) and video facts."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("video cannot be opened")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # apply the container's rotation metadata (phones store portrait video rotated)
    except Exception:
        pass
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not 0 < fps <= 240:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / opts.candidate_hz)))
    if total > 0 and total / stride > opts.max_candidates:
        stride = int(np.ceil(total / opts.max_candidates))
    cands: list[Candidate] = []
    idx = 0
    work_size = None
    while True:
        if not cap.grab():
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                if work_size is None:
                    s = min(1.0, opts.work_max_side / max(h, w))
                    work_size = (int(round(w * s)), int(round(h * s)))
                work = frame if work_size == (w, h) else cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
                aw = opts.analysis_width
                small = cv2.resize(work, (aw, int(round(work.shape[0] * aw / work.shape[1]))), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                ok2, buf = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok2:
                    cands.append(Candidate(idx, idx / fps, sharpness(gray), gray, buf.tobytes()))
        idx += 1
    cap.release()
    return cands, {"fps": fps, "frames_decoded": idx, "stride": stride, "work_size": work_size}


def extract_keyframes(video: str | Path, out_dir: Path | None, opts: KeyframeOptions | None = None) -> KeyframeResult:
    opts = opts or KeyframeOptions()
    cands, facts = collect_candidates(video, opts)
    if not cands:
        raise RuntimeError("no frame could be decoded")
    by_index = {c.index: c for c in cands}
    res = select_keyframes(cands, opts)
    if len(res.keyframes) < opts.target_min:
        relaxed = select_keyframes(cands, opts, relax=0.4)
        if len(relaxed.keyframes) > len(res.keyframes):
            res = relaxed
            res.relaxed = True
            res.warnings.append("fewer than the target number of keyframes at normal thresholds; spacing and parallax "
                                "thresholds were relaxed")
    if len(res.keyframes) > opts.target_max:
        res.keyframes, res.thinned = thin_keyframes(res.keyframes, opts.target_max)
    if len(res.keyframes) < opts.target_min:
        res.warnings.append(f"only {len(res.keyframes)} keyframes could be selected (target {opts.target_min}-{opts.target_max}); "
                            "the video may be short, static, blurred or fast-moving")
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, key in enumerate(res.keyframes):
            name = f"kf_{k:04d}.jpg"
            (out_dir / name).write_bytes(by_index[key.frame_index].jpeg)
            key.file = name
    res.video_facts = facts
    return res
