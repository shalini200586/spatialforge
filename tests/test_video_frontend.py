"""Video validation and deterministic keyframe selection."""

import numpy as np
import pytest
from helpers_video import make_test_video

cv2 = pytest.importorskip("cv2")  # the video tier is an optional install (pip install -e .[video])

from spatialforge.video.keyframes import (
    Candidate, KeyframeOptions, extract_keyframes, select_keyframes, sharpness, thin_keyframes, Keyframe,
)
from spatialforge.video.validator import validate_video


# ---------- validator ----------


def test_validator_rejects_a_missing_file(tmp_path):
    r = validate_video(tmp_path / "nope.mp4")
    assert not r.ok and "does not exist" in r.errors[0]


def test_validator_rejects_unreadable_and_unsupported_files(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    r = validate_video(bad)
    assert not r.ok and any("could not be opened" in e or "no frame" in e for e in r.errors)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert not validate_video(empty).ok
    txt = tmp_path / "clip.txt"
    txt.write_text("x")
    assert "unsupported container" in validate_video(txt).errors[0]


def test_validator_rejects_a_trivially_short_video_and_accepts_a_real_one(tmp_path):
    short = make_test_video(tmp_path / "short.mp4", seconds=2)
    r = validate_video(short)
    assert not r.ok and any("too short" in e for e in r.errors)
    good = make_test_video(tmp_path / "good.mp4", seconds=6)
    r = validate_video(good)
    assert r.ok, r.errors
    i = r.info
    assert (i.width, i.height) == (640, 480) and 15 <= i.fps <= 25 and i.frame_count >= 100 and i.probed_frames > 0
    assert i.duration_s == pytest.approx(6.0, abs=0.6)
    assert r.to_dict()["info"]["container"] == "mp4"


def test_validator_flags_a_tiny_resolution(tmp_path):
    p = make_test_video(tmp_path / "tiny.mp4", seconds=6, size=(160, 120))
    assert any("too small" in e for e in validate_video(p).errors)


# ---------- keyframes ----------


def textured(width=320, height=240, seed=1):
    rng = np.random.default_rng(seed)
    base = (rng.random((height, width * 4)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(base, (0, 0), 1.5)


def make_candidates(n=30, dt=1 / 6, shift=8, blur=(), static=False):
    base = textured()
    cands = []
    for i in range(n):
        off = 0 if static else i * shift
        g = base[:, off:off + 320].copy()
        if i in blur:
            g = cv2.GaussianBlur(g, (0, 0), 6.0)
        cands.append(Candidate(i * 4, i * dt, sharpness(g), g))
    return cands


def test_keyframe_selection_is_deterministic():
    opts = KeyframeOptions()
    a = select_keyframes(make_candidates(), opts)
    b = select_keyframes(make_candidates(), opts)
    assert [k.frame_index for k in a.keyframes] == [k.frame_index for k in b.keyframes]
    assert [k.parallax for k in a.keyframes] == [k.parallax for k in b.keyframes]
    assert len(a.keyframes) >= 5


def test_blurred_frames_are_filtered_out():
    blur = {5, 6, 13, 14, 15}
    res = select_keyframes(make_candidates(blur=blur), KeyframeOptions())
    chosen = {k.frame_index // 4 for k in res.keyframes}
    assert not (chosen & blur)
    assert res.rejected_blur >= len(blur)
    assert sharpness(make_candidates(blur={0})[0].gray) < sharpness(make_candidates()[0].gray) / 5


def test_keyframes_respect_the_minimum_temporal_gap():
    opts = KeyframeOptions(min_gap_s=0.5)
    res = select_keyframes(make_candidates(n=40), opts)
    gaps = np.diff([k.t for k in res.keyframes])
    assert len(res.keyframes) >= 4 and (gaps >= 0.5 - 1e-9).all()


def test_static_camera_gets_keyframes_only_after_the_maximum_gap():
    opts = KeyframeOptions(max_gap_s=2.0)
    res = select_keyframes(make_candidates(n=40, static=True), opts)
    gaps = np.diff([k.t for k in res.keyframes])
    assert len(res.keyframes) >= 2 and (gaps >= 2.0 - 1e-9).all()  # no parallax: only the max-gap rule fires
    moving = select_keyframes(make_candidates(n=40), opts)
    assert len(moving.keyframes) > len(res.keyframes)


def test_thinning_keeps_the_ends_and_the_requested_count():
    keys = [Keyframe(i, i * 0.3, 100.0 + i, 0.05) for i in range(40)]
    out, dropped = thin_keyframes(keys, 15)
    assert len(out) == 15 and dropped == 25 and out[0].frame_index == 0 and out[-1].frame_index == 39
    assert [k.t for k in out] == sorted(k.t for k in out)
    again, _ = thin_keyframes(keys, 15)
    assert [k.frame_index for k in again] == [k.frame_index for k in out]


def test_extract_keyframes_from_a_real_video_is_deterministic_and_writes_files(tmp_path):
    video = make_test_video(tmp_path / "v.mp4", seconds=8, fps=20)
    opts = KeyframeOptions(work_max_side=320, target_min=3)
    a = extract_keyframes(video, tmp_path / "a", opts)
    b = extract_keyframes(video, tmp_path / "b", opts)
    assert len(a.keyframes) >= 4
    assert [k.frame_index for k in a.keyframes] == [k.frame_index for k in b.keyframes]
    files = sorted(p.name for p in (tmp_path / "a").glob("*.jpg"))
    assert files == [k.file for k in a.keyframes] and a.video_facts["work_size"] == (320, 240)
    assert (tmp_path / "a" / files[0]).read_bytes() == (tmp_path / "b" / files[0]).read_bytes()
    d = a.to_dict()
    assert d["count"] == len(a.keyframes) and all(k["sharpness"] > 0 for k in d["keyframes"])
