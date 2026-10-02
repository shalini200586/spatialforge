import struct

import pytest
from test_lidar_validator import make_capture

from spatialforge.cli import main
from spatialforge.lidar.reconstruction import (
    ReconstructionError,
    ReconstructionOptions,
    reconstruct_lidar,
    select_frames,
)
from spatialforge.lidar.video_info import read_video_size

OPTS = ReconstructionOptions(frame_step=1, max_frames=None, voxel_size=0.02, source_size=(1920, 1440))


def read_ply(path):
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    return lines[: end + 1], [list(map(float, ln.split())) for ln in lines[end + 1 :]]


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def test_synthetic_reconstruction_writes_valid_ply(tmp_path):
    cap = make_capture(tmp_path / "cap")
    out = tmp_path / "out" / "cloud.ply"
    r = reconstruct_lidar(cap, out, OPTS)

    header, points = read_ply(out)
    assert header[:2] == ["ply", "format ascii 1.0"]
    assert header[2] == f"element vertex {len(points)}"
    assert header[3:6] == ["property float x", "property float y", "property float z"]
    assert len(points) == r.points_after_downsample > 0
    # Identity poses at the origin: world z equals the depth in metres (1.0 - 1.003 m).
    assert all(0.99 < p[2] < 1.01 for p in points)
    assert r.frames_used == 4
    assert r.candidate_points == 4 * 48
    assert r.removed_by_depth == 4  # one zero-depth pixel per frame
    assert r.points_after_filter == 4 * 47


def test_confidence_threshold_removes_everything_clearly(tmp_path):
    cap = make_capture(tmp_path / "cap")  # all confidence values are 2
    opts = ReconstructionOptions(frame_step=1, max_frames=None, min_confidence=3, source_size=(1920, 1440))
    with pytest.raises(ReconstructionError, match="no valid depth points"):
        reconstruct_lidar(cap, tmp_path / "x.ply", opts)


def test_invalid_capture_is_rejected(tmp_path):
    cap = make_capture(tmp_path / "cap", depth=False)
    with pytest.raises(ReconstructionError, match="INVALID"):
        reconstruct_lidar(cap, tmp_path / "x.ply", OPTS)


def test_unknown_source_size_is_a_clear_error(tmp_path):
    cap = make_capture(tmp_path / "cap")  # rgb.mp4 is fake bytes
    with pytest.raises(ReconstructionError, match="--source-size"):
        reconstruct_lidar(cap, tmp_path / "x.ply", ReconstructionOptions(frame_step=1))


def test_mismatched_aspect_ratio_is_rejected(tmp_path):
    cap = make_capture(tmp_path / "cap")
    opts = ReconstructionOptions(frame_step=1, source_size=(1920, 1080))
    with pytest.raises(ReconstructionError, match="aspect"):
        reconstruct_lidar(cap, tmp_path / "x.ply", opts)


def test_unwritable_output_is_a_clear_error(tmp_path):
    cap = make_capture(tmp_path / "cap")
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(ReconstructionError, match="cannot write"):
        reconstruct_lidar(cap, blocker / "out.ply", OPTS)  # parent is a file


def test_frame_selection_is_deterministic():
    ids = list(range(100))
    assert select_frames(ids, 10, None) == list(range(0, 100, 10))
    assert select_frames(ids, 1, 5) == [0, 25, 50, 74, 99]
    assert select_frames(ids, 1, 5) == select_frames(ids, 1, 5)
    assert select_frames(ids, 1, 1) == [0]


def test_cli_reports_error_without_traceback(tmp_path, capsys):
    code = main(["reconstruct-lidar", str(tmp_path / "missing"), "--output", str(tmp_path / "o.ply")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_read_video_size_from_minimal_mp4(tmp_path):
    # tkhd body: version/flags, times, track id, reserved, duration, ..., matrix, width, height
    tkhd_body = bytes(4 + 8 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2) + bytes(36) + struct.pack(">II", 1920 << 16, 1440 << 16)
    mp4 = box(b"moov", box(b"trak", box(b"tkhd", tkhd_body)))
    path = tmp_path / "rgb.mp4"
    path.write_bytes(mp4)
    assert read_video_size(path) == (1920, 1440)
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        read_video_size(path)
