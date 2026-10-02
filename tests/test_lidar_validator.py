"""Tests use small synthetic captures built in tmp_path (no real data needed)."""

import numpy as np
from PIL import Image

from spatialforge.cli import main
from spatialforge.lidar import Status, validate_lidar_capture

HEADER = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n"


def pose_row(frame_id):
    return f"{frame_id}.0, {frame_id:06d}, 0, 0, 0, 0, 0, 0, 1, 1500, 1500, 960, 720, , \n"


def make_capture(root, n=4, *, imu=True, depth=True, confidence=True, odometry_rows=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "rgb.mp4").write_bytes(b"fake")
    (root / "camera_matrix.csv").write_text("1500.0, 0.0, 960.0\n0.0, 1500.0, 720.0\n0.0, 0.0, 1.0\n")
    if imu:
        (root / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n1,0,0,0,0,0,0\n")
    if odometry_rows is None:
        odometry_rows = [pose_row(i) for i in range(n)]
    (root / "odometry.csv").write_text(HEADER + "".join(odometry_rows))
    if depth:
        (root / "depth").mkdir()
        for i in range(n):
            arr = np.full((6, 8), 1000 + i, dtype=np.uint16)
            arr[0, 0] = 0  # one invalid pixel
            Image.fromarray(arr).save(root / "depth" / f"{i:06d}.png")
    if confidence:
        (root / "confidence").mkdir()
        for i in range(n):
            Image.fromarray(np.full((6, 8), 2, dtype=np.uint8)).save(root / "confidence" / f"{i:06d}.png")
    return root


def test_valid_minimal_capture(tmp_path):
    r = validate_lidar_capture(make_capture(tmp_path / "cap"))
    assert r.status is Status.VALID, (r.warnings, r.errors)
    assert (r.depth_frame_count, r.confidence_frame_count, r.pose_frame_count) == (4, 4, 4)
    assert r.metadata["depth_dtype"] == "uint16"
    assert r.metadata["depth_resolution"].startswith("8x6")
    assert r.metadata["depth_stats"]["min"] == 1000
    assert r.metadata["depth_stats"]["max"] == 1003
    assert r.metadata["confidence_histogram"] == {2: 4 * 48}
    assert r.metadata["intrinsics"]["fx"] == 1500.0
    assert r.metadata["pose_depth_aligned"] is True


def test_missing_depth_folder_is_invalid(tmp_path):
    r = validate_lidar_capture(make_capture(tmp_path / "cap", depth=False))
    assert r.status is Status.INVALID
    assert any("depth/" in e for e in r.errors)


def test_missing_imu_is_warning_not_crash(tmp_path):
    r = validate_lidar_capture(make_capture(tmp_path / "cap", imu=False))
    assert r.status is Status.VALID_WITH_WARNINGS
    assert any("imu" in w for w in r.warnings)
    assert not r.errors


def test_mismatched_depth_and_confidence(tmp_path):
    root = make_capture(tmp_path / "cap")
    (root / "confidence" / "000002.png").unlink()
    r = validate_lidar_capture(root)
    assert r.status is Status.VALID_WITH_WARNINGS
    assert r.missing_frames["depth frame(s) without confidence frame"] == [2]
    assert r.metadata["depth_confidence_aligned"] is False


def test_duplicate_and_malformed_odometry(tmp_path):
    rows = [
        pose_row(0),
        pose_row(1),
        pose_row(1),  # duplicate
        "oops, abc, not, a, row\n",  # malformed
        pose_row(3),
    ]
    r = validate_lidar_capture(make_capture(tmp_path / "cap", odometry_rows=rows))
    assert r.status is Status.INVALID
    assert r.duplicate_pose_frames == [1]
    assert any("malformed" in w for w in r.warnings)
    assert any("duplicate" in e for e in r.errors)


def test_missing_pose_for_depth_frames(tmp_path):
    r = validate_lidar_capture(make_capture(tmp_path / "cap", odometry_rows=[pose_row(0)]))
    assert r.missing_frames["depth frame(s) without pose row"] == [1, 2, 3]
    assert r.status is Status.VALID_WITH_WARNINGS


def test_nested_capture_folder_and_cli(tmp_path, capsys):
    make_capture(tmp_path / "outer" / "abc123")
    assert main(["validate-lidar", str(tmp_path / "outer")]) == 0
    assert "RESULT: VALID" in capsys.readouterr().out


def test_nonexistent_path_is_invalid(tmp_path):
    assert validate_lidar_capture(tmp_path / "nope").status is Status.INVALID
