import numpy as np
import pytest

from spatialforge.lidar import geometry as g
from spatialforge.lidar.geometry import Intrinsics


def test_intrinsics_scaling_same_aspect_ratio():
    out = g.scale_intrinsics(Intrinsics(1600, 1600, 960, 720), 1920, 1440, 256, 192)
    assert out.fx == pytest.approx(1600 * 256 / 1920)
    assert out.fy == pytest.approx(1600 * 192 / 1440)
    assert out.cx == pytest.approx(128.0)
    assert out.cy == pytest.approx(96.0)


def test_principal_point_backprojects_to_axis():
    depth = np.zeros((5, 5), dtype=np.uint16)
    depth[2, 3] = 2000  # u=3, v=2
    mask = depth > 0
    pts = g.backproject(depth, mask, Intrinsics(100, 100, 3, 2))
    assert pts.shape == (1, 3)
    assert pts[0, 0] == pytest.approx(0.0)
    assert pts[0, 1] == pytest.approx(0.0)
    assert pts[0, 2] == pytest.approx(2.0)  # 2000 raw * 0.001


def test_off_centre_pixel_backprojection():
    depth = np.zeros((10, 10), dtype=np.uint16)
    depth[7, 9] = 4000  # u=9, v=7
    pts = g.backproject(depth, depth > 0, Intrinsics(fx=50, fy=25, cx=4, cy=3))
    # X = (9-4)*4/50 = 0.4, Y = (7-3)*4/25 = 0.64, Z = 4
    assert pts[0] == pytest.approx([0.4, 0.64, 4.0])


def test_identity_quaternion():
    assert np.allclose(g.quat_to_rotation(0, 0, 0, 1), np.eye(3))


def test_90_degree_rotation_about_z():
    s = np.sqrt(0.5)
    rot = g.quat_to_rotation(0, 0, s, s)
    assert np.allclose(rot @ [1, 0, 0], [0, 1, 0])  # x axis -> y axis
    assert np.allclose(rot @ [0, 1, 0], [-1, 0, 0])
    assert np.allclose(rot @ rot.T, np.eye(3))


def test_non_unit_and_zero_quaternion():
    assert np.allclose(g.quat_to_rotation(0, 0, 0, 5), np.eye(3))
    with pytest.raises(ValueError):
        g.quat_to_rotation(0, 0, 0, 0)


def test_translation_only_transform():
    out = g.transform_points(np.array([[1.0, 2.0, 3.0]]), np.eye(3), [10, 20, 30])
    assert np.allclose(out, [[11, 22, 33]])


def test_rotation_and_translation():
    s = np.sqrt(0.5)
    rot = g.quat_to_rotation(0, 0, s, s)
    out = g.transform_points(np.array([[1.0, 0.0, 0.0]]), rot, [5, 0, 0])
    assert np.allclose(out, [[5, 1, 0]])  # rotate first, then translate


def test_zero_depth_and_range_rejected():
    depth = np.array([[0, 500, 1000, 9000]], dtype=np.uint16)
    mask = g.valid_depth_mask(depth, None, 0, min_range_m=0.1, max_range_m=5.0)
    assert mask.tolist() == [[False, True, True, False]]


def test_confidence_filtering():
    depth = np.full((1, 4), 1000, dtype=np.uint16)
    conf = np.array([[0, 1, 2, 2]], dtype=np.uint8)
    mask = g.valid_depth_mask(depth, conf, 2, 0.0, 10.0)
    assert mask.tolist() == [[False, False, True, True]]
    assert g.valid_depth_mask(depth, conf, 1, 0.0, 10.0).sum() == 3


def test_voxel_downsample_is_deterministic_and_keeps_one_per_voxel():
    pts = np.array(
        [[0.001, 0.001, 0.001], [0.004, 0.002, 0.003], [0.051, 0.0, 0.0], [0.0, 0.0, 0.0099]],
        dtype=np.float64,
    )
    a = g.voxel_downsample(pts, 0.05)
    b = g.voxel_downsample(pts.copy(), 0.05)
    assert np.array_equal(a, b)
    assert len(a) == 2  # voxel (0,0,0) holds three points, voxel (1,0,0) holds one
    assert np.allclose(a[0], pts[0])  # first point in input order is the representative


def test_voxel_downsample_empty_and_bad_size():
    assert len(g.voxel_downsample(np.empty((0, 3)), 0.1)) == 0
    with pytest.raises(ValueError):
        g.voxel_downsample(np.zeros((1, 3)), 0)
