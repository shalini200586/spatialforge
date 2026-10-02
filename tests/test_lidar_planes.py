"""Floor/ceiling detection tests on synthetic point clouds (no real captures)."""

import json

import numpy as np

from spatialforge.lidar.planes import PlaneOptions, analyze_horizontal_planes

FLOOR_Y = -1.4
CAMERA = np.array([[x, 0.0, z] for x in (1.0, 3.0, 5.0) for z in (1.0, 2.5, 4.0)])  # camera ~1.4 m above floor


def plane_points(y_of_xz, x_range=(0, 6), z_range=(0, 5), step=0.05, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    x, z = np.meshgrid(np.arange(*x_range, step), np.arange(*z_range, step))
    x, z = x.ravel(), z.ravel()
    return np.column_stack([x, y_of_xz(x, z) + rng.normal(0, noise, x.size), z])


def flat(y):
    return lambda x, z: np.full_like(x, y)


def tilted(y0, deg):
    return lambda x, z: y0 + np.tan(np.radians(deg)) * x


def walls(y_lo, y_hi, step=0.05, noise=0.01, seed=5):
    """Four vertical walls around the 6 x 5 m room."""
    rng = np.random.default_rng(seed)
    out = []
    for x in (0.0, 6.0):
        yy, zz = np.meshgrid(np.arange(y_lo, y_hi, step), np.arange(0, 5, step))
        out.append(np.column_stack([np.full(yy.size, x) + rng.normal(0, noise, yy.size), yy.ravel(), zz.ravel()]))
    for z in (0.0, 5.0):
        yy, xx = np.meshgrid(np.arange(y_lo, y_hi, step), np.arange(0, 6, step))
        out.append(np.column_stack([xx.ravel(), yy.ravel(), np.full(yy.size, z) + rng.normal(0, noise, yy.size)]))
    return np.vstack(out)


def room(ceiling_y=None, noise=0.01, with_walls=False, seed=0):
    parts = [plane_points(flat(FLOOR_Y), noise=noise, seed=seed)]
    if ceiling_y is not None:
        parts.append(plane_points(flat(ceiling_y), noise=noise, seed=seed + 1))
    if with_walls:
        parts.append(walls(FLOOR_Y, ceiling_y if ceiling_y is not None else 1.1, noise=noise))
    return np.vstack(parts)


# ---------- floor ----------


def test_perfect_horizontal_floor_is_detected():
    a = analyze_horizontal_planes(room(), CAMERA)
    f = a.floor
    assert f.observed
    assert abs(f.height_m - FLOOR_Y) < 0.005
    assert f.tilt_deg < 0.3
    assert f.inlier_ratio > 0.9
    assert abs(f.plane[1]) > 0.999  # unit normal along world Y
    assert f.residual_median_m < 0.02


def test_tilted_floor_within_tolerance_is_accepted():
    pts = plane_points(tilted(FLOOR_Y, 3.0), x_range=(0, 4), z_range=(0, 4))
    a = analyze_horizontal_planes(pts, CAMERA - [0, 0.0, 0])
    assert a.floor.observed
    assert abs(a.floor.tilt_deg - 3.0) < 0.3


def test_tilted_floor_beyond_tolerance_is_rejected():
    pts = plane_points(tilted(FLOOR_Y, 9.0), x_range=(0, 4), z_range=(0, 4))
    a = analyze_horizontal_planes(pts, CAMERA)
    assert not a.floor.observed
    assert a.ceiling_levels == []
    # The same geometry is accepted if the tolerance is relaxed, so the gate is the tilt, not the data.
    relaxed = analyze_horizontal_planes(pts, CAMERA, PlaneOptions(max_tilt_deg=12.0))
    assert relaxed.floor.observed and abs(relaxed.floor.tilt_deg - 9.0) < 0.5


def test_floor_must_be_below_the_camera_path():
    a = analyze_horizontal_planes(room(), np.array([[1.0, FLOOR_Y + 0.2, 1.0], [2.0, FLOOR_Y + 0.2, 2.0]]))
    assert not a.floor.observed
    assert "below the camera" in a.floor.reject_reason


# ---------- ceiling ----------


def test_floor_and_ceiling_give_correct_height():
    a = analyze_horizontal_planes(room(ceiling_y=1.1), CAMERA)  # 2.5 m above the floor
    assert a.ceiling.observed
    assert abs(a.ceiling_height["value_m"] - 2.5) < 0.01
    lo, hi = a.ceiling_height["confidence_interval_m"]
    assert lo < 2.5 < hi
    assert len(a.ceiling_levels) == 1
    assert a.warnings == []


def test_floor_only_room_has_no_ceiling():
    a = analyze_horizontal_planes(room(with_walls=True), CAMERA)  # floor + four walls, nothing overhead
    assert a.floor.observed
    assert not a.ceiling.observed and a.ceiling_levels == []
    assert a.ceiling_height["value_m"] is None and a.ceiling_height["confidence_interval_m"] is None
    assert a.ceiling_height["confidence"] == 0.0


def test_walls_alone_do_not_become_a_ceiling():
    # Wall points exist at every height, including the ceiling search range, but only as thin lines.
    a = analyze_horizontal_planes(room(ceiling_y=None, with_walls=True), CAMERA)
    assert not a.ceiling.observed


def test_sparse_high_points_do_not_become_a_ceiling():
    rng = np.random.default_rng(7)
    sparse = np.column_stack([rng.uniform(0, 6, 400), rng.uniform(0.9, 1.3, 400), rng.uniform(0, 5, 400)])
    a = analyze_horizontal_planes(np.vstack([room(), sparse]), CAMERA)
    assert a.floor.observed and not a.ceiling.observed


def test_small_horizontal_patch_is_not_a_ceiling():
    # A dense 0.8 x 0.8 m panel (lamp / shelf top) at ceiling height: well fitted but far too small.
    patch = plane_points(flat(1.1), x_range=(2.6, 3.4), z_range=(2.1, 2.9), step=0.01, noise=0.003)
    a = analyze_horizontal_planes(np.vstack([room(), patch]), CAMERA)
    assert not a.ceiling.observed
    assert any("solid area" in c["rejected"] for c in a.candidate_log if c.get("rejected"))


def test_tilted_ceiling_beyond_tolerance_is_rejected():
    ceiling = plane_points(tilted(1.1, 9.0), x_range=(0, 4), z_range=(0, 4))
    a = analyze_horizontal_planes(np.vstack([room(), ceiling]), CAMERA)
    assert not a.ceiling.observed


def test_two_ceiling_levels_are_both_reported():
    left = plane_points(flat(1.0), x_range=(0, 3), z_range=(0, 5))
    right = plane_points(flat(1.9), x_range=(3, 6), z_range=(0, 5), seed=4)
    a = analyze_horizontal_planes(np.vstack([room(), left, right]), CAMERA)
    heights = sorted(h["value_m"] for h in a.ceiling_level_heights)
    assert len(heights) == 2 and abs(heights[0] - 2.4) < 0.02 and abs(heights[1] - 3.3) < 0.02
    assert any("ceiling levels" in w for w in a.warnings)


# ---------- height under tilt, uncertainty, determinism, serialisation ----------


def test_height_uses_plane_separation_under_small_tilt():
    tilt = 2.0
    pts = np.vstack([
        plane_points(tilted(FLOOR_Y, tilt), noise=0.003),
        plane_points(tilted(FLOOR_Y + 2.5, tilt), noise=0.003, seed=3),  # 2.5 m higher *vertically*
    ])
    a = analyze_horizontal_planes(pts, CAMERA)
    expected = 2.5 * np.cos(np.radians(tilt))  # perpendicular separation of parallel tilted planes
    assert abs(a.ceiling_height["value_m"] - expected) < 0.005
    assert abs(a.ceiling_height["value_m"] - 2.5) > 0.0005  # and it is NOT simply the vertical difference
    assert abs(a.floor.tilt_deg - tilt) < 0.2 and abs(a.ceiling.tilt_deg - tilt) < 0.2


def test_interval_is_narrow_for_clean_and_wider_for_noisy_geometry():
    clean = analyze_horizontal_planes(room(ceiling_y=1.1, noise=0.003), CAMERA)
    noisy = analyze_horizontal_planes(room(ceiling_y=1.1, noise=0.04), CAMERA)
    assert clean.ceiling.observed and noisy.ceiling.observed
    hw_clean = clean.ceiling_height["interval_halfwidth_m"]
    hw_noisy = noisy.ceiling_height["interval_halfwidth_m"]
    assert hw_clean < 0.02
    assert hw_noisy > 2 * hw_clean
    assert clean.ceiling_height["confidence"] > noisy.ceiling_height["confidence"]


def test_interval_widens_when_subsets_disagree():
    # Ceiling that is 3 cm higher on one side of the room: the subsets see different heights.
    half = lambda x, z: np.where(x < 3.0, 1.1, 1.14)
    pts = np.vstack([room(), plane_points(half, noise=0.003, seed=9)])
    uneven = analyze_horizontal_planes(pts, CAMERA)
    clean = analyze_horizontal_planes(room(ceiling_y=1.1, noise=0.003), CAMERA)
    assert uneven.ceiling_height["interval_halfwidth_m"] > clean.ceiling_height["interval_halfwidth_m"]


def test_analysis_is_deterministic():
    pts = room(ceiling_y=1.1, noise=0.02, with_walls=True)
    a = analyze_horizontal_planes(pts, CAMERA).to_dict()
    b = analyze_horizontal_planes(pts.copy(), CAMERA.copy()).to_dict()
    assert a == b


def test_result_serialises_to_json():
    with_ceiling = analyze_horizontal_planes(room(ceiling_y=1.1), CAMERA).to_dict()
    without = analyze_horizontal_planes(room(), CAMERA).to_dict()
    for d in (with_ceiling, without):
        restored = json.loads(json.dumps(d))
        assert restored == d
        assert {"floor", "ceiling", "ceiling_height", "ceiling_levels", "warnings", "options"} <= set(restored)
    assert with_ceiling["ceiling"]["observed"] is True
    assert len(with_ceiling["floor"]["plane"]) == 4
    assert without["ceiling"]["observed"] is False and without["ceiling_height"]["value_m"] is None
