"""Structural wall extraction tests on synthetic point clouds (no real captures)."""

import json

import numpy as np

from spatialforge.lidar.planes import PlaneFit
from spatialforge.lidar.walls import WallOptions, extract_walls

FLOOR_Y = -1.4
FLOOR = PlaneFit(0.0, 0.0, FLOOR_Y, 0.01, 0.03)


def wall(a, b, h_lo=0.12, h_hi=2.3, step=0.03, noise=0.005, tilt_deg=0.0, seed=0):
    """Dense vertical wall from (x, z) = a to b, between heights h_lo..h_hi above the floor."""
    rng = np.random.default_rng(seed)
    a, b = np.array(a, float), np.array(b, float)
    length = np.linalg.norm(b - a)
    d = (b - a) / length
    n = np.array([-d[1], d[0]])
    t, h = np.meshgrid(np.arange(0, length, step), np.arange(h_lo, h_hi, step))
    t, h = t.ravel(), h.ravel()
    lateral = rng.normal(0, noise, t.size) + np.tan(np.radians(tilt_deg)) * (h - h_lo)
    xz = a + np.outer(t, d) + np.outer(lateral, n)
    return np.column_stack([xz[:, 0], FLOOR_Y + h, xz[:, 1]])


def horizontal(y, x_range, z_range, step=0.03, noise=0.005, seed=1):
    rng = np.random.default_rng(seed)
    x, z = np.meshgrid(np.arange(*x_range, step), np.arange(*z_range, step))
    return np.column_stack([x.ravel(), y + rng.normal(0, noise, x.size), z.ravel()])


def room(rotate_top_deg=0.0, noise=0.005):
    top_end = (6.0, 5.0 + 6.0 * np.tan(np.radians(rotate_top_deg)))
    return np.vstack([
        wall((0, 0), (6, 0), noise=noise, seed=1),
        wall((6, 0), (6, 5), noise=noise, seed=2),
        wall((0, 5), top_end, noise=noise, seed=3),
        wall((0, 0), (0, 5), noise=noise, seed=4),
    ])


def ang_close(a, b, tol):
    d = abs(a - b) % 180
    return min(d, 180 - d) <= tol


# ---------- detection ----------


def test_single_perfect_wall_is_detected():
    a = extract_walls(wall((0, 0), (4, 0)), FLOOR)
    assert len(a.walls) == 1
    w = a.walls[0]
    assert abs(w.length_m - 4.0) < 0.2
    assert ang_close(w.orientation_deg, 0.0, 1.0)
    assert w.plane[1] == 0.0  # a vertical wall plane has no y component
    assert w.inlier_count > 1000


def test_rectangular_room_gives_four_walls():
    a = extract_walls(room(), FLOOR)
    assert len(a.walls) == 4
    lengths = sorted(round(w.length_m) for w in a.walls)
    assert lengths == [5, 5, 6, 6]
    horizontal_walls = [w for w in a.walls if ang_close(w.orientation_deg, 0, 2)]
    vertical_walls = [w for w in a.walls if ang_close(w.orientation_deg, 90, 2)]
    assert len(horizontal_walls) == 2 and len(vertical_walls) == 2
    assert all(w.snapped for w in a.walls)
    assert a.manhattan_strength > 0.95


def test_horizontal_floor_and_table_are_not_walls():
    cloud = np.vstack([horizontal(FLOOR_Y, (0, 6), (0, 5)), horizontal(FLOOR_Y + 0.9, (1, 3), (1, 3))])
    a = extract_walls(cloud, FLOOR)
    assert a.walls == []  # the floor is removed, a table top occupies one height bin only


def test_horizontal_ceiling_is_not_a_wall():
    ceiling = PlaneFit(0.0, 0.0, FLOOR_Y + 2.0, 0.01, 0.03)  # ceiling plane inside the height band
    cloud = np.vstack([horizontal(FLOOR_Y + 2.0, (0, 6), (0, 5)), horizontal(FLOOR_Y, (0, 6), (0, 5))])
    assert extract_walls(cloud, FLOOR, [ceiling]).walls == []
    # a ceiling above the band is excluded by the band itself
    assert extract_walls(horizontal(FLOOR_Y + 2.5, (0, 6), (0, 5)), FLOOR).walls == []


# ---------- furniture rejection ----------


def test_short_vertical_furniture_plane_is_rejected():
    cabinet = wall((0, 0), (1.5, 0), h_lo=0.12, h_hi=0.9)  # 1.5 m wide, only 0.9 m tall
    a = extract_walls(cabinet, FLOOR)
    assert a.walls == []
    assert any("vertical" in str(r) for r in a.rejected)


def test_tall_narrow_object_is_rejected():
    pole = wall((0, 0), (0.25, 0), h_lo=0.12, h_hi=2.3)  # 2 m tall but 25 cm wide
    assert extract_walls(pole, FLOOR).walls == []


def test_floating_plane_is_rejected():
    floating = wall((0, 0), (3, 0), h_lo=1.3, h_hi=2.3)  # starts 1.3 m above the floor
    a = extract_walls(floating, FLOOR)
    assert a.walls == []


def test_tilted_plane_is_rejected():
    leaning = wall((0, 0), (3, 0), tilt_deg=12.0)
    a = extract_walls(leaning, FLOOR)
    assert a.walls == [] or all(w.segments[0].tilt_deg > 5 for w in a.walls)
    assert not any(w for w in a.walls if w.segments[0].tilt_deg > 5)


def test_isolated_short_piece_is_rejected_but_attached_return_is_kept():
    long_wall = wall((0, 0), (5, 0), seed=1)
    isolated = wall((2.0, 3.0), (2.8, 3.0), seed=2)  # 0.8 m piece in the middle of nowhere
    a = extract_walls(np.vstack([long_wall, isolated]), FLOOR)
    assert len(a.walls) == 1
    assert any("isolated short" in str(r) for r in a.rejected)
    attached = wall((5, 0), (5, 0.8), seed=3)  # 0.8 m return at the end of the long wall
    b = extract_walls(np.vstack([long_wall, attached]), FLOOR)
    assert len(b.walls) == 2


# ---------- merging and separation ----------


def test_noisy_nearby_detections_of_one_wall_merge():
    cloud = np.vstack([wall((0, 0), (4, 0), seed=1), wall((0, 0.06), (4, 0.06), seed=2)])
    a = extract_walls(cloud, FLOOR)
    assert len(a.walls) == 1
    cloud2 = wall((0, 0), (4, 0), noise=0.03, seed=3)
    assert len(extract_walls(cloud2, FLOOR).walls) == 1


def test_genuinely_separated_parallel_walls_stay_separate():
    cloud = np.vstack([wall((0, 0), (4, 0), seed=1), wall((0, 1.5), (4, 1.5), seed=2)])
    a = extract_walls(cloud, FLOOR)
    assert len(a.walls) == 2


# ---------- orientation ----------


def test_slightly_misaligned_wall_snaps_to_dominant_direction():
    # A 2 m wall that is 3 degrees off the room's axes: snapping moves its ends only ~5 cm, so it snaps.
    short = wall((1.0, 5.0), (3.0, 5.0 + 2.0 * np.tan(np.radians(3.0))), seed=7)
    cloud = np.vstack([
        wall((0, 0), (6, 0), seed=1), wall((6, 0), (6, 5), seed=2), wall((0, 0), (0, 5), seed=4), short,
    ])
    a = extract_walls(cloud, FLOOR)
    w = next(w for w in a.walls if ang_close(w.raw_orientation_deg, 3.0, 1.5))
    assert w.snapped
    assert ang_close(w.orientation_deg, 0.0, 0.8)  # snapped to the axis
    assert 2.0 < w.snap_delta_deg < 4.0  # and the raw deviation is reported
    assert abs(w.raw_orientation_deg - 3.0) < 1.0


def test_long_wall_is_not_snapped_when_it_would_move_its_ends_too_far():
    # Same 3 degree deviation on a 6 m wall would shift its ends by ~16 cm, more than the wall's blur.
    a = extract_walls(room(rotate_top_deg=3.0), FLOOR)
    top = max((w for w in a.walls if ang_close(w.raw_orientation_deg, 3.0, 1.5)), key=lambda w: w.length_m)
    assert not top.snapped
    assert ang_close(top.orientation_deg, 3.0, 0.5)  # it keeps its measured orientation
    assert top.residual_p90_m < 0.02  # and therefore still fits the data tightly


def test_strongly_diagonal_wall_is_not_snapped():
    diag = wall((1.0, 1.0), (4.3, 3.3), seed=9)  # ~35 degrees
    a = extract_walls(np.vstack([room(), diag]), FLOOR)
    d = next(w for w in a.walls if ang_close(w.raw_orientation_deg, 34.9, 3.0))
    assert not d.snapped
    assert ang_close(d.orientation_deg, d.raw_orientation_deg, 0.01)


def test_short_off_axis_plane_is_rejected_but_long_diagonal_is_kept():
    short_diag = wall((1.0, 1.0), (2.2, 1.9), seed=11)  # 1.5 m, ~37 degrees off both axes of the room
    a = extract_walls(np.vstack([room(), short_diag]), FLOOR)
    assert len(a.walls) == 4  # only the four room walls survive
    assert any("dominant wall axes" in str(r) for r in a.rejected)


def test_evidence_tiers_and_axis_deviation_are_reported():
    diag = wall((1.0, 1.0), (4.3, 3.3), seed=9)
    a = extract_walls(np.vstack([room(), diag]), FLOOR)
    by_orientation = {round(w.raw_orientation_deg): w for w in a.walls}
    assert all(w.evidence == "strong" for w in a.walls if w.deviation_from_axes_deg < 1)
    d = next(w for w in a.walls if w.deviation_from_axes_deg > 15)
    assert d.evidence == "weak"  # kept, because it is long, but flagged
    assert 30 < d.deviation_from_axes_deg < 45
    assert by_orientation  # sanity


# ---------- geometry ----------


def test_endpoints_resist_outlier_points():
    rng = np.random.default_rng(5)
    stray = np.column_stack([rng.uniform(4.5, 10, 40), rng.uniform(-1.0, 0.5, 40), rng.normal(0, 0.01, 40)])
    a = extract_walls(np.vstack([wall((0, 0), (4, 0)), stray]), FLOOR)
    assert len(a.walls) == 1
    assert abs(a.walls[0].length_m - 4.0) < 0.25  # not stretched toward the stray points at x = 10


def test_wall_length_and_endpoints_of_a_diagonal_wall():
    a = extract_walls(wall((1.0, 1.0), (4.0, 5.0)), FLOOR)  # 3-4-5 triangle: exactly 5 m long
    w = a.walls[0]
    assert abs(w.length_m - 5.0) < 0.2
    ends = sorted([w.start, w.end])
    assert np.allclose(ends[0], (1.0, 1.0), atol=0.15)
    assert np.allclose(ends[1], (4.0, 5.0), atol=0.15)


def test_uncertainty_grows_with_plane_noise():
    clean = extract_walls(wall((0, 0), (4, 0), noise=0.004, seed=1), FLOOR).walls[0]
    noisy = extract_walls(wall((0, 0), (4, 0), noise=0.03, seed=1), FLOOR).walls[0]
    assert noisy.position_uncertainty_m > clean.position_uncertainty_m
    assert noisy.orientation_uncertainty_deg > clean.orientation_uncertainty_deg
    assert noisy.residual_p90_m > clean.residual_p90_m


def test_door_sized_gap_is_preserved():
    cloud = np.vstack([wall((0, 0), (2, 0), seed=1), wall((3, 0), (5, 0), seed=2)])  # 1.0 m gap
    a = extract_walls(cloud, FLOOR)
    assert len(a.walls) == 1
    w = a.walls[0]
    assert len(w.segments) == 2  # observed pieces are kept separate
    assert len(w.gaps) == 1 and abs(w.gaps[0]["length_m"] - 1.0) < 0.15
    assert abs(w.length_m - 5.0) < 0.2 and abs(w.observed_length_m - 4.0) < 0.2
    # a gap wider than a plausible opening splits the group into separate walls instead
    far = np.vstack([wall((0, 0), (2, 0), seed=1), wall((6, 0), (8, 0), seed=2)])
    assert len(extract_walls(far, FLOOR).walls) == 2


# ---------- determinism and serialisation ----------


def test_extraction_is_deterministic():
    cloud = np.vstack([room(rotate_top_deg=2.0), wall((1.0, 1.0), (4.3, 3.3), seed=9)])
    a = extract_walls(cloud, FLOOR).to_dict()
    b = extract_walls(cloud.copy(), FLOOR).to_dict()
    assert a == b


def test_result_serialises_to_json():
    d = extract_walls(room(), FLOOR).to_dict()
    restored = json.loads(json.dumps(d))
    assert restored == d
    w = restored["walls"][0]
    for key in ("id", "plane", "start", "end", "length_m", "orientation_deg", "raw_orientation_deg", "snapped",
                "vertical_span_m", "inlier_count", "residual_median_m", "residual_p90_m",
                "position_uncertainty_m", "segments", "gaps"):
        assert key in w
    assert restored["accepted_walls"] == 4 and restored["dominant_directions_deg"]


def test_empty_and_floor_only_clouds_do_not_crash():
    assert extract_walls(horizontal(FLOOR_Y, (0, 6), (0, 5)), FLOOR).walls == []
    tiny = extract_walls(wall((0, 0), (0.3, 0)), FLOOR)
    assert tiny.walls == [] and tiny.warnings
