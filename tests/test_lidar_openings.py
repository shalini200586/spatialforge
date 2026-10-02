"""Opening detection tests on synthetic walls and point clouds (no real captures)."""

import json

import numpy as np

from spatialforge.lidar import openings as op
from spatialforge.lidar.planes import PlaneFit
from spatialforge.lidar.rooms import WallInput, build_topology

FLOOR_Y = -1.4
FLOOR = PlaneFit(0.0, 0.0, FLOOR_Y, 0.01, 0.03)


def make_wall(wid="w1", a=(0.0, 0.0), b=(6.0, 0.0), segs=((0.0, 6.0),), tier="strong", unc=0.03):
    a, b = np.array(a, float), np.array(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    segments = [(tuple(a + s0 * d), tuple(a + s1 * d)) for s0, s1 in segs]
    gaps = [{"start": segments[i][1], "end": segments[i + 1][0], "length_m": segs[i + 1][0] - segs[i][1]}
            for i in range(len(segs) - 1)]
    return WallInput(wid, tier, a, d, unc, segments, gaps)


def cloud_on(wall, extent=(0.0, 6.0), holes=(), step=0.02, noise=0.008, seed=0, v_hi=2.3, jamb_jitter=0.0):
    """Dense wall points on the wall plane; `holes` = (u0, u1, v0, v1) rectangles without points."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.arange(extent[0], extent[1], step), np.arange(0.12, v_hi, step))
    u, v = u.ravel(), v.ravel()
    keep = np.ones(u.size, bool)
    for (u0, u1, v0, v1) in holes:
        if jamb_jitter:  # each 10 cm row gets its own jamb offset: ragged / noisy jambs
            offs = rng.uniform(-jamb_jitter, jamb_jitter, size=(2, 64))
            row = np.clip(((v - 0.12) / 0.1).astype(int), 0, 63)
            keep &= ~((u >= u0 + offs[0][row]) & (u < u1 + offs[1][row]) & (v >= v0) & (v < v1))
        else:
            keep &= ~((u >= u0) & (u < u1) & (v >= v0) & (v < v1))
    u, v = u[keep], v[keep]
    xz = wall.centre + np.outer(u, wall.direction) + np.outer(rng.normal(0, noise, u.size), wall.normal)
    return np.column_stack([xz[:, 0], FLOOR_Y + v, xz[:, 1]])


def detect(cloud, walls, topo=None, subsets=None, **kw):
    return op.detect_openings(cloud, FLOOR, walls, topo, subsets, op.OpeningOptions(), **kw)


DOOR = (2.5, 3.4, 0.0, 2.05)  # 0.9 m wide, 2.05 m high, reaches the floor


def door_scene(holes=(DOOR,), segs=((0.0, 2.5), (3.4, 6.0)), **kw):
    wall = make_wall(segs=segs)
    return wall, cloud_on(wall, holes=holes, **kw)


# ---------- detection, width, type ----------


def test_clear_doorway_is_detected_and_classified_as_a_door():
    wall, cloud = door_scene()
    res = detect(cloud, [wall])
    assert len(res.accepted) == 1
    o = res.accepted[0]
    assert o.type == "door" and o.sill_height_m == 0.0
    assert o.existence_quality in ("strong", "moderate") and o.observability == "strong"


def test_door_width_is_measured_from_jambs():
    wall, cloud = door_scene()
    o = detect(cloud, [wall]).accepted[0]
    assert abs(o.width_m - 0.90) < 0.03
    assert o.width_interval_m[0] < 0.90 < o.width_interval_m[1]
    # the Ticket 5 gap said 0.90 here too; make the raw gap wrong and the refined width must still be right
    wall2 = make_wall(segs=((0.0, 2.3), (3.7, 6.0)))  # raw gap 1.4 m, true opening 0.9 m
    o2 = detect(cloud, [wall2]).accepted[0]
    assert abs(o2.width_m - 0.90) < 0.03 and abs(o2.candidate.raw_width_m - 1.4) < 1e-6


def test_opening_reaching_the_floor_with_observed_head_has_a_height():
    wall, cloud = door_scene()
    o = detect(cloud, [wall]).accepted[0]
    assert o.type == "door" and o.type_quality == "strong"
    assert o.height_m is not None and abs(o.height_m - 2.05) < 0.12


def test_window_with_wall_below_is_classified_as_a_window():
    wall, cloud = door_scene(holes=[(2.5, 3.5, 0.9, 1.9)])
    o = detect(cloud, [wall]).accepted[0]
    assert o.type == "window"
    assert abs(o.width_m - 1.0) <= 0.06  # one 5 cm grid cell of quantisation
    assert o.width_interval_m[0] <= 1.0 <= o.width_interval_m[1]  # and the interval honestly contains the truth


def test_window_sill_and_height_are_computed():
    wall, cloud = door_scene(holes=[(2.5, 3.5, 0.9, 1.9)], segs=((0.0, 2.5), (3.5, 6.0)))
    o = detect(cloud, [wall]).accepted[0]
    assert abs(o.sill_height_m - 0.9) < 0.12
    assert o.height_m is not None and abs(o.height_m - 1.0) < 0.2
    # no wall above the opening -> height is not invented
    wall2, cloud2 = door_scene(holes=[(2.5, 3.5, 0.9, 2.4)], segs=((0.0, 2.5), (3.5, 6.0)))
    o2 = detect(cloud2, [wall2]).accepted[0]
    assert o2.type == "window" and o2.height_m is None and o2.type_quality == "moderate"


def test_strong_opening_with_ambiguous_profile_is_reported_as_generic_opening():
    wall, cloud = door_scene(holes=[(2.5, 3.5, 0.3, 1.9)], segs=((0.0, 2.5), (3.5, 6.0)))  # sill 0.3: neither door nor window
    o = detect(cloud, [wall]).accepted[0]
    assert o.type == "opening" and o.type_quality == "weak"
    assert o.existence_quality in ("strong", "moderate")  # the opening itself is proven
    assert o.sill_height_m is not None


def test_wide_legitimate_opening_is_accepted_as_generic_opening():
    wall, cloud = door_scene(holes=[(2.0, 4.6, 0.0, 2.1)], segs=((0.0, 2.0), (4.6, 6.6)), extent=(0.0, 6.6))
    o = detect(cloud, [wall]).accepted[0]
    assert o.type == "opening" and abs(o.width_m - 2.6) < 0.05


# ---------- things that must NOT become openings ----------


def test_sparse_unobserved_region_is_not_an_opening():
    wall = make_wall(segs=((0.0, 2.5), (3.4, 6.0)))
    cloud = cloud_on(wall, holes=[(1.0, 5.0, 0.0, 2.4)])  # nothing was scanned between u = 1 and 5, not just the gap
    res = detect(cloud, [wall])
    assert res.accepted == [] and res.low_confidence == []
    assert res.rejected[0]["stage"] in ("3d_verification", "wall_end", "observability")


def test_gap_that_is_solid_wall_in_3d_is_rejected():
    wall = make_wall(segs=((0.0, 2.5), (3.4, 6.0)))  # Ticket 5 claims a gap...
    res = detect(cloud_on(wall), [wall])  # ...but the 3D wall is complete
    assert res.accepted == [] and "present across the gap" in res.rejected[0]["reason"]


def test_furniture_in_front_of_the_wall_does_not_become_a_strong_opening():
    wall, cloud = door_scene(holes=[(2.3, 3.6, 0.0, 2.4)], segs=((0.0, 2.5), (3.4, 6.0)))
    cabinet = cloud_on(wall, extent=(2.3, 3.6), holes=[(0, 0, 0, 0)], v_hi=1.7, seed=3)
    cabinet[:, [0, 2]] += 0.30 * wall.normal  # a cabinet standing 30 cm in front of where the wall would be
    res = detect(np.vstack([cloud, cabinet]), [wall])
    assert not any(o.existence_quality == "strong" for o in res.openings)
    assert res.accepted == [] and any(r["stage"] in ("occlusion", "shape") for r in res.rejected)


def test_wall_end_gap_is_rejected():
    wall = make_wall(segs=((0.0, 2.5), (3.4, 3.55)))  # the wall simply stops: only 15 cm of wall beyond the 'gap'
    cloud = cloud_on(wall, extent=(0.0, 3.55), holes=[(2.5, 3.4, 0.0, 2.4)])
    res = detect(cloud, [wall])
    assert res.accepted == [] and any(r["stage"] == "wall_end" for r in res.rejected)


def test_very_narrow_gap_is_rejected():
    wall, cloud = door_scene(holes=[(2.5, 2.75, 0.0, 2.4)], segs=((0.0, 2.5), (2.75, 6.0)))  # 25 cm
    res = detect(cloud, [wall])
    assert res.accepted == []
    assert any(r["stage"] == "size" for r in res.rejected)


def test_jamb_outlier_points_do_not_change_the_width():
    wall, cloud = door_scene()
    rng = np.random.default_rng(11)
    u = rng.uniform(2.55, 3.35, 12)  # a few stray points inside the doorway, on the wall plane
    v = rng.uniform(0.3, 1.9, 12)
    stray = np.column_stack([wall.centre[0] + u, FLOOR_Y + v, np.zeros(12)])
    clean = detect(cloud, [wall]).accepted[0]
    dirty = detect(np.vstack([cloud, stray]), [wall]).accepted[0]
    assert abs(dirty.width_m - clean.width_m) < 0.03 and abs(dirty.width_m - 0.90) < 0.04


# ---------- uncertainty ----------


def test_width_uncertainty_grows_with_ragged_jambs():
    wall = make_wall(segs=((0.0, 2.4), (3.5, 6.0)))
    clean = detect(cloud_on(wall, holes=[DOOR]), [wall]).openings[0]
    ragged = detect(cloud_on(wall, holes=[DOOR], jamb_jitter=0.12), [wall]).openings[0]
    half = lambda o: (o.width_interval_m[1] - o.width_interval_m[0]) / 2
    assert half(ragged) > 1.3 * half(clean)
    assert ragged.left_jamb["sigma_m"] > clean.left_jamb["sigma_m"]


def subsets_of(cloud, k=3):
    """Randomly scattered partition (fixed seed, so deterministic): like frames that each see different points."""
    bucket = np.random.default_rng(5).integers(0, k, len(cloud))
    return [cloud[bucket == i] for i in range(k)]


def test_stable_deterministic_subsets_give_a_narrow_interval():
    wall, cloud = door_scene()
    o = detect(cloud, [wall], subsets=subsets_of(cloud)).accepted[0]
    assert o.metrics["subsets_used"] == 3 and o.metrics["subset_rms_deviation_m"] < 0.03
    assert o.width_interval_m[1] - o.width_interval_m[0] < 0.15
    assert o.existence_quality == "strong"


def test_existence_is_never_strong_without_independent_subset_confirmation():
    wall, cloud = door_scene()
    assert detect(cloud, [wall]).accepted[0].existence_quality == "moderate"  # no subsets given
    only_one = detect(cloud, [wall], subsets=[cloud_on(wall, holes=[DOOR], seed=9), cloud[:100]])  # 2nd subset has no wall
    assert only_one.openings[0].existence_quality != "strong"


def test_unstable_subsets_widen_the_interval_and_weaken_quality():
    wall, cloud = door_scene()
    stable = detect(cloud, [wall], subsets=subsets_of(cloud)).accepted[0]
    shifted = [cloud_on(wall, holes=[h], seed=i) for i, h in enumerate([(2.5, 3.4, 0.0, 2.05), (2.35, 3.6, 0.0, 2.05), (2.62, 3.3, 0.0, 2.05)])]
    res = detect(cloud, [wall], subsets=shifted)
    o = res.openings[0]
    assert (o.width_interval_m[1] - o.width_interval_m[0]) > 2 * (stable.width_interval_m[1] - stable.width_interval_m[0])
    assert o.width_quality != "strong" and stable.width_quality != "weak"
    # existence is rated separately: every subset still finds the doorway, so disagreement on its width does not
    # make the opening itself doubtful
    assert o.existence_quality == "strong" and o.metrics["subsets_used"] == 3
    # a hugely inconsistent set of subsets is rejected outright
    wild = [cloud_on(wall, holes=[h], seed=i) for i, h in enumerate([(2.5, 3.4, 0.0, 2.05), (2.0, 4.2, 0.0, 2.05), (2.9, 3.1, 0.0, 2.05)])]
    assert detect(cloud, [wall], subsets=wild).accepted == []


# ---------- connectivity ----------


def two_room_scene():
    walls = [
        make_wall("south", (0, 0), (10, 0), ((0, 10),)), make_wall("east", (10, 0), (10, 5), ((0, 5),)),
        make_wall("north", (10, 5), (0, 5), ((0, 10),)), make_wall("west", (0, 5), (0, 0), ((0, 5),)),
        make_wall("partition", (5, 0), (5, 5), ((0, 2.0), (2.9, 5.0))),  # a 0.9 m door in the partition
    ]
    clouds = [cloud_on(walls[i], extent=(0, e)) for i, e in enumerate((10, 5, 10, 5))]
    clouds.append(cloud_on(walls[4], extent=(0, 5), holes=[(2.0, 2.9, 0.0, 2.05)]))
    return walls, np.vstack(clouds)


def test_verified_opening_between_adjacent_rooms_updates_connectivity():
    walls, cloud = two_room_scene()
    topo = build_topology(walls)
    assert len(topo.rooms) == 2 and topo.rooms[0].adjacent  # adjacent as a geometric fact
    res = detect(cloud, walls, topo)
    o = [x for x in res.accepted if x.candidate.wall_id == "partition"][0]
    assert sorted(o.room_ids) == ["room_001", "room_002"] and o.type == "door"
    assert res.connectivity["room_001"]["connected_room_ids"] == ["room_002"]
    assert res.connectivity["room_002"]["connected_room_ids"] == ["room_001"]
    assert sorted(o.connects) == ["room_001", "room_002"]


def test_adjacent_rooms_without_an_opening_are_not_connected():
    walls, cloud = two_room_scene()
    walls[4] = make_wall("partition", (5, 0), (5, 5), ((0, 5),))  # solid partition
    cloud = np.vstack([cloud_on(walls[i], extent=(0, e)) for i, e in enumerate((10, 5, 10, 5, 5))])
    topo = build_topology(walls)
    res = detect(cloud, walls, topo)
    assert res.accepted == []
    assert all(v["connected_room_ids"] == [] for v in res.connectivity.values())
    assert res.connectivity["room_001"]["adjacent_room_ids"] == ["room_002"]


def test_opening_to_outside_or_unmodelled_space_is_supported():
    walls = [make_wall("south", (0, 0), (6, 0), ((0, 2.5), (3.4, 6.0))), make_wall("east", (6, 0), (6, 5), ((0, 5),)),
             make_wall("north", (6, 5), (0, 5), ((0, 6),)), make_wall("west", (0, 5), (0, 0), ((0, 5),))]
    cloud = np.vstack([cloud_on(walls[0], holes=[DOOR]), cloud_on(walls[1], extent=(0, 5)),
                       cloud_on(walls[2]), cloud_on(walls[3], extent=(0, 5))])
    topo = build_topology(walls)
    res = detect(cloud, walls, topo)
    o = res.accepted[0]
    assert o.room_ids == ["room_001"] and o.connects == ["room_001", "unmodelled"]
    assert res.connectivity["room_001"]["opens_to_unmodelled"] == [o.id]
    assert res.connectivity["room_001"]["connected_room_ids"] == []


def test_openings_are_found_where_no_room_polygon_exists():
    wall, cloud = door_scene()
    res = detect(cloud, [wall], topo=None)
    assert len(res.accepted) == 1 and res.accepted[0].connects == ["unmodelled"] and res.accepted[0].room_ids == []


def test_collinear_walls_of_different_groups_produce_a_candidate():
    a = make_wall("a", (0, 0), (3, 0), ((0, 3),))
    b = make_wall("b", (4.2, 0), (7, 0), ((0, 2.8),))
    b = WallInput("b", "strong", np.array([4.2, 0.0]), np.array([1.0, 0.0]), 0.03, [((4.2, 0.0), (7.0, 0.0))])
    cands, gaps = op.generate_candidates([a, b], None, op.OpeningOptions())
    assert len(cands) == 1 and cands[0].source == "collinear_walls" and abs(cands[0].raw_width_m - 1.2) < 1e-6


# ---------- determinism and serialisation ----------


def test_detection_is_deterministic():
    walls, cloud = two_room_scene()
    topo = build_topology(walls)
    subs = subsets_of(cloud)
    a = op.openings_to_dict(detect(cloud, walls, topo, subs))
    b = op.openings_to_dict(detect(cloud.copy(), list(walls), build_topology(walls), [s.copy() for s in subs]))
    assert a == b


def test_results_serialise_to_json():
    walls, cloud = two_room_scene()
    res = detect(cloud, walls, build_topology(walls), subsets_of(cloud))
    d = op.openings_to_dict(res)
    assert json.loads(json.dumps(d)) == d
    o = d["openings"][0]
    for key in ("id", "wall_id", "type", "room_ids", "width_m", "width_interval_m", "height_m", "sill_height_m",
                "observability", "existence_quality", "type_quality", "width_quality", "left_jamb", "right_jamb", "connects"):
        assert key in o
    cd = json.loads(json.dumps(op.candidates_to_dict(res)))
    assert cd["candidates"] and all("outcome" in c for c in cd["candidates"])
