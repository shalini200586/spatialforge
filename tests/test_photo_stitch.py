"""Cross-room stitching: evidence, relative transforms, constraints, graph solve, loops, overlaps, adjacency."""

import math
import random

import numpy as np
import pytest
from helpers_stitch import DOORS, LAYOUT, global_error, make_geoms, rect, to_local, true_constraint, true_pose
from helpers_video import camera_pose

from spatialforge.photos.stitch import (
    OpeningGeom, RoomGeom, StitchOptions, adjacent_edges, apply, build_doorway_constraint, build_visual_constraint, compose,
    components, cross_room_evidence, doorway_hypotheses, initial_poses, invert, loop_report, match_doorways, overlap_of,
    overlap_report, place_component, rot, solve_poses, visual_pair_transform,
)

OPTS = StitchOptions()
IDS = sorted(LAYOUT)


# ---------------- planar algebra ----------------


def test_planar_transform_algebra():
    p, q = (0.7, 2.0, -1.0), (-1.9, 0.5, 3.0)
    pts = np.array([[1.0, 2.0], [-3.0, 0.5]])
    assert np.allclose(apply(compose(p, q), pts), apply(p, apply(q, pts)))
    assert np.allclose(apply(invert(p), apply(p, pts)), pts)
    c = compose(p, invert(p))
    assert abs(c[0]) < 1e-12 and abs(c[1]) < 1e-12 and abs(c[2]) < 1e-12
    assert np.allclose(rot(math.pi / 2) @ [1, 0], [0, 1])


# ---------------- cross-room feature evidence ----------------


def pair(a, b, raw, ver):
    return {"a": a, "b": b, "raw": raw, "verified": ver}


ROOM_OF = {"r1/a.jpg": "room_001", "r1/b.jpg": "room_001", "r2/a.jpg": "room_002", "r2/b.jpg": "room_002", "r3/a.jpg": "room_003"}


def test_cross_room_feature_evidence_is_grouped_by_room_pair_and_reports_every_pair():
    stats = [pair("r1/a.jpg", "r2/a.jpg", 300, 150), pair("r1/b.jpg", "r2/b.jpg", 200, 60), pair("r1/a.jpg", "r1/b.jpg", 500, 400),
             pair("r2/a.jpg", "r3/a.jpg", 40, 3)]
    ev = {(e.room_a, e.room_b): e for e in cross_room_evidence(stats, ROOM_OF, OPTS)}
    assert set(ev) == {("room_001", "room_002"), ("room_001", "room_003"), ("room_002", "room_003")}  # all room pairs reported
    e12 = ev[("room_001", "room_002")]
    assert e12.accepted and e12.verified_total == 210 and e12.best_pair_verified == 150
    assert e12.best_inlier_ratio == pytest.approx(0.5) and len(e12.image_pairs) == 2
    assert e12.image_pairs[0]["raw_matches"] == 300 and e12.image_pairs[0]["verified_matches"] == 150  # within-room pairs excluded
    assert not ev[("room_001", "room_003")].accepted and "no image pair" in ev[("room_001", "room_003")].reason


def test_geometric_match_rejection():
    # many raw matches but few geometrically verified ones: raw matches never count
    e = cross_room_evidence([pair("r1/a.jpg", "r2/a.jpg", 900, 10)], ROOM_OF, OPTS)
    r12 = next(x for x in e if (x.room_a, x.room_b) == ("room_001", "room_002"))
    assert not r12.accepted and "verified matches" in r12.reason and r12.verified_total == 10
    # a good pair but too little in total
    low = cross_room_evidence([pair("r1/a.jpg", "r2/a.jpg", 60, 30)], ROOM_OF, OPTS)
    r = next(x for x in low if (x.room_a, x.room_b) == ("room_001", "room_002"))
    assert not r.accepted and "only 30 verified" in r.reason
    # plenty verified but a terrible inlier ratio
    ratio = cross_room_evidence([pair("r1/a.jpg", "r2/a.jpg", 1000, 90)], ROOM_OF, OPTS)
    assert not next(x for x in ratio if x.room_b == "room_002" and x.room_a == "room_001").accepted
    assert cross_room_evidence([pair("r1/a.jpg", "r2/a.jpg", 5, 3)], ROOM_OF, OPTS)[0].accepted is False


# ---------------- relative transform from a joint model ----------------


class JointPose:
    def __init__(self, R_cw, t_cw):
        self.rotation_cw, self.translation_cw = R_cw, t_cw


def planar_to_3d(pose):
    th, tx, tz = pose
    psi = -th
    R = np.array([[math.cos(psi), 0, math.sin(psi)], [0, 1, 0], [-math.sin(psi), 0, math.cos(psi)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, 0.0, tz]
    return T  # local -> global


def cameras_in_room(rid, n=3, seed=0):
    rng = random.Random(seed + int(rid[-1]))
    r = LAYOUT[rid]
    out = []
    for k in range(n):
        pos = [rng.uniform(r[0, 0] + 0.8, r[2, 0] - 0.8), 1.4, rng.uniform(r[0, 1] + 0.8, r[2, 1] - 0.8)]
        out.append(camera_pose(pos, yaw=rng.uniform(0, 6.28), pitch=rng.uniform(-0.2, 0.2)))
    return out  # camera -> global


def joint_scene(a, b, n=3, s=0.37, tilt=0.0):
    ids = IDS
    Qj = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    if np.linalg.det(Qj) < 0:
        Qj[:, 0] *= -1
    tj = np.array([3.0, -1.0, 2.0])
    poses_a, poses_b, joint = {}, {}, {}
    for rid, poses in ((a, poses_a), (b, poses_b)):
        F = planar_to_3d(true_pose(ids.index(rid)))
        if tilt:
            c, s_ = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
            F[:3, :3] = F[:3, :3] @ np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])  # a room frame that is not truly gravity-aligned
        for k, Tg in enumerate(cameras_in_room(rid, n)):
            name = f"{rid}/img_{k:02d}.jpg"
            poses[name] = np.linalg.inv(F) @ Tg  # camera -> room-local
            R_j = Qj @ Tg[:3, :3]
            C_j = s * Qj @ Tg[:3, 3] + tj
            joint[name] = JointPose(R_j.T, -R_j.T @ C_j)
    return poses_a, poses_b, joint


def test_relative_2d_transform_from_a_joint_model_is_recovered():
    a, b = "room_001", "room_002"
    pa, pb, joint = joint_scene(a, b)
    tf, why = visual_pair_transform(pa, pb, joint, OPTS)
    assert tf is not None, why
    truth = compose(invert(true_pose(0)), true_pose(1))
    assert tf.tx == pytest.approx(truth[1], abs=1e-6) and tf.tz == pytest.approx(truth[2], abs=1e-6)
    assert abs(math.degrees((tf.theta_rad - truth[0] + math.pi) % (2 * math.pi) - math.pi)) < 1e-4
    assert tf.tilt_deg < 1e-3 and tf.scale_ratio == pytest.approx(1.0, abs=1e-6) and tf.fit_rms_m < 1e-6
    assert tf.images_a == 3 and tf.images_b == 3 and tf.sigma_t_m < 0.2 and 0 < tf.sigma_theta_deg < 3
    assert set(tf.to_dict()) >= {"rotation_deg", "translation_x_m", "translation_z_m", "joint_scale_ratio"}


def test_visual_transform_rejects_degenerate_or_inconsistent_joint_models():
    a, b = "room_001", "room_002"
    pa, pb, joint = joint_scene(a, b)
    only_a = {k: v for k, v in joint.items() if k.startswith(a)}
    tf, why = visual_pair_transform(pa, pb, only_a, OPTS)
    assert tf is None and "only one room" in why
    pa1 = {k: v for k, v in list(pa.items())[:1]}
    pb1 = {k: v for k, v in list(pb.items())[:1]}
    tf, why = visual_pair_transform(pa1, pb1, joint, OPTS)
    assert tf is None and "joint-model scale" in why  # one image per room: no baseline to fix the scale
    pa, pb, joint = joint_scene(a, b, tilt=15.0)  # rooms whose vertical axes disagree
    tf, why = visual_pair_transform(pa, pb, joint, OPTS)
    assert tf is None and "tilted" in why
    # a joint model whose scale differs between the two rooms
    pa, pb, joint = joint_scene(a, b)
    for k in list(joint):
        if k.startswith(b):
            R_j = joint[k].rotation_cw.T
            C_j = -R_j @ joint[k].translation_cw
            C_j = C_j * 2.0
            joint[k] = JointPose(R_j.T, -R_j.T @ C_j)
    tf, why = visual_pair_transform(pa, pb, joint, OPTS)
    assert tf is None and "disagree" in why


# ---------------- doorways ----------------


def two_rooms():
    layout = {"room_001": LAYOUT["room_001"], "room_002": LAYOUT["room_002"]}
    doors = [DOORS[0]]
    g = make_geoms(layout, doors)
    return g["room_001"], g["room_002"]


def test_shared_doorway_creates_a_strong_constraint_from_a_visual_match():
    a, b = two_rooms()
    truth = compose(invert(true_pose(0)), true_pose(1))
    pa, pb, joint = joint_scene("room_001", "room_002")
    tf, _ = visual_pair_transform(pa, pb, joint, OPTS)
    ev = cross_room_evidence([pair("room_001/img_00.jpg", "room_002/img_00.jpg", 400, 220)],
                             {"room_001/img_00.jpg": "room_001", "room_002/img_00.jpg": "room_002"}, OPTS)[0]
    c = build_visual_constraint(ev, tf, a, b, OPTS)
    assert c.evidence_type == "visual+doorway" and c.quality == "strong"
    assert (c.source_opening_id, c.target_opening_id) == ("opening_001", "opening_001")
    assert c.verified_match_count == 220 and c.inlier_ratio == pytest.approx(220 / 400)
    assert any("doorway" in n for n in c.notes)
    no_doors = (RoomGeom(a.id, a.polygon, []), RoomGeom(b.id, b.polygon, []))
    assert build_visual_constraint(ev, tf, *no_doors, OPTS).evidence_type == "visual"
    # weaker visual evidence: moderate on its own, and a coinciding doorway raises it one level
    weak_ev = cross_room_evidence([pair("room_001/img_00.jpg", "room_002/img_00.jpg", 300, 90)],
                                  {"room_001/img_00.jpg": "room_001", "room_002/img_00.jpg": "room_002"}, OPTS)[0]
    only_visual = build_visual_constraint(weak_ev, tf, *no_doors, OPTS)
    assert only_visual.evidence_type == "visual" and only_visual.quality == "moderate" and only_visual.source_opening_id is None
    with_door = build_visual_constraint(weak_ev, tf, a, b, OPTS)
    assert with_door.evidence_type == "visual+doorway" and with_door.quality == "strong"
    assert math.hypot(c.translation_x_m - truth[1], c.translation_z_m - truth[2]) < 1e-6


def test_doorway_match_requires_coincidence_width_and_parallel_walls():
    a, b = two_rooms()
    truth = compose(invert(true_pose(0)), true_pose(1))
    assert len(match_doorways(a, b, truth, OPTS)) == 1
    off = (truth[0], truth[1] + 1.5, truth[2])  # shifted by 1.5 m
    assert match_doorways(a, b, off, OPTS) == []
    turned = (truth[0] + math.radians(60), truth[1], truth[2])
    assert match_doorways(a, b, turned, OPTS) == []
    wide = RoomGeom(b.id, b.polygon, [OpeningGeom("opening_001", "w", b.openings[0].left, b.openings[0].left + 3 * (b.openings[0].right - b.openings[0].left), 2.7)])
    assert match_doorways(a, wide, truth, OPTS) == []


def test_doorway_only_constraint_is_weak_and_needs_a_unique_placement():
    a, b = two_rooms()
    c, why = build_doorway_constraint(a, b, OPTS)
    assert c is not None, why
    truth = compose(invert(true_pose(0)), true_pose(1))
    assert c.evidence_type == "doorway" and c.quality == "weak" and c.verified_match_count == 0
    assert math.hypot(c.translation_x_m - truth[1], c.translation_z_m - truth[2]) < 1e-6
    assert abs((c.rotation_deg - math.degrees(truth[0]) + 180) % 360 - 180) < 1e-6
    # two equally good doorways in both rooms -> several different placements -> refuse to guess
    g = make_geoms(dict(list(LAYOUT.items())[:2]), [DOORS[0], ("room_001", "room_002", (5.0, 0.3), (5.0, 1.2))])
    c2, why2 = build_doorway_constraint(g["room_001"], g["room_002"], OPTS)
    assert c2 is None and "ambiguous" in why2
    assert build_doorway_constraint(RoomGeom("x", None, a.openings), b, OPTS)[0] is None  # no polygon, no validation
    assert doorway_hypotheses(a, RoomGeom("y", b.polygon, []), OPTS) == []


# ---------------- global placement ----------------


def solved(ids, edges, noise=None, anchor=None):
    geoms = make_geoms()
    cons = [true_constraint(a, b, IDS, (noise or {}).get((a, b), (0.0, 0.0))) for a, b in edges]
    comps = components(ids, cons)
    return geoms, cons, comps


def test_two_room_stitching_recovers_the_layout():
    geoms, cons, comps = solved(["room_001", "room_002"], [("room_001", "room_002")])
    assert comps == [["room_001", "room_002"]]
    p = place_component(comps[0], cons, geoms, OPTS)
    assert p.anchor == "room_001" and set(p.rooms) == {"room_001", "room_002"}
    assert global_error(p.poses, IDS, "room_001") == (pytest.approx(0, abs=1e-5), pytest.approx(0, abs=1e-4))
    placed_b = apply(p.poses["room_002"], geoms["room_002"].polygon)
    expect = LAYOUT["room_002"] - LAYOUT["room_001"][0]  # relative to the anchor's own placement
    inter, small, frac = overlap_of(apply(p.poses["room_001"], geoms["room_001"].polygon), placed_b, 0.05)
    assert frac < 0.02 and expect.shape == placed_b.shape


def test_three_room_chain():
    ids = ["room_001", "room_002", "room_004"]
    geoms, cons, comps = solved(ids, [("room_001", "room_002"), ("room_002", "room_004")])
    p = place_component(comps[0], cons, geoms, OPTS)
    assert set(p.rooms) == set(ids) and not p.rejected and p.loops == []
    t, r = global_error(p.poses, IDS, "room_001")
    assert t < 1e-4 and r < 1e-3


def test_three_room_loop_closes():
    edges = [("room_001", "room_002"), ("room_002", "room_004"), ("room_003", "room_004"), ("room_001", "room_003")]
    noise = {("room_001", "room_002"): (0.3, 0.04), ("room_002", "room_004"): (-0.2, -0.03), ("room_003", "room_004"): (0.2, 0.02)}
    geoms, cons, comps = solved(["room_001", "room_002", "room_003", "room_004"], edges, noise)
    p = place_component(comps[0], cons, geoms, OPTS)
    assert not p.rejected and len(p.loops) == 1
    assert p.loops[0]["translation_residual_m"] < 0.2 and p.loops[0]["rotation_residual_deg"] < 1.0
    t, r = global_error(p.poses, IDS, "room_001")
    assert t < 0.15 and r < 1.0  # the loop is distributed, not forced to zero on one edge
    assert all(c.residual_sigma is not None and c.residual_sigma < 2.0 for c in p.constraints)


def test_an_inconsistent_loop_edge_is_rejected_not_averaged_into_everything():
    edges = [("room_001", "room_002"), ("room_002", "room_004"), ("room_003", "room_004"), ("room_001", "room_003")]
    geoms, cons, comps = solved(["room_001", "room_002", "room_003", "room_004"], edges, {("room_003", "room_004"): (25.0, 3.0)})
    cons[2].quality = "weak"
    p = place_component(comps[0], cons, geoms, OPTS)
    assert [(c.source_room_id, c.target_room_id) for c in p.rejected] == [("room_003", "room_004")]
    assert p.rejected[0].status == "rejected" and "loop" in p.rejected[0].notes[-1]
    assert set(p.rooms) == {"room_001", "room_002", "room_003", "room_004"}  # still connected through the other edges
    t, r = global_error(p.poses, IDS, "room_001")
    assert t < 1e-3 and r < 1e-2  # the good edges are untouched by the bad one
    assert any("rejected room_003->room_004" in a for a in p.actions)


def test_room_and_constraint_ordering_does_not_change_the_result():
    edges = [("room_001", "room_002"), ("room_002", "room_004"), ("room_003", "room_004"), ("room_001", "room_003")]
    noise = {("room_001", "room_002"): (0.3, 0.04), ("room_002", "room_004"): (-0.2, -0.03), ("room_003", "room_004"): (0.2, 0.02)}
    results = []
    for seed in (0, 1, 2):
        geoms, cons, _ = solved(IDS[:4], edges, noise)
        rnd = random.Random(seed)
        rnd.shuffle(cons)
        rooms = IDS[:4][:]
        rnd.shuffle(rooms)
        p = place_component(rooms, cons, geoms, OPTS)
        results.append({k: (round(v[0], 9), round(v[1], 6), round(v[2], 6)) for k, v in sorted(p.poses.items())})
    assert results[0] == results[1] == results[2]


def test_a_disconnected_room_forms_its_own_component_and_is_never_attached():
    ids = ["room_001", "room_002", "room_005"]
    geoms, cons, comps = solved(ids, [("room_001", "room_002")])
    assert comps == [["room_001", "room_002"], ["room_005"]]
    assert initial_poses(ids, cons, "room_001").keys() == {"room_001", "room_002"}  # room_005 gets no pose


# ---------------- overlap ----------------


def test_overlap_detection():
    a, b = rect(0, 0, 4, 4), rect(3, 0, 7, 4)
    inter, small, frac = overlap_of(a, b, 0.05)
    assert inter == pytest.approx(4.0, abs=0.1) and small == pytest.approx(16.0, abs=0.1) and frac == pytest.approx(0.25, abs=0.01)
    assert overlap_of(a, rect(4, 0, 8, 4), 0.05)[2] < 0.01  # touching rooms
    assert overlap_of(a, rect(1, 1, 2, 2), 0.05)[2] == pytest.approx(1.0, abs=0.01)  # a room inside a room: fully overlapped
    geoms = make_geoms()
    poses = {r: compose(true_pose(i), (0, 0, 0)) for i, r in enumerate(IDS)}
    rep = overlap_report(IDS, poses, geoms, OPTS)
    assert len(rep) == 10 and all(set(r) >= {"room_a", "room_b", "intersection_area_m2", "fraction_of_smaller"} for r in rep)


def test_valid_adjacent_rooms_do_not_significantly_overlap():
    ids = IDS[:4]
    edges = [("room_001", "room_002"), ("room_002", "room_004"), ("room_003", "room_004"), ("room_001", "room_003")]
    geoms, cons, comps = solved(ids, edges)
    p = place_component(comps[0], cons, geoms, OPTS)
    rep = overlap_report(p.rooms, p.poses, geoms, OPTS)
    assert max(r["fraction_of_smaller"] for r in rep) < 0.01


def test_an_overlap_producing_constraint_is_rejected_and_the_room_left_unplaced():
    ids = ["room_001", "room_002", "room_004"]
    geoms, cons, comps = solved(ids, [("room_001", "room_002"), ("room_002", "room_004")])
    # make room_004's constraint wrong so that it lands on top of room_001 (true location shifted by (-5, -4) m)
    bad = cons[1]
    wrong = compose(invert(true_pose(1)), compose((0.0, -5.0, -4.0), true_pose(3)))
    bad.rotation_deg, bad.translation_x_m, bad.translation_z_m = math.degrees(wrong[0]), wrong[1], wrong[2]
    bad.quality = "weak"
    p = place_component(comps[0], cons, geoms, OPTS)
    assert [(c.source_room_id, c.target_room_id) for c in p.rejected] == [("room_002", "room_004")]
    assert "overlap" in p.rejected[0].notes[-1] and "room_004" not in p.rooms
    assert any("overlapped" in a for a in p.actions)


def test_an_alternative_doorway_hypothesis_is_tried_before_rejecting():
    a, b = two_rooms()
    c, _ = build_doorway_constraint(a, b, OPTS)
    assert c is not None and isinstance(c.alternatives, list)
    from spatialforge.photos.stitch import Hypothesis

    truth_pose = c.pose
    onto_a = compose(invert(true_pose(0)), compose((0.0, -4.0, 0.0), true_pose(1)))  # B shifted 4 m left: on top of A
    c.rotation_deg, c.translation_x_m, c.translation_z_m = math.degrees(onto_a[0]), onto_a[1], onto_a[2]
    c.alternatives = [Hypothesis(truth_pose, "opening_001", "opening_001", 1.0, 0.0, 0.0)]
    p = place_component(["room_001", "room_002"], [c], {"room_001": a, "room_002": b}, OPTS)
    assert not p.rejected and any("next doorway hypothesis" in x for x in p.actions)
    assert any("alternative doorway hypothesis used" in n for n in c.notes)
    assert p.poses["room_002"][1:] == pytest.approx(tuple(truth_pose[1:]), abs=1e-3)  # the alternative placement was applied


# ---------------- adjacency ----------------


def test_adjacency_geometry():
    assert adjacent_edges(rect(0, 0, 5, 4), rect(5, 0, 9, 4))
    assert adjacent_edges(rect(0, 0, 5, 4), rect(5.3, 1, 9, 3))  # a wall's thickness apart
    assert not adjacent_edges(rect(0, 0, 5, 4), rect(7, 0, 9, 4))  # a 2 m gap
    assert not adjacent_edges(rect(0, 0, 5, 4), rect(5, 3.8, 9, 7))  # touching along only 0.2 m
    assert adjacent_edges(rect(0, 0, 5, 4), rect(0, 4, 5, 7))
