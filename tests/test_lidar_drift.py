"""Drift-correction tests: synthetic geometry only, no real captures."""

import json

import numpy as np
import pytest
from test_lidar_validator import make_capture

from spatialforge.lidar import drift
from spatialforge.lidar.drift import DriftOptions
from spatialforge.lidar.drift_run import run_drift_ablation
from spatialforge.lidar.reconstruction import ReconstructionOptions

OPTS = DriftOptions()


def rot_y(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def corner_points(step=0.02):
    """Three perpendicular planes in front of a camera at the origin: constrains all 6 DoF."""
    g = lambda lo, hi: np.arange(lo, hi, step)
    a, b = np.meshgrid(g(-1, 1.5), g(1, 3))
    wall_x = np.column_stack([np.full(a.size, 2.0), a.ravel(), b.ravel()])
    a, b = np.meshgrid(g(-1, 2), g(1, 3))
    floor = np.column_stack([a.ravel(), np.full(a.size, 1.5), b.ravel()])
    a, b = np.meshgrid(g(-1, 2), g(-1, 1.5))
    wall_z = np.column_stack([a.ravel(), b.ravel(), np.full(a.size, 3.0)])
    return np.vstack([wall_x, floor, wall_z])


def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def small_motion_scene(n=6, step_m=0.05):
    """Camera poses moving forward; every frame sees the same fixed corner. Poses are exact."""
    world = corner_points()
    poses = [drift.pose_matrix(np.eye(3), [0.0, 0.0, step_m * k]) for k in range(n)]
    cams = [transform(np.linalg.inv(p), world) for p in poses]  # world -> camera
    return list(range(n)), poses, cams


def build_clouds(cams):
    return [drift.make_registration_cloud(c, OPTS) for c in cams]


# ---------- pose algebra ----------


def test_relative_transform_maps_source_camera_to_target_camera():
    a = drift.pose_matrix(rot_y(20), [1, 0, 0])
    b = drift.pose_matrix(rot_y(-35) @ rot_x(10), [0, 2, 1])
    world_point = np.array([[0.3, -0.2, 2.0]])
    p_a = transform(np.linalg.inv(a), world_point)
    p_b = transform(np.linalg.inv(b), world_point)
    assert np.allclose(transform(drift.relative_transform(a, b), p_a), p_b)


def test_applying_a_pose_correction():
    original = drift.pose_matrix(rot_y(30), [1, 2, 3])
    optimised = drift.pose_matrix(rot_y(33), [1.1, 2.0, 2.9])
    correction = drift.correction_between(original, optimised)
    assert np.allclose(drift.apply_pose_correction(original, correction), optimised)
    assert drift.transform_delta(np.eye(4)) == (0.0, 0.0)


def test_keep_original_tilt_keeps_heading_change_only():
    original = drift.pose_matrix(rot_x(5), [0, 0, 0])
    optimised = drift.pose_matrix(rot_y(10) @ rot_x(8) @ rot_x(5), [1, 0, 0])  # +10 deg heading, +8 deg tilt
    out = drift.keep_original_tilt(original, optimised)
    assert drift.tilt_change_deg(original[:3, :3], out[:3, :3]) == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(out[:3, 3], [1, 0, 0])
    assert drift.rotation_angle_deg(out[:3, :3] @ original[:3, :3].T) == pytest.approx(10.0, abs=0.5)


# ---------- registration and acceptance ----------


def test_icp_recovers_known_transform():
    world = corner_points()
    true_rel = drift.pose_matrix(rot_y(1.5), [0.03, -0.02, 0.05])  # source cam -> target cam
    tgt = drift.make_registration_cloud(world, OPTS)
    src = drift.make_registration_cloud(transform(np.linalg.inv(true_rel), world), OPTS)
    guess = drift.pose_matrix(rot_y(0.5), [0.0, 0.0, 0.0])  # odometry guess is off by ~1 deg / 6 cm
    refined, fitness, rmse = drift.refine_edge(src, tgt, guess, [0.10])
    err_t, err_r = drift.transform_delta(np.linalg.inv(true_rel) @ refined)
    assert err_t < 0.01 and err_r < 0.3
    assert fitness > 0.8 and rmse < 0.03


def test_bad_registration_is_rejected():
    limits = dict(min_fitness=0.5, max_rmse=0.05, max_translation=0.15, max_rotation_deg=5.0)
    I = np.eye(4)
    assert drift.judge_edge(I, I, 0.9, 0.02, **limits)[0] is True
    assert "fitness" in drift.judge_edge(I, I, 0.2, 0.02, **limits)[1]
    assert "RMSE" in drift.judge_edge(I, I, 0.9, 0.2, **limits)[1]
    far = drift.pose_matrix(np.eye(3), [0.5, 0, 0])
    assert "correction" in drift.judge_edge(I, far, 0.9, 0.02, **limits)[1]
    assert "deg" in drift.judge_edge(I, drift.pose_matrix(rot_y(20), [0, 0, 0]), 0.9, 0.02, **limits)[1]

    # Unrelated random clouds must not pass the gates.
    rng = np.random.default_rng(1)
    a = drift.make_registration_cloud(rng.uniform(-2, 2, (4000, 3)), OPTS)
    b = drift.make_registration_cloud(rng.uniform(-2, 2, (4000, 3)), OPTS)
    assert drift.register_pair(0, 1, a, b, np.eye(4), "sequential", OPTS).accepted is False


def test_too_few_points_is_rejected():
    tiny = drift.make_registration_cloud(np.random.default_rng(0).uniform(0, 1, (50, 3)), OPTS)
    e = drift.register_pair(0, 1, tiny, tiny, np.eye(4), "sequential", OPTS)
    assert not e.accepted and e.reason == "too few points"


# ---------- loop closure candidates ----------


def test_loop_candidates_respect_distance_and_separation():
    opts = DriftOptions(loop_min_separation=10, loop_max_odom_distance=0.5)
    # Out along +x for 15 poses, then back to near the start: pose 20 revisits pose 2's area.
    xs = [0.3 * k for k in range(15)] + [0.3 * k for k in range(14, -1, -1)]
    poses = [drift.pose_matrix(np.eye(3), [x, 0, 0]) for x in xs]
    pairs = drift.find_loop_candidates(poses, opts)
    assert pairs, "the return leg must produce candidates"
    for i, j in pairs:
        assert j - i >= 10  # minimum frame separation
        assert np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3]) <= 0.5  # spatial threshold
    assert (0, 29) in pairs  # the true revisit, found
    assert drift.find_loop_candidates(poses, DriftOptions(loop_min_separation=100)) == []
    # Poses 0 and 29 are at exactly the same position, so they survive even a tiny distance threshold,
    # while near-misses (0.3 m apart) do not.
    tight = drift.find_loop_candidates(poses, DriftOptions(loop_max_odom_distance=0.01, loop_min_separation=10))
    assert (0, 29) in tight
    assert all(np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3]) <= 0.01 for i, j in tight)


def test_loop_candidates_require_similar_heading():
    opts = DriftOptions(loop_min_separation=2, loop_max_odom_distance=1.0, loop_max_heading_deg=60)
    poses = [drift.pose_matrix(np.eye(3), [0, 0, 0]), drift.pose_matrix(np.eye(3), [5, 0, 0]),
             drift.pose_matrix(rot_y(180), [0.1, 0, 0])]  # position matches pose 0 but faces backwards
    assert drift.find_loop_candidates(poses, opts) == []


def test_loop_candidates_are_capped():
    poses = [drift.pose_matrix(np.eye(3), [0.001 * k, 0, 0]) for k in range(200)]
    assert len(drift.find_loop_candidates(poses, DriftOptions(loop_max_candidates=7))) == 7


# ---------- pose graph and the full correction ----------


def test_pose_graph_edge_convention_matches_open3d():
    """Consistent edges leave poses alone; a disagreeing loop edge pulls the chain toward it."""
    poses = [drift.pose_matrix(rot_y(8 * k), [0.4 * k, 0.0, 0.2 * k]) for k in range(6)]
    info = np.eye(6) * 100

    def optimise(extra_dx=None):
        g = drift.reg.PoseGraph()
        for p in poses:
            g.nodes.append(drift.reg.PoseGraphNode(p))
        for i in range(5):
            g.edges.append(drift.reg.PoseGraphEdge(i, i + 1, drift.relative_transform(poses[i], poses[i + 1]), info))
        if extra_dx:
            moved = poses[5].copy()
            moved[0, 3] += extra_dx
            g.edges.append(drift.reg.PoseGraphEdge(0, 5, drift.relative_transform(poses[0], moved), info * 10))
        drift.reg.global_optimization(
            g, drift.reg.GlobalOptimizationLevenbergMarquardt(), drift.reg.GlobalOptimizationConvergenceCriteria(),
            drift.reg.GlobalOptimizationOption(max_correspondence_distance=0.1, edge_prune_threshold=0.25, reference_node=0),
        )
        return [np.asarray(n.pose) for n in g.nodes]

    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(optimise(), poses))
    pulled = optimise(0.3)
    assert pulled[5][0, 3] - poses[5][0, 3] > 0.2
    assert np.allclose(pulled[0], poses[0])  # node 0 is the fixed reference


def test_perfectly_aligned_clouds_give_negligible_correction():
    frames, poses, cams = small_motion_scene()
    result = drift.estimate_corrected_poses(frames, poses, build_clouds(cams), OPTS)
    assert sum(e.accepted for e in result.edges if e.kind == "sequential") == len(frames) - 1
    assert result.max_translation_m < 2e-3
    assert result.max_rotation_deg < 0.1
    assert result.max_tilt_change_deg < 0.05


def test_correction_is_deterministic():
    frames, poses, cams = small_motion_scene()
    # Give the odometry a small error so the optimiser actually has something to do.
    noisy = [drift.pose_matrix(np.eye(3), p[:3, 3] + [0.01 * (k % 2), 0, 0.01]) for k, p in enumerate(poses)]
    a = drift.estimate_corrected_poses(frames, noisy, build_clouds(cams), OPTS)
    b = drift.estimate_corrected_poses(frames, noisy, build_clouds(cams), OPTS)
    assert all(np.allclose(x, y, atol=1e-9) for x, y in zip(a.corrected_poses, b.corrected_poses))
    assert [e.accepted for e in a.edges] == [e.accepted for e in b.edges]
    assert a.max_translation_m > 0  # the noise really was corrected, not ignored


def test_safe_fallback_when_nothing_passes_quality_gates():
    rng = np.random.default_rng(3)
    clouds = [drift.make_registration_cloud(rng.uniform(-3, 3, (5000, 3)), OPTS) for _ in range(5)]
    poses = [drift.pose_matrix(rot_y(5 * k), [0.1 * k, 0, 0]) for k in range(5)]
    result = drift.estimate_corrected_poses(list(range(5)), poses, clouds, OPTS)
    assert not any(e.accepted for e in result.edges)
    assert all(np.array_equal(o, c) for o, c in zip(result.original_poses, result.corrected_poses))
    assert result.max_translation_m == 0.0 and result.max_rotation_deg == 0.0


def test_judge_correction_rejects_worse_and_unchanged():
    frames, poses, cams = small_motion_scene()
    result = drift.estimate_corrected_poses(frames, poses, build_clouds(cams), OPTS)
    base = {"overlap": 0.5, "neighbour_residual_median_m": 0.02, "floor_band_thickness_m": 0.1,
            "floor_peak_share": 0.2, "wall_slab_cells_per_1000_points": 15.0, "footprint_x_m": 10.0, "footprint_z_m": 10.0}
    rules = drift.AcceptanceRules()
    assert drift.judge_correction(base, dict(base), result, rules)[:2] == (False, "no metric improved meaningfully")
    better = dict(base, floor_peak_share=0.25)
    assert drift.judge_correction(base, better, result, rules)[0] is True
    worse = dict(better, floor_band_thickness_m=0.2)
    ok, reason, _ = drift.judge_correction(base, worse, result, rules)
    assert not ok and "floor_band_thickness_m" in reason
    shrunk = dict(better, footprint_x_m=8.0)
    assert not drift.judge_correction(base, shrunk, result, rules)[0]


# ---------- end-to-end ablation on a synthetic capture ----------


def test_ablation_report_and_artifacts(tmp_path):
    cap = make_capture(tmp_path / "cap", n=5)
    out = tmp_path / "out"
    opts = ReconstructionOptions(frame_step=1, max_frames=None, source_size=(1920, 1440))
    report = run_drift_ablation(cap, out, opts)

    for name in ("before.ply", "after.ply", "before_topdown.png", "after_topdown.png",
                 "drift_report.json", "trajectory_before.csv", "trajectory_after.csv"):
        assert (out / name).stat().st_size > 0, name
    saved = json.loads((out / "drift_report.json").read_text())
    assert saved == json.loads(json.dumps(report))  # serialisable and identical to the returned report
    for key in ("capture", "frames_used", "method", "sequential_edges", "loop_closures", "before", "after",
                "pose_correction", "accepted", "fallback_reason", "production_poses"):
        assert key in saved
    assert saved["sequential_edges"]["attempted"] == 4
    # These flat 8x6 frames are too small to register, so the safe fallback must engage.
    assert saved["accepted"] is False and saved["production_poses"] == "original"
    assert saved["pose_correction"]["max_translation_m"] == 0.0

    # BEFORE and AFTER use exactly the same frames.
    frames = lambda f: [line.split(",")[0] for line in (out / f).read_text().splitlines()[1:]]
    assert frames("trajectory_before.csv") == frames("trajectory_after.csv") == [str(i) for i in range(5)]
    assert saved["before"]["footprint_x_m"] == saved["after"]["footprint_x_m"]


def test_ablation_is_deterministic(tmp_path):
    cap = make_capture(tmp_path / "cap", n=5)
    opts = ReconstructionOptions(frame_step=1, max_frames=None, source_size=(1920, 1440))
    a = run_drift_ablation(cap, tmp_path / "a", opts)
    b = run_drift_ablation(cap, tmp_path / "b", opts)
    a.pop("runtime_s"), b.pop("runtime_s")
    assert a == b
    assert (tmp_path / "a" / "before.ply").read_bytes() == (tmp_path / "b" / "before.ply").read_bytes()
    assert (tmp_path / "a" / "after.ply").read_bytes() == (tmp_path / "b" / "after.ply").read_bytes()
