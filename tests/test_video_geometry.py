"""SfM result model, pose conventions, metric scale, back-projection, gravity, MetricScene and uncertainty."""

import json

import numpy as np
import pytest
from helpers_video import SyntheticWalkthrough, camera_pose

from spatialforge.pipeline.lidar_adapter import MeasureContext
from spatialforge.pipeline.scene import MetricScene, SceneError
from spatialforge.video.fusion import FusionFrame, FusionOptions, backproject_frame, fuse_frames, multiview_support
from spatialforge.video.gravity import estimate_gravity, rotation_to_y_up
from spatialforge.video.scale import (
    FrameScale, ScaleOptions, estimate_metric_scale, frame_alignment_factor, frame_scale, metric_camera_to_world,
    sample_depth,
)
from spatialforge.video.sfm import (
    SfmCameraInfo, SfmOptions, SfmResult, cam_to_world, classify_tracking, quat_xyzw_to_rotation,
)
from spatialforge.video.uncertainty import propagate


# ---------- SfM result model and tracking quality ----------


def test_sfm_result_serialises_to_plain_json():
    cam = SfmCameraInfo("SIMPLE_RADIAL", 1280, 960, [1100.0, 640.0, 480.0, 0.02], 1100.0, "estimated by COLMAP")
    r = SfmResult(100, 90, 0.9, 2500, 0.8, 5.2, 1, [90], cam, ["kf_0003.jpg"], "strong", ["ok"], {"mapping_s": 3.2}, 640)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["registered_images"] == 90 and d["camera"]["focal_px"] == 1100.0 and d["unregistered"] == ["kf_0003.jpg"]
    assert d["tracking_quality"] == "strong" and d["seconds"] == {"mapping_s": 3.2}


def test_tracking_quality_classes():
    o = SfmOptions()
    assert classify_tracking(100, 110, 3000, 0.7, o)[0] == "strong"
    assert classify_tracking(70, 110, 1200, 0.9, o)[0] == "moderate"
    assert classify_tracking(100, 110, 3000, 2.5, o)[0] != "strong"  # a big reprojection error blocks STRONG
    assert classify_tracking(30, 110, 500, 0.9, o)[0] == "weak"
    assert classify_tracking(6, 110, 400, 0.9, o)[0] == "failure"  # too few images
    assert classify_tracking(12, 150, 450, 0.7, o)[0] == "failure"  # 8% registered: no coherent trajectory
    assert classify_tracking(0, 50, 0, None, o)[0] == "failure"


# ---------- pose conventions ----------


def test_colmap_world_to_camera_converts_to_camera_to_world():
    T_wc = camera_pose([1.0, 1.4, 2.0], yaw=0.7, pitch=0.1)
    R_cw = T_wc[:3, :3].T
    t_cw = -R_cw @ T_wc[:3, 3]
    assert np.allclose(cam_to_world(R_cw, t_cw), T_wc, atol=1e-12)
    X = np.array([0.5, 0.2, 3.0])  # a point in the camera frame
    assert np.allclose(R_cw @ (T_wc[:3, :3] @ X + T_wc[:3, 3]) + t_cw, X)


def test_quaternion_xyzw_convention():
    s = np.sqrt(0.5)
    R = quat_xyzw_to_rotation([0, 0, s, s])  # +90 degrees about z
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert np.allclose(quat_xyzw_to_rotation([0, 0, 0, 2.0]), np.eye(3))  # not unit length: normalised


def test_pose_convention_matches_pycolmap_when_installed():
    pycolmap = pytest.importorskip("pycolmap")
    q = np.array([-0.203325, -0.0606779, -0.192829, 0.958016])
    t = np.array([-3.1583, 1.58004, 4.14757])
    rig = pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)
    R = quat_xyzw_to_rotation(rig.rotation.quat)
    assert np.allclose(R, rig.rotation.matrix(), atol=1e-6)
    M = np.vstack([rig.matrix(), [0, 0, 0, 1]])
    assert np.allclose(cam_to_world(R, t), np.linalg.inv(M), atol=1e-6)


# ---------- metric scale ----------


def synthetic_observations(n=200, scale=2.5, noise=0.03, outliers=0.0, seed=0, shape=(96, 128), size=(1280, 960)):
    """Sparse points (pixels + SfM depth) and a depth map that equals scale * SfM depth (plus noise)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    xy = np.column_stack([rng.uniform(40, size[0] - 40, n), rng.uniform(40, size[1] - 40, n)])
    # smooth, gently varying SfM depth field so a 5x5 window median is representative
    zs = 1.0 + 0.8 * (xy[:, 0] / size[0]) + 0.5 * (xy[:, 1] / size[1])
    depth = np.zeros(shape)
    for j in range(h):
        for i in range(w):
            depth[j, i] = scale * (1.0 + 0.8 * (i / w) + 0.5 * (j / h))
    depth = depth * np.exp(rng.normal(0, noise, shape))
    zs_obs = zs.copy()
    k = int(outliers * n)
    if k:  # a share of correspondences is wrong by a large factor
        zs_obs[:k] *= rng.uniform(3, 8, k)
    return xy, zs_obs, depth.astype(np.float32), size


def test_relative_scale_is_metric_depth_over_sfm_depth():
    xy, z, depth, size = synthetic_observations(scale=2.5, noise=0.0)
    fs = frame_scale("f", xy, z, depth, size, ScaleOptions())
    assert fs.scale == pytest.approx(2.5, rel=0.03) and fs.used > 100 and fs.spread_rel < 0.05
    vals, ok = sample_depth(depth, xy[:5], size, ScaleOptions())
    assert ok.all() and np.all(vals > 2.4)


def test_robust_estimator_rejects_outlier_correspondences_and_outlier_frames():
    xy, z, depth, size = synthetic_observations(scale=2.5, outliers=0.25, noise=0.02)
    fs = frame_scale("f", xy, z, depth, size, ScaleOptions())
    assert fs.scale == pytest.approx(2.5, rel=0.06)  # 25% gross outliers do not move the median
    frames = [FrameScale(f"f{i}", 200, 150, 2.5 * (1 + 0.02 * ((i % 5) - 2)), 0.05) for i in range(12)]
    frames[3] = FrameScale("bad", 200, 150, 9.0, 0.05)  # one wildly inconsistent frame
    est = estimate_metric_scale(frames)
    assert est.available and est.scale == pytest.approx(2.5, rel=0.05)


def test_inconsistent_per_frame_scales_lower_the_quality():
    rng = np.random.default_rng(1)

    def est(sd, opts=None):
        fr = [FrameScale(f"f{i}", 300, 200, float(2.5 * np.exp(rng.normal(0, sd))), 0.05) for i in range(30)]
        return estimate_metric_scale(fr, opts)

    tight, mid, wild = est(0.03), est(0.65), est(1.2)
    order = {"strong": 3, "moderate": 2, "weak": 1, "unavailable": 0}
    assert order[tight.quality] > order[mid.quality] > order[wild.quality]
    assert (tight.quality, mid.quality, wild.quality) == ("moderate", "weak", "unavailable")
    assert tight.sigma_rel < mid.sigma_rel < (wild.sigma_rel or 99)
    assert not wild.available and wild.scale is None  # no metres from scales that disagree this much
    assert tight.sigma_rel >= 0.10  # never below the unvalidated-model floor, however consistent the frames are
    assert est(0.03, ScaleOptions(allow_strong=True, model_floor_rel=0.04)).quality == "strong"  # only a calibrated source may


def test_insufficient_evidence_gives_no_scale():
    few = [FrameScale(f"f{i}", 300, 200, 2.5, 0.05) for i in range(3)]
    e = estimate_metric_scale(few)
    assert not e.available and e.scale is None and e.quality == "unavailable" and "only 3" in e.reasons[0]
    thin = [FrameScale(f"f{i}", 30, 10, 2.5, 0.05) for i in range(10)]
    assert not estimate_metric_scale(thin).available  # 100 correspondences < 200
    assert estimate_metric_scale([]).scale is None


def test_valid_scales_produce_a_metric_trajectory():
    walk = SyntheticWalkthrough(n=6)
    a, Q = walk.a, walk.Q
    centres_sfm, metric = [], []
    for T in walk.poses:
        R_cw = T[:3, :3].T @ Q.T
        C = a * Q @ T[:3, 3]
        t_cw = -R_cw @ C
        centres_sfm.append(C)
        metric.append(metric_camera_to_world(R_cw, t_cw, 1.0 / a))
    d_sfm = np.linalg.norm(np.diff(centres_sfm, axis=0), axis=1)
    d_m = np.linalg.norm(np.diff([m[:3, 3] for m in metric], axis=0), axis=1)
    assert np.allclose(d_m, d_sfm / a)  # distances are in metres once multiplied by the scale
    assert np.allclose(d_m, np.linalg.norm(np.diff([T[:3, 3] for T in walk.poses], axis=0), axis=1), atol=1e-9)
    assert np.allclose(np.array([m[:3, :3] for m in metric]) @ np.array([m[:3, :3] for m in metric]).transpose(0, 2, 1),
                       np.eye(3), atol=1e-9)  # orientation untouched: still a proper rotation


def test_frame_alignment_factor_and_exclusion():
    f = FrameScale("f", 300, 200, 2.0, 0.05)
    assert frame_alignment_factor(f, 2.5) == pytest.approx(1.25)
    assert frame_alignment_factor(FrameScale("g", 300, 200, 0.5, 0.05), 2.5) is None  # 5x off: inconsistent, excluded
    assert frame_alignment_factor(FrameScale("h", 0, 0, None, None, "none"), 2.5) is None


# ---------- back-projection and fusion ----------


def test_metric_depth_back_projects_through_the_camera_pose():
    depth = np.full((20, 40), 2.0)
    T = camera_pose([1.0, 1.4, 0.5], yaw=np.pi / 2)  # looking along +X
    fr = FusionFrame("f", depth, 40.0, 40.0, 20.0, 10.0, T)
    pts, n = backproject_frame(fr, FusionOptions(pixel_stride=1, edge_threshold=1.0, min_range_m=0.1))
    assert n == 800 and len(pts) == 800
    assert np.allclose(pts[:, 0], 1.0 + 2.0, atol=1e-6)  # a fronto-parallel wall 2 m ahead along +X
    centre = pts[np.argmin(np.abs(pts[:, 2] - 0.5) + np.abs(pts[:, 1] - 1.4))]
    assert np.allclose(centre[[1, 2]], [1.4, 0.5], atol=0.06)  # the principal ray hits straight ahead
    assert pts[:, 1].min() < 1.4 < pts[:, 1].max()  # image rows map to height (y down in the image = lower in the world)


def test_back_projection_undoes_radial_distortion():
    depth = np.full((21, 41), 3.0)
    T = np.eye(4)
    plain = backproject_frame(FusionFrame("f", depth, 40, 40, 20.5, 10.5, T), FusionOptions(pixel_stride=1, edge_threshold=9))[0]
    dist = backproject_frame(FusionFrame("f", depth, 40, 40, 20.5, 10.5, T, k1=0.1), FusionOptions(pixel_stride=1, edge_threshold=9))[0]
    r_plain, r_dist = np.hypot(plain[:, 0], plain[:, 1]), np.hypot(dist[:, 0], dist[:, 1])
    assert r_dist.max() < r_plain.max()  # positive k1: edge rays move toward the centre once undistorted
    xu, yu = dist[:, 0] / 3.0, dist[:, 1] / 3.0  # undistorted normalised coordinates
    f = 1.0 + 0.1 * (xu ** 2 + yu ** 2)  # the forward model: x_distorted = x_undistorted * (1 + k1 r_u^2)
    assert np.allclose(np.column_stack([xu * f, yu * f]), plain[:, :2] / 3.0, atol=1e-6)  # recovers the pixel rays
    centre = np.argmin(np.hypot(plain[:, 0], plain[:, 1]))
    assert np.allclose(plain[centre], dist[centre], atol=1e-3)


def test_multiview_support_drops_single_view_points():
    pts = np.array([[0, 0, 0], [0.01, 0, 0], [5, 5, 5]], dtype=float)
    keep = multiview_support(pts, np.array([0, 1, 0]), 0.1, 2)
    assert keep.tolist() == [True, True, False]


def test_fusion_of_a_synthetic_room_is_metric_and_deterministic():
    walk = SyntheticWalkthrough(n=10, depth_noise=0.0)
    frames = []
    for i, T in enumerate(walk.poses):
        frames.append(FusionFrame(walk.names[i], walk.depth_gt[i], 1000.0 * 160 / 1280, 1000.0 * 120 / 960, 80.0, 60.0, T))
    a = fuse_frames(frames, FusionOptions(voxel_m=0.05, pixel_stride=1))
    b = fuse_frames(frames, FusionOptions(voxel_m=0.05, pixel_stride=1))
    assert np.array_equal(a.points, b.points) and a.frames_used == 10
    lo, hi = a.points.min(axis=0), a.points.max(axis=0)
    assert lo.min() > -0.1 and (hi - np.array(walk.room)).max() < 0.1  # every point lies inside the real room
    assert a.consistency_removed_fraction < 0.5 and "after_voxel_downsample" in a.stats()


# ---------- gravity ----------


def rotate(points, cams, R):
    return points @ R.T, cams @ R.T


def fused_room(walk):
    frames = [FusionFrame(walk.names[i], walk.depth_gt[i], 1000.0 * 160 / 1280, 1000.0 * 120 / 960, 80.0, 60.0, T)
              for i, T in enumerate(walk.poses)]
    return fuse_frames(frames, FusionOptions(voxel_m=0.05)).points, np.array([T[:3, 3] for T in walk.poses])


def random_rotations(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        yield R, rng


def test_gravity_alignment_recovers_world_up_on_a_synthetic_room():
    walk = SyntheticWalkthrough(n=24, depth_noise=0.0)
    cloud, cams = fused_room(walk)
    image_up = np.array([-T[:3, 1] for T in walk.poses])  # an upright video: image-up is the camera -y axis
    for R, rng in random_rotations(3, 11):
        pts, c = cloud @ R.T, cams @ R.T
        noisy = pts + rng.normal(0, 0.03, pts.shape)
        g = estimate_gravity(noisy, c, camera_up_hints=image_up @ R.T)
        true_up = R @ np.array([0, 1.0, 0])
        err = np.degrees(np.arccos(np.clip(g.up @ true_up, -1, 1)))
        assert err < 3.0, err
        assert g.sign_basis.startswith(("floor below", "camera image-up"))  # decisive evidence, not a default
        # after rotating into +Y-up the floor is the lowest plane and the camera sits ~1.4 m above it
        R_up = rotation_to_y_up(g.up)
        h = (noisy @ R_up.T)[:, 1]
        # sanity: a residual tilt below 3 degrees over a 5 m room and 3 cm noise blur the extremes by a few decimetres
        assert np.percentile(h, 99.8) - np.percentile(h, 0.2) == pytest.approx(2.6, abs=0.45)  # the room is 2.6 m tall
        assert np.median((c @ R_up.T)[:, 1]) - np.percentile(h, 0.2) == pytest.approx(1.4, abs=0.35)  # cameras 1.4 m up
    assert g.confidence > 0.2 and g.to_dict()["method"]


def box_cloud(room, n=60000, seed=0):
    """Points sampled uniformly (by area) on the six faces of a box room: floor and ceiling are clearly visible."""
    rng = np.random.default_rng(seed)
    x, y, z = room
    areas = np.array([x * z, x * z, y * z, y * z, x * y, x * y], dtype=float)
    face = rng.choice(6, size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    p = np.zeros((n, 3))
    for k in range(6):
        m = face == k
        if k < 2:
            p[m] = np.column_stack([u[m] * x, np.full(m.sum(), 0.0 if k == 0 else y), v[m] * z])
        elif k < 4:
            p[m] = np.column_stack([np.full(m.sum(), 0.0 if k == 2 else x), u[m] * y, v[m] * z])
        else:
            p[m] = np.column_stack([u[m] * x, v[m] * y, np.full(m.sum(), 0.0 if k == 4 else z)])
    return p


def test_gravity_sign_is_decisive_when_only_one_orientation_has_a_plausible_floor():
    room = (5.0, 4.4, 4.0)  # the ceiling is 3 m above the cameras: it cannot be a floor seen from a handheld camera
    cloud = box_cloud(room)
    cams = np.array([T[:3, 3] for T in SyntheticWalkthrough(n=24).poses])
    for R, rng in random_rotations(3, 5):
        g = estimate_gravity(cloud @ R.T + rng.normal(0, 0.02, cloud.shape), cams @ R.T, camera_up_hints=None)
        assert g.sign_basis.startswith("floor below")
        assert np.degrees(np.arccos(np.clip(g.up @ (R @ np.array([0, 1.0, 0])), -1, 1))) < 3.0
        assert g.concentration > g.baseline_concentration * 1.5 and g.quality in ("moderate", "strong")


def test_gravity_does_not_trust_a_rotated_image_up():
    """Phone video stored rotated by 90 degrees: image-up is horizontal. It must carry no weight, and the estimate must
    admit that the orientation rests on weaker evidence."""
    walk = SyntheticWalkthrough(n=24, depth_noise=0.0)
    cloud, cams = fused_room(walk)
    sideways = np.array([T[:3, 0] for T in walk.poses])  # image-up pointing along the camera x axis (a horizontal direction)
    g = estimate_gravity(cloud, cams, sideways)
    assert abs(g.up @ np.array([0, 1.0, 0])) > 0.99  # the axis still comes from the geometry
    assert "image-up" not in g.sign_basis
    assert g.quality != "strong" and g.notes  # and it says so


def test_rotation_to_y_up_maps_the_vector_and_is_a_rotation():
    rng = np.random.default_rng(2)
    for _ in range(5):
        u = rng.normal(size=3)
        R = rotation_to_y_up(u)
        assert np.allclose(R @ (u / np.linalg.norm(u)), [0, 1, 0], atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(rotation_to_y_up([0, -1, 0]) @ [0, -1, 0], [0, 1, 0])


def test_gravity_is_not_fooled_by_a_rotated_image_up():
    """The image-up hint is only a tie-breaker: a wrong (90-degree rotated) hint must not override the geometry."""
    walk = SyntheticWalkthrough(n=24, depth_noise=0.0)
    frames = [FusionFrame(walk.names[i], walk.depth_gt[i], 1000.0 * 160 / 1280, 1000.0 * 120 / 960, 80.0, 60.0, T)
              for i, T in enumerate(walk.poses)]
    cloud = fuse_frames(frames, FusionOptions(voxel_m=0.05)).points
    cams = np.array([T[:3, 3] for T in walk.poses])
    wrong_hint = np.array([[1.0, 0, 0]] * len(cams))  # "up" along world X: 90 degrees off
    g = estimate_gravity(cloud, cams, wrong_hint)
    assert abs(g.up @ np.array([0, 1.0, 0])) > 0.99


# ---------- MetricScene ----------


def test_metric_scene_summary_and_roundtrip(tmp_path):
    pts = np.random.default_rng(0).random((500, 3))
    poses = np.array([camera_pose([0, 1.4, 0], 0.0), camera_pose([1, 1.4, 0], 0.3)])
    s = MetricScene(pts, "video", "moderate", "weak", poses, ["kf_0000.jpg", "kf_0001.jpg"], ["a warning"], {"k": 1})
    d = s.summary()
    assert d["source_tier"] == "video" and d["points"] == 500 and d["cameras"] == 2 and d["scale_quality"] == "weak"
    assert len(d["bounds_min_m"]) == 3 and json.loads(json.dumps(d)) == d
    s.save(tmp_path / "scene.npz")
    t = MetricScene.load(tmp_path / "scene.npz")
    assert np.allclose(t.points_xyz_m, pts.astype(np.float32)) and np.allclose(t.camera_poses, poses)
    assert t.frame_refs == s.frame_refs and t.warnings == ["a warning"] and t.metadata == {"k": 1}
    assert t.source_tier == "video" and t.scale_quality == "weak"
    assert t.camera_positions.shape == (2, 3) and t.frame_subsets() == []


def test_metric_scene_validation():
    ok = np.zeros((3, 3))
    with pytest.raises(SceneError):
        MetricScene(np.array([[np.nan, 0, 0]]), "video", "strong", "strong")
    with pytest.raises(SceneError):
        MetricScene(ok, "radar", "strong", "strong")
    with pytest.raises(SceneError):
        MetricScene(ok, "video", "great", "strong")
    with pytest.raises(SceneError):
        MetricScene(ok, "video", "strong", "magic")
    with pytest.raises(SceneError):
        MetricScene(ok, "video", "strong", "strong", np.zeros((2, 4, 4)), ["only one"])
    lidar = MetricScene(ok, "lidar", "strong", "sensor")  # LiDAR fits the same model
    assert lidar.scale_quality == "sensor" and len(lidar.camera_positions) == 0


# ---------- video uncertainty ----------


def test_video_uncertainty_widens_with_scale_variance_and_weak_tracking():
    base = propagate(0.06, 0.6, 1.0, 0.1)
    noisy_scale = propagate(0.25, 0.6, 1.0, 0.1)
    assert noisy_scale.total_rel_sigma > base.total_rel_sigma * 2 and noisy_scale.scale_rel_sigma == 0.25
    assert propagate(0.06, 2.0, 1.0, 0.1).sfm_rel_sigma > base.sfm_rel_sigma  # worse reprojection
    assert propagate(0.06, 0.6, 0.3, 0.1).sfm_rel_sigma > base.sfm_rel_sigma  # unregistered keyframes
    assert propagate(0.06, 0.6, 1.0, 0.7).depth_rel_sigma > base.depth_rel_sigma  # multi-view disagreement
    assert base.total_rel_sigma == pytest.approx(np.sqrt(base.scale_rel_sigma ** 2 + base.sfm_rel_sigma ** 2 + base.depth_rel_sigma ** 2))
    assert "not calibrated" in base.to_dict()["note"]


def test_measure_context_widens_intervals_and_leaves_lidar_untouched():
    lidar = MeasureContext("lidar").measure(4.0, "m", (3.9, 4.1), quality="strong")
    assert (lidar.low, lidar.high, lidar.source_tier) == (3.9, 4.1, "lidar")
    assert MeasureContext("lidar").measure(4.0, "m").low is None  # nothing invented without a sigma
    small = MeasureContext("video", 0.05).measure(4.0, "m", (3.9, 4.1), quality="strong")
    big = MeasureContext("video", 0.20).measure(4.0, "m", (3.9, 4.1), quality="strong")
    assert small.source_tier == "video" and small.value == 4.0
    assert (big.high - big.low) > (small.high - small.low) > (lidar.high - lidar.low)
    assert big.low < small.low < 3.9 and big.high > small.high > 4.1
    assert small.low <= small.value <= small.high
    # no stage interval: the video interval comes from the scale uncertainty alone (k * sigma * value)
    bare = MeasureContext("video", 0.10, k=2.0).measure(5.0, "m")
    assert bare.low == pytest.approx(4.0) and bare.high == pytest.approx(6.0)
    area = MeasureContext("video", 0.10, k=2.0).measure(10.0, "m2")
    assert area.high - area.value == pytest.approx(4.0)  # area error is twice the length error
    ang = MeasureContext("video", 0.10).measure(90.0, "deg", (89.0, 91.0))
    assert (ang.low, ang.high) == (89.0, 91.0)  # angles do not depend on scale
    capped = MeasureContext("video", 0.1, quality_cap="weak").measure(3.0, "m", quality="strong")
    assert capped.quality == "weak"
    assert MeasureContext("video", 0.1, quality_cap="moderate").measure(3.0, "m", quality="weak").quality == "weak"
