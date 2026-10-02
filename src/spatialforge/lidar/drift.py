"""Pose refinement and drift correction for LiDAR captures (ICP + pose graph, Open3D, CPU).

Idea in plain English
---------------------
The supplied odometry is the starting trajectory. For neighbouring sampled frames we
re-align their depth clouds with ICP, starting from the odometry guess. Good alignments
become pose-graph constraints, poor ones fall back to the odometry relation with low
weight. A few conservative loop-closure constraints may be added where the camera
revisits a place. The pose graph is optimised and the result is only used if the
before/after metrics agree it is safe (see `judge_correction`).

Pose convention (from Ticket 2): every pose is camera-to-world, world = R @ p_cam + t,
world +Y is up. `relative_transform(a, b)` maps points in camera a into camera b.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import open3d as o3d

from spatialforge.lidar import geometry

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
reg = o3d.pipelines.registration


# ---------- options and thresholds ----------


@dataclass(frozen=True)
class DriftOptions:
    # Registration clouds are much coarser than the 0.02 m output voxel: ICP only needs
    # a few thousand well-spread points per frame, and 5 cm is above the depth noise.
    reg_voxel: float = 0.05
    normal_radius: float = 0.15  # 3 x reg_voxel
    normal_max_nn: int = 30
    min_points: int = 300  # a frame cloud smaller than this is not registered

    # Sequential edges (odometry guess is good, so correspondences are searched close by).
    seq_max_corr: float = 0.10
    seq_min_fitness: float = 0.50
    seq_max_rmse: float = 0.05
    seq_max_translation: float = 0.15  # ICP may not move the odometry guess further than this
    seq_max_rotation_deg: float = 5.0

    # Loop closures are stricter than sequential edges: a false loop is worse than a miss.
    loop_min_separation: int = 15  # in sampled-frame indices
    loop_max_odom_distance: float = 1.5  # camera positions must already be this close
    loop_max_heading_deg: float = 60.0  # and look roughly the same way
    loop_max_candidates: int = 20
    loop_coarse_corr: float = 0.25
    loop_max_corr: float = 0.10
    loop_min_fitness: float = 0.60
    loop_max_rmse: float = 0.04
    loop_max_translation: float = 0.40
    loop_max_rotation_deg: float = 8.0

    enable_loop_closure: bool = True
    # Pose-graph weights. Every consecutive pair keeps an isotropic odometry edge (the anchor);
    # accepted ICP edges are added on top. A sequential ICP edge is weak evidence: on the samples
    # it deviates ~20 mm / 0.8 deg per edge from odometry, while the odometry closes whole loops
    # to ~2.5 cm / 2 deg (~1 mm per edge), so by inverse variance ICP deserves a tiny weight.
    # ICP-only graphs random-walked away from the odometry in first experiments (see README).
    # Loop closures are the global evidence and keep full weight.
    odometry_edge_weight: float = 1.0
    icp_edge_weight: float = 0.02
    loop_edge_weight: float = 1.0
    constrain_gravity: bool = True  # keep the supplied roll/pitch, allow only heading refinement
    metric_voxel: float = 0.03  # per-frame voxel used for before/after metrics


@dataclass(frozen=True)
class AcceptanceRules:
    """Whole-capture decision rules. Fixed in advance, not tuned per capture."""

    min_accepted_sequential_fraction: float = 0.5
    min_relative_improvement: float = 0.02  # at least one metric must improve by 2 %
    max_relative_degradation: float = 0.03  # and none may get worse by more than 3 %
    max_footprint_change: float = 0.05  # X and Z extents may change by at most 5 %
    max_translation_m: float = 0.5
    max_rotation_deg: float = 5.0
    max_tilt_change_deg: float = 1.0


# ---------- pose algebra ----------


def pose_matrix(rotation: np.ndarray, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def relative_transform(src_pose: np.ndarray, tgt_pose: np.ndarray) -> np.ndarray:
    """Transform taking points in the source camera frame to the target camera frame.

    Both arguments are camera-to-world 4x4 matrices.
    """
    return np.linalg.inv(tgt_pose) @ src_pose


def correction_between(original: np.ndarray, optimised: np.ndarray) -> np.ndarray:
    """World-frame correction D with D @ original == optimised."""
    return optimised @ np.linalg.inv(original)


def apply_pose_correction(pose: np.ndarray, correction: np.ndarray) -> np.ndarray:
    return correction @ pose


def rotation_angle_deg(R: np.ndarray) -> float:
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def transform_delta(T: np.ndarray) -> tuple[float, float]:
    """(translation metres, rotation degrees) of a transform's deviation from identity."""
    return float(np.linalg.norm(T[:3, 3])), rotation_angle_deg(T[:3, :3])


def tilt_change_deg(R_before: np.ndarray, R_after: np.ndarray) -> float:
    """Roll/pitch change: angle between where world-up points in the camera before vs after.

    A pure heading (yaw) change leaves this at zero.
    """
    up = np.array([0.0, 1.0, 0.0])
    a, b = R_before.T @ up, R_after.T @ up
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def keep_original_tilt(original: np.ndarray, optimised: np.ndarray) -> np.ndarray:
    """Keep the optimised position and heading but the original roll/pitch.

    The optimiser's world-frame rotation change is reduced to its rotation about world +Y.
    """
    delta = optimised[:3, :3] @ original[:3, :3].T
    theta = np.arctan2(delta[0, 2] - delta[2, 0], delta[0, 0] + delta[2, 2])
    c, s = np.cos(theta), np.sin(theta)
    yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    out = optimised.copy()
    out[:3, :3] = yaw @ original[:3, :3]
    return out


def constrain_relative_gravity(
    refined: np.ndarray, src_pose: np.ndarray, tgt_pose: np.ndarray
) -> np.ndarray:
    """Remove roll/pitch change from an ICP relative transform (source cam -> target cam).

    The refined transform implies a new target pose given the source pose; that pose keeps
    its heading and position but gets the target's original roll/pitch back.
    """
    implied_tgt = src_pose @ np.linalg.inv(refined)
    return relative_transform(src_pose, keep_original_tilt(tgt_pose, implied_tgt))


def refine_translation_only(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    transform: np.ndarray,
    max_corr: float,
    iterations: int = 5,
    ridge: float = 0.01,
) -> np.ndarray:
    """Re-solve only the translation of `transform` (point-to-plane), keeping its rotation fixed.

    Used after the gravity constraint changes an ICP rotation: the ICP translation was estimated
    together with the old rotation, so keeping it would be inconsistent (a 0.5 deg tilt over a 3 m
    lever arm is a 2.6 cm shift). `ridge` (a fraction of the correspondence count) keeps the
    solution at the current value along directions the geometry does not constrain.
    """
    T = transform.copy()
    src_pts, tgt_pts, tgt_normals = (np.asarray(x) for x in (src.points, tgt.points, tgt.normals))
    for _ in range(iterations):
        corr = np.asarray(reg.evaluate_registration(src, tgt, max_corr, T).correspondence_set)
        if len(corr) < 50:
            break
        p = src_pts[corr[:, 0]] @ T[:3, :3].T + T[:3, 3]
        n = tgt_normals[corr[:, 1]]
        dist = np.einsum("ij,ij->i", n, p - tgt_pts[corr[:, 1]])  # signed point-to-plane distances
        A = n.T @ n + ridge * len(corr) * np.eye(3)
        T[:3, 3] += np.linalg.solve(A, -n.T @ dist)
    return T


# ---------- registration ----------


def make_registration_cloud(camera_points: np.ndarray, opts: DriftOptions) -> o3d.geometry.PointCloud:
    """Downsampled camera-frame cloud with normals pointing toward the sensor."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(camera_points, dtype=np.float64))
    pcd = pcd.voxel_down_sample(opts.reg_voxel)
    if len(pcd.points) >= 3:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(opts.normal_radius, opts.normal_max_nn))
        pcd.orient_normals_towards_camera_location(np.zeros(3))
    return pcd


@dataclass
class EdgeRecord:
    source: int  # frame IDs
    target: int
    kind: str  # "sequential" or "loop"
    accepted: bool
    reason: str
    fitness: float
    inlier_rmse: float
    translation_correction_m: float
    rotation_correction_deg: float
    initial: np.ndarray = field(repr=False, default_factory=lambda: np.eye(4))
    refined: np.ndarray = field(repr=False, default_factory=lambda: np.eye(4))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("initial")
        d.pop("refined")
        for key in ("fitness", "inlier_rmse", "translation_correction_m", "rotation_correction_deg"):
            d[key] = round(d[key], 5)
        return d


def refine_edge(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    initial: np.ndarray,
    corr_distances: list[float],
) -> tuple[np.ndarray, float, float]:
    """Point-to-plane ICP from the odometry guess; coarse-to-fine over `corr_distances`."""
    T = initial
    result = None
    for dist in corr_distances:
        result = reg.registration_icp(
            src, tgt, dist, T, reg.TransformationEstimationPointToPlane(),
            reg.ICPConvergenceCriteria(max_iteration=50),
        )
        T = result.transformation
    return np.asarray(T), float(result.fitness), float(result.inlier_rmse)


def judge_edge(
    initial: np.ndarray,
    refined: np.ndarray,
    fitness: float,
    rmse: float,
    *,
    min_fitness: float,
    max_rmse: float,
    max_translation: float,
    max_rotation_deg: float,
) -> tuple[bool, str, float, float]:
    """Accept/reject one ICP result. Returns (accepted, reason, translation_m, rotation_deg)."""
    trans, rot = transform_delta(np.linalg.inv(initial) @ refined)
    if fitness < min_fitness:
        return False, f"fitness {fitness:.2f} < {min_fitness}", trans, rot
    if rmse > max_rmse:
        return False, f"inlier RMSE {rmse:.4f} > {max_rmse}", trans, rot
    if trans > max_translation:
        return False, f"correction {trans:.3f} m > {max_translation}", trans, rot
    if rot > max_rotation_deg:
        return False, f"correction {rot:.1f} deg > {max_rotation_deg}", trans, rot
    return True, "ok", trans, rot


def register_pair(
    fid_src: int,
    fid_tgt: int,
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    initial: np.ndarray,
    kind: str,
    opts: DriftOptions,
) -> EdgeRecord:
    if min(len(src.points), len(tgt.points)) < opts.min_points:
        return EdgeRecord(fid_src, fid_tgt, kind, False, "too few points", 0.0, 0.0, 0.0, 0.0, initial, initial)
    if kind == "sequential":
        corr = [opts.seq_max_corr]
        limits = dict(min_fitness=opts.seq_min_fitness, max_rmse=opts.seq_max_rmse,
                      max_translation=opts.seq_max_translation, max_rotation_deg=opts.seq_max_rotation_deg)
    else:
        corr = [opts.loop_coarse_corr, opts.loop_max_corr]
        limits = dict(min_fitness=opts.loop_min_fitness, max_rmse=opts.loop_max_rmse,
                      max_translation=opts.loop_max_translation, max_rotation_deg=opts.loop_max_rotation_deg)
    refined, fitness, rmse = refine_edge(src, tgt, initial, corr)
    ok, reason, trans, rot = judge_edge(initial, refined, fitness, rmse, **limits)
    return EdgeRecord(fid_src, fid_tgt, kind, ok, reason, fitness, rmse, trans, rot, initial, refined)


# ---------- loop-closure candidates ----------


def find_loop_candidates(
    poses: list[np.ndarray], opts: DriftOptions
) -> list[tuple[int, int]]:
    """Index pairs (i, j), i < j, that are far apart in time but close in odometry space.

    Conditions: j - i >= loop_min_separation, camera positions within loop_max_odom_distance,
    and viewing directions within loop_max_heading_deg. At most one partner per frame (the
    spatially closest), then at most loop_max_candidates pairs overall (closest first).
    """
    n = len(poses)
    pos = np.array([p[:3, 3] for p in poses])
    fwd = np.array([p[:3, 2] for p in poses])  # camera +Z (forward) in world
    cos_limit = np.cos(np.radians(opts.loop_max_heading_deg))
    found = []
    for i in range(n):
        best = None
        for j in range(i + opts.loop_min_separation, n):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist > opts.loop_max_odom_distance or float(fwd[i] @ fwd[j]) < cos_limit:
                continue
            if best is None or dist < best[0]:
                best = (dist, j)
        if best is not None:
            found.append((best[0], i, best[1]))
    found.sort()
    return [(i, j) for _, i, j in found[: opts.loop_max_candidates]]


# ---------- pose graph ----------


def optimise_pose_graph(
    poses: list[np.ndarray],
    clouds: list[o3d.geometry.PointCloud],
    edges: list[EdgeRecord],
    index_of: dict[int, int],
    opts: DriftOptions,
) -> list[np.ndarray]:
    """Optimise node poses (camera-to-world). Node 0 is held fixed, so scale and frame are kept.

    Every consecutive pair has an odometry edge with isotropic weight. Accepted ICP edges
    (sequential and loop) are added on top, weighted by the information matrix of their point
    correspondences, which is small along directions the geometry does not constrain.
    Rejected edges contribute nothing beyond the odometry edge. Loop edges are 'uncertain' so
    the optimiser may prune ones that contradict the rest of the graph.
    """
    # Open3D edge convention: T maps source points to the target frame, i.e.
    # T = inv(pose_target) @ pose_source (checked in tests/test_lidar_drift.py).
    icp_infos = {}
    for k, e in enumerate(edges):
        if e.accepted:
            i, j = index_of[e.source], index_of[e.target]
            icp_infos[k] = np.asarray(
                reg.get_information_matrix_from_point_clouds(clouds[i], clouds[j], opts.seq_max_corr * 1.5, e.refined)
            )
    traces = [np.trace(m) / 6 for m in icp_infos.values()]
    scale = float(np.median(traces)) if traces else 1000.0  # typical information per axis
    odometry_info = np.eye(6) * scale * opts.odometry_edge_weight

    graph = reg.PoseGraph()
    for pose in poses:
        graph.nodes.append(reg.PoseGraphNode(pose))
    for k, e in enumerate(edges):
        i, j = index_of[e.source], index_of[e.target]
        if e.kind == "sequential":
            graph.edges.append(reg.PoseGraphEdge(i, j, e.initial, odometry_info, uncertain=False))
        if e.accepted:
            weight = opts.loop_edge_weight if e.kind == "loop" else opts.icp_edge_weight
            graph.edges.append(
                reg.PoseGraphEdge(i, j, e.refined, icp_infos[k] * weight, uncertain=(e.kind == "loop"))
            )
    reg.global_optimization(
        graph,
        reg.GlobalOptimizationLevenbergMarquardt(),
        reg.GlobalOptimizationConvergenceCriteria(),
        reg.GlobalOptimizationOption(
            max_correspondence_distance=opts.seq_max_corr,
            edge_prune_threshold=0.25,
            preference_loop_closure=1.0,
            reference_node=0,
        ),
    )
    return [np.asarray(node.pose).copy() for node in graph.nodes]


# ---------- result of a full correction run ----------


@dataclass
class CorrectionResult:
    frames: list[int]
    original_poses: list[np.ndarray]
    corrected_poses: list[np.ndarray]  # after optional gravity constraint
    edges: list[EdgeRecord]
    loop_candidates: int
    max_raw_tilt_change_deg: float  # optimiser output, before the gravity constraint
    mean_translation_m: float = 0.0
    max_translation_m: float = 0.0
    mean_rotation_deg: float = 0.0
    max_rotation_deg: float = 0.0
    max_tilt_change_deg: float = 0.0  # of the poses actually returned
    # How far the poses are from satisfying the accepted ICP constraints, mean (metres, degrees).
    # "before" is the supplied odometry, "after" the corrected poses. Lower is better.
    residual_before: dict = field(default_factory=dict)
    residual_after: dict = field(default_factory=dict)


def estimate_corrected_poses(
    frames: list[int],
    original_poses: list[np.ndarray],
    clouds: list[o3d.geometry.PointCloud],
    opts: DriftOptions,
) -> CorrectionResult:
    """Register sampled frames, optimise the pose graph, return corrected poses.

    `clouds` are camera-frame registration clouds (see make_registration_cloud), one per frame.
    Never raises for poor registrations: if nothing passes the quality gates the original
    poses are returned unchanged.
    """
    index_of = {fid: k for k, fid in enumerate(frames)}

    edges: list[EdgeRecord] = []
    for k in range(len(frames) - 1):
        init = relative_transform(original_poses[k], original_poses[k + 1])
        edges.append(register_pair(frames[k], frames[k + 1], clouds[k], clouds[k + 1], init, "sequential", opts))

    candidates: list[tuple[int, int]] = []
    if opts.enable_loop_closure:
        candidates = find_loop_candidates(original_poses, opts)
        for i, j in candidates:
            init = relative_transform(original_poses[i], original_poses[j])
            edges.append(register_pair(frames[i], frames[j], clouds[i], clouds[j], init, "loop", opts))

    if opts.constrain_gravity:
        for e in edges:
            if e.accepted:
                i, j = index_of[e.source], index_of[e.target]
                constrained = constrain_relative_gravity(e.refined, original_poses[i], original_poses[j])
                max_corr = opts.seq_max_corr if e.kind == "sequential" else opts.loop_max_corr
                e.refined = refine_translation_only(clouds[i], clouds[j], constrained, max_corr)

    n_accepted = sum(e.accepted for e in edges)
    if n_accepted == 0:
        corrected = [p.copy() for p in original_poses]
        raw_tilt = 0.0
    else:
        optimised = optimise_pose_graph(original_poses, clouds, edges, index_of, opts)
        raw_tilt = max(tilt_change_deg(o[:3, :3], n[:3, :3]) for o, n in zip(original_poses, optimised))
        corrected = (
            [keep_original_tilt(o, n) for o, n in zip(original_poses, optimised)]
            if opts.constrain_gravity
            else optimised
        )

    result = CorrectionResult(frames, original_poses, corrected, edges, len(candidates), raw_tilt)
    summarise_correction(result)
    for kind in ("sequential", "loop"):
        accepted = [e for e in edges if e.kind == kind and e.accepted]
        if not accepted:
            continue
        before, after = [], []
        for e in accepted:
            i, j = index_of[e.source], index_of[e.target]
            before.append(transform_delta(np.linalg.inv(e.refined) @ e.initial))
            after.append(
                transform_delta(np.linalg.inv(e.refined) @ relative_transform(corrected[i], corrected[j]))
            )
        result.residual_before[kind] = tuple(float(x) for x in np.mean(before, axis=0))
        result.residual_after[kind] = tuple(float(x) for x in np.mean(after, axis=0))
    return result


def summarise_correction(r: CorrectionResult) -> None:
    trans = np.array([np.linalg.norm(c[:3, 3] - o[:3, 3]) for o, c in zip(r.original_poses, r.corrected_poses)])
    rots = np.array(
        [rotation_angle_deg(c[:3, :3] @ o[:3, :3].T) for o, c in zip(r.original_poses, r.corrected_poses)]
    )
    r.mean_translation_m, r.max_translation_m = float(trans.mean()), float(trans.max())
    r.mean_rotation_deg, r.max_rotation_deg = float(rots.mean()), float(rots.max())
    r.max_tilt_change_deg = max(
        tilt_change_deg(o[:3, :3], c[:3, :3]) for o, c in zip(r.original_poses, r.corrected_poses)
    )


# ---------- clouds and metrics ----------


def neighbour_overlap(world_clouds: list[np.ndarray], voxel: float = 0.05) -> float:
    """Mean voxel overlap between consecutive frames (the Ticket 2 diagnostic, higher is better)."""
    sets = [set(map(tuple, np.floor(c / voxel).astype(np.int64)[::3])) for c in world_clouds]
    scores = [len(a & b) / max(1, min(len(a), len(b))) for a, b in zip(sets, sets[1:])]
    return float(np.mean(scores)) if scores else 0.0


def neighbour_residual(world_clouds: list[np.ndarray], overlap_radius: float = 0.3) -> tuple[float, float]:
    """Median nearest-neighbour distance between consecutive frames, within the overlap region.

    Returns (mean over pairs of the median distance in metres, mean inlier fraction).
    Lower distance and higher fraction are better. Note ICP optimises something similar, so this
    metric favours the corrected result more than the structural ones below.
    """
    clouds = []
    for c in world_clouds:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(c.astype(np.float64))
        clouds.append(p)
    medians, fractions = [], []
    for a, b in zip(clouds, clouds[1:]):
        d = np.asarray(a.compute_point_cloud_distance(b))
        near = d[d < overlap_radius]
        if len(near):
            medians.append(float(np.median(near)))
            fractions.append(len(near) / len(d))
    if not medians:
        return 0.0, 0.0
    return float(np.mean(medians)), float(np.mean(fractions))


def floor_band_metrics(cloud: np.ndarray, band: float = 0.15, peak_halfwidth: float = 0.03) -> dict:
    """Thickness of the dominant horizontal surface (the floor), world +Y up.

    floor_y is the fullest 2 cm Y bin. Thickness is P90-P10 of Y among points within +-band
    of it. peak_share is the fraction of all points within +-peak_halfwidth (higher = sharper).
    """
    y = cloud[:, 1].astype(np.float64)
    edges = np.arange(y.min(), y.max() + 0.02, 0.02)
    hist, edges = np.histogram(y, bins=edges)
    floor_y = float(edges[int(np.argmax(hist))] + 0.01)
    near = y[np.abs(y - floor_y) <= band]
    p10, p90 = np.percentile(near, [10, 90])
    return {
        "floor_y_m": floor_y,
        "floor_band_thickness_m": float(p90 - p10),
        "floor_peak_share": float(np.mean(np.abs(y - floor_y) <= peak_halfwidth)),
    }


def wall_slab_metrics(cloud: np.ndarray, floor_y: float, cell: float = 0.05) -> dict:
    """Occupied top-down cells for points 0.4-1.6 m above the floor (mostly walls and furniture).

    Ghosted or doubled walls occupy more cells for the same number of points.
    """
    y = cloud[:, 1]
    slab = cloud[(y >= floor_y + 0.4) & (y <= floor_y + 1.6)]
    if len(slab) == 0:
        return {"wall_slab_points": 0, "wall_slab_cells": 0, "wall_slab_cells_per_1000_points": 0.0}
    cells = np.unique(np.floor(slab[:, [0, 2]] / cell).astype(np.int64), axis=0)
    return {
        "wall_slab_points": int(len(slab)),
        "wall_slab_cells": int(len(cells)),
        "wall_slab_cells_per_1000_points": float(1000 * len(cells) / len(slab)),
    }


def compute_metrics(
    metric_points: list[np.ndarray], poses: list[np.ndarray], global_cloud: np.ndarray
) -> dict:
    """All before/after metrics for one set of poses.

    `metric_points` are per-frame camera-space points, already voxel-downsampled to
    DriftOptions.metric_voxel; `global_cloud` is the final world cloud for the same poses.
    """
    per_frame = [
        geometry.transform_points(p, T[:3, :3], T[:3, 3]) for p, T in zip(metric_points, poses)
    ]
    resid, inlier_frac = neighbour_residual(per_frame)
    floor = floor_band_metrics(global_cloud)
    wall = wall_slab_metrics(global_cloud, floor["floor_y_m"])
    return {
        "overlap": neighbour_overlap(per_frame),
        "neighbour_residual_median_m": resid,
        "neighbour_inlier_fraction": inlier_frac,
        **floor,
        **wall,
        "footprint_x_m": float(np.ptp(global_cloud[:, 0])),
        "footprint_z_m": float(np.ptp(global_cloud[:, 2])),
        "points": int(len(global_cloud)),
    }


# metric name -> True if higher is better
METRIC_DIRECTION = {
    "overlap": True,
    "neighbour_residual_median_m": False,
    "floor_band_thickness_m": False,
    "floor_peak_share": True,
    "wall_slab_cells_per_1000_points": False,
}


def judge_correction(
    before: dict, after: dict, correction: CorrectionResult, rules: AcceptanceRules
) -> tuple[bool, str | None, dict]:
    """Decide whether to use the corrected poses. Returns (accepted, fallback_reason, relative_changes)."""
    changes = {}
    for name, higher_better in METRIC_DIRECTION.items():
        b, a = before[name], after[name]
        rel = (a - b) / b if b else 0.0
        changes[name] = rel if higher_better else -rel  # positive = improvement

    attempted_seq = [e for e in correction.edges if e.kind == "sequential"]
    accepted_seq = [e for e in attempted_seq if e.accepted]
    if attempted_seq and len(accepted_seq) / len(attempted_seq) < rules.min_accepted_sequential_fraction:
        return False, "too few sequential registrations passed the quality gates", changes
    worse = [n for n, c in changes.items() if c < -rules.max_relative_degradation]
    if worse:
        return False, "metrics got worse: " + ", ".join(worse), changes
    if not any(c >= rules.min_relative_improvement for c in changes.values()):
        return False, "no metric improved meaningfully", changes
    for axis in ("footprint_x_m", "footprint_z_m"):
        if abs(after[axis] - before[axis]) / before[axis] > rules.max_footprint_change:
            return False, f"{axis} changed by more than {rules.max_footprint_change:.0%}", changes
    if correction.max_translation_m > rules.max_translation_m:
        return False, f"max translation correction {correction.max_translation_m:.2f} m is too large", changes
    if correction.max_rotation_deg > rules.max_rotation_deg:
        return False, f"max rotation correction {correction.max_rotation_deg:.1f} deg is too large", changes
    if correction.max_tilt_change_deg > rules.max_tilt_change_deg:
        return False, f"roll/pitch changed by {correction.max_tilt_change_deg:.1f} deg", changes
    return True, None, changes
