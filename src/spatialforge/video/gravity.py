"""Gravity (world-up) estimation for a video scene, from geometry. No IMU, no ARKit metadata, no assumption that a
COLMAP axis or the image "up" is vertical (phone video is often stored rotated by 90 degrees).

Evidence, in order of weight:
  1. Floors and ceilings are large horizontal planes: along the true vertical, point heights pile up in a few thin
     slabs. Candidate axes are scored by how concentrated the heights are (top slabs' share of all points).
  2. The camera path of a walkthrough is roughly planar and horizontal, so the axis of least camera-position variance
     restricts the search (a cone around it); a non-planar path searches the whole sphere with lower confidence.
  3. Sign: the floor must lie below the cameras at a plausible handheld distance and a ceiling, if seen, above them.
     When this is ambiguous, the image-up hint of the cameras (if given) breaks the tie and confidence drops.
The metric scale must already be applied (the sign test uses metres).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GravityOptions:
    max_points: int = 150_000
    bin_m: float = 0.05
    slabs: int = 3  # strongest height slabs counted in the concentration score
    cone_deg: float = 35.0
    cone_step_deg: float = 2.0
    refine_step_deg: float = 0.5
    sphere_directions: int = 2500
    planar_ratio: float = 0.35  # smallest/middle camera-covariance eigenvalue below this = planar path
    floor_to_camera_m: tuple[float, float] = (0.6, 2.3)  # plausible handheld camera height above the floor
    wall_refine_points: int = 40_000
    wall_normal_radius_m: float = 0.30
    wall_refine_cone_deg: float = 6.0
    wall_refine_min_normals: int = 300
    wall_sigma: float = 0.04  # |n . up| of a truly vertical wall is 0; this is the sharpness of the alignment score


@dataclass
class GravityEstimate:
    up: np.ndarray  # unit vector pointing up, in the input frame
    confidence: float  # heuristic 0-1, not a probability
    quality: str  # strong | moderate | weak
    method: str
    concentration: float  # share of points in the strongest height slabs along `up`
    baseline_concentration: float  # median share over candidate axes (how special `up` is)
    path_planarity: float | None
    prior_angle_deg: float | None  # angle between the refined axis and the trajectory-derived prior
    sign_basis: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"up": [round(float(v), 5) for v in self.up], "confidence": round(self.confidence, 3), "quality": self.quality,
                "method": self.method, "concentration": round(self.concentration, 4),
                "baseline_concentration": round(self.baseline_concentration, 4),
                "path_planarity": None if self.path_planarity is None else round(self.path_planarity, 4),
                "prior_angle_deg": None if self.prior_angle_deg is None else round(self.prior_angle_deg, 2),
                "sign_basis": self.sign_basis, "notes": self.notes}


def rotation_to_y_up(up: np.ndarray) -> np.ndarray:
    """Rotation R (3x3) with R @ up = +Y (shortest rotation; yaw about the vertical is left as it is)."""
    u = np.asarray(up, dtype=np.float64)
    u = u / np.linalg.norm(u)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, y)
    c = float(u @ y)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)


def _fibonacci_hemisphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = i / n  # 0..1 : upper hemisphere (axes are sign-free)
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _cone_directions(axis: np.ndarray, half_deg: float, step_deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    t = np.deg2rad(np.arange(-half_deg, half_deg + 1e-9, step_deg))
    out = []
    for x in t:
        for y in t:
            if np.hypot(x, y) <= np.deg2rad(half_deg) + 1e-12:
                d = a + np.tan(x) * e1 + np.tan(y) * e2
                out.append(d / np.linalg.norm(d))
    return np.array(out)


def concentration(points: np.ndarray, u: np.ndarray, bin_m: float, slabs: int) -> float:
    """Share of points inside the `slabs` strongest, mutually separated height slabs along axis u."""
    h = points @ u
    n_bins = max(1, int(np.ceil((h.max() - h.min()) / bin_m)) + 1)
    hist = np.bincount(np.floor((h - h.min()) / bin_m).astype(np.int64), minlength=n_bins).astype(np.float64)
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)  # a slab straddling two bins still counts
    total, taken = 0.0, smooth.copy()
    sep = max(2, int(round(0.4 / bin_m)))
    for _ in range(slabs):
        j = int(np.argmax(taken))
        if taken[j] <= 0:
            break
        total += smooth[j]
        taken[max(0, j - sep):j + sep + 1] = 0
    return float(min(1.0, total / len(h)))


def _subsample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[np.linspace(0, len(points) - 1, n).astype(np.int64)]  # deterministic stride


def _slab_peaks(h: np.ndarray, bin_m: float, slabs: int) -> list[tuple[float, float]]:
    """(height, share) of the strongest separated slabs of heights h."""
    lo = float(h.min())
    n_bins = max(1, int(np.ceil((h.max() - lo) / bin_m)) + 1)
    hist = np.bincount(np.floor((h - lo) / bin_m).astype(np.int64), minlength=n_bins).astype(np.float64)
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    taken = smooth.copy()
    sep = max(2, int(round(0.4 / bin_m)))
    peaks = []
    for _ in range(slabs):
        j = int(np.argmax(taken))
        if taken[j] <= 0:
            break
        # centroid of heights inside the slab
        sel = (h >= lo + (j - 1) * bin_m) & (h < lo + (j + 2) * bin_m)
        peaks.append((float(h[sel].mean()) if sel.any() else lo + (j + 0.5) * bin_m, float(smooth[j] / len(h))))
        taken[max(0, j - sep):j + sep + 1] = 0
    return peaks


def surface_normals(points: np.ndarray, radius: float, max_points: int) -> np.ndarray:
    """Unsigned unit normals of a deterministic subsample (Open3D, CPU)."""
    import open3d as o3d

    pts = _subsample(np.asarray(points, dtype=np.float64), max_points)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    n = np.asarray(pcd.normals)
    return n[np.isfinite(n).all(axis=1)]


def wall_alignment(normals: np.ndarray, u: np.ndarray, sigma: float) -> float:
    """Mean sharpness of |n . u| around zero: vertical walls have normals perpendicular to the vertical axis."""
    return float(np.mean(np.exp(-0.5 * ((normals @ u) / sigma) ** 2)))


def estimate_gravity(points: np.ndarray, camera_positions: np.ndarray, camera_up_hints: np.ndarray | None = None,
                     opts: GravityOptions | None = None) -> GravityEstimate:
    """Estimate world-up for a metric point cloud and its camera path. `camera_up_hints`: (M,3) image-up directions."""
    opts = opts or GravityOptions()
    pts = _subsample(np.asarray(points, dtype=np.float64), opts.max_points)
    centre = pts.mean(axis=0)
    pts = pts - centre
    cams = np.asarray(camera_positions, dtype=np.float64) - centre
    if len(pts) < 500:
        raise ValueError("too few points to estimate gravity")
    notes: list[str] = []

    prior, planarity = None, None
    if len(cams) >= 6:
        w, v = np.linalg.eigh(np.cov((cams - cams.mean(axis=0)).T))  # ascending
        planarity = float(w[0] / w[1]) if w[1] > 1e-12 else 0.0
        if planarity <= opts.planar_ratio:
            prior = v[:, 0]
    if prior is not None:
        dirs = _cone_directions(prior, opts.cone_deg, opts.cone_step_deg)
        method = "height-slab concentration within a cone around the least-variance axis of the camera path"
    else:
        dirs = _fibonacci_hemisphere(opts.sphere_directions)
        method = "height-slab concentration over the whole sphere (camera path is not planar: lower confidence)"
        notes.append("camera path is not planar; vertical axis chosen from point-cloud structure alone")

    scores = np.array([concentration(pts, d, opts.bin_m, opts.slabs) for d in dirs])
    best = dirs[int(np.argmax(scores))]
    fine = _cone_directions(best, opts.cone_step_deg * 1.5, opts.refine_step_deg)
    fine_scores = np.array([concentration(pts, d, opts.bin_m, opts.slabs) for d in fine])
    axis = fine[int(np.argmax(fine_scores))]
    axis_slab, refined = axis, False
    # Walls dominate most walkthroughs and show little floor or ceiling, which under-constrains the slab score.
    # Refine with wall verticality: horizontal-ish normals must be perpendicular to up (combined with the slab score).
    try:
        normals = surface_normals(pts, opts.wall_normal_radius_m, opts.wall_refine_points)
        wall_like = normals[np.abs(normals @ axis) < 0.5]
        if len(wall_like) >= opts.wall_refine_min_normals:
            cone = _cone_directions(axis, opts.wall_refine_cone_deg, opts.refine_step_deg)
            wall = np.array([wall_alignment(wall_like, d, opts.wall_sigma) for d in cone])
            slab = np.array([concentration(pts, d, opts.bin_m, opts.slabs) for d in cone])
            combined = 0.5 * wall / max(wall.max(), 1e-12) + 0.5 * slab / max(slab.max(), 1e-12)
            axis = cone[int(np.argmax(combined))]
            refined = True
            method += " + wall-verticality refinement"
        else:
            notes.append("too few wall-like surface normals for the wall-verticality refinement")
    except Exception as exc:  # the refinement is an improvement, not a requirement
        notes.append(f"wall-verticality refinement skipped ({type(exc).__name__})")
    conc = concentration(pts, axis, opts.bin_m, opts.slabs)
    baseline = float(np.median(scores))

    # sign, in priority order of reliability:
    #   A. exactly one orientation puts a floor slab below the cameras at a plausible handheld distance (decisive);
    #   B. the cameras' image-up directions agree strongly with the vertical axis (a 90-degree rotated video gives none);
    #   C. weak scene cues: clutter piles up near the floor, and the camera sits near a typical handheld height.
    def describe(s: float) -> dict:
        u = axis * s
        h = pts @ u
        peaks = _slab_peaks(h, opts.bin_m, opts.slabs)
        cam_h = float(np.median(cams @ u)) if len(cams) else 0.0
        strong = [p for p in peaks if p[1] >= 0.5 * peaks[0][1]]
        below = [p[0] for p in strong if p[0] < cam_h]
        above = [p[0] for p in strong if p[0] > cam_h]
        floor_h = min(below) if below else None
        ceil_h = max(above) if above else None
        d_floor = None if floor_h is None else cam_h - floor_h
        clutter = 0.0
        if floor_h is not None and ceil_h is not None and ceil_h - floor_h >= 1.5:
            near_floor = float(((h > floor_h + 0.15) & (h < floor_h + 1.2)).mean())
            near_ceil = float(((h < ceil_h - 0.15) & (h > ceil_h - 1.2)).mean())
            clutter = near_floor - near_ceil
        plausible = d_floor is not None and opts.floor_to_camera_m[0] <= d_floor <= opts.floor_to_camera_m[1]
        height_prior = -abs(d_floor - 1.45) if d_floor is not None else -1.0
        return {"plausible": plausible, "clutter": clutter, "height_prior": height_prior}

    pos, neg = describe(1.0), describe(-1.0)
    hint_cos = None
    if camera_up_hints is not None and len(camera_up_hints):
        hints = np.asarray(camera_up_hints, dtype=np.float64)
        hints = hints / np.maximum(np.linalg.norm(hints, axis=1, keepdims=True), 1e-12)
        hint_cos = float(np.mean(hints @ axis))
    if pos["plausible"] != neg["plausible"]:
        sign = 1.0 if pos["plausible"] else -1.0
        sign_basis, sign_conf = "floor below the cameras at a plausible distance", 1.0
    elif hint_cos is not None and abs(hint_cos) >= 0.6:
        sign = 1.0 if hint_cos > 0 else -1.0
        sign_basis, sign_conf = "camera image-up direction agrees with the vertical axis", 0.8
    else:
        weak = 4.0 * (pos["clutter"] - neg["clutter"]) + (pos["height_prior"] - neg["height_prior"])
        if abs(weak) >= 0.1:
            sign = 1.0 if weak > 0 else -1.0
            sign_basis, sign_conf = "weak scene cues (clutter near the floor, handheld camera height)", 0.5
            notes.append("floor/ceiling orientation rests on weak cues: both orientations looked plausible")
        else:
            sign, sign_basis, sign_conf = 1.0, "undetermined (defaulted)", 0.3
            notes.append("up direction sign could not be determined from geometry; the scene may be upside down")
    up = axis * sign

    prior_angle = None
    if prior is not None:
        prior_angle = float(np.degrees(np.arccos(np.clip(abs(prior @ axis), -1, 1))))
    lift = (conc - baseline) / max(1e-9, 1.0 - baseline)
    conf = float(np.clip(0.15 + 1.6 * lift, 0, 1))  # evidence from floor/ceiling slabs
    if refined:
        # independent cues (slab axis, wall-normal axis, camera-path normal) agreeing is evidence even when little floor
        # or ceiling is visible; disagreement beyond a few degrees is not
        disagree = float(np.degrees(np.arccos(np.clip(abs(axis_slab @ axis), -1, 1))))
        if prior_angle is not None:
            disagree = max(disagree, prior_angle)
        conf = max(conf, float(np.clip(1.0 - (disagree - 2.0) / 10.0, 0, 1)) * 0.75)
        notes.append(f"slab, wall-normal and camera-path axes agree to within {disagree:.1f} degrees")
    conf *= 0.5 + 0.5 * sign_conf
    if planarity is None:
        conf *= 0.6
        notes.append("too few cameras to assess the path")
    elif prior is None:
        conf *= 0.7
    elif prior_angle is not None and prior_angle > 20:
        conf *= 0.7
        notes.append(f"vertical axis differs from the camera-path normal by {prior_angle:.0f} degrees")
    quality = "strong" if conf >= 0.6 else "moderate" if conf >= 0.35 else "weak"
    return GravityEstimate(up, conf, quality, method, conc, baseline, planarity, prior_angle, sign_basis, notes)
