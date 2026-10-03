"""Cross-room stitching: separate room reconstructions -> ONE placed property.

Planar convention (shared with the rest of SpatialForge): points are (x, z) in metres, a rotation by theta is
    [x', z'] = [[cos t, -sin t], [sin t, cos t]] [x, z]
and a pose (theta, tx, tz) maps a room's LOCAL frame to the global frame: x_global = R(theta) x_local + t.
A constraint T_AB (theta, tx, tz) maps room B's local frame into room A's local frame.

Evidence, strongest first:
  visual  - images of different room folders share geometrically verified feature matches. The images of both rooms are
            mapped TOGETHER (one joint SfM model); aligning each room's metric local poses to the joint poses gives the
            relative rotation and translation (a full 6-DoF fit reduced to yaw + x/z because both rooms are gravity-aligned).
  doorway - an opening of room A and an opening of room B describe the same doorway: same width, parallel walls, and
            coincident after the transform. Used to confirm a visual transform, or (weakly) on its own: candidate
            transforms that align the two doorways are generated, scored (width, no room overlap) and accepted only if
            exactly one is clearly best.
Nothing is placed by guesswork: rooms without enough evidence stay unplaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

QUALITY_RANK = {"weak": 1, "moderate": 2, "strong": 3}
RANK_QUALITY = {1: "weak", 2: "moderate", 3: "strong"}


@dataclass
class StitchOptions:
    # cross-room match evidence (geometrically VERIFIED matches only)
    min_pair_verified: int = 25  # at least one image pair with this many verified inliers ...
    min_pair_inlier_ratio: float = 0.25  # ... and this inlier ratio
    min_total_verified: int = 50  # and this many over all image pairs of the two rooms
    # visual transform quality
    strong_total_verified: int = 120
    strong_inlier_ratio: float = 0.40
    max_tilt_deg: float = 6.0  # residual roll/pitch between two gravity-aligned rooms: beyond this the transform is rejected
    max_scale_ratio: float = 1.6  # joint-model scale seen from the two rooms: beyond this the transform is rejected
    ok_scale_ratio: float = 1.25
    min_baseline_m: float = 0.05
    # doorway matching
    door_distance_m: float = 0.6
    door_angle_deg: float = 15.0
    door_width_ratio: float = 1.35
    hypothesis_margin: float = 0.15  # doorway-only: the best hypothesis must beat the next by this score margin ...
    hypothesis_distinct_m: float = 0.5  # ... unless the runners-up are the same transform (within this distance)
    # solver
    sigma_theta_floor_deg: float = 1.0
    sigma_t_floor_m: float = 0.05
    huber_k: float = 2.5
    max_normalized_residual: float = 3.0
    # overlap
    overlap_max_fraction: float = 0.10  # of the smaller room
    overlap_min_area_m2: float = 0.4
    raster_cell_m: float = 0.05


# ---------------- planar transforms ----------------


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def apply(pose, pts: np.ndarray) -> np.ndarray:
    th, tx, tz = pose
    return np.asarray(pts, dtype=np.float64) @ rot(th).T + np.array([tx, tz])


def compose(p, q):
    """p ∘ q: apply q first, then p."""
    R = rot(p[0])
    t = np.array([p[1], p[2]]) + R @ np.array([q[1], q[2]])
    return (wrap(p[0] + q[0]), float(t[0]), float(t[1]))


def invert(p):
    th = -p[0]
    t = -rot(th) @ np.array([p[1], p[2]])
    return (wrap(th), float(t[0]), float(t[1]))


# ---------------- what stitching needs to know about a room ----------------


@dataclass
class OpeningGeom:
    id: str
    wall_id: str
    left: np.ndarray
    right: np.ndarray
    width: float
    quality: str = "moderate"
    width_low: float | None = None
    width_high: float | None = None

    @property
    def mid(self) -> np.ndarray:
        return (self.left + self.right) / 2

    @property
    def direction(self) -> np.ndarray:
        d = self.right - self.left
        return d / max(np.linalg.norm(d), 1e-9)


@dataclass
class RoomGeom:
    id: str
    polygon: np.ndarray | None  # (n, 2) local frame, or None when no closed room was found
    openings: list[OpeningGeom] = field(default_factory=list)
    rank: tuple = (0,)  # anchor preference: larger is stronger
    wall_points: np.ndarray | None = None  # (m, 2) wall endpoints, used for the footprint of polygon-less rooms


# ---------------- cross-room image evidence ----------------


@dataclass
class PairEvidence:
    room_a: str
    room_b: str
    image_pairs: list[dict]
    verified_total: int
    best_pair_verified: int
    best_inlier_ratio: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return {"room_a": self.room_a, "room_b": self.room_b, "verified_total": self.verified_total,
                "best_pair_verified": self.best_pair_verified, "best_inlier_ratio": round(self.best_inlier_ratio, 3),
                "accepted": self.accepted, "reason": self.reason, "image_pairs": self.image_pairs}


def cross_room_evidence(pair_stats: list[dict], room_of: dict[str, str], opts: StitchOptions) -> list[PairEvidence]:
    """Group the match table by room pair. Only geometrically verified matches count towards acceptance."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in pair_stats:
        ra, rb = room_of.get(p["a"]), room_of.get(p["b"])
        if ra is None or rb is None or ra == rb:
            continue
        key = (ra, rb) if ra < rb else (rb, ra)
        a, b = (p["a"], p["b"]) if ra < rb else (p["b"], p["a"])
        raw, ver = int(p["raw"]), int(p["verified"])
        groups.setdefault(key, []).append({"a": a, "b": b, "raw_matches": raw, "verified_matches": ver,
                                           "inlier_ratio": round(ver / raw, 4) if raw else 0.0})
    rooms = sorted(set(room_of.values()))
    out = []
    for i, ra in enumerate(rooms):
        for rb in rooms[i + 1:]:
            pairs = sorted(groups.get((ra, rb), []), key=lambda d: (-d["verified_matches"], d["a"], d["b"]))
            total = sum(d["verified_matches"] for d in pairs)
            best = max((d["verified_matches"] for d in pairs), default=0)
            strong_pairs = [d for d in pairs if d["verified_matches"] >= opts.min_pair_verified and d["inlier_ratio"] >= opts.min_pair_inlier_ratio]
            ratio = max((d["inlier_ratio"] for d in strong_pairs), default=max((d["inlier_ratio"] for d in pairs), default=0.0))
            if not pairs:
                ok, why = False, "no image pair of these rooms shares any feature match"
            elif not strong_pairs:
                ok, why = False, (f"no image pair has >= {opts.min_pair_verified} verified matches with inlier ratio "
                                  f">= {opts.min_pair_inlier_ratio} (best: {best})")
            elif total < opts.min_total_verified:
                ok, why = False, f"only {total} verified matches over all image pairs (need {opts.min_total_verified})"
            else:
                ok, why = True, f"{len(strong_pairs)} image pair(s) with verified matches, {total} in total"
            out.append(PairEvidence(ra, rb, pairs, total, best, float(ratio), ok, why))
    return out


# ---------------- visual relative transform ----------------


@dataclass
class PairTransform:
    theta_rad: float
    tx: float
    tz: float
    sigma_theta_deg: float
    sigma_t_m: float
    tilt_deg: float
    scale_ratio: float | None  # joint-model scale seen from room A / from room B (1.0 = consistent), None if unknown
    images_a: int
    images_b: int
    fit_rms_m: float
    notes: list[str] = field(default_factory=list)
    log_scale_a_over_b: float | None = None  # signed ln(s_A / s_B): the two rooms' metric scales as seen in the joint model

    @property
    def pose(self):
        return (self.theta_rad, self.tx, self.tz)

    def to_dict(self) -> dict:
        return {"rotation_deg": round(math.degrees(self.theta_rad), 3), "translation_x_m": round(self.tx, 4),
                "translation_z_m": round(self.tz, 4), "rotation_uncertainty_deg": round(self.sigma_theta_deg, 3),
                "translation_uncertainty_m": round(self.sigma_t_m, 4), "tilt_deg": round(self.tilt_deg, 3),
                "joint_scale_ratio": None if self.scale_ratio is None else round(self.scale_ratio, 4),
                "images_a": self.images_a, "images_b": self.images_b, "fit_rms_m": round(self.fit_rms_m, 4), "notes": self.notes,
                "log_scale_a_over_b": None if self.log_scale_a_over_b is None else round(self.log_scale_a_over_b, 5)}


def _chordal_mean(rots: list[np.ndarray]) -> tuple[np.ndarray, float]:
    M = sum(rots)
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    spread = float(np.mean([math.degrees(math.acos(np.clip((np.trace(R.T @ r) - 1) / 2, -1, 1))) for r in rots]))
    return R, spread


def _centres(Ts: list[np.ndarray]) -> np.ndarray:
    return np.array([T[:3, 3] for T in Ts])


def visual_pair_transform(poses_a: dict, poses_b: dict, joint: dict, opts: StitchOptions) -> tuple[PairTransform | None, str]:
    """Relative planar transform of room B in room A's frame from a joint SfM model of both rooms.

    poses_x: image name -> 4x4 camera-to-world in that room's LOCAL metric frame (+Y up).
    joint:   image name -> object with rotation_cw / translation_cw (the joint model, arbitrary scale and orientation).
    Returns (transform, "") or (None, reason).
    """
    def prep(poses):
        names = sorted(n for n in poses if n in joint)
        loc = [poses[n] for n in names]
        Jr = [joint[n].rotation_cw.T for n in names]  # camera-to-world rotation in the joint frame
        Jc = [-joint[n].rotation_cw.T @ joint[n].translation_cw for n in names]
        return names, loc, Jr, np.array(Jc).reshape(-1, 3)

    na, la, Ja, Ca = prep(poses_a)
    nb, lb, Jb, Cb = prep(poses_b)
    if not na or not nb:
        return None, f"the joint model registers images of only one room (room A: {len(na)}, room B: {len(nb)})"

    def room_frame(loc, Jr, Jc):
        R, spread = _chordal_mean([j @ l[:3, :3].T for j, l in zip(Jr, loc)])
        ratios = []
        Lc = _centres(loc)
        for i in range(len(loc)):
            for j in range(i + 1, len(loc)):
                dl = np.linalg.norm(Lc[i] - Lc[j])
                if dl >= opts.min_baseline_m:
                    ratios.append(np.linalg.norm(Jc[i] - Jc[j]) / dl)
        return R, spread, ratios, Lc

    Ra, spread_a, ratios_a, La = room_frame(la, Ja, Ca)
    Rb, spread_b, ratios_b, Lb = room_frame(lb, Jb, Cb)
    allr = ratios_a + ratios_b
    if not allr:
        return None, "cannot determine the joint-model scale: no room has two registered images with a usable baseline"
    s = float(np.median(allr))
    sa = float(np.median(ratios_a)) if ratios_a else None
    sb = float(np.median(ratios_b)) if ratios_b else None
    scale_ratio = None if sa is None or sb is None else float(max(sa, sb) / min(sa, sb))
    notes = []
    if scale_ratio is not None and scale_ratio > opts.max_scale_ratio:
        return None, f"the two rooms' metric scales disagree by {scale_ratio:.2f}x in the joint model"
    if scale_ratio is None:
        notes.append("joint-model scale taken from one room only (the other has a single registered image)")
    ta = (Ca - s * (La @ Ra.T)).mean(axis=0)
    tb = (Cb - s * (Lb @ Rb.T)).mean(axis=0)
    rms = float(np.sqrt(np.mean(np.concatenate([np.sum((Ca - (s * (La @ Ra.T) + ta)) ** 2, axis=1),
                                                 np.sum((Cb - (s * (Lb @ Rb.T) + tb)) ** 2, axis=1)])))) / s
    R_ab = Ra.T @ Rb
    t_ab = Ra.T @ (tb - ta) / s
    tilt = math.degrees(math.acos(float(np.clip(R_ab[1, 1], -1, 1))))
    if tilt > opts.max_tilt_deg:
        return None, f"the relative rotation is tilted {tilt:.1f} deg out of the horizontal plane (rooms are gravity-aligned)"
    theta = -math.atan2(R_ab[0, 2], R_ab[0, 0])  # rotation about +Y, expressed in the (x, z) plane convention above
    sigma_theta = float(np.hypot(np.hypot(spread_a, spread_b), np.hypot(tilt / 2, opts.sigma_theta_floor_deg * 0.5)))
    dist = float(np.hypot(t_ab[0], t_ab[2]))
    mismatch = 0.0 if scale_ratio is None else math.log(scale_ratio) / 2
    sigma_t = float(np.sqrt(rms ** 2 + (0.03 * dist) ** 2 + (mismatch * dist) ** 2 + 0.05 ** 2))
    log_ab = None if sa is None or sb is None else float(math.log(sa / sb))
    return PairTransform(theta, float(t_ab[0]), float(t_ab[2]), sigma_theta, sigma_t, tilt, scale_ratio, len(na), len(nb), rms,
                         notes, log_ab), ""


# ---------------- doorways ----------------


@dataclass
class DoorMatch:
    opening_a: str
    opening_b: str
    distance_m: float
    width_ratio: float
    angle_deg: float

    def to_dict(self) -> dict:
        return {"opening_a": self.opening_a, "opening_b": self.opening_b, "distance_m": round(self.distance_m, 3),
                "width_ratio": round(self.width_ratio, 3), "angle_deg": round(self.angle_deg, 2)}


def _line_angle_deg(d: np.ndarray) -> float:
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def match_doorways(room_a: RoomGeom, room_b: RoomGeom, pose_b_in_a, opts: StitchOptions) -> list[DoorMatch]:
    """Openings of A and B that describe the same doorway once B is placed in A's frame (greedy, one-to-one)."""
    cands = []
    for oa in room_a.openings:
        for ob in room_b.openings:
            mb = apply(pose_b_in_a, ob.mid[None])[0]
            db = rot(pose_b_in_a[0]) @ ob.direction
            dist = float(np.linalg.norm(mb - oa.mid))
            ang = _angle_diff(_line_angle_deg(oa.direction), _line_angle_deg(db))
            ratio = max(oa.width, ob.width) / max(min(oa.width, ob.width), 1e-9)
            if dist <= opts.door_distance_m and ang <= opts.door_angle_deg and ratio <= opts.door_width_ratio:
                cands.append((dist, oa.id, ob.id, DoorMatch(oa.id, ob.id, dist, ratio, ang)))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    used_a, used_b, out = set(), set(), []
    for _, ia, ib, m in cands:
        if ia not in used_a and ib not in used_b:
            used_a.add(ia)
            used_b.add(ib)
            out.append(m)
    return out


@dataclass
class Hypothesis:
    pose: tuple
    opening_a: str
    opening_b: str
    width_ratio: float
    overlap_fraction: float
    score: float


def raster_polygon(poly: np.ndarray, bounds, cell: float) -> np.ndarray:
    """Boolean occupancy of a polygon on a grid (even-odd test at cell centres); deterministic."""
    x0, z0, x1, z1 = bounds
    xs = x0 + (np.arange(int(math.ceil((x1 - x0) / cell))) + 0.5) * cell
    zs = z0 + (np.arange(int(math.ceil((z1 - z0) / cell))) + 0.5) * cell
    gx, gz = np.meshgrid(xs, zs)
    px, pz = gx.ravel(), gz.ravel()
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1_, z1_ = poly[i]
        x2_, z2_ = poly[(i + 1) % n]
        crosses = (z1_ > pz) != (z2_ > pz)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1_ + (pz - z1_) * (x2_ - x1_) / (z2_ - z1_)
        inside ^= crosses & (px < x_at)
    return inside.reshape(gx.shape)


def overlap_of(poly_a: np.ndarray, poly_b: np.ndarray, cell: float) -> tuple[float, float, float]:
    """(intersection area, area of the smaller polygon, fraction of the smaller one that is overlapped)."""
    both = np.vstack([poly_a, poly_b])
    pad = 0.1
    bounds = (both[:, 0].min() - pad, both[:, 1].min() - pad, both[:, 0].max() + pad, both[:, 1].max() + pad)
    a, b = raster_polygon(poly_a, bounds, cell), raster_polygon(poly_b, bounds, cell)
    inter = float((a & b).sum()) * cell * cell
    small = float(min(a.sum(), b.sum())) * cell * cell
    return inter, small, (inter / small if small > 0 else 0.0)


def doorway_hypotheses(room_a: RoomGeom, room_b: RoomGeom, opts: StitchOptions) -> list[Hypothesis]:
    """All transforms of B into A that make one opening of each coincide (parallel walls, B on the far side)."""
    if room_a.polygon is None or room_b.polygon is None:
        return []
    hyps = []
    for oa in room_a.openings:
        for ob in room_b.openings:
            ratio = max(oa.width, ob.width) / max(min(oa.width, ob.width), 1e-9)
            if ratio > opts.door_width_ratio:
                continue
            da, db = oa.direction, ob.direction
            base = math.atan2(da[1], da[0]) - math.atan2(db[1], db[0])
            for theta in (wrap(base), wrap(base + math.pi)):
                t = oa.mid - rot(theta) @ ob.mid
                pose = (theta, float(t[0]), float(t[1]))
                _, _, frac = overlap_of(room_a.polygon, apply(pose, room_b.polygon), opts.raster_cell_m)
                if frac > opts.overlap_max_fraction:
                    continue
                hyps.append(Hypothesis(pose, oa.id, ob.id, ratio, frac, abs(math.log(ratio)) + 3.0 * frac))
    hyps.sort(key=lambda h: (h.score, h.opening_a, h.opening_b, h.pose[0]))
    return hyps


def pick_unique_hypothesis(hyps: list[Hypothesis], opts: StitchOptions) -> tuple[Hypothesis | None, str]:
    if not hyps:
        return None, "no doorway pair yields a physically consistent placement"
    best = hyps[0]
    for other in hyps[1:]:
        if other.score - best.score >= opts.hypothesis_margin:
            break
        same = (np.hypot(other.pose[1] - best.pose[1], other.pose[2] - best.pose[2]) < opts.hypothesis_distinct_m
                and abs(wrap(other.pose[0] - best.pose[0])) < math.radians(10))
        if not same:
            return None, (f"ambiguous: {sum(1 for h in hyps if h.score - best.score < opts.hypothesis_margin)} different "
                          "doorway placements score about equally")
    return best, ""


# ---------------- constraints ----------------


@dataclass
class RoomStitchConstraint:
    source_room_id: str
    target_room_id: str
    rotation_deg: float
    translation_x_m: float
    translation_z_m: float
    evidence_type: str  # visual | doorway | visual+doorway
    verified_match_count: int
    inlier_ratio: float
    source_opening_id: str | None
    target_opening_id: str | None
    translation_uncertainty_m: float
    rotation_uncertainty_deg: float
    quality: str
    status: str = "active"  # active | rejected
    notes: list[str] = field(default_factory=list)
    alternatives: list = field(default_factory=list, repr=False)  # other doorway hypotheses, for overlap resolution
    residual_sigma: float | None = None

    @property
    def pose(self):
        return (math.radians(self.rotation_deg), self.translation_x_m, self.translation_z_m)

    def to_dict(self) -> dict:
        return {"source_room_id": self.source_room_id, "target_room_id": self.target_room_id,
                "rotation_deg": round(self.rotation_deg, 3), "translation_x_m": round(self.translation_x_m, 4),
                "translation_z_m": round(self.translation_z_m, 4), "evidence_type": self.evidence_type,
                "verified_match_count": self.verified_match_count, "inlier_ratio": round(self.inlier_ratio, 3),
                "source_opening_id": self.source_opening_id, "target_opening_id": self.target_opening_id,
                "translation_uncertainty_m": round(self.translation_uncertainty_m, 4),
                "rotation_uncertainty_deg": round(self.rotation_uncertainty_deg, 3), "quality": self.quality,
                "status": self.status, "notes": self.notes,
                "normalized_residual": None if self.residual_sigma is None else round(self.residual_sigma, 3)}


def visual_quality(ev: PairEvidence, tf: PairTransform, opts: StitchOptions) -> str:
    q = 3
    if ev.verified_total < opts.strong_total_verified or ev.best_inlier_ratio < opts.strong_inlier_ratio:
        q = 2
    if tf.tilt_deg > opts.max_tilt_deg / 2 or (tf.scale_ratio or 1.0) > opts.ok_scale_ratio or tf.fit_rms_m > 0.15:
        q = min(q, 2)
    if tf.images_a < 2 or tf.images_b < 2 or tf.fit_rms_m > 0.35 or tf.sigma_t_m > 0.4:
        q = 1
    return RANK_QUALITY[q]


def build_visual_constraint(ev: PairEvidence, tf: PairTransform, room_a: RoomGeom, room_b: RoomGeom,
                            opts: StitchOptions) -> RoomStitchConstraint:
    quality = visual_quality(ev, tf, opts)
    doors = match_doorways(room_a, room_b, tf.pose, opts)
    kind, oa, ob, notes = "visual", None, None, list(tf.notes)
    if doors:
        kind, oa, ob = "visual+doorway", doors[0].opening_a, doors[0].opening_b
        quality = RANK_QUALITY[min(3, QUALITY_RANK[quality] + 1)]
        notes.append(f"doorway {oa} <-> {ob} coincides within {doors[0].distance_m:.2f} m after placement")
    return RoomStitchConstraint(ev.room_a, ev.room_b, math.degrees(tf.theta_rad), tf.tx, tf.tz, kind, ev.verified_total,
                                ev.best_inlier_ratio, oa, ob, tf.sigma_t_m, tf.sigma_theta_deg, quality, notes=notes)


def build_doorway_constraint(room_a: RoomGeom, room_b: RoomGeom, opts: StitchOptions) -> tuple[RoomStitchConstraint | None, str]:
    hyps = doorway_hypotheses(room_a, room_b, opts)
    best, why = pick_unique_hypothesis(hyps, opts)
    if best is None:
        return None, why
    st = 0.25 + 0.15 * abs(math.log(best.width_ratio)) * 5  # a doorway fixes the placement along the wall only loosely
    c = RoomStitchConstraint(room_a.id, room_b.id, math.degrees(best.pose[0]), best.pose[1], best.pose[2], "doorway", 0, 0.0,
                             best.opening_a, best.opening_b, st, 3.0, "weak",
                             notes=["no cross-room image evidence: placement from a unique doorway match only"])
    c.alternatives = [h for h in hyps if h is not best]
    return c, ""


# ---------------- global placement ----------------


def _residuals(poses: dict, cons: list[RoomStitchConstraint], opts: StitchOptions) -> list[np.ndarray]:
    out = []
    for c in cons:
        pa, pb = poses[c.source_room_id], poses[c.target_room_id]
        th, tx, tz = c.pose
        st = max(c.translation_uncertainty_m, opts.sigma_t_floor_m)
        sa = math.radians(max(c.rotation_uncertainty_deg, opts.sigma_theta_floor_deg))
        t = np.array([pa[1], pa[2]]) + rot(pa[0]) @ np.array([tx, tz]) - np.array([pb[1], pb[2]])
        out.append(np.array([wrap(pa[0] + th - pb[0]) / sa, t[0] / st, t[1] / st]))
    return out


def initial_poses(nodes: list[str], cons: list[RoomStitchConstraint], anchor: str) -> dict:
    """Compose constraints outward from the anchor along the best constraints first (deterministic)."""
    poses = {anchor: (0.0, 0.0, 0.0)}
    order = sorted(cons, key=lambda c: (-QUALITY_RANK[c.quality], c.translation_uncertainty_m, c.source_room_id, c.target_room_id))
    changed = True
    while changed:
        changed = False
        for c in order:
            a, b = c.source_room_id, c.target_room_id
            if a in poses and b not in poses:
                poses[b] = compose(poses[a], c.pose)
                changed = True
            elif b in poses and a not in poses:
                poses[a] = compose(poses[b], invert(c.pose))
                changed = True
    return poses


def solve_poses(nodes: list[str], cons: list[RoomStitchConstraint], anchor: str, opts: StitchOptions, iters: int = 30) -> dict:
    """Deterministic robust (Huber IRLS) Gauss-Newton on (theta, tx, tz) per room; the anchor is fixed at the identity."""
    poses = initial_poses(nodes, cons, anchor)
    free = [n for n in sorted(nodes) if n != anchor and n in poses]
    if not free or not cons:
        return poses
    idx = {n: i for i, n in enumerate(free)}
    x = np.array([v for n in free for v in poses[n]], dtype=np.float64)

    def to_poses(x):
        p = {anchor: (0.0, 0.0, 0.0)}
        for n, i in idx.items():
            p[n] = (float(x[3 * i]), float(x[3 * i + 1]), float(x[3 * i + 2]))
        return p

    active = [c for c in cons if c.source_room_id in to_poses(x) and c.target_room_id in to_poses(x)]

    def stacked(x, w):
        p = to_poses(x)
        r = _residuals(p, active, opts)
        return np.concatenate([np.sqrt(wi) * ri for wi, ri in zip(w, r)])

    for _ in range(iters):
        p = to_poses(x)
        w = []
        for r in _residuals(p, active, opts):
            n = float(np.linalg.norm(r))
            w.append(1.0 if n <= opts.huber_k else opts.huber_k / n)
        r0 = stacked(x, w)
        J = np.zeros((len(r0), len(x)))
        eps = 1e-6
        for k in range(len(x)):
            xp = x.copy()
            xp[k] += eps
            J[:, k] = (stacked(xp, w) - r0) / eps
        H = J.T @ J + 1e-9 * np.eye(len(x))
        step = np.linalg.solve(H, -J.T @ r0)
        x = x + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return {k: (wrap(v[0]), v[1], v[2]) for k, v in to_poses(x).items()}


def edge_residuals(poses: dict, cons: list[RoomStitchConstraint], opts: StitchOptions) -> list[float]:
    """Normalised residual per constraint (RMS of the three sigma-scaled components); ~1 means consistent."""
    return [float(np.sqrt(np.mean(r ** 2))) for r in _residuals(poses, cons, opts)]


def components(nodes: list[str], cons: list[RoomStitchConstraint]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in cons:
        if c.status == "active" and c.source_room_id in parent and c.target_room_id in parent:
            parent[find(c.source_room_id)] = find(c.target_room_id)
    groups: dict[str, list[str]] = {}
    for n in sorted(nodes):
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def loop_report(nodes: list[str], cons: list[RoomStitchConstraint]) -> list[dict]:
    """Closure error of every fundamental cycle (the loop of constraints composed back to its start)."""
    active = [c for c in cons if c.status == "active"]
    out = []
    for comp in components(nodes, active):
        if len(comp) < 3:
            continue
        cs = [c for c in active if c.source_room_id in comp]
        anchor = comp[0]
        poses = {anchor: (0.0, 0.0, 0.0)}
        tree = []
        order = sorted(cs, key=lambda c: (-QUALITY_RANK[c.quality], c.source_room_id, c.target_room_id))
        changed = True
        while changed:
            changed = False
            for c in order:
                a, b = c.source_room_id, c.target_room_id
                if c in tree:
                    continue
                if a in poses and b not in poses:
                    poses[b] = compose(poses[a], c.pose)
                    tree.append(c)
                    changed = True
                elif b in poses and a not in poses:
                    poses[a] = compose(poses[b], invert(c.pose))
                    tree.append(c)
                    changed = True
        for c in cs:
            if c in tree:
                continue
            pred = compose(poses[c.source_room_id], c.pose)  # where this edge says its target is
            have = poses[c.target_room_id]
            d = math.hypot(pred[1] - have[1], pred[2] - have[2])
            out.append({"closing_edge": [c.source_room_id, c.target_room_id], "rooms": comp,
                        "rotation_residual_deg": round(abs(math.degrees(wrap(pred[0] - have[0]))), 3),
                        "translation_residual_m": round(d, 4)})
    return out


def has_alternative_path(a: str, b: str, cons: list[RoomStitchConstraint], skip: RoomStitchConstraint) -> bool:
    """Is a connected to b without the edge `skip`? (True means the edge lies on a loop.)"""
    adj: dict[str, list[str]] = {}
    for c in cons:
        if c is skip or c.status != "active":
            continue
        adj.setdefault(c.source_room_id, []).append(c.target_room_id)
        adj.setdefault(c.target_room_id, []).append(c.source_room_id)
    seen, stack = {a}, [a]
    while stack:
        n = stack.pop()
        if n == b:
            return True
        for m in adj.get(n, []):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False


@dataclass
class ComponentPlacement:
    rooms: list[str]
    anchor: str
    poses: dict
    initial_poses: dict
    constraints: list[RoomStitchConstraint]
    rejected: list[RoomStitchConstraint]
    loops: list[dict]
    actions: list[str]


def choose_anchor(rooms: list[str], geoms: dict[str, RoomGeom]) -> str:
    return sorted(rooms, key=lambda r: (tuple(-v for v in geoms[r].rank), r))[0]


def place_component(rooms: list[str], cons: list[RoomStitchConstraint], geoms: dict[str, RoomGeom],
                    opts: StitchOptions) -> ComponentPlacement:
    """Solve one connected component; reject loop-inconsistent edges and edges that make rooms overlap."""
    cons = [c for c in cons if c.status == "active" and c.source_room_id in rooms and c.target_room_id in rooms]
    anchor = choose_anchor(rooms, geoms)
    actions: list[str] = []
    rejected: list[RoomStitchConstraint] = []
    init = initial_poses(rooms, cons, anchor)
    for _ in range(4 * max(1, len(cons))):
        keep = [c for c in cons if c.status == "active"]
        comp = [r for r in rooms if r in initial_poses(rooms, keep, anchor)]
        poses = solve_poses(comp, keep, anchor, opts)
        res = edge_residuals(poses, keep, opts)
        for c, r in zip(keep, res):
            c.residual_sigma = r
        # 1. loop consistency: drop the worst edge that lies on a loop and disagrees with the others
        bad = [(r, c) for c, r in zip(keep, res) if r > opts.max_normalized_residual and has_alternative_path(
            c.source_room_id, c.target_room_id, keep, c)]
        if bad:
            r, c = max(bad, key=lambda t: (t[0], -QUALITY_RANK[t[1].quality]))
            c.status = "rejected"
            c.notes.append(f"rejected: inconsistent with the other constraints in a loop (normalised residual {r:.1f})")
            rejected.append(c)
            actions.append(f"rejected {c.source_room_id}->{c.target_room_id} ({c.evidence_type}): loop residual {r:.1f} sigma")
            continue
        # 2. overlap: placed rooms must not occupy the same floor space
        offender = _worst_overlap(comp, poses, geoms, opts)
        if offender is not None:
            a, b, frac, area = offender
            path = _path_edges(a, b, keep)
            if path:
                weakest = sorted(path, key=lambda c: (QUALITY_RANK[c.quality], -c.translation_uncertainty_m,
                                                       c.source_room_id, c.target_room_id))[0]
                if weakest.alternatives:
                    alt = weakest.alternatives.pop(0)
                    weakest.rotation_deg, weakest.translation_x_m, weakest.translation_z_m = math.degrees(alt.pose[0]), alt.pose[1], alt.pose[2]
                    weakest.source_opening_id, weakest.target_opening_id = alt.opening_a, alt.opening_b
                    weakest.notes.append(f"alternative doorway hypothesis used after {a}/{b} overlapped by {frac:.0%}")
                    actions.append(f"{a} and {b} overlapped ({frac:.0%} of the smaller): switched {weakest.source_room_id}->"
                                   f"{weakest.target_room_id} to the next doorway hypothesis")
                    continue
                weakest.status = "rejected"
                weakest.notes.append(f"rejected: placing {a} and {b} made them overlap by {frac:.0%} of the smaller room")
                rejected.append(weakest)
                actions.append(f"rejected {weakest.source_room_id}->{weakest.target_room_id} ({weakest.evidence_type}): "
                               f"{a} and {b} overlapped by {frac:.0%} of the smaller room ({area:.1f} m2)")
                continue
        break
    keep = [c for c in cons if c.status == "active"]
    placed = [r for r in rooms if r in initial_poses(rooms, keep, anchor)]
    poses = solve_poses(placed, keep, anchor, opts)
    for c, r in zip(keep, edge_residuals(poses, keep, opts)):
        c.residual_sigma = r
    return ComponentPlacement(placed, anchor, poses, init, keep, rejected, loop_report(rooms, keep), actions)


def _placed_polygon(room: str, poses: dict, geoms: dict[str, RoomGeom]) -> np.ndarray | None:
    g = geoms[room]
    return None if g.polygon is None else apply(poses[room], g.polygon)


def _overlaps(rooms: list[str], poses: dict, geoms: dict[str, RoomGeom], opts: StitchOptions) -> list[dict]:
    out = []
    polys = {r: _placed_polygon(r, poses, geoms) for r in rooms}
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if polys[a] is None or polys[b] is None:
                continue
            inter, small, frac = overlap_of(polys[a], polys[b], opts.raster_cell_m)
            out.append({"room_a": a, "room_b": b, "intersection_area_m2": round(inter, 3),
                        "smaller_room_area_m2": round(small, 3), "fraction_of_smaller": round(frac, 4)})
    return out


def _worst_overlap(rooms, poses, geoms, opts):
    worst = None
    for o in _overlaps(rooms, poses, geoms, opts):
        if o["fraction_of_smaller"] > opts.overlap_max_fraction and o["intersection_area_m2"] >= opts.overlap_min_area_m2:
            if worst is None or o["fraction_of_smaller"] > worst[2]:
                worst = (o["room_a"], o["room_b"], o["fraction_of_smaller"], o["intersection_area_m2"])
    return worst


def overlap_report(rooms: list[str], poses: dict, geoms: dict[str, RoomGeom], opts: StitchOptions) -> list[dict]:
    return _overlaps(rooms, poses, geoms, opts)


def _path_edges(a: str, b: str, cons: list[RoomStitchConstraint]) -> list[RoomStitchConstraint]:
    """Edges on the (BFS, deterministic) path from a to b."""
    adj: dict[str, list[tuple[str, RoomStitchConstraint]]] = {}
    for c in sorted(cons, key=lambda c: (c.source_room_id, c.target_room_id)):
        adj.setdefault(c.source_room_id, []).append((c.target_room_id, c))
        adj.setdefault(c.target_room_id, []).append((c.source_room_id, c))
    prev = {a: None}
    queue = [a]
    while queue:
        n = queue.pop(0)
        if n == b:
            break
        for m, c in adj.get(n, []):
            if m not in prev:
                prev[m] = (n, c)
                queue.append(m)
    if b not in prev:
        return []
    path, n = [], b
    while prev[n] is not None:
        n, c = prev[n]
        path.append(c)
    return path[::-1]


# ---------------- adjacency ----------------


def adjacent_edges(poly_a: np.ndarray, poly_b: np.ndarray, dist_tol: float = 0.5, min_overlap: float = 0.5) -> bool:
    """Geometric adjacency: some edge of A is (anti)parallel to an edge of B within dist_tol and they overlap >= min_overlap."""
    def edges(p):
        return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]

    for a0, a1 in edges(poly_a):
        da = a1 - a0
        la = np.linalg.norm(da)
        if la < min_overlap:
            continue
        ua = da / la
        for b0, b1 in edges(poly_b):
            db = b1 - b0
            lb = np.linalg.norm(db)
            if lb < min_overlap:
                continue
            ub = db / lb
            if abs(ua[0] * ub[1] - ua[1] * ub[0]) > math.sin(math.radians(12)):
                continue
            n = np.array([-ua[1], ua[0]])
            if abs((b0 - a0) @ n) > dist_tol or abs((b1 - a0) @ n) > dist_tol:
                continue
            s0, s1 = sorted([(b0 - a0) @ ua, (b1 - a0) @ ua])
            if min(la, s1) - max(0.0, s0) >= min_overlap:
                return True
    return False
