"""Structural vertical-wall extraction from a metric point cloud (NumPy only, deterministic).

World +Y is vertical. Walls are found in the top-down X-Z plane, then verified in 3D.

Plain-English summary
---------------------
1. Keep points in a structural height band above the floor (floor and ceiling points excluded).
2. On a 5 cm X-Z grid mark "wall-like" columns: many points AND points at many different heights.
   Floors, ceilings and low furniture cannot produce these.
3. Find lines in that mask with an iterative Hough transform (deterministic, no random sampling).
4. Merge duplicate parallel detections of one surface; estimate dominant (Manhattan) directions
   from long lines and snap near-aligned walls to them (raw orientation is kept).
5. For every line gather 3D inliers, split them into contiguous observed segments, and accept a
   segment only if it is long enough, spans enough height, is observed at many heights, reaches
   toward the floor, is near-vertical and fits tightly. Gaps between segments are kept, not filled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from spatialforge.lidar.planes import PlaneFit


@dataclass(frozen=True)
class WallOptions:
    # Structural height band above the floor. Below 0.3 m: skirting, floor noise, low clutter.
    # Above 2.2 m: ceiling edges, lights, cornices (ceilings seen at 2.4 m and 3.0 m).
    band_bottom_m: float = 0.3
    band_top_m: float = 2.2
    floor_margin_m: float = 0.12  # points closer to the floor than this belong to the floor
    ceiling_margin_m: float = 0.10  # extra clearance around accepted ceiling planes

    cell_m: float = 0.05
    height_bin_m: float = 0.10
    cell_min_points: int = 6
    cell_min_height_bins: int = 5  # a column must show points at >= 5 distinct 10 cm heights

    hough_theta_step_deg: float = 0.5
    hough_rho_step_m: float = 0.05
    min_line_cells: int = 12  # a Hough line needs this many wall-like cells (12 cells = 0.6 m)
    max_candidate_lines: int = 80
    cell_line_tol_m: float = 0.12  # cells within this distance support a line (walls are blurred by drift)
    ribbon_halfwidth_m: float = 0.25  # after a line is found, cells this close are consumed with it

    inlier_min_m: float = 0.04  # 3D inlier tolerance = clip(2.5 sigma, min, max)
    inlier_max_m: float = 0.10
    max_tilt_deg: float = 5.0  # maximum deviation of a wall plane from vertical

    run_bin_m: float = 0.05
    run_bin_min_points: int = 4
    gap_close_m: float = 0.20  # smaller gaps are noise/thin occluders; bigger gaps are kept as gaps
    # A gap up to this width (a wide door or opening) stays inside one wall group as a recorded gap;
    # pieces further apart than this are different walls that merely happen to be collinear.
    max_opening_gap_m: float = 2.5

    min_horizontal_span_m: float = 0.6
    min_vertical_span_m: float = 1.0
    min_vertical_coverage: float = 0.35  # share of 10 cm height bins observed within the band
    max_floor_gap_m: float = 0.8  # lowest support may not float further above the floor
    mask_fill_tol_m: float = 0.06
    # Segments that really follow a wall lie on the structural mask ribbon for most of their length
    # (measured 0.5-1.0 on visibly good walls, 0.0-0.44 on lines cutting across ribbons).
    min_mask_fill: float = 0.5
    # Short pieces are kept only if they join a long wall (door jambs, wall returns); an isolated short
    # vertical plane in the middle of a room is far more likely a wardrobe, pillar or door leaf.
    structural_min_length_m: float = 1.5
    junction_tol_m: float = 0.35
    # Soft Manhattan prior: when the building clearly has two wall axes, a SHORT plane far from both axes is
    # far more likely clutter than a wall. Long off-axis walls are kept (they may be real diagonal walls).
    off_axis_deg: float = 15.0
    off_axis_max_length_m: float = 2.0
    # Evidence tiers (a heuristic summary, not a calibrated probability)
    strong_min_length_m: float = 2.5
    strong_max_axis_deviation_deg: float = 8.0
    strong_min_vertical_coverage: float = 0.7
    # Robust plane spread. Long structural walls measure 0.05-0.08 m here (residual drift blurs walls to
    # ~3x the floor's spread), so this only rejects genuinely smeared, non-planar clutter.
    max_sigma_m: float = 0.08
    min_inliers: int = 300

    # Duplicate detections of one surface are merged when close in angle, close at the ends of the
    # shorter line, strongly overlapping AND the points between them form one mode, not two.
    dup_angle_deg: float = 6.0
    dup_offset_m: float = 0.30  # walls here are smeared to ~0.5 m wide; the valley test protects real double walls
    dup_overlap: float = 0.5
    two_surface_min_sep_m: float = 0.15
    two_surface_valley_ratio: float = 0.35

    snap_tol_deg: float = 5.0
    # Snapping may move a wall's ends sideways by at most this much (half length * tan(deviation)), so long
    # walls only snap for small deviations (6 m: ~1.9 deg) while short walls can snap up to snap_tol_deg.
    snap_max_shift_m: float = 0.10
    min_manhattan_strength: float = 0.6  # how concentrated line orientations are around two axes

    position_patch_m: float = 1.0  # independent samples along a wall (errors correlate over ~1 m)
    min_position_uncertainty_m: float = 0.01


# ---------- 2D line helper ----------


@dataclass
class Line:
    centre: np.ndarray  # a point on the line (x, z)
    phi: float  # direction angle in radians, in [0, pi)
    merged_from: int = 1

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.phi), np.sin(self.phi)])

    @property
    def normal(self) -> np.ndarray:
        return np.array([-np.sin(self.phi), np.cos(self.phi)])

    def perp(self, xz: np.ndarray) -> np.ndarray:
        return (xz - self.centre) @ self.normal

    def along(self, xz: np.ndarray) -> np.ndarray:
        return (xz - self.centre) @ self.direction

    @property
    def offset(self) -> float:
        return float(self.centre @ self.normal)


def _wrap_phi(phi: float) -> float:
    return float(phi % np.pi)


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two undirected line angles (radians), in degrees."""
    d = abs(a - b) % np.pi
    return float(np.degrees(min(d, np.pi - d)))


def fit_line_tls(xz: np.ndarray) -> Line:
    """Total-least-squares line through 2D points."""
    c = xz.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((xz - c).T))
    d = vecs[:, -1]  # largest-variance direction
    return Line(c, _wrap_phi(np.arctan2(d[1], d[0])))


# ---------- results ----------


@dataclass
class WallSegment:
    start: tuple[float, float]  # (x, z)
    end: tuple[float, float]
    length_m: float
    vertical_span_m: float
    vertical_coverage: float
    floor_gap_m: float
    inlier_count: int
    density_per_m2: float
    tilt_deg: float
    residual_median_m: float
    residual_p90_m: float
    sigma_m: float  # robust spread (1.4826 * MAD) of perpendicular residuals; not truncated by the inlier tolerance
    mask_fill: float  # share of the segment lying on the structural wall-like cell ribbon
    accepted: bool
    reject_reasons: list[str] = field(default_factory=list)


@dataclass
class Wall:
    id: str
    plane: list[float]  # [a, b, c, d]; vertical so b = 0
    orientation_deg: float  # final line direction in X-Z, [0, 180)
    raw_orientation_deg: float
    snapped: bool
    snap_delta_deg: float
    start: tuple[float, float]
    end: tuple[float, float]
    length_m: float  # extent from first to last observed segment (gaps included)
    observed_length_m: float  # sum of observed segments
    horizontal_span_m: float
    vertical_span_m: float
    inlier_count: int
    residual_median_m: float
    residual_p90_m: float
    position_uncertainty_m: float
    orientation_uncertainty_deg: float
    merged_from: int
    deviation_from_axes_deg: float  # raw orientation vs the nearest dominant wall axis (0 if no dominant axis)
    evidence: str  # "strong" | "moderate" | "weak": heuristic tier, not a calibrated probability
    segments: list[WallSegment]
    gaps: list[dict]  # kept, not filled: potential openings for a later stage
    inliers: np.ndarray = field(repr=False, default=None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("inliers")
        return d


@dataclass
class WallAnalysis:
    point_count: int
    zone_point_count: int
    floor_height_m: float
    dominant_directions_deg: list[float]
    manhattan_strength: float
    candidate_lines: int
    merged_duplicates: int
    walls: list[Wall]
    rejected: list[dict]  # candidate lines/segments that failed, with reasons
    options: WallOptions
    warnings: list[str] = field(default_factory=list)
    suppressed_duplicate_segments: int = 0

    @property
    def segment_count(self) -> int:
        return sum(len(w.segments) for w in self.walls)

    def to_dict(self) -> dict:
        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            if isinstance(v, (float, np.floating)):
                return round(float(v), 4)
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.bool_):
                return bool(v)
            return v

        return clean({
            "point_count": self.point_count,
            "zone_point_count": self.zone_point_count,
            "floor_height_m": self.floor_height_m,
            "dominant_directions_deg": self.dominant_directions_deg,
            "manhattan_strength": self.manhattan_strength,
            "candidate_lines": self.candidate_lines,
            "merged_duplicate_lines": self.merged_duplicates,
            "suppressed_duplicate_segments": self.suppressed_duplicate_segments,
            "accepted_walls": len(self.walls),
            "accepted_segments": self.segment_count,
            "rejected_candidates": len(self.rejected),
            "walls": [w.to_dict() for w in self.walls],
            "rejected": self.rejected,
            "warnings": self.warnings,
            "options": asdict(self.options),
        })


# ---------- stage 1-2: zone points and wall-like cells ----------


def _wall_zone(points, floor: PlaneFit, ceilings: list[PlaneFit], opts: WallOptions):
    """Points above the floor margin and below the band top, minus ceiling-plane inliers."""
    h = points[:, 1] - floor.height_at(points[:, 0], points[:, 2])
    keep = (h >= opts.floor_margin_m) & (h <= opts.band_top_m)
    for c in ceilings:
        dist = np.abs((points[:, 1] - c.height_at(points[:, 0], points[:, 2])) * c.normal[1])
        keep &= dist > c.threshold + opts.ceiling_margin_m
    return points[keep], h[keep]


def _wall_like_cells(P: np.ndarray, H: np.ndarray, opts: WallOptions, origin: np.ndarray):
    """Centres (n, 2) of 5 cm columns with many points at many heights (inside the band)."""
    inband = (H >= opts.band_bottom_m) & (H <= opts.band_top_m)
    xz = P[inband][:, [0, 2]]
    hb = np.floor((H[inband] - opts.band_bottom_m) / opts.height_bin_m).astype(np.int64)
    idx = np.floor((xz - origin) / opts.cell_m).astype(np.int64)
    nx, nz = int(idx[:, 0].max()) + 1, int(idx[:, 1].max()) + 1
    nb = int(hb.max()) + 1
    cell_key = idx[:, 0] * nz + idx[:, 1]
    counts = np.bincount(cell_key, minlength=nx * nz)
    occupied = np.zeros(nx * nz * nb, dtype=bool)
    occupied[cell_key * nb + hb] = True
    bins_per_cell = occupied.reshape(nx * nz, nb).sum(axis=1)
    ok = (counts >= opts.cell_min_points) & (bins_per_cell >= opts.cell_min_height_bins)
    keys = np.nonzero(ok)[0]
    cells = np.column_stack([keys // nz, keys % nz])
    return origin + (cells + 0.5) * opts.cell_m


# ---------- stage 3: deterministic Hough lines ----------


def _hough_best(xz: np.ndarray, opts: WallOptions):
    thetas = np.deg2rad(np.arange(0.0, 180.0, opts.hough_theta_step_deg))
    cos, sin = np.cos(thetas), np.sin(thetas)
    rho = xz[:, 0:1] * cos + xz[:, 1:2] * sin  # (n, T) signed distance for each direction
    ridx = np.round(rho / opts.hough_rho_step_m).astype(np.int64)
    ridx -= ridx.min()
    R = int(ridx.max()) + 3
    flat = (np.arange(len(thetas))[None, :] * R + ridx + 1).ravel()
    acc = np.bincount(flat, minlength=len(thetas) * R).reshape(len(thetas), R).astype(np.int64)
    acc3 = acc + np.roll(acc, 1, axis=1) + np.roll(acc, -1, axis=1)  # tolerate +-1 rho bin
    t, r = np.unravel_index(int(np.argmax(acc3)), acc3.shape)  # first maximum: deterministic
    return int(acc3[t, r]), float(thetas[t])


def _detect_lines(cells: np.ndarray, opts: WallOptions) -> list[tuple[Line, np.ndarray]]:
    """Iteratively peel the strongest line off the wall-like cells. Returns (line, cell xz) pairs."""
    remaining = cells.copy()
    out = []
    for _ in range(opts.max_candidate_lines):
        if len(remaining) < opts.min_line_cells:
            break
        votes, theta = _hough_best(remaining, opts)
        if votes < opts.min_line_cells:
            break
        # Hough normal direction is (cos, sin); the line direction is perpendicular to it.
        line = Line(remaining.mean(axis=0), _wrap_phi(theta + np.pi / 2))
        # rough support for the Hough direction, then refine with TLS (twice)
        n = np.array([np.cos(theta), np.sin(theta)])
        rho_vals = remaining @ n
        hist_bins = np.round(rho_vals / opts.hough_rho_step_m).astype(np.int64)
        best = np.bincount(hist_bins - hist_bins.min())
        smooth = np.convolve(best, np.ones(3), mode="same")  # not np.roll: that would wrap the last bin to the first
        peak = (int(np.argmax(smooth)) + hist_bins.min()) * opts.hough_rho_step_m
        sel = np.abs(rho_vals - peak) <= opts.cell_line_tol_m
        if sel.sum() < 3:
            remaining = remaining[~sel] if sel.any() else remaining[1:]
            continue
        line = fit_line_tls(remaining[sel])
        for _ in range(2):
            sel = np.abs(line.perp(remaining)) <= opts.cell_line_tol_m
            if sel.sum() < 3:
                break
            line = fit_line_tls(remaining[sel])
        sel = np.abs(line.perp(remaining)) <= opts.cell_line_tol_m
        if sel.sum() < opts.min_line_cells:
            remaining = remaining[~sel]
            continue
        out.append((line, remaining[sel]))
        # Remove the whole blurred ribbon around the line (walls are smeared by drift to ~0.5 m),
        # over the line's observed extent, so leftovers are not re-detected as parallel duplicates.
        t_cells = line.along(remaining[sel])
        t_all = line.along(remaining)
        ribbon = (
            (np.abs(line.perp(remaining)) <= opts.ribbon_halfwidth_m)
            & (t_all >= t_cells.min() - 0.1) & (t_all <= t_cells.max() + 0.1)
        )
        remaining = remaining[~(ribbon | sel)]
    return out


# ---------- stage 4: duplicates, dominant directions, snapping ----------


def _extent(line: Line, xz: np.ndarray) -> tuple[float, float]:
    t = line.along(xz)
    return float(np.percentile(t, 2)), float(np.percentile(t, 98))


def two_distinct_surfaces(la: Line, lb: Line, common: np.ndarray, zone_xz: np.ndarray, opts: WallOptions) -> bool:
    """True if points between two close, parallel-ish lines form two separate surfaces.

    The perpendicular offsets of points in the shared corridor are histogrammed (3 cm bins). Two
    peaks at least `two_surface_min_sep_m` apart with a deep valley between them mean two surfaces;
    one broad or blurred mode (a single wall smeared by residual drift) does not.
    """
    mid = Line((la.centre + lb.centre) / 2, la.phi)
    t = mid.along(zone_xz)
    s = mid.perp(zone_xz)
    t_range = np.sort(mid.along(common))  # `common` = the two end points (x, z) of the shared stretch
    half = 0.5 * abs(la.perp(lb.centre[None])[0]) + 0.25
    sel = (t >= t_range[0]) & (t <= t_range[1]) & (np.abs(s) <= half)
    if sel.sum() < 200:
        return False
    hist, _ = np.histogram(s[sel], bins=np.arange(-half, half + 0.03, 0.03))
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    sep = int(round(opts.two_surface_min_sep_m / 0.03))
    peaks = [i for i in range(1, len(hist) - 1) if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= 0.2 * hist.max()]
    best = 0.0
    for i in peaks:
        for j in peaks:
            if j - i >= sep:
                valley = float(hist[i:j + 1].min())
                best = max(best, min(hist[i], hist[j]) - valley)  # prominence of the weaker peak over the valley
                if valley < opts.two_surface_valley_ratio * min(hist[i], hist[j]):
                    return True
    return False


def _is_duplicate(a: tuple[Line, np.ndarray], b: tuple[Line, np.ndarray], opts: WallOptions, zone_xz: np.ndarray) -> bool:
    la, ca = a
    lb, cb = b
    if _angle_diff_deg(la.phi, lb.phi) > opts.dup_angle_deg:
        return False
    a0, a1 = _extent(la, ca)
    b0, b1 = _extent(la, cb)  # both projected on a's axis
    overlap = min(a1, b1) - max(a0, b0)
    shorter = min(a1 - a0, b1 - b0)
    if not (shorter > 0 and overlap / shorter >= opts.dup_overlap):
        return False
    # perpendicular distance of the shorter line's two ends from the other line
    short_line, short_cells, other = (la, ca, lb) if (a1 - a0) <= (b1 - b0) else (lb, cb, la)
    e0, e1 = _extent(short_line, short_cells)
    ends = np.array([short_line.centre + e * short_line.direction for e in (e0, e1)])
    if np.abs(other.perp(ends)).max() > opts.dup_offset_m:
        return False
    common = np.array([la.centre + max(a0, b0) * la.direction, la.centre + min(a1, b1) * la.direction])
    return not two_distinct_surfaces(la, lb, common, zone_xz, opts)


def _merge_duplicates(lines: list[tuple[Line, np.ndarray]], opts: WallOptions, zone_xz: np.ndarray):
    lines = list(lines)
    merged = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if _is_duplicate(lines[i], lines[j], opts, zone_xz):
                    cells = np.vstack([lines[i][1], lines[j][1]])
                    line = fit_line_tls(cells)
                    line.merged_from = lines[i][0].merged_from + lines[j][0].merged_from
                    lines[i] = (line, cells)
                    del lines[j]
                    merged += 1
                    changed = True
                    break
            if changed:
                break
    return lines, merged


def dominant_directions(lines: list[tuple[Line, np.ndarray]], opts: WallOptions) -> tuple[float | None, float]:
    """Dominant wall direction (degrees, in [0, 90)) and its strength in [0, 1].

    Orientations are averaged modulo 90 degrees, weighted by line length, so walls in a
    rectangular building reinforce each other. Strength near 1 means a clear two-axis structure.
    """
    if not lines:
        return None, 0.0
    w = [max(1.0, _extent(l, c)[1] - _extent(l, c)[0]) for l, c in lines]
    return dominant_from_angles([l.phi for l, _ in lines], w, opts)


def dominant_from_angles(angles: list[float], weights: list[float], opts: WallOptions) -> tuple[float | None, float]:
    """Dominant direction (degrees, [0, 90)) and strength from line angles (radians) and weights (lengths)."""
    if not len(angles):
        return None, 0.0
    w = np.asarray(weights, dtype=float)
    ang = np.asarray(angles, dtype=float)
    z = np.sum(w * np.exp(4j * ang)) / w.sum()
    strength = float(abs(z))
    # Mode-seeking estimate: the orientation cluster (+-snap_tol/2, modulo 90) holding the most wall length,
    # averaged only within the cluster. A plain mean would let a slightly rotated wall or a diagonal wall
    # drag the dominant direction away from the walls that do agree.
    cluster = np.radians(opts.snap_tol_deg / 2)
    best_score, best_dom = -1.0, 0.0
    for c in sorted(ang):
        dev = np.abs(((ang - c + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
        inside = dev <= cluster
        score = float(w[inside].sum())
        if score > best_score + 1e-9:
            best_score = score
            best_dom = np.angle(np.sum(w[inside] * np.exp(4j * ang[inside]))) / 4
    return float(np.degrees(best_dom) % 90.0), strength


def snap_angle(phi: float, dominant_deg: float | None, strength: float, opts: WallOptions) -> tuple[float, bool, float]:
    """Snap a line angle (radians) to the nearest dominant axis if close enough.

    Returns (angle, snapped, delta_deg). Walls that are not near an axis are left untouched.
    """
    if dominant_deg is None or strength < opts.min_manhattan_strength:
        return phi, False, 0.0
    dom = np.radians(dominant_deg)
    k = np.round((phi - dom) / (np.pi / 2))
    snapped = dom + k * np.pi / 2
    delta = float(np.degrees(abs(phi - snapped)))
    if delta <= opts.snap_tol_deg:
        return _wrap_phi(snapped), True, delta
    return phi, False, delta


# ---------- stage 5: 3D verification, segments, gates ----------


def _runs(t: np.ndarray, opts: WallOptions) -> list[tuple[float, float]]:
    """Contiguous observed runs along a wall; gaps up to gap_close_m are closed, larger ones kept."""
    if len(t) == 0:
        return []
    t0 = float(t.min())
    bins = np.floor((t - t0) / opts.run_bin_m).astype(np.int64)
    counts = np.bincount(bins)
    occ = np.nonzero(counts >= opts.run_bin_min_points)[0]
    if len(occ) == 0:
        return []
    gap_bins = int(round(opts.gap_close_m / opts.run_bin_m))
    breaks = np.nonzero(np.diff(occ) > gap_bins + 1)[0]
    starts = np.concatenate([[occ[0]], occ[breaks + 1]])
    ends = np.concatenate([occ[breaks], [occ[-1]]])
    return [(t0 + s * opts.run_bin_m, t0 + (e + 1) * opts.run_bin_m) for s, e in zip(starts, ends)]


def _measured_tilt_deg(pts: np.ndarray) -> float:
    """Deviation of the best-fit plane from vertical (0 = vertical wall, 90 = horizontal surface)."""
    if len(pts) < 10:
        return 90.0
    sub = pts[:: max(1, len(pts) // 20000)]
    _, vecs = np.linalg.eigh(np.cov((sub - sub.mean(axis=0)).T))
    return float(np.degrees(np.arcsin(np.clip(abs(vecs[1, 0]), 0, 1))))  # y-component of the plane normal


def _refine_line(line: Line, P: np.ndarray, opts: WallOptions, fixed_direction: bool) -> tuple[Line, float, float]:
    """Refit the line to nearby 3D points. Returns (line, sigma, inlier tolerance)."""
    tol = opts.cell_line_tol_m
    sigma = 0.02
    for _ in range(3):
        sel = np.abs(line.perp(P[:, [0, 2]])) <= tol
        if sel.sum() < 20:
            break
        xz = P[sel][:, [0, 2]]
        if fixed_direction:
            shift = float(np.median(line.perp(xz)))
            line = Line(line.centre + shift * line.normal, line.phi, line.merged_from)
        else:
            merged = line.merged_from
            line = fit_line_tls(xz)
            line.merged_from = merged
            line.centre = line.centre + np.median(line.perp(xz)) * line.normal
        r = line.perp(xz)
        sigma = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.005)
        tol = float(np.clip(2.5 * sigma, opts.inlier_min_m, opts.inlier_max_m))
    return line, sigma, tol


def _cell_runs(line: Line, cells: np.ndarray, opts: WallOptions) -> list[tuple[float, float]]:
    """Contiguous runs along a line made of wall-like cells within its corridor.

    Segment extents come from the structural mask (columns with points at many heights), not from raw
    3D points, so clutter that merely lies near the line cannot extend a wall.
    """
    near = cells[np.abs(line.perp(cells)) <= opts.cell_line_tol_m]
    if len(near) == 0:
        return []
    t = line.along(near)
    t0 = float(t.min())
    bins = np.unique(np.floor((t - t0) / opts.run_bin_m).astype(np.int64))
    gap_bins = int(round(opts.gap_close_m / opts.run_bin_m))
    breaks = np.nonzero(np.diff(bins) > gap_bins + 1)[0]
    starts = np.concatenate([[bins[0]], bins[breaks + 1]])
    ends = np.concatenate([bins[breaks], [bins[-1]]])
    return [(t0 + s * opts.run_bin_m, t0 + (e + 1) * opts.run_bin_m) for s, e in zip(starts, ends)]


def _verify_line(line: Line, P: np.ndarray, H: np.ndarray, cells: np.ndarray, opts: WallOptions, fixed_direction: bool):
    """Mask runs -> observed segments; 3D inliers give each segment's evidence and gate decisions."""
    line, sigma, tol = _refine_line(line, P, opts, fixed_direction)
    xz = P[:, [0, 2]]
    perp = line.perp(xz)
    inl = np.abs(perp) <= tol
    t_all = line.along(xz[inl])
    n_bins = int(round((opts.band_top_m - opts.floor_margin_m) / opts.height_bin_m))
    segments, seg_pts = [], []
    for t0, t1 in _cell_runs(line, cells, opts):
        in_seg = inl.copy()
        in_seg[inl] = (t_all >= t0) & (t_all <= t1)
        pts, h, r = P[in_seg], H[in_seg], perp[in_seg]
        length = t1 - t0
        reasons = []
        if len(pts) == 0:
            continue
        v_lo, v_hi = np.percentile(h, [2, 98])
        hb = np.floor((h - opts.floor_margin_m) / opts.height_bin_m).astype(np.int64)
        hist = np.bincount(np.clip(hb, 0, n_bins - 1), minlength=n_bins)
        occupied = np.nonzero(hist >= 3)[0]
        coverage = len(occupied) / n_bins
        floor_gap = float(opts.floor_margin_m + occupied[0] * opts.height_bin_m) if len(occupied) else float(v_lo)
        tilt = _measured_tilt_deg(pts)
        absr = np.abs(r)
        p90 = float(np.percentile(absr, 90))
        seg_sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        # How much of the segment actually lies on the structural mask ribbon (tight +-6 cm corridor).
        tight = cells[np.abs(line.perp(cells)) <= opts.mask_fill_tol_m]
        tt = line.along(tight)
        tt = tt[(tt >= t0) & (tt <= t1)]
        n_run_bins = max(1, int(round(length / opts.run_bin_m)))
        mask_fill = float(len(np.unique(np.floor((tt - t0) / opts.run_bin_m).astype(np.int64))) / n_run_bins)
        if mask_fill < opts.min_mask_fill:
            reasons.append(f"line follows the wall mask for only {mask_fill:.0%} of its length")
        if length < opts.min_horizontal_span_m:
            reasons.append(f"horizontal span {length:.2f} m < {opts.min_horizontal_span_m}")
        if v_hi - v_lo < opts.min_vertical_span_m:
            reasons.append(f"vertical span {v_hi - v_lo:.2f} m < {opts.min_vertical_span_m}")
        if coverage < opts.min_vertical_coverage:
            reasons.append(f"vertical coverage {coverage:.2f} < {opts.min_vertical_coverage}")
        if floor_gap > opts.max_floor_gap_m:
            reasons.append(f"support floats {floor_gap:.2f} m above the floor")
        if tilt > opts.max_tilt_deg:
            reasons.append(f"tilt {tilt:.1f} deg > {opts.max_tilt_deg}")
        if seg_sigma > opts.max_sigma_m:
            reasons.append(f"plane spread sigma {seg_sigma:.3f} m > {opts.max_sigma_m}")
        if len(pts) < opts.min_inliers:
            reasons.append(f"only {len(pts)} inliers")
        a = line.centre + t0 * line.direction
        b = line.centre + t1 * line.direction
        seg = WallSegment(
            start=(float(a[0]), float(a[1])), end=(float(b[0]), float(b[1])), length_m=float(length),
            vertical_span_m=float(v_hi - v_lo), vertical_coverage=float(coverage), floor_gap_m=floor_gap,
            inlier_count=int(len(pts)), density_per_m2=float(len(pts) / max(length * (v_hi - v_lo), 1e-6)),
            tilt_deg=tilt, residual_median_m=float(np.median(absr)), residual_p90_m=p90, sigma_m=seg_sigma,
            mask_fill=mask_fill, accepted=not reasons, reject_reasons=reasons,
        )
        segments.append(seg)
        seg_pts.append(pts)
    return line, sigma, segments, seg_pts


def _snap_line(line: Line, phi: float, segments: list[WallSegment], seg_pts: list[np.ndarray]) -> Line:
    """Rotate a verified wall to the snapped direction about the centre of its accepted segments.

    The plane offset is re-estimated as the median distance of the inliers, and segment end points are
    projected onto the snapped line. Segment evidence metrics stay those measured on the free fit.
    """
    acc = [(s, p) for s, p in zip(segments, seg_pts) if s.accepted]
    mids = np.array([[(s.start[0] + s.end[0]) / 2, (s.start[1] + s.end[1]) / 2] for s, _ in acc])
    weights = np.array([s.length_m for s, _ in acc])
    pivot = (mids * weights[:, None]).sum(axis=0) / weights.sum()
    snapped = Line(pivot, phi, line.merged_from)
    inl = np.vstack([p for _, p in acc])[:, [0, 2]]
    snapped.centre = pivot + float(np.median(snapped.perp(inl))) * snapped.normal
    for s in segments:
        for attr in ("start", "end"):
            p = np.array(getattr(s, attr))
            q = p - snapped.perp(p[None])[0] * snapped.normal
            setattr(s, attr, (float(q[0]), float(q[1])))
        if snapped.along(np.array(s.end)[None])[0] < snapped.along(np.array(s.start)[None])[0]:
            s.start, s.end = s.end, s.start  # keep start -> end along the (possibly flipped) line direction
        s.length_m = float(np.hypot(s.end[0] - s.start[0], s.end[1] - s.start[1]))
    return snapped


def _segments_duplicate(la: Line, sa: WallSegment, lb: Line, sb: WallSegment, zone_xz: np.ndarray, opts: WallOptions) -> bool:
    """True if segment `sa` (on line la) repeats segment `sb` (on line lb): same orientation, strongly
    overlapping, close at the ends, and not two separate surfaces."""
    if _angle_diff_deg(la.phi, lb.phi) > opts.dup_angle_deg:
        return False
    pa = np.array([sa.start, sa.end])
    pb = np.array([sb.start, sb.end])
    ta, tb = lb.along(pa), lb.along(pb)
    lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
    shorter = min(ta.max() - ta.min(), tb.max() - tb.min())
    if shorter <= 0 or (hi - lo) / shorter < opts.dup_overlap:
        return False
    short = pa if (ta.max() - ta.min()) <= (tb.max() - tb.min()) else pb
    other = lb if short is pa else la
    if np.abs(other.perp(short)).max() > opts.dup_offset_m:
        return False
    common = np.array([lb.centre + lo * lb.direction, lb.centre + hi * lb.direction])
    return not two_distinct_surfaces(la, lb, common, zone_xz, opts)


def _axis_deviation_deg(phi: float, dominant_deg: float | None) -> float:
    """Angle between a line direction (radians) and the nearest of the two dominant wall axes, in degrees."""
    if dominant_deg is None:
        return 0.0
    d = (np.degrees(phi) - dominant_deg) % 90.0
    return float(min(d, 90.0 - d))


def _point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _touches(short: WallSegment, other: WallSegment, tol: float) -> bool:
    """True if either end of `short` lies within `tol` of the segment `other` (a junction or return)."""
    a, b = np.array(other.start), np.array(other.end)
    return any(_point_to_segment(np.array(p), a, b) <= tol for p in (short.start, short.end))


def _finish_wall(idx: int, line: Line, raw_phi: float, snapped: bool, delta: float, sigma: float,
                 chunk: list, opts: WallOptions, dominant_deg: float | None = None, strength: float = 0.0) -> Wall:
    axis_dev = _axis_deviation_deg(raw_phi, dominant_deg) if strength >= opts.min_manhattan_strength else 0.0
    segs = [s for s, _ in chunk]
    pts = np.vstack([p for _, p in chunk])
    start, end = segs[0].start, segs[-1].end
    gaps = []
    for s0, s1 in zip(segs, segs[1:]):
        gaps.append({
            "start": s0.end, "end": s1.start,
            "length_m": float(np.hypot(s1.start[0] - s0.end[0], s1.start[1] - s0.end[1])),
        })
    observed = float(sum(s.length_m for s in segs))
    resid = np.abs(line.perp(pts[:, [0, 2]]))
    n_patches = max(1.0, observed / opts.position_patch_m)
    pos_unc = max(1.96 * sigma / np.sqrt(n_patches), opts.min_position_uncertainty_m)
    ori_unc = float(np.degrees(np.arctan(np.sqrt(12) * sigma / (observed * np.sqrt(max(2.0, observed / 0.5))))))
    n = line.normal
    coverage = float(np.median([s.vertical_coverage for s in segs]))
    if observed >= opts.strong_min_length_m and axis_dev <= opts.strong_max_axis_deviation_deg \
            and coverage >= opts.strong_min_vertical_coverage:
        evidence = "strong"
    elif observed < opts.structural_min_length_m or axis_dev > opts.off_axis_deg:
        evidence = "weak"
    else:
        evidence = "moderate"
    return Wall(
        id=f"wall_{idx:03d}",
        plane=[float(n[0]), 0.0, float(n[1]), float(-line.offset)],
        orientation_deg=float(np.degrees(line.phi)),
        raw_orientation_deg=float(np.degrees(raw_phi)),
        snapped=snapped, snap_delta_deg=delta,
        start=start, end=end,
        length_m=float(np.hypot(end[0] - start[0], end[1] - start[1])),
        observed_length_m=observed,
        horizontal_span_m=float(np.hypot(end[0] - start[0], end[1] - start[1])),
        vertical_span_m=float(np.median([s.vertical_span_m for s in segs])),
        inlier_count=int(len(pts)),
        residual_median_m=float(np.median(resid)), residual_p90_m=float(np.percentile(resid, 90)),
        position_uncertainty_m=float(pos_unc), orientation_uncertainty_deg=ori_unc,
        merged_from=line.merged_from, deviation_from_axes_deg=axis_dev, evidence=evidence,
        segments=segs, gaps=gaps, inliers=pts,
    )


def extract_walls(
    points: np.ndarray,
    floor: PlaneFit,
    ceilings: list[PlaneFit] | None = None,
    opts: WallOptions | None = None,
) -> WallAnalysis:
    """Find structural vertical walls in a world-frame cloud (+Y up) given the floor and ceiling planes."""
    opts = opts or WallOptions()
    ceilings = ceilings or []
    points = np.asarray(points, dtype=np.float64)
    P, H = _wall_zone(points, floor, ceilings, opts)
    warnings: list[str] = []
    floor_h = float(floor.height_at(*points[:, [0, 2]].mean(axis=0)))
    empty = WallAnalysis(len(points), len(P), floor_h, [], 0.0, 0, 0, [], [], opts, warnings)
    if len(P) == 0:
        warnings.append("no points in the structural height band")
        return empty
    origin = P[:, [0, 2]].min(axis=0) - 0.5
    cells = _wall_like_cells(P, H, opts, origin)
    if len(cells) < opts.min_line_cells:
        warnings.append("too few wall-like columns to form any line")
        return empty

    detected = _detect_lines(cells, opts)
    candidate_lines = len(detected)
    merged_lines, merged = _merge_duplicates(detected, opts, P[:, [0, 2]])

    walls: list[Wall] = []
    rejected: list[dict] = []
    finished = []

    # Pass 1: verify every line at its TRUE (unconstrained) orientation, so its evidence cannot depend on
    # snapping. Dominant directions then come from the lines that actually have accepted support.
    verified = []
    for line, cell_xz in merged_lines:
        ref, sigma, segments, seg_pts = _verify_line(
            Line(line.centre, line.phi, line.merged_from), P, H, cells, opts, fixed_direction=False
        )
        verified.append((line, cell_xz, ref, sigma, segments, seg_pts))
    supported = [(v[2].phi, sum(s.length_m for s in v[4] if s.accepted)) for v in verified if any(s.accepted for s in v[4])]
    dom, strength = dominant_from_angles([a for a, _ in supported], [w for _, w in supported], opts)

    # Pass 2: collect results. Snapping happens last, per final wall piece (see below), because later
    # filters can remove segments and a rotation about the wrong pivot would swing what remains sideways.
    for line, cell_xz, ref, sigma, segments, seg_pts in verified:
        raw_phi = ref.phi
        if not any(s.accepted for s in segments):
            rejected.append({
                "kind": "line", "orientation_deg": float(np.degrees(raw_phi)),
                "centre": [float(c) for c in line.centre], "cells": int(len(cell_xz)),
                "segments": [
                    {"start": s.start, "end": s.end, "length_m": s.length_m, "vertical_span_m": s.vertical_span_m,
                     "sigma_m": s.sigma_m, "reasons": s.reject_reasons} for s in segments
                ] or "no observed segment with enough support in 3D",
            })
            continue
        for s in segments:
            if not s.accepted:
                rejected.append({"kind": "segment", "start": s.start, "end": s.end, "length_m": s.length_m,
                                 "vertical_span_m": s.vertical_span_m, "sigma_m": s.sigma_m, "reasons": s.reject_reasons})
        finished.append((ref, raw_phi, False, 0.0, sigma, segments, seg_pts))

    # Suppress duplicate segments across lines: strongest (most inliers) first.
    zone_xz = P[:, [0, 2]]
    entries = [(fi, s, pts) for fi, f in enumerate(finished) for s, pts in zip(f[5], f[6]) if s.accepted]
    entries.sort(key=lambda e: -e[1].inlier_count)
    kept: list[tuple[int, WallSegment, np.ndarray]] = []
    suppressed = 0
    for fi, s, pts in entries:
        twin = next((k for k in kept if _segments_duplicate(finished[fi][0], s, finished[k[0]][0], k[1], zone_xz, opts)), None)
        if twin is None:
            kept.append((fi, s, pts))
            continue
        s.accepted = False
        s.reject_reasons.append(f"duplicate of the wall segment starting at ({twin[1].start[0]:.2f}, {twin[1].start[1]:.2f})")
        rejected.append({"kind": "duplicate", "start": s.start, "end": s.end, "length_m": s.length_m,
                         "vertical_span_m": s.vertical_span_m, "sigma_m": s.sigma_m, "reasons": s.reject_reasons})
        suppressed += 1

    # Soft Manhattan prior for short, strongly off-axis planes.
    if dom is not None and strength >= opts.min_manhattan_strength:
        still = []
        for k in kept:
            seg, line = k[1], finished[k[0]][0]
            off = _axis_deviation_deg(line.phi, dom)
            if off > opts.off_axis_deg and seg.length_m < opts.off_axis_max_length_m:
                seg.accepted = False
                seg.reject_reasons.append(
                    f"short ({seg.length_m:.2f} m) plane {off:.0f} deg away from both dominant wall axes"
                )
                rejected.append({"kind": "off_axis_short", "start": seg.start, "end": seg.end, "length_m": seg.length_m,
                                 "vertical_span_m": seg.vertical_span_m, "sigma_m": seg.sigma_m, "reasons": seg.reject_reasons})
            else:
                still.append(k)
        kept = still

    # Isolated short pieces are probably furniture: keep a short segment only if it touches a long one.
    primaries = [k for k in kept if k[1].length_m >= opts.structural_min_length_m]
    still_kept = []
    for k in kept:
        seg = k[1]
        if seg.length_m >= opts.structural_min_length_m or any(
            k is not o and _touches(seg, o[1], opts.junction_tol_m) for o in primaries
        ):
            still_kept.append(k)
            continue
        seg.accepted = False
        seg.reject_reasons.append(
            f"isolated short vertical plane ({seg.length_m:.2f} m, no junction with a wall >= {opts.structural_min_length_m} m)"
        )
        rejected.append({"kind": "isolated_short", "start": seg.start, "end": seg.end, "length_m": seg.length_m,
                         "vertical_span_m": seg.vertical_span_m, "sigma_m": seg.sigma_m, "reasons": seg.reject_reasons})
    kept = still_kept

    # Group the kept segments per line and split where the gap is wider than a plausible opening.
    pieces = []
    for fi, (ref, raw_phi, snapped, delta, sigma, segments, seg_pts) in enumerate(finished):
        mine = sorted(((s, p) for f2, s, p in kept if f2 == fi), key=lambda sp: ref.along(np.array(sp[0].start)[None])[0])
        chunk: list = []
        for s, p in mine:
            if chunk and np.hypot(s.start[0] - chunk[-1][0].end[0], s.start[1] - chunk[-1][0].end[1]) > opts.max_opening_gap_m:
                pieces.append((ref, raw_phi, snapped, delta, sigma, chunk))
                chunk = []
            chunk.append((s, p))
        if chunk:
            pieces.append((ref, raw_phi, snapped, delta, sigma, chunk))
    pieces.sort(key=lambda pc: -sum(s.length_m for s, _ in pc[5]))
    for i, (ref, raw_phi, _, _, sigma, chunk) in enumerate(pieces, 1):
        segs = [s for s, _ in chunk]
        phi, snapped, delta = snap_angle(raw_phi, dom, strength, opts)
        line = ref
        if snapped:
            # Snapping rotates the wall, moving its ends sideways by (half length) * tan(delta). Only snap if
            # that stays within the wall's own blur; otherwise the snapped line would no longer fit the data.
            t_ends = ref.along(np.array([p for s in segs for p in (s.start, s.end)]))
            shift = 0.5 * float(t_ends.max() - t_ends.min()) * np.tan(np.radians(delta))
            if shift <= opts.snap_max_shift_m:
                line = _snap_line(ref, phi, segs, [p for _, p in chunk])
            else:
                snapped = False
        walls.append(_finish_wall(i, line, raw_phi, snapped, delta, sigma, chunk, opts, dom, strength))

    analysis = WallAnalysis(
        len(points), len(P), floor_h,
        [] if dom is None else [dom, (dom + 90.0) % 180.0], strength,
        candidate_lines, merged, walls, rejected, opts, warnings,
    )
    analysis.suppressed_duplicate_segments = suppressed
    return analysis
