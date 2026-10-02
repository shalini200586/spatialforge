"""Floor and ceiling detection from a metric point cloud (NumPy only, deterministic).

World +Y is vertical (Ticket 2/3), so horizontal planes are fitted as y = p*x + q*z + r with a
robust (Tukey) reweighted least squares, then converted to a*x + b*y + c*z + d = 0 with a unit
normal pointing up. There is no random sampling anywhere, so the same input gives the same output.

Plain-English summary
---------------------
Floor:   candidate Y-density peaks that lie below the camera path; the lowest peak whose solid
         horizontal area is at least half of the best one is fitted and kept.
Ceiling: candidate Y-density peaks at a plausible height above the floor and above the camera
         path. A candidate is accepted only if the fitted plane is near-horizontal, tightly fitted,
         and covers enough *solid* X-Z area (thin wall lines and small object tops do not count).
         Several well-supported levels may be returned; none is invented when evidence is weak.
Height:  perpendicular distance from the ceiling plane's inlier centroid to the floor plane, with an
         interval from deterministic spatial subsets (see `estimate_height_uncertainty`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class PlaneOptions:
    # Search range for the ceiling above the floor. 2.0 m is below any normal habitable ceiling;
    # 4.5 m covers high residential ceilings. The sample building has levels near 2.4 m and 3.1 m.
    min_ceiling_height_m: float = 2.0
    max_ceiling_height_m: float = 4.5
    max_tilt_deg: float = 5.0  # planes tilted more than this from horizontal are rejected

    hist_bin_m: float = 0.02
    peak_min_share: float = 0.0005  # of all points; only for generating candidates
    peak_window_m: float = 0.20  # a peak must be the maximum within +- this
    band_halfwidth_m: float = 0.15  # initial fitting band around a peak
    inlier_min_m: float = 0.03  # inlier threshold = clip(2.5 sigma, min, max)
    inlier_max_m: float = 0.08

    min_floor_clearance_m: float = 0.5  # floor must be this far below the lowest camera position
    min_ceiling_clearance_m: float = 0.3  # ceiling this far above the highest camera position
    floor_relative_area: float = 0.5  # floor = lowest candidate with >= this share of the best area

    cell_m: float = 0.10  # grid for solid-coverage measurement
    cell_min_points: int = 3
    min_floor_area_m2: float = 2.0
    # A lamp, shelf or table top is well under 3 m2 of solid area; a ceiling over a room is not.
    min_ceiling_area_m2: float = 3.0
    min_ceiling_area_ratio: float = 0.10  # ... and at least this share of the floor's solid area
    max_ceiling_sigma_m: float = 0.06  # robust spread of the ceiling fit

    correlation_cell_m: float = 1.0  # scale over which surface errors are treated as correlated
    subset_block_m: float = 1.0  # deterministic spatial subsets for the uncertainty estimate
    subset_count: int = 4
    min_interval_halfwidth_m: float = 0.01  # depth is quantised to 1 mm and clouds are voxelised


# ---------- plane representation and fitting ----------


@dataclass
class PlaneFit:
    p: float  # y = p*x + q*z + r
    q: float
    r: float
    sigma: float  # robust spread of perpendicular residuals (m)
    threshold: float  # inlier threshold used (m)
    inlier_mask: np.ndarray = field(repr=False, default=None)

    @property
    def normal(self) -> np.ndarray:
        n = np.array([-self.p, 1.0, -self.q])
        return n / np.linalg.norm(n)

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.normal @ WORLD_UP, -1.0, 1.0))))

    @property
    def equation(self) -> list[float]:
        """[a, b, c, d] with a*x + b*y + c*z + d = 0, unit normal pointing up."""
        n = self.normal
        return [float(n[0]), float(n[1]), float(n[2]), float(-self.r * n[1])]

    def height_at(self, x, z):
        return self.p * x + self.q * z + self.r


def _tukey_weights(resid: np.ndarray, scale: float) -> np.ndarray:
    u = resid / (4.685 * scale)
    w = (1 - u * u) ** 2
    w[np.abs(u) >= 1] = 0.0
    return w


def fit_horizontal_plane(points: np.ndarray, y0: float, opts: PlaneOptions, min_points: int = 200) -> PlaneFit | None:
    """Robustly fit y = p*x + q*z + r near height y0. Returns None if there is too little support.

    The band of candidate points follows the current plane, so a tilted plane is still found if
    the first band catches part of it.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xm, zm = float(np.mean(x)), float(np.mean(z))
    xc, zc = x - xm, z - zm
    p = q = 0.0
    r = y0  # height at (xm, zm)
    sigma = 0.02
    for _ in range(4):
        sel = np.abs(y - (p * xc + q * zc + r)) <= opts.band_halfwidth_m
        if int(sel.sum()) < min_points:
            return None
        A = np.column_stack([xc[sel], zc[sel], np.ones(int(sel.sum()))])
        ys = y[sel]
        w = np.ones(len(ys))
        for _ in range(10):
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(A * sw[:, None], ys * sw, rcond=None)
            resid = ys - A @ coef
            med = np.median(resid)
            sigma = max(1.4826 * float(np.median(np.abs(resid - med))), 0.005)
            w = _tukey_weights(resid, sigma)
            if w.sum() < min_points * 0.1:
                return None
        p, q, r = float(coef[0]), float(coef[1]), float(coef[2])
    # Convert the height at the centroid back to y = p*x + q*z + r0.
    r0 = r - p * xm - q * zm
    plane = PlaneFit(p, q, r0, sigma, 0.0)
    dist = (y - plane.height_at(x, z)) * plane.normal[1]  # perpendicular distance
    sel = np.abs(dist) <= opts.band_halfwidth_m
    sigma_perp = max(1.4826 * float(np.median(np.abs(dist[sel] - np.median(dist[sel])))), 0.005)
    plane.sigma = sigma_perp
    plane.threshold = float(np.clip(2.5 * sigma_perp, opts.inlier_min_m, opts.inlier_max_m))
    plane.inlier_mask = np.abs(dist) <= plane.threshold
    return plane


# ---------- coverage ----------


def solid_cells(xz: np.ndarray, cell: float, min_points: int, origin: np.ndarray) -> np.ndarray:
    """Cell indices (n, 2) that are densely occupied and have all 8 neighbours occupied too.

    Eroding by one cell removes thin lines (a wall cross-section is not a ceiling) and isolated blobs.
    """
    if len(xz) == 0:
        return np.empty((0, 2), dtype=np.int64)
    idx = np.floor((xz - origin) / cell).astype(np.int64)
    nx, nz = int(idx[:, 0].max()) + 1, int(idx[:, 1].max()) + 1
    if idx.min() < 0 or nx * nz > 50_000_000:
        raise ValueError("point cloud X-Z extent is unreasonable for the coverage grid")
    counts = np.bincount(idx[:, 0] * nz + idx[:, 1], minlength=nx * nz).reshape(nx, nz)
    occupied = np.pad(counts >= min_points, 1)
    eroded = np.ones((nx, nz), dtype=bool)
    for i in range(3):
        for j in range(3):
            eroded &= occupied[i:i + nx, j:j + nz]
    return np.argwhere(eroded).astype(np.int64)


def refit_on_solid_columns(
    points: np.ndarray, fit: PlaneFit, opts: PlaneOptions, origin: np.ndarray
) -> PlaneFit:
    """Second fitting pass using only X-Z columns where the first fit has solid horizontal coverage.

    Walls cross a height band as thin lines with points spread over all heights; they inflate the robust
    spread of a first fit. Restricting to columns with solid inlier coverage removes them, so sigma then
    describes the surface itself. The result is returned as a plane over all points.
    """
    solid = solid_cells(points[fit.inlier_mask][:, [0, 2]], opts.cell_m, opts.cell_min_points, origin)
    if len(solid) == 0:
        return fit
    cols = np.floor((points[:, [0, 2]] - origin) / opts.cell_m).astype(np.int64)
    width = int(max(cols[:, 1].max(), solid[:, 1].max())) + 2
    solid_keys = np.unique(solid[:, 0] * width + solid[:, 1])
    in_solid = np.isin(cols[:, 0] * width + cols[:, 1], solid_keys)
    y0 = fit.height_at(points[fit.inlier_mask][:, 0].mean(), points[fit.inlier_mask][:, 2].mean())
    refined = fit_horizontal_plane(points[in_solid], y0, opts)
    if refined is None:
        return fit
    # Re-evaluate the refined plane against ALL points so inliers / coverage are measured consistently.
    dist = (points[:, 1] - refined.height_at(points[:, 0], points[:, 2])) * refined.normal[1]
    refined.inlier_mask = np.abs(dist) <= refined.threshold
    return refined


@dataclass
class PlaneResult:
    observed: bool
    plane: list[float] | None = None  # [a, b, c, d]
    height_m: float | None = None  # plane height at the cloud's X-Z centre (see `reference_xz`)
    tilt_deg: float | None = None
    inlier_count: int = 0
    inlier_ratio: float = 0.0  # of all cloud points
    residual_median_m: float | None = None
    residual_p90_m: float | None = None
    sigma_m: float | None = None  # robust spread (thickness proxy)
    inlier_threshold_m: float | None = None
    solid_area_m2: float = 0.0
    coverage_ratio: float = 0.0  # solid area / solid area of the whole scanned footprint
    x_extent_m: float | None = None
    z_extent_m: float | None = None
    reject_reason: str | None = None
    centroid: list[float] | None = None
    # Not serialised:
    fit: PlaneFit | None = field(default=None, repr=False)
    inliers: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("fit", "inliers"):
            d.pop(k)
        return d


def describe_plane(points: np.ndarray, fit: PlaneFit, footprint_solid: int, opts: PlaneOptions, origin: np.ndarray) -> PlaneResult:
    inl = points[fit.inlier_mask]
    dist = np.abs((inl[:, 1] - fit.height_at(inl[:, 0], inl[:, 2])) * fit.normal[1])
    solid = solid_cells(inl[:, [0, 2]], opts.cell_m, opts.cell_min_points, origin)
    centroid = inl.mean(axis=0)
    return PlaneResult(
        observed=True,
        plane=fit.equation,
        height_m=float(fit.height_at(centroid[0], centroid[2])),
        tilt_deg=fit.tilt_deg,
        inlier_count=int(len(inl)),
        inlier_ratio=float(len(inl) / len(points)),
        residual_median_m=float(np.median(dist)),
        residual_p90_m=float(np.percentile(dist, 90)),
        sigma_m=float(fit.sigma),
        inlier_threshold_m=float(fit.threshold),
        solid_area_m2=float(len(solid) * opts.cell_m ** 2),
        coverage_ratio=float(len(solid) / footprint_solid) if footprint_solid else 0.0,
        x_extent_m=float(np.ptp(inl[:, 0])),
        z_extent_m=float(np.ptp(inl[:, 2])),
        centroid=[float(c) for c in centroid],
        fit=fit,
        inliers=inl,
    )


# ---------- candidates ----------


def histogram_peaks(y: np.ndarray, opts: PlaneOptions) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """(bin centres, counts, peak heights strongest first) of the vertical density profile."""
    edges = np.arange(y.min(), y.max() + opts.hist_bin_m, opts.hist_bin_m)
    counts, edges = np.histogram(y, bins=edges)
    centres = edges[:-1] + opts.hist_bin_m / 2
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    half = max(1, int(round(opts.peak_window_m / opts.hist_bin_m)))
    floor_count = opts.peak_min_share * len(y) / 3
    peaks = []
    for i in range(len(smooth)):
        window = smooth[max(0, i - half): i + half + 1]
        is_max = smooth[i] == window.max()
        starts_plateau = i == 0 or smooth[i] > smooth[i - 1]  # equal neighbours are not both peaks
        if smooth[i] >= floor_count and is_max and starts_plateau:
            peaks.append((float(smooth[i]), float(centres[i])))
    peaks.sort(key=lambda t: (-t[0], t[1]))
    return centres, counts, [h for _, h in peaks]


@dataclass
class PlaneAnalysis:
    point_count: int
    floor: PlaneResult
    ceiling: PlaneResult  # primary ceiling level (largest solid area), observed=False if none
    ceiling_levels: list[PlaneResult]
    ceiling_height: dict  # primary level: value_m, confidence_interval_m, confidence, ...
    ceiling_level_heights: list[dict]
    camera_y_min: float
    camera_y_max: float
    warnings: list[str]
    options: PlaneOptions
    candidate_log: list = field(default_factory=list)  # every ceiling candidate and why it was kept/rejected
    profile_centres: np.ndarray = field(repr=False, default=None)
    profile_counts: np.ndarray = field(repr=False, default=None)

    def to_dict(self) -> dict:
        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            if isinstance(v, (float, np.floating)):
                return round(float(v), 5)
            if isinstance(v, np.integer):
                return int(v)
            return v

        out = {
            "point_count": self.point_count,
            "camera_y_range_m": [self.camera_y_min, self.camera_y_max],
            "floor": self.floor.to_dict(),
            "ceiling": self.ceiling.to_dict(),
            "ceiling_height": self.ceiling_height,
            "ceiling_levels": [
                {**lvl.to_dict(), "height": h} for lvl, h in zip(self.ceiling_levels, self.ceiling_level_heights)
            ],
            "ceiling_candidates": self.candidate_log,
            "warnings": self.warnings,
            "options": asdict(self.options),
        }
        return clean(out)


def _reject(reason: str) -> PlaneResult:
    return PlaneResult(observed=False, reject_reason=reason)


def _spatial_subsets(points: np.ndarray, opts: PlaneOptions) -> list[np.ndarray]:
    """Deterministic interleaved subsets: 1 m blocks dealt out round-robin, so each spans the scan."""
    bx = np.floor(points[:, 0] / opts.subset_block_m).astype(np.int64)
    bz = np.floor(points[:, 2] / opts.subset_block_m).astype(np.int64)
    sid = (bx + 2 * bz) % opts.subset_count
    return [points[sid == k] for k in range(opts.subset_count)]


def _fit_like_main(points: np.ndarray, y0: float, opts: PlaneOptions, origin: np.ndarray, min_points: int = 200):
    """The same two-pass fit used for the headline planes (so subsets are comparable to the full fit)."""
    fit = fit_horizontal_plane(points, y0, opts, min_points)
    return None if fit is None else refit_on_solid_columns(points, fit, opts, origin)


def estimate_height_uncertainty(
    points: np.ndarray, floor: PlaneResult, ceiling: PlaneResult, opts: PlaneOptions, origin: np.ndarray
) -> dict:
    """Ceiling height with an interval, from deterministic spatial subsets.

    Two families of deterministic subsets are used. (1) `subset_count` interleaved 1 m blocks, each
    spanning the scan: floor and ceiling are re-fitted and the perpendicular height measured at one
    shared reference point (the ceiling's inlier centroid). (2) The four quadrants around the ceiling's
    centre, each giving the local height at its own location. The spread of those heights captures
    inconsistency of the planes across the scan (noise, drift, tilt, steps).
    Half-width = 1.96 * sqrt(s^2 + sigma_f^2/n_f + sigma_c^2/n_c), s = the larger of the two
    standard deviations, n = solid area / (1 m)^2. Points on a surface are strongly
    correlated, and pose drift is smooth over about a metre, so independent samples are counted as 1 m
    patches, not points or 10 cm cells. Minimum half-width: 1 cm.
    The interval does NOT include the depth-scale assumption (0.001 m per unit).
    """
    ref = np.array(ceiling.centroid)

    def perpendicular_height(f: PlaneFit, c: PlaneFit) -> float:
        return float((c.height_at(ref[0], ref[2]) - f.height_at(ref[0], ref[2])) * f.normal[1])

    full = perpendicular_height(floor.fit, ceiling.fit)
    heights = []
    for sub in _spatial_subsets(points, opts):
        f = _fit_like_main(sub, floor.fit.height_at(floor.centroid[0], floor.centroid[2]), opts, origin, 100)
        c = _fit_like_main(sub, ceiling.fit.height_at(ref[0], ref[2]), opts, origin, 100)
        if f is not None and c is not None:
            heights.append(perpendicular_height(f, c))
    # Regional subsets: the four quadrants around the ceiling's centre. Each measures the local height
    # at its own location, so drift, tilt or steps between regions show up as spread (interleaved
    # subsets cannot see spatial structure).
    cx, cz = np.median(ceiling.inliers[:, 0]), np.median(ceiling.inliers[:, 2])
    regional = []
    for west in (True, False):
        for south in (True, False):
            sel = ((points[:, 0] < cx) == west) & ((points[:, 2] < cz) == south)
            sub = points[sel]
            f = _fit_like_main(sub, floor.fit.height_at(floor.centroid[0], floor.centroid[2]), opts, origin, 100)
            c = _fit_like_main(sub, ceiling.fit.height_at(ref[0], ref[2]), opts, origin, 100)
            if f is None or c is None or int(c.inlier_mask.sum()) < 500:
                continue
            at = sub[c.inlier_mask].mean(axis=0)
            regional.append(float((c.height_at(at[0], at[2]) - f.height_at(at[0], at[2])) * f.normal[1]))
    n_f = max(1.0, floor.solid_area_m2 / opts.correlation_cell_m ** 2)
    n_c = max(1.0, ceiling.solid_area_m2 / opts.correlation_cell_m ** 2)
    s_interleaved = float(np.std(heights, ddof=1)) if len(heights) >= 3 else None
    s_regional = float(np.std(regional, ddof=1)) if len(regional) >= 2 else 0.0
    if s_interleaved is None:
        s, spread_note = 0.10, "fewer than 3 subsets could be fitted; interval set conservatively wide"
    else:
        s, spread_note = max(s_interleaved, s_regional), None
    half = 1.96 * float(np.sqrt(s ** 2 + floor.sigma_m ** 2 / n_f + ceiling.sigma_m ** 2 / n_c))
    half = max(half, opts.min_interval_halfwidth_m)
    # Confidence: 1 when the interval is far below 5 cm and the ceiling covers >= 10 m2 of solid area.
    precision = float(np.exp(-half / 0.05))
    support = float(min(1.0, ceiling.solid_area_m2 / 10.0))
    out = {
        "value_m": full,
        "confidence_interval_m": [full - half, full + half],
        "interval_halfwidth_m": half,
        "confidence": precision * support,
        "subset_heights_m": heights,
        "regional_heights_m": regional,
        "subset_std_m": s,
        "excludes": "depth-scale assumption (0.001 m per raw unit)",
    }
    if spread_note:
        out["note"] = spread_note
    return out


def analyze_horizontal_planes(
    points: np.ndarray, camera_positions: np.ndarray, opts: PlaneOptions | None = None
) -> PlaneAnalysis:
    """Detect floor and ceiling planes in a world-frame point cloud (+Y up) given the camera path."""
    opts = opts or PlaneOptions()
    points = np.asarray(points, dtype=np.float64)
    cam_y_min, cam_y_max = float(camera_positions[:, 1].min()), float(camera_positions[:, 1].max())
    warnings: list[str] = []
    origin = points[:, [0, 2]].min(axis=0)
    footprint = solid_cells(points[:, [0, 2]], opts.cell_m, opts.cell_min_points, origin)
    footprint_solid = max(len(footprint), 1)

    centres, counts, peaks = histogram_peaks(points[:, 1], opts)
    empty = _reject("not evaluated")

    # ---- floor: candidates below the camera path ----
    floor_cands = []
    for yc in (h for h in peaks if h <= cam_y_min - opts.min_floor_clearance_m):
        fit = fit_horizontal_plane(points, yc, opts)
        if fit is None:
            continue
        fit = refit_on_solid_columns(points, fit, opts, origin)
        res = describe_plane(points, fit, footprint_solid, opts, origin)
        if res.tilt_deg <= opts.max_tilt_deg:
            floor_cands.append(res)
    floor = _reject("no horizontal band below the camera path with enough support")
    if floor_cands:
        best_area = max(c.solid_area_m2 for c in floor_cands)
        strong = [c for c in floor_cands if c.solid_area_m2 >= opts.floor_relative_area * best_area]
        floor = min(strong, key=lambda c: c.height_m)
        if floor.solid_area_m2 < opts.min_floor_area_m2:
            floor = _reject(f"floor candidate covers only {floor.solid_area_m2:.1f} m2 of solid area")

    levels: list[PlaneResult] = []
    level_heights: list[dict] = []
    ceiling = _reject("floor not observed" if not floor.observed else "no candidate passed the ceiling checks")
    candidate_log = []
    if floor.observed:
        floor_y_ref = floor.height_m
        lo = max(floor_y_ref + opts.min_ceiling_height_m, cam_y_max + opts.min_ceiling_clearance_m)
        hi = floor_y_ref + opts.max_ceiling_height_m
        accepted: list[PlaneResult] = []
        for yc in (h for h in peaks if lo <= h <= hi):
            fit = fit_horizontal_plane(points, yc, opts)
            if fit is None:
                candidate_log.append({"y_peak": yc, "rejected": "too few points near peak"})
                continue
            fit = refit_on_solid_columns(points, fit, opts, origin)
            res = describe_plane(points, fit, footprint_solid, opts, origin)
            reason = None
            if res.tilt_deg > opts.max_tilt_deg:
                reason = f"tilt {res.tilt_deg:.1f} deg exceeds {opts.max_tilt_deg}"
            elif res.sigma_m > opts.max_ceiling_sigma_m:
                reason = f"plane spread {res.sigma_m:.3f} m exceeds {opts.max_ceiling_sigma_m}"
            elif res.solid_area_m2 < opts.min_ceiling_area_m2:
                reason = f"solid area {res.solid_area_m2:.1f} m2 below {opts.min_ceiling_area_m2}"
            elif res.solid_area_m2 < opts.min_ceiling_area_ratio * floor.solid_area_m2:
                reason = f"solid area is only {res.solid_area_m2 / floor.solid_area_m2:.0%} of the floor's"
            elif not (lo <= res.height_m <= hi):
                reason = "fitted plane left the allowed height range"
            candidate_log.append({"y_peak": yc, "fitted_height_m": res.height_m, "solid_area_m2": res.solid_area_m2,
                                  "sigma_m": res.sigma_m, "tilt_deg": res.tilt_deg, "rejected": reason})
            if reason is None:
                accepted.append(res)
        # Several peaks can be the same surface: keep the best of any pair closer than 0.15 m.
        accepted.sort(key=lambda c: -c.solid_area_m2)
        for c in accepted:
            if all(abs(c.height_m - k.height_m) > 0.15 for k in levels):
                levels.append(c)
        if not candidate_log:
            ceiling = _reject(
                f"no horizontal-density peak between {lo - floor_y_ref:.2f} and {hi - floor_y_ref:.2f} m above the floor "
                "(and above the camera path)"
            )
        if levels:
            ceiling = levels[0]
            for lvl in levels:
                level_heights.append(estimate_height_uncertainty(points, floor, lvl, opts, origin))
    ceiling_height = level_heights[0] if level_heights else {"value_m": None, "confidence_interval_m": None, "confidence": 0.0}

    # ---- sanity checks ----
    if floor.observed:
        if floor.height_m >= cam_y_min:
            warnings.append("floor is not below the camera trajectory")
        if floor.tilt_deg > 2.0:
            warnings.append(f"floor tilt {floor.tilt_deg:.1f} deg is large for a levelled capture")
    for lvl, h in zip(levels, level_heights):
        v = h["value_m"]
        if lvl.height_m <= cam_y_max:
            warnings.append("a ceiling level is not above the camera trajectory")
        if v < 1.8 or v > 5.0:
            warnings.append(f"ceiling height {v:.2f} m is outside the plausible 1.8-5.0 m range")
        if lvl.solid_area_m2 < 2 * opts.min_ceiling_area_m2:
            warnings.append(f"ceiling at {v:.2f} m is supported by only {lvl.solid_area_m2:.1f} m2 of solid area")
        angle = float(np.degrees(np.arccos(np.clip(np.array(lvl.plane[:3]) @ np.array(floor.plane[:3]), -1, 1))))
        if angle > opts.max_tilt_deg:
            warnings.append(f"floor and ceiling normals differ by {angle:.1f} deg")
        regional = h["regional_heights_m"]
        if len(regional) >= 2 and np.ptp(regional) > 0.05:
            warnings.append(
                f"ceiling at {v:.2f} m: local heights differ by {np.ptp(regional):.2f} m between regions "
                "(plane tilt, residual drift, or separate rooms with different ceiling heights); interval widened accordingly"
            )
        if h["interval_halfwidth_m"] < 0.25 * max(floor.sigma_m, lvl.sigma_m) and h["interval_halfwidth_m"] > opts.min_interval_halfwidth_m:
            warnings.append("confidence interval is unrealistically small relative to the plane residuals")
    if len(levels) > 1:
        heights = ", ".join(f"{h['value_m']:.2f} m" for h in level_heights)
        warnings.append(
            f"{len(levels)} separate ceiling levels found ({heights}); a single global height is not meaningful, "
            "assigning levels to rooms needs room segmentation (not implemented)"
        )

    analysis = PlaneAnalysis(
        point_count=len(points), floor=floor, ceiling=ceiling, ceiling_levels=levels,
        ceiling_height=ceiling_height, ceiling_level_heights=level_heights,
        camera_y_min=cam_y_min, camera_y_max=cam_y_max, warnings=warnings, options=opts,
        candidate_log=candidate_log, profile_centres=centres, profile_counts=counts,
    )
    return analysis
