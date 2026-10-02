"""Structural opening (door / window / generic) detection with metric widths. NumPy only, deterministic.

Coordinates for every candidate (the "local wall frame"):
    u = distance along the wall line, v = height above the floor plane, n = perpendicular distance from the wall.

Plain-English summary
---------------------
1. Candidates: gaps inside a wall group (Ticket 5), gaps between collinear walls, and unobserved stretches along
   room boundaries (Ticket 6). A gap alone is NOT an opening.
2. Each candidate gets a u-v occupancy map of 3D points lying on the wall plane (floor and ceiling excluded, isolated
   cells removed). An opening exists only where the middle of the gap is empty over a band of rows while solid wall
   runs exist on BOTH sides in those rows. Wall present across the gap in 3D = rejected (a spurious gap).
3. Jambs: for every opening row the nearest solid run on each side gives an edge; the jamb is the median over rows
   (outlier points and isolated cells are ignored). width = right jamb - left jamb.
4. Type from the vertical profile: reaches the floor -> door; solid wall below a gap band -> window; otherwise a
   generic opening. Heights/sills are reported only when the head/sill is actually observed.
5. Guards: observability (wall support around the gap), wall ends, furniture/occlusion in front of the gap, shape
   consistency and stability across deterministic frame subsets. Existence, type and width are rated separately.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from spatialforge.lidar.planes import PlaneFit
from spatialforge.lidar.rooms import TopologyResult, WallInput, point_in_polygon

TIER_RANK = {"strong": 0, "moderate": 1, "weak": 2}
LABEL_RANK = {"strong": 0, "moderate": 1, "weak": 2}


@dataclass(frozen=True)
class OpeningOptions:
    # occupancy grid
    du_m: float = 0.05
    dv_m: float = 0.10
    # Points this close to the wall plane count as wall. Walls are blurred by residual drift, so the tolerance
    # follows each wall's own measured blur: 2 x its p90 residual, clamped to this range.
    plane_tol_min_m: float = 0.12
    plane_tol_max_m: float = 0.20
    slab_max_m: float = 0.60  # points in front of / behind the wall within this distance may be occluding clutter
    v_min_m: float = 0.12  # below this the floor itself lies on the wall plane
    v_max_m: float = 2.40
    ceiling_clearance_m: float = 0.15
    side_window_m: float = 1.0  # how far either side of the candidate is examined
    cell_min_points: int = 3
    subset_cell_min_points: int = 2  # a subset holds ~1/3 of the frames
    solid_neighbours: int = 3  # a cell is solid if occupied with this many occupied neighbours (of 8)
    min_run_cols: int = 3  # a solid run needs this many consecutive columns (15 cm)
    centre_half_m: float = 0.15

    # candidates / sanity
    min_width_m: float = 0.40
    max_width_m: float = 4.0
    max_candidate_gap_m: float = 4.0
    collinear_angle_deg: float = 6.0
    collinear_offset_m: float = 0.30
    room_edge_margin_m: float = 0.5  # ignore unobserved stretches this close to a room corner
    room_distance_m: float = 0.45

    # classification
    door_sill_max_m: float = 0.25
    door_min_height_m: float = 1.6
    door_max_width_m: float = 1.6
    window_min_sill_m: float = 0.40
    window_min_band_m: float = 0.30
    head_min_rows: int = 2
    below_support_fraction: float = 0.7

    # jambs
    jamb_min_rows: int = 3
    jamb_row_tolerance_m: float = 0.25  # measured jamb edges are ragged by ~+-0.2 m between rows on real walls
    jamb_min_consistency: float = 0.6

    # observability and guards
    side_strong_m: float = 0.8
    side_moderate_m: float = 0.4
    fill_strong: float = 0.6
    fill_moderate: float = 0.4
    empty_strong: float = 0.9
    empty_moderate: float = 0.8
    empty_reject: float = 0.7
    blockage_strong: float = 0.10
    blockage_moderate: float = 0.25
    blockage_reject: float = 0.40

    # frame subsets
    subset_block_frames: int = 20
    subset_count: int = 3
    stability_strong_m: float = 0.05
    stability_moderate_m: float = 0.12
    stability_reject_m: float = 0.35
    min_subsets: int = 2
    fallback_subset_std_m: float = 0.10
    wall_position_weight: float = 0.25  # share of the wall position uncertainty carried into the width


# ---------- data model ----------


@dataclass
class Candidate:
    id: str
    wall_id: str
    source: str  # wall_gap | collinear_walls | room_boundary
    evidence: str
    u0: float
    u1: float
    start: tuple[float, float]
    end: tuple[float, float]
    raw_width_m: float
    room_ids: list[str] = field(default_factory=list)


@dataclass
class Grid:
    u0: float
    v0: float
    du: float
    dv: float
    occupied: np.ndarray  # (nv, nu) bool
    solid: np.ndarray  # (nv, nu) bool
    slab: np.ndarray  # (nv, nu) bool: points in front of / behind the wall plane


@dataclass
class Opening:
    id: str
    candidate: Candidate
    status: str  # accepted | low_confidence
    type: str  # door | window | opening
    room_ids: list[str]
    connects: list[str]  # room ids and/or "unmodelled"
    width_m: float
    width_interval_m: tuple[float, float]
    height_m: float | None
    sill_height_m: float | None
    left_jamb: dict
    right_jamb: dict
    observability: str
    existence_quality: str
    type_quality: str
    width_quality: str
    metrics: dict
    grid: Grid | None = None

    def to_dict(self) -> dict:
        c = self.candidate
        return {
            "id": self.id, "status": self.status, "wall_id": c.wall_id, "type": self.type,
            "room_ids": self.room_ids, "connects": self.connects,
            "width_m": self.width_m, "width_interval_m": list(self.width_interval_m),
            "height_m": self.height_m, "sill_height_m": self.sill_height_m,
            "left_jamb": self.left_jamb, "right_jamb": self.right_jamb,
            "observability": self.observability, "existence_quality": self.existence_quality,
            "type_quality": self.type_quality, "width_quality": self.width_quality,
            "candidate": {"id": c.id, "source": c.source, "wall_evidence": c.evidence,
                          "raw_gap_width_m": c.raw_width_m, "raw_start": list(c.start), "raw_end": list(c.end)},
            "metrics": self.metrics,
        }


@dataclass
class OpeningResult:
    candidates: list[Candidate]
    openings: list[Opening]  # accepted + low_confidence
    rejected: list[dict]
    connectivity: dict
    wall_gaps_considered: int
    options: OpeningOptions
    warnings: list[str] = field(default_factory=list)
    grids: dict = field(default_factory=dict)  # candidate id -> occupancy Grid (for diagnostics)

    @property
    def accepted(self) -> list[Opening]:
        return [o for o in self.openings if o.status == "accepted"]

    @property
    def low_confidence(self) -> list[Opening]:
        return [o for o in self.openings if o.status == "low_confidence"]


# ---------- candidate generation ----------


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _dedupe(cands: list[Candidate], walls: dict[str, WallInput]) -> list[Candidate]:
    kept: list[Candidate] = []
    for c in cands:
        w = walls[c.wall_id]
        dup = False
        for k in kept:
            wk = walls[k.wall_id]
            if abs(_cross2(w.direction, wk.direction)) > np.sin(np.radians(10)):
                continue
            if abs((np.array(c.start) - wk.centre) @ wk.normal) > 0.35:
                continue
            a0, a1 = sorted((wk.t(c.start), wk.t(c.end)))
            b0, b1 = sorted((wk.t(k.start), wk.t(k.end)))
            overlap = min(a1, b1) - max(a0, b0)
            if overlap / max(min(a1 - a0, b1 - b0), 1e-9) >= 0.5:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def generate_candidates(walls: list[WallInput], topo: TopologyResult | None, opts: OpeningOptions) -> tuple[list[Candidate], int]:
    by_id = {w.id: w for w in walls}
    raw: list[Candidate] = []
    gaps_seen = 0
    # A. internal gaps of a wall group (Ticket 5 preserved them)
    for w in sorted(walls, key=lambda w: w.id):
        for g in w.gaps:
            gaps_seen += 1
            a, b = g["start"], g["end"]
            ta, tb = sorted((w.t(a), w.t(b)))
            if tb - ta < opts.min_width_m * 0.5 or tb - ta > opts.max_candidate_gap_m:
                continue
            raw.append(Candidate("", w.id, "wall_gap", w.evidence, ta, tb, tuple(a), tuple(b), tb - ta))
    # B. gaps between collinear walls of different groups
    for wa, wb in itertools.combinations(sorted(walls, key=lambda w: w.id), 2):
        if np.degrees(np.arcsin(min(1.0, abs(_cross2(wa.direction, wb.direction))))) > opts.collinear_angle_deg:
            continue
        ends_b = [wb.point(wb.t0), wb.point(wb.t1)]
        if max(abs((p - wa.centre) @ wa.normal) for p in ends_b) > opts.collinear_offset_m:
            continue
        tb0, tb1 = sorted(wa.t(p) for p in ends_b)
        gap = (tb0 - wa.t1) if tb0 >= wa.t1 else ((wa.t0 - tb1) if tb1 <= wa.t0 else None)
        if gap is None or gap < opts.min_width_m or gap > opts.max_candidate_gap_m:
            continue
        lo, hi = (wa.t1, tb0) if tb0 >= wa.t1 else (tb1, wa.t0)
        gaps_seen += 1
        tier = max(wa.evidence, wb.evidence, key=lambda t: TIER_RANK[t])
        raw.append(Candidate("", wa.id, "collinear_walls", tier, lo, hi, tuple(wa.point(lo)), tuple(wa.point(hi)), hi - lo))
    # C. unobserved stretches along room boundaries (away from corners)
    if topo is not None:
        for room in topo.rooms:
            pts = room.polygon
            for i in range(len(pts)):
                p0, p1 = pts[i], pts[(i + 1) % len(pts)]
                length = float(np.linalg.norm(p1 - p0))
                if length < 2 * opts.room_edge_margin_m + opts.min_width_m:
                    continue
                d = (p1 - p0) / length
                nrm = np.array([-d[1], d[0]])
                support = []
                for w in walls:
                    if np.degrees(np.arcsin(min(1.0, abs(_cross2(w.direction, d))))) > opts.collinear_angle_deg:
                        continue
                    if abs((p0 - w.centre) @ w.normal) > opts.collinear_offset_m:
                        continue
                    for a, b in w.intervals:
                        sa, sb = sorted(((w.point(a) - p0) @ d, (w.point(b) - p0) @ d))
                        support.append((max(sa, 0.0), min(sb, length)))
                support = sorted(s for s in support if s[1] > s[0])
                cursor = opts.room_edge_margin_m
                holes = []
                for s0, s1 in support + [(length - opts.room_edge_margin_m, length)]:
                    if s0 > cursor + opts.min_width_m:
                        holes.append((cursor, min(s0, length - opts.room_edge_margin_m)))
                    cursor = max(cursor, s1)
                for h0, h1 in holes:
                    if h1 - h0 < opts.min_width_m or h1 - h0 > opts.max_candidate_gap_m:
                        continue
                    ref = max((w for w in walls if abs((p0 - w.centre) @ w.normal) <= opts.collinear_offset_m
                               and np.degrees(np.arcsin(min(1.0, abs(_cross2(w.direction, d))))) <= opts.collinear_angle_deg),
                              key=lambda w: (-TIER_RANK[w.evidence], w.observed_length), default=None)
                    if ref is None:
                        continue
                    a, b = p0 + h0 * d, p0 + h1 * d
                    ta, tb = sorted((ref.t(a), ref.t(b)))
                    raw.append(Candidate("", ref.id, "room_boundary", ref.evidence, ta, tb, tuple(a), tuple(b), tb - ta))
    cands = _dedupe(raw, by_id)
    cands.sort(key=lambda c: (c.wall_id, round(c.u0, 3)))
    for i, c in enumerate(cands, 1):
        c.id = f"candidate_{i:03d}"
        mid = np.array([(c.start[0] + c.end[0]) / 2, (c.start[1] + c.end[1]) / 2])
        c.room_ids = _rooms_near(mid, topo, opts) if topo is not None else []
    return cands, gaps_seen


def _rooms_near(mid: np.ndarray, topo: TopologyResult, opts: OpeningOptions) -> list[str]:
    dists = []
    for room in topo.rooms:
        p = room.polygon
        best = min(_seg_dist(mid, p[i], p[(i + 1) % len(p)]) for i in range(len(p)))
        if best <= opts.room_distance_m:
            dists.append((best, room.id))
    return [rid for _, rid in sorted(dists)[:2]]


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


# ---------- occupancy ----------


def build_grid(points: np.ndarray, floor: PlaneFit, wall: WallInput, u_lo: float, u_hi: float,
               v_hi: float, opts: OpeningOptions, min_points: int) -> Grid:
    xz = points[:, [0, 2]]
    rel = xz - wall.centre
    u = rel @ wall.direction
    sel = (u >= u_lo) & (u < u_hi)
    p, u = points[sel], u[sel]
    n = rel[sel] @ wall.normal
    v = p[:, 1] - floor.height_at(p[:, 0], p[:, 2])
    ok = (v >= opts.v_min_m) & (v < v_hi)
    p, u, n, v = p[ok], u[ok], n[ok], v[ok]
    nu = max(1, int(np.ceil((u_hi - u_lo) / opts.du_m)))
    nv = max(1, int(round((v_hi - opts.v_min_m) / opts.dv_m)))
    iu = np.clip(((u - u_lo) / opts.du_m).astype(int), 0, nu - 1)
    iv = np.clip(((v - opts.v_min_m) / opts.dv_m).astype(int), 0, nv - 1)
    tol = float(np.clip(2.0 * wall.residual_p90_m, opts.plane_tol_min_m, opts.plane_tol_max_m))
    on = np.abs(n) <= tol
    near = (np.abs(n) > tol) & (np.abs(n) <= opts.slab_max_m)
    counts = np.bincount(iv[on] * nu + iu[on], minlength=nv * nu).reshape(nv, nu)
    slab = np.bincount(iv[near] * nu + iu[near], minlength=nv * nu).reshape(nv, nu)
    occupied = counts >= min_points
    pad = np.pad(occupied, 1).astype(int)
    neighbours = sum(pad[i:i + nv, j:j + nu] for i in range(3) for j in range(3) if (i, j) != (1, 1))
    solid = occupied & (neighbours >= opts.solid_neighbours)
    return Grid(u_lo, opts.v_min_m, opts.du_m, opts.dv_m, occupied, solid, slab >= min_points)


# ---------- row analysis ----------


def _run_from(row: np.ndarray, start: int, step: int, min_run: int):
    """Nearest solid run (>= min_run cells) moving from `start` in direction `step`.
    Returns (index of the run cell nearest to the start, run length in cells) or None."""
    n = len(row)
    i = start
    while 0 <= i < n:
        if row[i] and all(0 <= i + s * step < n and row[i + s * step] for s in range(min_run)):
            # a real run starts here; measure its extent, tolerating a single empty cell
            j, length, misses = i, 0, 0
            while 0 <= j < n:
                if row[j]:
                    length += 1
                    misses = 0
                else:
                    misses += 1
                    if misses >= 2:
                        break
                j += step
            return i, length
        i += step  # empty cell or a speck shorter than min_run: keep looking outward
    return None


@dataclass
class RowScan:
    open_rows: list[int]
    left_edge: dict[int, int]
    right_edge: dict[int, int]
    left_len: dict[int, int]
    right_len: dict[int, int]
    centre_col: int
    half_cols: int


def scan_rows(grid: Grid, centre_u: float, opts: OpeningOptions, centre_half_m: float) -> RowScan:
    S = grid.solid
    nv, nu = S.shape
    c = int((centre_u - grid.u0) / grid.du)
    k = max(1, int(np.ceil(centre_half_m / grid.du)))
    lo, hi = max(0, c - k), min(nu, c + k + 1)
    scan = RowScan([], {}, {}, {}, {}, c, k)
    for r in range(nv):
        row = S[r]
        if row[lo:hi].any():
            continue  # the middle of the gap is not empty in this row
        left = _run_from(row, lo - 1, -1, opts.min_run_cols)
        right = _run_from(row, hi, 1, opts.min_run_cols)
        if left is None or right is None:
            continue
        scan.open_rows.append(r)
        scan.left_edge[r], scan.left_len[r] = left
        scan.right_edge[r], scan.right_len[r] = right
    return scan


def _row_runs(rows: list[int]) -> list[list[int]]:
    """Group rows into runs of consecutive indices, bridging a single missing row."""
    runs: list[list[int]] = []
    for r in rows:
        if runs and r - runs[-1][-1] <= 2:
            runs[-1].append(r)
        else:
            runs.append([r])
    return runs


def jamb_positions(grid: Grid, scan: RowScan, rows: list[int]):
    """Median jamb u over the given rows (right edge of the left solid run, left edge of the right one)."""
    rows = [r for r in rows if r in scan.left_edge]
    if len(rows) < 1:
        return None
    left = np.array([grid.u0 + (scan.left_edge[r] + 1) * grid.du for r in rows])
    right = np.array([grid.u0 + scan.right_edge[r] * grid.du for r in rows])
    return rows, left, right


# ---------- one candidate, one point cloud ----------


def _centre_solid_fraction(S: np.ndarray, r: int, c: int, k: int) -> float:
    return float(S[r, max(0, c - k): c + k + 1].mean())


def analyse_candidate(cand: Candidate, wall: WallInput, points: np.ndarray, floor: PlaneFit, v_hi: float,
                      opts: OpeningOptions, min_points: int, rows_hint: list[int] | None = None):
    """Measure one candidate in one cloud. Returns (measurement dict | None, reject reason | None, grid)."""
    margin = opts.side_window_m + 0.3
    u_lo, u_hi = min(cand.u0, cand.u1) - margin, max(cand.u0, cand.u1) + margin
    grid = build_grid(points, floor, wall, u_lo, u_hi, v_hi, opts, min_points)
    centre_u = 0.5 * (cand.u0 + cand.u1)
    scan = scan_rows(grid, centre_u, opts, min(opts.centre_half_m, 0.25 * cand.raw_width_m))
    nv = grid.solid.shape[0]
    if rows_hint is not None:  # subset measurement: reuse the full-cloud vertical decision
        got = jamb_positions(grid, scan, [r for r in rows_hint if r in scan.left_edge])
        if got is None or len(got[0]) < opts.jamb_min_rows:
            return None, "too few open rows in this subset", grid
        rows, left, right = got
        return {"left": float(np.median(left)), "right": float(np.median(right)), "rows": rows}, None, grid
    if not scan.open_rows:
        return None, "the wall is present across the gap in 3D, or there is no wall support on both sides", grid
    runs = _row_runs(scan.open_rows)
    run = max(runs, key=lambda r: (len(r), -r[0]))
    if len(run) < 2:
        return None, "no vertical opening band (fewer than 2 open rows)", grid
    rows = list(range(run[0], run[-1] + 1))
    got = jamb_positions(grid, scan, [r for r in rows if r in scan.left_edge])
    if got is None or len(got[0]) < opts.jamb_min_rows:
        return None, f"jamb evidence in fewer than {opts.jamb_min_rows} rows", grid
    used, left, right = got
    jl, jr = float(np.median(left)), float(np.median(right))
    tol = opts.jamb_row_tolerance_m
    cons_l = float(np.mean(np.abs(left - jl) <= tol))
    cons_r = float(np.mean(np.abs(right - jr) <= tol))
    if min(cons_l, cons_r) < opts.jamb_min_consistency:
        return None, f"jamb position inconsistent across rows ({min(cons_l, cons_r):.0%} agree)", grid
    sigma_l = 1.4826 * float(np.median(np.abs(left - jl)))
    sigma_r = 1.4826 * float(np.median(np.abs(right - jr)))

    dv = grid.dv
    start_row, end_row = run[0], run[-1]
    start_v = grid.v0 + start_row * dv
    top_v = grid.v0 + (end_row + 1) * dv
    c, k = scan.centre_col, scan.half_cols
    S = grid.solid
    # head: solid wall above the open band at the centre for >= head_min_rows rows
    # (one transitional row is allowed: the head edge usually falls inside a 10 cm row)
    r = end_row + 1
    transitional = 0
    if r < nv and _centre_solid_fraction(S, r, c, k) < 0.6:
        transitional = 1
        r += 1
    head_rows = 0
    while r < nv and _centre_solid_fraction(S, r, c, k) >= 0.6:
        head_rows += 1
        r += 1
    head_observed = head_rows >= opts.head_min_rows
    head_v = (top_v + 0.5 * dv * transitional) if head_observed else None
    # support below the band at the centre (needed for a window)
    below = 0
    for r in range(start_row - 1, -1, -1):
        if _centre_solid_fraction(S, r, c, k) >= 0.6:
            below += 1
        else:
            break
    below_needed = int(np.ceil(opts.below_support_fraction * start_row)) if start_row > 0 else 0
    # shape and blockage inside the refined opening rectangle
    i0 = max(0, int(np.floor((jl - grid.u0) / grid.du)))
    i1 = min(grid.occupied.shape[1], int(np.ceil((jr - grid.u0) / grid.du)))
    rect_occ = grid.occupied[start_row:end_row + 1, i0:i1]
    rect_slab = grid.slab[start_row:end_row + 1, i0:i1]
    empty_fraction = float(1.0 - rect_occ.mean()) if rect_occ.size else 0.0
    blockage = float(rect_slab.mean()) if rect_slab.size else 0.0
    # observability: wall support left and right of the opening
    wl = int(opts.side_window_m / grid.du)
    il, ir = int((jl - grid.u0) / grid.du), int(np.ceil((jr - grid.u0) / grid.du))
    left_fill = float(S[:, max(0, il - wl):il].mean()) if il > 0 else 0.0
    right_fill = float(S[:, ir:ir + wl].mean()) if ir < S.shape[1] else 0.0
    left_support = float(np.median([scan.left_len[r] for r in used]) * grid.du)
    right_support = float(np.median([scan.right_len[r] for r in used]) * grid.du)
    return {
        "left": jl, "right": jr, "sigma_left": sigma_l, "sigma_right": sigma_r, "rows": used,
        "consistency_left": cons_l, "consistency_right": cons_r,
        "band_rows": (start_row, end_row), "start_v": start_v, "top_v": top_v,
        "head_observed": head_observed, "head_v": head_v, "below_rows": below, "below_needed": below_needed,
        "empty_fraction": empty_fraction, "blockage": blockage,
        "left_fill": left_fill, "right_fill": right_fill,
        "left_support_m": left_support, "right_support_m": right_support,
    }, None, grid


# ---------- decisions ----------


def _classify(m: dict, width: float, opts: OpeningOptions) -> tuple[str, str, float | None, float | None, list[str]]:
    """(type, type_quality, height_m, sill_height_m, notes) from the vertical profile."""
    notes = []
    start_v, head_v = m["start_v"], m["head_v"]
    reaches_floor = start_v <= opts.door_sill_max_m
    band = (head_v if head_v is not None else m["top_v"]) - start_v
    if reaches_floor:
        height = head_v
        if width > opts.door_max_width_m:
            notes.append(f"wider than {opts.door_max_width_m} m: reported as a generic opening")
            return "opening", "moderate", height, 0.0, notes
        if head_v is not None and head_v < opts.door_min_height_m:
            notes.append("open from the floor but lower than a door")
            return "opening", "weak", height, 0.0, notes
        return "door", "strong" if head_v is not None else "moderate", height, 0.0, notes
    solid_below = m["below_rows"] >= max(m["below_needed"], 1)
    if start_v >= opts.window_min_sill_m and solid_below and band >= opts.window_min_band_m:
        height = (head_v - start_v) if head_v is not None else None
        return "window", "strong" if head_v is not None else "moderate", height, start_v, notes
    notes.append("sill/support profile does not clearly match a door or a window")
    return "opening", "weak", None, start_v if solid_below else None, notes


def _observability(m: dict, opts: OpeningOptions) -> str:
    side = min(m["left_support_m"], m["right_support_m"])
    fill = min(m["left_fill"], m["right_fill"])
    if side >= opts.side_strong_m and fill >= opts.fill_strong:
        return "strong"
    if side >= opts.side_moderate_m and fill >= opts.fill_moderate:
        return "moderate"
    return "weak"


def _stability(widths: list[float], full_width: float) -> tuple[float | None, int]:
    """RMS deviation of the subset widths from the full-cloud width. Unlike a plain standard deviation this also
    counts a systematic offset between the subsets and the full measurement."""
    if len(widths) >= 2:
        return float(np.sqrt(np.mean((np.asarray(widths) - full_width) ** 2))), len(widths)
    return None, len(widths)


# ---------- main entry point ----------


def detect_openings(
    points: np.ndarray,
    floor: PlaneFit,
    walls: list[WallInput],
    topo: TopologyResult | None = None,
    subset_clouds: list[np.ndarray] | None = None,
    opts: OpeningOptions | None = None,
    ceiling_height_m: float | None = None,
) -> OpeningResult:
    opts = opts or OpeningOptions()
    by_id = {w.id: w for w in walls}
    v_hi = opts.v_max_m if ceiling_height_m is None else min(opts.v_max_m, ceiling_height_m - opts.ceiling_clearance_m)
    cands, gaps_seen = generate_candidates(walls, topo, opts)
    accepted: list[Opening] = []
    rejected: list[dict] = []
    grids: dict = {}

    def reject(c: Candidate, reason: str, stage: str, extra: dict | None = None):
        rejected.append({"candidate_id": c.id, "wall_id": c.wall_id, "source": c.source, "wall_evidence": c.evidence,
                         "raw_gap_width_m": c.raw_width_m, "raw_start": list(c.start), "raw_end": list(c.end),
                         "room_ids": c.room_ids, "stage": stage, "reason": reason, **(extra or {})})

    for c in cands:
        wall = by_id[c.wall_id]
        if c.raw_width_m < opts.min_width_m * 0.5:
            reject(c, "gap narrower than half the minimum opening width", "size")
            continue
        m, reason, grid = analyse_candidate(c, wall, points, floor, v_hi, opts, opts.cell_min_points)
        grids[c.id] = grid
        if m is None:
            reject(c, reason, "3d_verification")
            continue
        width = m["right"] - m["left"]
        if width < opts.min_width_m:
            reject(c, f"refined width {width:.2f} m is below {opts.min_width_m} m", "size", {"refined_width_m": width})
            continue
        if width > opts.max_width_m:
            reject(c, f"refined width {width:.2f} m is above {opts.max_width_m} m", "size", {"refined_width_m": width})
            continue
        obs = _observability(m, opts)
        if min(m["left_support_m"], m["right_support_m"]) < opts.side_moderate_m:
            reject(c, "only one jamb (or neither) is backed by enough wall: a wall end, not an opening", "wall_end",
                   {"left_support_m": m["left_support_m"], "right_support_m": m["right_support_m"]})
            continue
        if obs == "weak":
            reject(c, "the wall around the gap was not scanned well enough to call it an opening", "observability",
                   {"left_fill": m["left_fill"], "right_fill": m["right_fill"]})
            continue
        if m["empty_fraction"] < opts.empty_reject:
            reject(c, f"opening region only {m['empty_fraction']:.0%} empty: not a clean vertical opening", "shape")
            continue
        if m["blockage"] >= opts.blockage_reject:
            reject(c, f"{m['blockage']:.0%} of the gap has points just in front of the wall: furniture/occlusion explains it better", "occlusion")
            continue
        otype, type_q, height, sill, notes = _classify(m, width, opts)

        # frame-subset stability: re-measure with the same vertical rows
        widths = []
        if subset_clouds:
            for sc in subset_clouds:
                sm, _, _ = analyse_candidate(c, wall, sc, floor, v_hi, opts, opts.subset_cell_min_points, rows_hint=m["rows"])
                if sm is not None:
                    widths.append(sm["right"] - sm["left"])
        std, n_sub = _stability(widths, width)
        if std is not None and std >= opts.stability_reject_m:
            reject(c, f"width changes by {std:.2f} m (std) between frame subsets", "stability", {"subset_widths_m": widths})
            continue

        # width uncertainty: jamb scatter, grid resolution, wall orientation/position, subset spread
        res = opts.du_m / np.sqrt(12)
        sl = np.hypot(m["sigma_left"], res)
        sr = np.hypot(m["sigma_right"], res)
        pos_term = opts.wall_position_weight * wall.position_uncertainty_m
        subset_term = std if std is not None else opts.fallback_subset_std_m
        half = 1.96 * float(np.sqrt(sl ** 2 + sr ** 2 + pos_term ** 2 + subset_term ** 2))
        interval = (max(0.0, width - half), width + half)

        existence = _existence(m, obs, n_sub, len(subset_clouds or []), opts)
        width_q = "strong" if half <= 0.06 else ("moderate" if half <= 0.15 else "weak")
        status = "accepted" if existence in ("strong", "moderate") else "low_confidence"
        accepted.append(Opening(
            id="", candidate=c, status=status, type=otype, room_ids=list(c.room_ids), connects=[],
            width_m=float(width), width_interval_m=interval, height_m=None if height is None else float(height),
            sill_height_m=None if sill is None else float(sill),
            left_jamb=_jamb(wall, m["left"], m["sigma_left"], opts), right_jamb=_jamb(wall, m["right"], m["sigma_right"], opts),
            observability=obs, existence_quality=existence, type_quality=type_q, width_quality=width_q,
            metrics={
                "empty_fraction": m["empty_fraction"], "blockage_fraction": m["blockage"],
                "left_support_m": m["left_support_m"], "right_support_m": m["right_support_m"],
                "left_side_fill": m["left_fill"], "right_side_fill": m["right_fill"],
                "jamb_row_consistency": [m["consistency_left"], m["consistency_right"]],
                "rows_used": len(m["rows"]), "head_observed": m["head_observed"],
                "subset_widths_m": widths, "subset_rms_deviation_m": std, "subsets_used": n_sub,
                "refined_vs_raw_width_m": float(width - c.raw_width_m), "notes": notes,
            },
            grid=grid,
        ))
    accepted.sort(key=lambda o: (o.candidate.wall_id, o.left_jamb["u_m"]))
    for i, o in enumerate(accepted, 1):
        o.id = f"opening_{i:03d}"
    connectivity = _connectivity(accepted, topo)
    for o in accepted:
        if o.status == "accepted":
            o.connects = [r for r in o.room_ids] if len(o.room_ids) == 2 else (o.room_ids + ["unmodelled"] if o.room_ids else ["unmodelled"])
    return OpeningResult(cands, accepted, rejected, connectivity, gaps_seen, opts, grids=grids)


def _jamb(wall: WallInput, u: float, sigma: float, opts: OpeningOptions) -> dict:
    p = wall.point(u)
    return {"u_m": float(u), "x": float(p[0]), "z": float(p[1]), "sigma_m": float(np.hypot(sigma, opts.du_m / np.sqrt(12)))}


def _existence(m: dict, obs: str, n_found: int, n_expected: int, opts: OpeningOptions) -> str:
    """Does the opening exist? Rated from observability, clean emptiness, no occlusion, jamb evidence and whether
    independent frame subsets each find it. How well those subsets AGREE ON THE WIDTH is a width question and
    only feeds the width interval and width quality, never existence."""
    cons = min(m["consistency_left"], m["consistency_right"])
    all_found = n_expected > 0 and n_found == n_expected
    enough = n_expected == 0 or n_found >= min(opts.min_subsets, n_expected)
    if (obs == "strong" and m["empty_fraction"] >= opts.empty_strong and m["blockage"] < opts.blockage_strong
            and all_found and cons >= 0.8):
        return "strong"  # (never "strong" without independent subset confirmation)
    if (obs in ("strong", "moderate") and m["empty_fraction"] >= opts.empty_moderate and m["blockage"] < opts.blockage_moderate
            and enough and cons >= opts.jamb_min_consistency):
        return "moderate"
    return "weak"


def _connectivity(openings: list[Opening], topo: TopologyResult | None) -> dict:
    rooms = {}
    if topo is not None:
        for r in topo.rooms:
            rooms[r.id] = {"adjacent_room_ids": sorted(a["room_id"] for a in r.adjacent), "connected_room_ids": [],
                           "opening_ids": [], "opens_to_unmodelled": []}
    for o in openings:
        if o.status != "accepted":
            continue
        rid = o.room_ids
        for r in rid:
            if r in rooms:
                rooms[r]["opening_ids"].append(o.id)
        if len(rid) == 2 and all(r in rooms for r in rid):
            a, b = rid
            if b not in rooms[a]["connected_room_ids"]:
                rooms[a]["connected_room_ids"].append(b)
            if a not in rooms[b]["connected_room_ids"]:
                rooms[b]["connected_room_ids"].append(a)
        else:
            for r in rid:
                if r in rooms:
                    rooms[r]["opens_to_unmodelled"].append(o.id)
    for v in rooms.values():
        v["connected_room_ids"].sort()
    return rooms


# ---------- serialisation ----------


def _round(v, nd=4):
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x, nd) for x in v]
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def candidates_to_dict(res: OpeningResult) -> dict:
    status = {o.candidate.id: o.status for o in res.openings}
    rej = {r["candidate_id"]: r for r in res.rejected}
    out = []
    for c in res.candidates:
        d = {"id": c.id, "wall_id": c.wall_id, "source": c.source, "wall_evidence": c.evidence,
             "raw_gap_width_m": c.raw_width_m, "raw_start": list(c.start), "raw_end": list(c.end), "room_ids": c.room_ids}
        if c.id in status:
            d["outcome"] = status[c.id]
        else:
            d["outcome"] = "rejected"
            d["stage"] = rej[c.id]["stage"]
            d["reason"] = rej[c.id]["reason"]
        out.append(d)
    return _round({"candidates": out, "rejected_details": res.rejected})


def openings_to_dict(res: OpeningResult) -> dict:
    types = {t: sum(1 for o in res.accepted if o.type == t) for t in ("door", "window", "opening")}
    return _round({
        "wall_gaps_considered": res.wall_gaps_considered,
        "candidate_openings": len(res.candidates),
        "accepted_openings": len(res.accepted),
        "low_confidence_openings": len(res.low_confidence),
        "rejected_candidates": len(res.rejected),
        "counts_by_type": types,
        "openings": [o.to_dict() for o in res.accepted],
        "low_confidence": [o.to_dict() for o in res.low_confidence],
        "connectivity": res.connectivity,
        "warnings": res.warnings,
        "options": asdict(res.options),
    })
