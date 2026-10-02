"""Room topology from structural wall candidates (NumPy only, deterministic).

Everything is in metric top-down X-Z coordinates (+X right, +Z up, polygons counter-clockwise).

Plain-English summary
---------------------
1. Walls (Ticket 5) become 2D lines with an observed extent, observed intervals and a position
   uncertainty. Near-duplicate parallel walls are suppressed (the stronger one is kept).
2. Corners: for every pair of sufficiently non-parallel walls the infinite lines are intersected. The
   point is a corner only if each wall needs at most a small extension to reach it, limited by a maximum
   and by the walls' position uncertainty. Nothing is extended further.
3. Nodes (merged corners) and wall pieces between consecutive nodes form a planar graph. A wall piece
   that spans an observed gap (a possible door) stays one edge: collinear gaps are never closed or
   erased, only recorded.
4. Rooms are the bounded faces of that graph (half-edge face walking; no cycle enumeration), kept only
   if they pass sanity filters. Weak walls take part only in a second pass and may close a face only if
   its other edges are strong/moderate and no accepted room is subdivided by them.
5. Each room gets area, perimeter, wall lengths, (rectangular) length/width from opposite wall lines,
   deterministic-perturbation intervals, an evidence-based quality label, a spatially assigned ceiling
   level and geometric adjacency.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from spatialforge.lidar.planes import solid_cells

TIER_RANK = {"strong": 0, "moderate": 1, "weak": 2}


@dataclass(frozen=True)
class RoomOptions:
    # Corners. A wall's observation can stop short of its true end because of occlusion or detection
    # thresholds. Measured on the three captures, the nearest non-parallel wall pairs miss by 0.43-0.49 m and
    # the next ones by 0.60 m or more, so 0.5 m reaches the blurred corners but not metre-scale gaps.
    # Strong walls may extend by the full cap, moderate 80%, weak 50% (plus the other wall's position
    # uncertainty / sin(angle), which shifts the intersection point along this wall).
    max_corner_extension_m: float = 0.50
    tier_extension_factor: dict = field(default_factory=lambda: {"strong": 1.0, "moderate": 0.8, "weak": 0.5})
    min_corner_angle_deg: float = 25.0  # nearer-parallel walls never intersect
    node_merge_m: float = 0.10

    # Walls this close, nearly parallel and overlapping are one structural wall (keep the stronger).
    parallel_merge_m: float = 0.35
    parallel_angle_deg: float = 6.0
    parallel_overlap: float = 0.5
    min_edge_support: float = 0.30  # an edge needs this share of observed wall (rest: gap/extension)

    # Room sanity filters
    min_room_area_m2: float = 1.2
    min_edge_length_m: float = 0.4
    min_thickness_m: float = 0.45  # 2 * area / perimeter: rejects slivers
    # A face that encloses this much structural wall inside itself is an outline of several spaces (a
    # property footprint), not a room.
    footprint_interior_fraction: float = 0.25  # interior wall length / perimeter
    footprint_interior_clearance_m: float = 0.35
    footprint_interior_min_wall_m: float = 1.0
    max_inferred_fraction: float = 0.30  # inferred extension / perimeter
    min_observed_fraction: float = 0.50  # observed wall / perimeter
    collinear_merge_deg: float = 3.0
    weak_max_edges: int = 1
    weak_max_fraction: float = 0.40

    # Dimensions
    rect_angle_tol_deg: float = 10.0
    rect_min_rectangularity: float = 0.90
    perturbation_patterns: int = 64
    weak_uncertainty_factor: float = 1.5

    # Ceiling association
    ceiling_cell_m: float = 0.10
    min_ceiling_coverage: float = 0.25
    ceiling_ambiguity_ratio: float = 0.60
    ceiling_height_gap_m: float = 0.15

    # Adjacency (geometric only; says nothing about doors)
    adjacency_distance_m: float = 0.40
    adjacency_overlap_m: float = 0.50
    adjacency_angle_deg: float = 6.0


# ---------- input model ----------


@dataclass
class WallInput:
    id: str
    evidence: str  # strong | moderate | weak
    centre: np.ndarray  # a point on the line (x, z)
    direction: np.ndarray  # unit vector along the line
    position_uncertainty_m: float
    segments: list[tuple[tuple[float, float], tuple[float, float]]]  # observed (start, end)
    gaps: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.centre = np.asarray(self.centre, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)
        self.direction = self.direction / np.linalg.norm(self.direction)
        ts = [self.t(p) for seg in self.segments for p in seg]
        self.t0, self.t1 = float(min(ts)), float(max(ts))
        self.intervals = sorted((min(self.t(a), self.t(b)), max(self.t(a), self.t(b))) for a, b in self.segments)

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    @property
    def offset(self) -> float:
        return float(self.centre @ self.normal)

    @property
    def observed_length(self) -> float:
        return float(sum(b - a for a, b in self.intervals))

    def t(self, p) -> float:
        return float((np.asarray(p, dtype=float) - self.centre) @ self.direction)

    def point(self, t: float) -> np.ndarray:
        return self.centre + t * self.direction

    @staticmethod
    def from_dict(d: dict) -> "WallInput":
        """Build from a Ticket 5 wall record (walls.json)."""
        a, _, c, dd = d["plane"]
        n = np.array([a, c], dtype=float)
        n /= np.linalg.norm(n)
        centre = n * (-dd)
        direction = np.array([-n[1], n[0]])
        return WallInput(
            id=d["id"], evidence=d["evidence"], centre=centre, direction=direction,
            position_uncertainty_m=float(d["position_uncertainty_m"]),
            segments=[(tuple(s["start"]), tuple(s["end"])) for s in d["segments"]],
            gaps=list(d.get("gaps", [])),
        )


def _angle_between(d1: np.ndarray, d2: np.ndarray) -> float:
    """Angle between two undirected directions, 0..90 degrees."""
    c = abs(float(d1 @ d2))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


# ---------- step 1: duplicate parallel walls ----------


def suppress_parallel_duplicates(walls: list[WallInput], opts: RoomOptions) -> tuple[list[WallInput], list[dict]]:
    """Drop the weaker of two near-identical parallel walls (stronger tier first, then more observed length)."""
    order = sorted(walls, key=lambda w: (TIER_RANK[w.evidence], -w.observed_length, w.id))
    kept: list[WallInput] = []
    dropped: list[dict] = []
    for w in order:
        twin = None
        for k in kept:
            if _angle_between(w.direction, k.direction) > opts.parallel_angle_deg:
                continue
            ends = np.array([w.point(w.t0), w.point(w.t1)])
            if np.abs((ends - k.centre) @ k.normal).max() > opts.parallel_merge_m:
                continue
            a0, a1 = sorted(k.t(p) for p in ends)
            overlap = min(a1, k.t1) - max(a0, k.t0)
            if overlap / max(min(a1 - a0, k.t1 - k.t0), 1e-9) >= opts.parallel_overlap:
                twin = k
                break
        if twin is None:
            kept.append(w)
        else:
            dropped.append({"wall_id": w.id, "duplicate_of": twin.id})
    kept.sort(key=lambda w: w.id)
    return kept, dropped


# ---------- step 2-3: corners, nodes, edges ----------


@dataclass
class Corner:
    point: np.ndarray
    wall_a: str
    wall_b: str
    ext_a: float
    ext_b: float
    angle_deg: float


def find_corners(walls: list[WallInput], opts: RoomOptions) -> list[Corner]:
    corners = []
    for wa, wb in itertools.combinations(walls, 2):
        theta = _angle_between(wa.direction, wb.direction)
        if theta < opts.min_corner_angle_deg:
            continue
        cross = wa.direction[0] * wb.direction[1] - wa.direction[1] * wb.direction[0]
        diff = wb.centre - wa.centre
        t_a = (diff[0] * wb.direction[1] - diff[1] * wb.direction[0]) / cross
        p = wa.centre + t_a * wa.direction
        t_b = wb.t(p)
        ext_a = max(0.0, wa.t0 - t_a, t_a - wa.t1)
        ext_b = max(0.0, wb.t0 - t_b, t_b - wb.t1)
        sin = np.sin(np.radians(theta))
        tol_a = min(opts.max_corner_extension_m, opts.max_corner_extension_m * opts.tier_extension_factor[wa.evidence]
                    + wb.position_uncertainty_m / sin)
        tol_b = min(opts.max_corner_extension_m, opts.max_corner_extension_m * opts.tier_extension_factor[wb.evidence]
                    + wa.position_uncertainty_m / sin)
        if ext_a <= tol_a and ext_b <= tol_b:
            corners.append(Corner(p, wa.id, wb.id, ext_a, ext_b, theta))
    corners.sort(key=lambda c: (round(c.point[0], 3), round(c.point[1], 3), c.wall_a, c.wall_b))
    return corners


@dataclass
class Node:
    id: int
    point: np.ndarray
    walls: set = field(default_factory=set)
    corner_indices: list = field(default_factory=list)
    extensions: dict = field(default_factory=dict)  # wall id -> metres the wall must be extended to reach the node


@dataclass
class Edge:
    id: int
    a: int  # node ids
    b: int
    wall_id: str
    evidence: str
    length_m: float
    observed_support_fraction: float
    inferred_extension_m: float  # part of the edge beyond the wall's observed extent
    gap_m: float  # unobserved stretch inside the wall's observed extent (possible opening)
    position_uncertainty_m: float


@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
    corners: list[Corner]
    dropped_edges: list[dict]


def build_graph(walls: list[WallInput], opts: RoomOptions) -> Graph:
    by_id = {w.id: w for w in walls}
    corners = find_corners(walls, opts)
    nodes: list[Node] = []
    members: list[list[np.ndarray]] = []
    for ci, c in enumerate(corners):
        for n, pts in zip(nodes, members):
            if np.linalg.norm(c.point - np.mean(pts, axis=0)) <= opts.node_merge_m:
                pts.append(c.point)
                n.walls.update((c.wall_a, c.wall_b))
                n.corner_indices.append(ci)
                n.point = np.mean(pts, axis=0)
                break
        else:
            nodes.append(Node(len(nodes), c.point.copy(), {c.wall_a, c.wall_b}, [ci]))
            members.append([c.point])

    for n in nodes:
        n.extensions = {wid: float(max(0.0, by_id[wid].t0 - by_id[wid].t(n.point), by_id[wid].t(n.point) - by_id[wid].t1))
                        for wid in sorted(n.walls)}

    edges: list[Edge] = []
    dropped: list[dict] = []
    for w in walls:
        attached =sorted(((w.t(n.point), n.id) for n in nodes if w.id in n.walls))
        uniq = []
        for t, nid in attached:
            if not uniq or nid != uniq[-1][1]:
                uniq.append((t, nid))
        for (ta, na), (tb, nb) in zip(uniq, uniq[1:]):
            length = tb - ta
            if length < opts.node_merge_m:
                continue
            obs = sum(max(0.0, min(tb, b) - max(ta, a)) for a, b in w.intervals)
            inside = max(0.0, min(tb, w.t1) - max(ta, w.t0))
            support = obs / length
            record = Edge(len(edges), na, nb, w.id, w.evidence, float(length), float(support),
                          float(length - inside), float(max(0.0, inside - obs)), w.position_uncertainty_m)
            if support < opts.min_edge_support:
                dropped.append({"wall_id": w.id, "length_m": float(length), "observed_support_fraction": float(support),
                                "reason": f"only {support:.0%} of the span is observed wall"})
                continue
            edges.append(record)
    return planarize(Graph(nodes, edges, corners, dropped), by_id)


def _seg_intersection(p, p2, q, q2):
    r, s = p2 - p, q2 - q
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
    eps = 1e-6
    if eps < t < 1 - eps and eps < u < 1 - eps:
        return t, u
    return None


def planarize(g: Graph, walls_by_id: dict) -> Graph:
    """Split edges at proper crossings that are not already nodes (safety net: keeps the graph planar)."""
    changed = True
    while changed:
        changed = False
        for e1, e2 in itertools.combinations(g.edges, 2):
            if {e1.a, e1.b} & {e2.a, e2.b}:
                continue
            p, p2 = g.nodes[e1.a].point, g.nodes[e1.b].point
            q, q2 = g.nodes[e2.a].point, g.nodes[e2.b].point
            hit = _seg_intersection(p, p2, q, q2)
            if hit is None:
                continue
            pt = p + hit[0] * (p2 - p)
            node = Node(len(g.nodes), pt, {e1.wall_id, e2.wall_id}, [], {e1.wall_id: 0.0, e2.wall_id: 0.0})
            g.nodes.append(node)
            g.edges.remove(e1)
            g.edges.remove(e2)
            for e, t in ((e1, hit[0]), (e2, hit[1])):
                for a, b, f in ((e.a, node.id, t), (node.id, e.b, 1 - t)):
                    g.edges.append(Edge(
                        -1, a, b, e.wall_id, e.evidence, e.length_m * f, e.observed_support_fraction,
                        e.inferred_extension_m * f, e.gap_m * f, e.position_uncertainty_m,
                    ))
            changed = True
            break
    for i, e in enumerate(g.edges):
        e.id = i
    return g


# ---------- step 4: planar faces (half-edge walking) ----------


def _prune_dangling(edges: list[Edge]) -> list[Edge]:
    edges = list(edges)
    while True:
        deg: dict[int, int] = {}
        for e in edges:
            deg[e.a] = deg.get(e.a, 0) + 1
            deg[e.b] = deg.get(e.b, 0) + 1
        keep = [e for e in edges if deg[e.a] > 1 and deg[e.b] > 1]
        if len(keep) == len(edges):
            return keep
        edges = keep


def signed_area(poly: np.ndarray) -> float:
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def extract_faces(g: Graph) -> tuple[list[dict], list[dict]]:
    """All faces of the planar graph: (bounded faces with CCW order, outer-face walks).

    Each face is {'nodes': [...], 'edges': [Edge...], 'area': signed area}.
    """
    edges = _prune_dangling(g.edges)
    if not edges:
        return [], []
    out: dict[int, list[tuple[float, int, Edge]]] = {}
    for e in edges:
        for u, v in ((e.a, e.b), (e.b, e.a)):
            d = g.nodes[v].point - g.nodes[u].point
            out.setdefault(u, []).append((float(np.arctan2(d[1], d[0])), v, e))
    for u in out:
        out[u].sort(key=lambda t: (t[0], t[1]))
    visited: set[tuple[int, int, int]] = set()
    bounded, outer = [], []
    for u0 in sorted(out):
        for _, v0, e0 in out[u0]:
            if (u0, v0, e0.id) in visited:
                continue
            u, v, e = u0, v0, e0
            nodes, face_edges = [], []
            while (u, v, e.id) not in visited:
                visited.add((u, v, e.id))
                nodes.append(u)
                face_edges.append(e)
                ring = out[v]
                idx = next(i for i, (_, w, ee) in enumerate(ring) if w == u and ee.id == e.id)
                _, w, ee = ring[idx - 1]  # next clockwise from the way we came: keeps the face on the left
                u, v, e = v, w, ee
            area = signed_area(np.array([g.nodes[n].point for n in nodes]))
            (bounded if area > 1e-9 else outer).append({"nodes": nodes, "edges": face_edges, "area": area})
    return bounded, outer


# ---------- polygon utilities ----------


def point_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd test."""
    x, z = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        crosses = (z1 > z) != (z2 > z)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
        inside ^= crosses & (x < x_at)
    return inside


def is_simple(poly: np.ndarray) -> bool:
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if _seg_intersection(a, b, poly[j], poly[(j + 1) % n]) is not None:
                return False
    return True


def perimeter_of(poly: np.ndarray) -> float:
    return float(np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1).sum())


def clean_polygon(nodes: list[int], edges: list[Edge], g: Graph, opts: RoomOptions):
    """Drop nodes that do not turn the boundary (T-junction points on a straight run), merging their edges."""
    pts = [g.nodes[n].point for n in nodes]
    n = len(pts)
    groups = [[e] for e in edges]  # edge i runs from nodes[i] to nodes[i+1]
    keep = []
    for i in range(n):
        prev, cur, nxt = pts[i - 1], pts[i], pts[(i + 1) % n]
        a, b = cur - prev, nxt - cur
        turn = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
        keep.append(abs(turn) >= opts.collinear_merge_deg)
    if sum(keep) < 3:
        return None
    start = next(i for i in range(n) if keep[i])
    order = list(range(start, n)) + list(range(0, start))
    verts, merged = [], []
    current: list[Edge] = []
    for i in order:
        if keep[i] and current:
            merged.append(current)
            current = []
        if keep[i]:
            verts.append(pts[i])
        current.extend(groups[i])
    merged.append(current)
    if len(merged) != len(verts):
        return None
    return np.array(verts), merged


# ---------- results ----------


@dataclass
class PolygonEdge:
    wall_ids: list[str]
    length_m: float
    tier: str  # weakest tier among the walls it uses
    observed_support_fraction: float
    inferred_extension_m: float
    gap_m: float
    uncertainty_m: float


@dataclass
class Room:
    id: str
    polygon: np.ndarray
    edges: list[PolygonEdge]
    wall_ids: list[str]
    area_m2: float
    perimeter_m: float
    wall_lengths_m: list[float]
    length_m: float | None
    width_m: float | None
    dimension_method: str | None
    area_interval_m2: tuple[float, float]
    wall_length_intervals_m: list[tuple[float, float]]
    length_interval_m: tuple[float, float] | None
    width_interval_m: tuple[float, float] | None
    strong_fraction: float
    moderate_fraction: float
    weak_fraction: float
    inferred_extension_total_m: float
    observed_fraction: float
    topology_quality: str
    ceiling: dict = field(default_factory=lambda: {"ceiling_observed": False})
    adjacent: list[dict] = field(default_factory=list)
    uses_weak_walls: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["polygon"] = [{"x": float(x), "z": float(z)} for x, z in self.polygon]
        d["adjacent_room_ids"] = [a["room_id"] for a in self.adjacent]
        return d


@dataclass
class TopologyResult:
    graph: Graph
    rooms: list[Room]
    rejected_faces: list[dict]
    outer_boundary: dict | None
    suppressed_walls: list[dict]
    wall_count: int
    candidate_faces: int
    options: RoomOptions
    warnings: list[str] = field(default_factory=list)


# ---------- measurement ----------


def _edge_lines(edges: list[list[Edge]], walls_by_id: dict) -> list[WallInput]:
    """For each polygon edge the wall that supplies most of its length (its fitted line)."""
    lines = []
    for group in edges:
        weight: dict[str, float] = {}
        for e in group:
            weight[e.wall_id] = weight.get(e.wall_id, 0.0) + e.length_m
        lines.append(walls_by_id[max(sorted(weight), key=lambda k: weight[k])])
    return lines


def _vertices_from_lines(lines: list[WallInput], offsets: list[float], base: np.ndarray) -> np.ndarray:
    """Polygon vertices as intersections of consecutive (offset) wall lines; falls back to the original
    vertex shifted by the mean offset when two consecutive lines are nearly parallel."""
    n = len(lines)
    out = np.zeros((n, 2))
    for i in range(n):
        la, lb = lines[i - 1], lines[i]  # vertex i joins edge i-1 and edge i
        da, db = la.direction, lb.direction
        cross = da[0] * db[1] - da[1] * db[0]
        if abs(cross) < np.sin(np.radians(10)):
            out[i] = base[i] + 0.5 * (offsets[i - 1] * la.normal + offsets[i] * lb.normal)
            continue
        ca = la.centre + offsets[i - 1] * la.normal
        cb = lb.centre + offsets[i] * lb.normal
        diff = cb - ca
        t = (diff[0] * db[1] - diff[1] * db[0]) / cross
        out[i] = ca + t * da
    return out


def _dimensions(poly: np.ndarray, lines: list[WallInput], offsets: list[float], opts: RoomOptions):
    """(length, width, method) or (None, None, None) for irregular rooms."""
    n = len(poly)
    if n == 4:
        angs = [_angle_between(lines[i].direction, lines[(i + 1) % 4].direction) for i in range(4)]
        if all(abs(a - 90) <= opts.rect_angle_tol_deg for a in angs) and \
                _angle_between(lines[0].direction, lines[2].direction) <= opts.rect_angle_tol_deg and \
                _angle_between(lines[1].direction, lines[3].direction) <= opts.rect_angle_tol_deg:
            def gap(i, j):  # distance between two roughly parallel (offset) wall lines
                ci = lines[i].centre + offsets[i] * lines[i].normal
                cj = lines[j].centre + offsets[j] * lines[j].normal
                nrm = lines[i].normal
                mid = 0.5 * (ci + cj)
                return abs(float((cj - ci) @ nrm)) if np.isfinite(mid).all() else 0.0
            d02, d13 = gap(0, 2), gap(1, 3)
            return max(d02, d13), min(d02, d13), "opposite_wall_distances"
    # near-rectangular with small jogs: oriented bounding box along the dominant edge direction
    lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    k = int(np.argmax(lengths))
    d = (np.roll(poly, -1, axis=0)[k] - poly[k]) / max(lengths[k], 1e-9)
    nrm = np.array([-d[1], d[0]])
    u, v = poly @ d, poly @ nrm
    box = (u.max() - u.min()) * (v.max() - v.min())
    if box <= 0 or abs(signed_area(poly)) / box < opts.rect_min_rectangularity:
        return None, None, None
    for i in range(n):
        e = np.roll(poly, -1, axis=0)[i] - poly[i]
        if np.linalg.norm(e) > 1e-9 and min(_angle_between(e / np.linalg.norm(e), d),
                                            _angle_between(e / np.linalg.norm(e), nrm)) > opts.rect_angle_tol_deg:
            return None, None, None
    a, b = u.max() - u.min(), v.max() - v.min()
    return max(a, b), min(a, b), "oriented_bounding_box"


def _sign_patterns(m: int, count: int) -> list[list[int]]:
    if m <= 6:
        return [list(p) for p in itertools.product((-1, 1), repeat=m)]
    out = []
    for k in range(count):  # fixed hash: deterministic, no random number generator
        h = (k * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
        out.append([1 if (h >> (j % 31)) & 1 else -1 for j in range(m)])
    return out


def measure_intervals(poly, lines, edge_sigmas, opts: RoomOptions) -> dict:
    """Deterministic perturbation: shift every wall by +-sigma over a fixed set of sign patterns and
    record the extremes of area, wall lengths and (rectangular) dimensions."""
    n = len(poly)
    areas, lens, dims = [], [], []
    for signs in _sign_patterns(n, opts.perturbation_patterns):
        offsets = [s * sig for s, sig in zip(signs, edge_sigmas)]
        p = _vertices_from_lines(lines, offsets, poly)
        areas.append(abs(signed_area(p)))
        lens.append(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1))
        length, width, _ = _dimensions(p, lines, offsets, opts)
        dims.append((length, width))
    lens = np.array(lens)
    out = {
        "area": (float(min(areas)), float(max(areas))),
        "lengths": [(float(lens[:, i].min()), float(lens[:, i].max())) for i in range(n)],
    }
    if all(d[0] is not None for d in dims):
        out["length"] = (float(min(d[0] for d in dims)), float(max(d[0] for d in dims)))
        out["width"] = (float(min(d[1] for d in dims)), float(max(d[1] for d in dims)))
    return out


# ---------- room acceptance ----------


def _interior_wall_length(poly: np.ndarray, walls: list[WallInput], opts: RoomOptions) -> float:
    """Total length of strong/moderate observed wall segments lying inside the polygon, away from its boundary."""
    total = 0.0
    n = len(poly)
    for w in walls:
        if w.evidence == "weak":
            continue
        for a, b in w.segments:
            a, b = np.asarray(a, float), np.asarray(b, float)
            length = float(np.linalg.norm(b - a))
            if length < opts.footprint_interior_min_wall_m:
                continue
            mid = 0.5 * (a + b)
            if not point_in_polygon(np.array([mid, a, b]), poly).all():
                continue
            dist = min(
                _point_segment_distance(mid, poly[i], poly[(i + 1) % n]) for i in range(n)
            )
            if dist > opts.footprint_interior_clearance_m:
                total += length
    return total


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _evaluate_face(face: dict, g: Graph, walls_by_id: dict, opts: RoomOptions):
    """Return (room-without-id | None, reason | None)."""
    cleaned = clean_polygon(face["nodes"], face["edges"], g, opts)
    if cleaned is None:
        return None, "fewer than 3 boundary turns after removing straight junction points"
    poly, groups = cleaned
    area = abs(signed_area(poly))
    perim = perimeter_of(poly)
    lens = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    if len(poly) < 3 or len({e.wall_id for grp in groups for e in grp}) < 3:
        return None, "fewer than 3 distinct walls"
    if not is_simple(poly):
        return None, "self-intersecting polygon"
    if area < opts.min_room_area_m2:
        return None, f"area {area:.2f} m2 below {opts.min_room_area_m2}"
    interior = _interior_wall_length(poly, list(walls_by_id.values()), opts)
    if interior >= opts.footprint_interior_fraction * perim:
        return None, (f"encloses {interior:.1f} m of interior structural wall: an outline of several spaces "
                      f"(property footprint candidate), not a single room")
    if lens.min() < opts.min_edge_length_m:
        return None, f"edge of {lens.min():.2f} m shorter than {opts.min_edge_length_m}"
    thickness = 2 * area / perim
    if thickness < opts.min_thickness_m:
        return None, f"too thin (2A/P = {thickness:.2f} m)"
    total = sum(e.length_m for grp in groups for e in grp)
    inferred = sum(e.inferred_extension_m for grp in groups for e in grp)
    observed = sum(e.length_m * e.observed_support_fraction for grp in groups for e in grp)
    if inferred / total > opts.max_inferred_fraction:
        return None, f"{inferred / total:.0%} of the boundary is inferred extension"
    if observed / total < opts.min_observed_fraction:
        return None, f"only {observed / total:.0%} of the boundary is observed wall"
    by_tier = {t: sum(e.length_m for grp in groups for e in grp if e.evidence == t) / total for t in TIER_RANK}
    edges_out, lines = [], _edge_lines(groups, walls_by_id)
    for grp, ln, length in zip(groups, lines, lens):
        glen = sum(e.length_m for e in grp)
        edges_out.append(PolygonEdge(
            wall_ids=sorted({e.wall_id for e in grp}), length_m=float(length),
            tier=max((e.evidence for e in grp), key=lambda t: TIER_RANK[t]),
            observed_support_fraction=float(sum(e.length_m * e.observed_support_fraction for e in grp) / glen),
            inferred_extension_m=float(sum(e.inferred_extension_m for e in grp)),
            gap_m=float(sum(e.gap_m for e in grp)), uncertainty_m=float(max(e.position_uncertainty_m for e in grp)),
        ))
    return {
        "poly": poly, "edges": edges_out, "lines": lines, "area": area, "perimeter": perim, "lens": lens,
        "by_tier": by_tier, "inferred": inferred, "observed": observed / total, "total": total,
    }, None


def _quality(info: dict, opts: RoomOptions) -> str:
    inferred_fraction = info["inferred"] / info["perimeter"]
    if info["by_tier"]["weak"] > 0.25 or inferred_fraction > 0.2 or info["observed"] < 0.6:
        return "weak"
    if info["by_tier"]["strong"] >= 0.6 and info["by_tier"]["weak"] == 0 and inferred_fraction <= 0.1:
        return "strong"
    return "moderate"


def _make_room(info: dict, opts: RoomOptions) -> Room:
    poly, edges, lines = info["poly"], info["edges"], info["lines"]
    sigmas = []
    for e, ln in zip(edges, lines):
        s = ln.position_uncertainty_m * (opts.weak_uncertainty_factor if e.tier == "weak" else 1.0)
        sigmas.append(s + 0.5 * e.inferred_extension_m)  # an uncertain corner moves the wall's end
    length, width, method = _dimensions(poly, lines, [0.0] * len(lines), opts)
    ivals = measure_intervals(poly, lines, sigmas, opts)
    wall_ids = sorted({w for e in edges for w in e.wall_ids})
    return Room(
        id="", polygon=poly, edges=edges, wall_ids=wall_ids, area_m2=info["area"], perimeter_m=info["perimeter"],
        wall_lengths_m=[float(x) for x in info["lens"]], length_m=length, width_m=width, dimension_method=method,
        area_interval_m2=ivals["area"], wall_length_intervals_m=ivals["lengths"],
        length_interval_m=ivals.get("length"), width_interval_m=ivals.get("width"),
        strong_fraction=info["by_tier"]["strong"], moderate_fraction=info["by_tier"]["moderate"],
        weak_fraction=info["by_tier"]["weak"], inferred_extension_total_m=info["inferred"],
        observed_fraction=info["observed"], topology_quality=_quality(info, opts),
        uses_weak_walls=info["by_tier"]["weak"] > 0,
    )


def _same_polygon(a: np.ndarray, b: np.ndarray, tol: float = 0.15) -> bool:
    if abs(abs(signed_area(a)) - abs(signed_area(b))) > 0.05 * max(abs(signed_area(a)), 1e-9) + 0.05:
        return False
    return all(np.min(np.linalg.norm(b - p, axis=1)) <= tol for p in a) and \
        all(np.min(np.linalg.norm(a - p, axis=1)) <= tol for p in b)


# ---------- ceiling association and adjacency ----------


def assign_ceilings(rooms: list[Room], levels: list[dict], opts: RoomOptions) -> None:
    """Attach a ceiling level to each room from spatial overlap of the level's solid inlier area.

    levels: [{"height_m": float, "interval_m": [lo, hi] | None, "points_xz": (n, 2) array}, ...]
    """
    cells = []
    for lv in levels:
        xz = np.asarray(lv["points_xz"], dtype=float)
        if len(xz) == 0:
            cells.append(np.empty((0, 2)))
            continue
        origin = xz.min(axis=0) - 0.5
        idx = solid_cells(xz, opts.ceiling_cell_m, 3, origin)
        cells.append(origin + (idx + 0.5) * opts.ceiling_cell_m)
    cell_area = opts.ceiling_cell_m ** 2
    for room in rooms:
        cover = []
        for lv, c in zip(levels, cells):
            inside = point_in_polygon(c, room.polygon).sum() if len(c) else 0
            cover.append(float(inside * cell_area / room.area_m2))
        info = {"levels": [{"height_m": float(lv["height_m"]), "coverage": round(cv, 3)} for lv, cv in zip(levels, cover)]}
        order = sorted(range(len(levels)), key=lambda i: -cover[i])
        if not order or cover[order[0]] < opts.min_ceiling_coverage:
            room.ceiling = {"ceiling_observed": False, "ambiguous": False, **info}
            continue
        best = order[0]
        rival = next((i for i in order[1:] if cover[i] >= opts.ceiling_ambiguity_ratio * cover[best]
                      and abs(levels[i]["height_m"] - levels[best]["height_m"]) > opts.ceiling_height_gap_m), None)
        if rival is not None:
            room.ceiling = {"ceiling_observed": False, "ambiguous": True, **info,
                            "note": "two ceiling levels overlap this room comparably; not assigned"}
            continue
        room.ceiling = {"ceiling_observed": True, "ambiguous": False, "ceiling_height_m": float(levels[best]["height_m"]),
                        "ceiling_height_interval_m": levels[best].get("interval_m"),
                        "coverage": round(cover[best], 3), **info}


def compute_adjacency(rooms: list[Room], opts: RoomOptions) -> None:
    """Rooms are geometrically adjacent where polygon edges run parallel, close together and overlap.
    This says nothing about doors or openings."""
    def edge_list(r: Room):
        pts = r.polygon
        return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]

    for a, b in itertools.combinations(rooms, 2):
        shared = 0.0
        for p0, p1 in edge_list(a):
            da = p1 - p0
            la = np.linalg.norm(da)
            if la < 1e-9:
                continue
            da /= la
            for q0, q1 in edge_list(b):
                db = q1 - q0
                lb = np.linalg.norm(db)
                if lb < 1e-9 or _angle_between(da, db / lb) > opts.adjacency_angle_deg:
                    continue
                nrm = np.array([-da[1], da[0]])
                if max(abs((q0 - p0) @ nrm), abs((q1 - p0) @ nrm)) > opts.adjacency_distance_m:
                    continue
                t = sorted([(q0 - p0) @ da, (q1 - p0) @ da])
                overlap = min(la, t[1]) - max(0.0, t[0])
                if overlap > 0:
                    shared += overlap
        if shared >= opts.adjacency_overlap_m:
            a.adjacent.append({"room_id": b.id, "shared_boundary_m": round(shared, 3)})
            b.adjacent.append({"room_id": a.id, "shared_boundary_m": round(shared, 3)})


# ---------- main entry point ----------


def build_topology(walls: list[WallInput], opts: RoomOptions | None = None, ceiling_levels: list[dict] | None = None) -> TopologyResult:
    opts = opts or RoomOptions()
    usable, suppressed = suppress_parallel_duplicates(walls, opts)
    by_id = {w.id: w for w in usable}
    rejected: list[dict] = []
    accepted: list[tuple[Room, bool]] = []
    candidate_faces = 0

    def poly_dict(face, g):
        return [[float(g.nodes[n].point[0]), float(g.nodes[n].point[1])] for n in face["nodes"]]

    # Pass A: strong + moderate walls only.
    core = [w for w in usable if w.evidence != "weak"]
    gA = build_graph(core, opts)
    facesA, outersA = extract_faces(gA)
    candidate_faces += len(facesA)
    for face in facesA:
        info, reason = _evaluate_face(face, gA, by_id, opts)
        if reason:
            rejected.append({"pass": "strong+moderate", "reason": reason, "area_m2": float(face["area"]), "polygon": poly_dict(face, gA)})
            continue
        room = _make_room(info, opts)
        if any(_same_polygon(room.polygon, r.polygon) for r, _ in accepted):
            rejected.append({"pass": "strong+moderate", "reason": "duplicate polygon", "area_m2": room.area_m2, "polygon": poly_dict(face, gA)})
            continue
        accepted.append((room, False))

    # Pass B: all walls. A face is new only if it is not inside an accepted room, and weak walls may supply
    # at most `weak_max_edges` edges and `weak_max_fraction` of the perimeter.
    gB, outersB = gA, outersA
    if any(w.evidence == "weak" for w in usable):
        gB = build_graph(usable, opts)
        facesB, outersB = extract_faces(gB)
        for face in facesB:
            info, reason = _evaluate_face(face, gB, by_id, opts)
            if reason:
                if any(e.evidence == "weak" for e in face["edges"]):
                    rejected.append({"pass": "all walls", "reason": reason, "area_m2": float(face["area"]), "polygon": poly_dict(face, gB)})
                continue
            if info["by_tier"]["weak"] == 0:
                continue  # same as a pass A candidate (already handled there)
            candidate_faces += 1
            weak_edges = sum(1 for e in info["edges"] if e.tier == "weak")
            room = _make_room(info, opts)
            centroid = room.polygon.mean(axis=0, keepdims=True)
            if weak_edges > opts.weak_max_edges or info["by_tier"]["weak"] > opts.weak_max_fraction:
                rejected.append({"pass": "all walls", "reason": f"too much weak wall ({weak_edges} edges, {info['by_tier']['weak']:.0%})",
                                 "area_m2": room.area_m2, "polygon": poly_dict(face, gB)})
                continue
            if any(point_in_polygon(centroid, r.polygon)[0] for r, _ in accepted):
                rejected.append({"pass": "all walls", "reason": "would subdivide a room using weak walls", "area_m2": room.area_m2,
                                 "polygon": poly_dict(face, gB)})
                continue
            if any(_same_polygon(room.polygon, r.polygon) for r, _ in accepted):
                continue
            accepted.append((room, True))

    # Nested enclosures (separate wall components entirely inside a room) are not rooms.
    rooms_only = [r for r, _ in accepted]
    final: list[Room] = []
    for r in rooms_only:
        host = next((o for o in rooms_only if o is not r and o.area_m2 > r.area_m2 and
                     point_in_polygon(r.polygon, o.polygon).all()), None)
        if host is not None:
            rejected.append({"pass": "nested", "reason": "enclosure entirely inside another room (furniture or column)",
                             "area_m2": r.area_m2, "polygon": [[float(x), float(z)] for x, z in r.polygon]})
            continue
        final.append(r)

    final.sort(key=lambda r: (round(float(r.polygon[:, 0].mean()), 3), round(float(r.polygon[:, 1].mean()), 3)))
    for i, r in enumerate(final, 1):
        r.id = f"room_{i:03d}"
    compute_adjacency(final, opts)
    for r in final:
        r.adjacent.sort(key=lambda a: a["room_id"])
    if ceiling_levels is not None:
        assign_ceilings(final, ceiling_levels, opts)

    outer = None
    simple_outers = [o for o in outersB if len(o["nodes"]) >= 3]
    if simple_outers:
        big = max(simple_outers, key=lambda o: abs(o["area"]))
        pts = np.array([gB.nodes[n].point for n in big["nodes"]])
        if is_simple(pts):
            outer = {"area_m2": float(abs(big["area"])), "perimeter_m": perimeter_of(pts),
                     "polygon": [{"x": float(x), "z": float(z)} for x, z in pts[::-1]],
                     "wall_ids": sorted({e.wall_id for e in big["edges"]}),
                     "note": "outer face of the wall graph; a property footprint candidate, not a room"}
    return TopologyResult(gB, final, rejected, outer, suppressed, len(walls), candidate_faces, opts)


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


def graph_to_dict(res: TopologyResult) -> dict:
    g = res.graph
    corner_id = lambda i: f"corner_{i + 1:03d}"
    return _round({
        "corners": [
            {"id": corner_id(n.id), "x": float(n.point[0]), "z": float(n.point[1]), "wall_ids": sorted(n.walls),
             "inferred": any(e > 0.02 for e in n.extensions.values()),
             "max_extension_m": max(n.extensions.values(), default=0.0), "extensions_m": dict(sorted(n.extensions.items()))}
            for n in g.nodes
        ],
        "edges": [
            {"id": f"edge_{e.id + 1:03d}", "wall_id": e.wall_id, "start": corner_id(e.a), "end": corner_id(e.b),
             "tier": e.evidence, "length_m": e.length_m, "observed_support_fraction": e.observed_support_fraction,
             "inferred_extension_m": e.inferred_extension_m, "gap_m": e.gap_m,
             "position_uncertainty_m": e.position_uncertainty_m}
            for e in g.edges
        ],
        "dropped_edges": g.dropped_edges,
        "suppressed_parallel_walls": res.suppressed_walls,
    })


def topology_to_dict(res: TopologyResult) -> dict:
    g = res.graph
    inferred_corners = [n for n in g.nodes if any(e > 0.02 for e in n.extensions.values())]
    return _round({
        "walls_consumed": res.wall_count,
        "suppressed_parallel_walls": res.suppressed_walls,
        "corners": graph_to_dict(res)["corners"],
        "corner_count": len(g.nodes),
        "inferred_corner_count": len(inferred_corners),
        "graph": {"nodes": len(g.nodes), "edges": len(g.edges), "dropped_edges": len(g.dropped_edges)},
        "candidate_faces": res.candidate_faces,
        "accepted_rooms": len(res.rooms),
        "rejected_faces": res.rejected_faces,
        "total_inferred_extension_m": float(sum(e.inferred_extension_m for e in g.edges)),
        "rooms": [r.to_dict() for r in res.rooms],
        "property_outer_boundary": res.outer_boundary,
        "warnings": res.warnings,
        "options": asdict(res.options),
    })


def rooms_to_geojson(res: TopologyResult) -> dict:
    """Room polygons as GeoJSON in the local metric (X, Z) frame (not geographic coordinates)."""
    feats = []
    for r in res.rooms:
        ring = [[float(x), float(z)] for x, z in r.polygon] + [[float(r.polygon[0][0]), float(r.polygon[0][1])]]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": _round({"id": r.id, "area_m2": r.area_m2, "perimeter_m": r.perimeter_m,
                                            "topology_quality": r.topology_quality,
                                            "ceiling_height_m": r.ceiling.get("ceiling_height_m")})})
    return {"type": "FeatureCollection", "properties": {"crs": "local metric X-Z, metres"}, "features": feats}
