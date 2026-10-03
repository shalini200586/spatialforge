"""Camera geometry for damage: project an image region onto the structural surfaces of a canonical property.

World frame = the property's frame: +Y up, plan coordinates (x, z) metres. A wall is the vertical plane through its start/end;
a floor/ceiling is the horizontal plane of a room (inside the room polygon). Pixel rays are cast from the camera; the nearest
valid surface wins (so a wall in front occludes one behind). Furniture and other unmodelled objects are NOT surfaces: damage on
an object in front of a wall would be projected onto the wall, which is a stated limitation.

A metric extent is only reported when the evidence is reliable (rays agree on ONE surface, the surface is not grazing, the
range is sensible, the surface itself is not weak). Otherwise the detection is kept and the extent is None, with a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from spatialforge.pipeline.models import Measurement, Property

MIN_RANGE_M, MAX_RANGE_M = 0.3, 10.0
MIN_COS_INCIDENCE = 0.25  # about 75 degrees from the surface normal
MIN_AGREEMENT_ASSOC, MIN_AGREEMENT_METRIC = 0.5, 0.8
PIXEL_ERROR = 2.0  # boundary localisation error of a detected region, pixels
WALL_TOP_IF_UNKNOWN_M = 4.5  # only a validity gate for ray hits; never reported as a ceiling height
SAMPLE_POINTS = 48
K = 2.0  # half-width in sigmas, as elsewhere in SpatialForge


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0  # SIMPLE_RADIAL: x_distorted = x_undistorted * (1 + k1 r^2)


@dataclass
class Surface:
    id: str
    kind: str  # wall | floor | ceiling
    room_ids: list[str]
    quality: str = "moderate"
    position_sigma_m: float = 0.0  # across the surface (diagnostic, from the property)
    scale_rel: float = 0.0  # relative half-width of the surface's own lengths (carries scale uncertainty for video/photos)
    ceiling_height: float | None = None  # walls: height of their room's observed ceiling; ceilings: their height above the floor
    # wall
    p0: np.ndarray | None = None
    direction: np.ndarray | None = None
    normal: np.ndarray | None = None
    length: float = 0.0
    openings: list[tuple[float, float, float | None]] = field(default_factory=list)  # (u0, u1, height)
    # floor / ceiling
    polygon: np.ndarray | None = None


def _rel_half_width(m: Measurement) -> float:
    if m.low is None or m.value <= 0:
        return 0.0
    return float(max(m.high - m.value, m.value - m.low) / m.value)


def build_surfaces(prop: Property) -> list[Surface]:
    rooms = {r.id: r for r in prop.rooms}
    out: list[Surface] = []
    for w in prop.walls:
        a, b = np.array([w.start.x, w.start.z]), np.array([w.end.x, w.end.z])
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        ch = next((rooms[r].ceiling.height.value for r in w.room_ids if r in rooms and rooms[r].ceiling.observed), None)
        ops = []
        for o in prop.openings:
            if o.wall_id != w.id:
                continue
            us = sorted([float((np.array([o.left_jamb.x, o.left_jamb.z]) - a) @ d), float((np.array([o.right_jamb.x, o.right_jamb.z]) - a) @ d)])
            ops.append((us[0], us[1], None if o.height is None else o.height.value))
        out.append(Surface(w.id, "wall", list(w.room_ids), w.evidence_quality, float(w.position_uncertainty.value),
                           _rel_half_width(w.length), ch, a, d, np.array([-d[1], d[0]]), L, ops))
    for r in prop.rooms:
        poly = np.array([[p.x, p.z] for p in r.polygon])
        rel = _rel_half_width(r.area) / 2.0  # an area's relative error is about twice a length's
        out.append(Surface(f"floor:{r.id}", "floor", [r.id], r.topology_quality, 0.0, rel, polygon=poly))
        if r.ceiling.observed:
            out.append(Surface(f"ceiling:{r.id}", "ceiling", [r.id], r.topology_quality, 0.0, rel,
                               ceiling_height=r.ceiling.height.value, polygon=poly))
    return out


def surface_in_room(s: Surface, room: str | None) -> bool:
    if room is None:
        return True
    return room in s.room_ids or s.id.startswith(room + "_") or s.id.endswith(":" + room)


def pixel_rays(cam: CameraModel, T_wc: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(camera centre (3,), unit ray directions (n, 3)) in the property frame. Camera axes: x right, y down, z forward."""
    px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    xd, yd = (px[:, 0] - cam.cx) / cam.fx, (px[:, 1] - cam.cy) / cam.fy
    xu, yu = xd, yd
    if cam.k1:
        for _ in range(8):
            f = 1.0 + cam.k1 * (xu * xu + yu * yu)
            xu, yu = xd / f, yd / f
    d = np.column_stack([xu, yu, np.ones_like(xu)]) @ T_wc[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return T_wc[:3, 3].copy(), d


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        crosses = (z1 > pts[:, 1]) != (z2 > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (pts[:, 1] - z1) * (x2 - x1) / (z2 - z1)
        inside ^= crosses & (pts[:, 0] < x_at)
    return inside


def cast(origin: np.ndarray, dirs: np.ndarray, surfaces: list[Surface], floor_y: float, room: str | None = None):
    """Nearest valid hit per ray: (surface index or -1, distance, hit points (n, 3), |cos| between ray and surface normal)."""
    n = len(dirs)
    best_t = np.full(n, np.inf)
    best_i = np.full(n, -1, dtype=int)
    best_c = np.zeros(n)
    o2 = origin[[0, 2]]
    for si, s in enumerate(surfaces):
        if not surface_in_room(s, room):
            continue
        if s.kind == "wall":
            d2 = dirs[:, [0, 2]]
            denom = d2 @ s.normal
            with np.errstate(divide="ignore", invalid="ignore"):
                t = float((s.p0 - o2) @ s.normal) / denom
            ok = (np.abs(denom) > 1e-6) & (t > 0.05)
            t = np.where(ok, t, 0.0)  # rays parallel to the wall have no hit (and no NaN/inf in the arithmetic below)
            hit = origin + t[:, None] * dirs
            u = (hit[:, [0, 2]] - s.p0) @ s.direction
            v = hit[:, 1] - floor_y
            top = (s.ceiling_height if s.ceiling_height is not None else WALL_TOP_IF_UNKNOWN_M) + 0.05
            ok &= (u >= -0.15) & (u <= s.length + 0.15) & (v >= -0.05) & (v <= top)
            for (u0, u1, hgt) in s.openings:  # a ray through a doorway does not hit the wall
                ok &= ~((u > u0) & (u < u1) & ((v < hgt + 0.05) if hgt is not None else True))
            cosv = np.abs(denom)
        else:
            plane_y = floor_y if s.kind == "floor" else floor_y + s.ceiling_height
            dy = dirs[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (plane_y - origin[1]) / dy
            ok = (np.abs(dy) > 1e-6) & (t > 0.05)
            hit = origin + np.where(ok, t, 0.0)[:, None] * dirs
            ok &= _inside(s.polygon, hit[:, [0, 2]])
            cosv = np.abs(dy)
        t = np.where(ok, t, np.inf)
        better = ok & (t < best_t)
        best_t = np.where(better, t, best_t)
        best_i = np.where(better, si, best_i)
        best_c = np.where(better, cosv, best_c)
    pts = origin + np.where(np.isfinite(best_t), best_t, 0.0)[:, None] * dirs
    return best_i, best_t, pts, best_c


def surface_coords(s: Surface, pts: np.ndarray, floor_y: float) -> np.ndarray:
    """(u, v) in metres: wall -> (distance along the wall, height above the floor); floor/ceiling -> plan (x, z)."""
    if s.kind == "wall":
        return np.column_stack([(pts[:, [0, 2]] - s.p0) @ s.direction, pts[:, 1] - floor_y])
    return pts[:, [0, 2]].copy()


def resample_polygon(poly: np.ndarray, n: int = SAMPLE_POINTS) -> np.ndarray:
    """Points spread evenly along the polygon outline (a thin crack outline keeps its shape)."""
    p = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return poly[:1]
    s = np.linspace(0, cum[-1], n, endpoint=False)
    return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])


@dataclass
class Mapping:
    surface: Surface | None = None
    agreement: float = 0.0
    uv: np.ndarray | None = None  # (m, 2) boundary points on the surface, in outline order
    range_m: float | None = None
    cos_incidence: float | None = None
    centre_uv: tuple[float, float] | None = None
    centre_world: np.ndarray | None = None
    reliable: bool = False
    reasons: list[str] = field(default_factory=list)
    px_m: float | None = None  # size of one pixel on the surface (m)


def map_polygon(polygon_px: np.ndarray, cam: CameraModel, T_wc: np.ndarray, floor_y: float, surfaces: list[Surface],
                room: str | None = None) -> Mapping:
    """Which surface does this image region lie on, and where on it?"""
    m = Mapping()
    outline = resample_polygon(np.asarray(polygon_px, dtype=np.float64))
    centre = np.asarray(polygon_px, dtype=np.float64).mean(axis=0)
    pts = np.vstack([outline, centre[None]])
    origin, dirs = pixel_rays(cam, T_wc, pts)
    idx, t, hits, cosv = cast(origin, dirs, surfaces, floor_y, room)
    out_idx, c_idx = idx[:-1], idx[-1]
    valid = out_idx[out_idx >= 0]
    if len(valid) == 0 and c_idx < 0:
        m.reasons.append("the region's rays hit no modelled surface (no wall, floor or ceiling there)")
        return m
    counts = np.bincount(valid, minlength=len(surfaces)) if len(valid) else np.zeros(len(surfaces), dtype=int)
    best = int(np.argmax(counts)) if counts.max() > 0 else int(c_idx)
    m.surface = surfaces[best]
    sel = out_idx == best
    m.agreement = float(sel.sum() / len(out_idx))
    if m.agreement < MIN_AGREEMENT_ASSOC:
        m.reasons.append(f"the region spans several surfaces (only {m.agreement:.0%} of its outline lies on {m.surface.id})")
    ts, cs = t[:-1][sel], cosv[:-1][sel]
    if sel.sum() >= 3:
        m.uv = surface_coords(m.surface, hits[:-1][sel], floor_y)
        m.range_m = float(np.median(ts))
        m.cos_incidence = float(np.median(cs))
        m.px_m = m.range_m / (0.5 * (cam.fx + cam.fy)) / max(m.cos_incidence, 1e-3)
    if c_idx == best:
        m.centre_world = hits[-1]
        m.centre_uv = tuple(float(v) for v in surface_coords(m.surface, hits[-1:], floor_y)[0])
    # --- reliability gates for a METRIC extent ---
    why = []
    if m.agreement < MIN_AGREEMENT_METRIC:
        why.append(f"only {m.agreement:.0%} of the outline agrees on one surface")
    if m.uv is None:
        why.append("too few outline points reached the surface")
    else:
        if m.cos_incidence < MIN_COS_INCIDENCE:
            why.append(f"the surface is seen at a grazing angle (cos {m.cos_incidence:.2f})")
        if not MIN_RANGE_M <= m.range_m <= MAX_RANGE_M:
            why.append(f"range {m.range_m:.1f} m is outside {MIN_RANGE_M}-{MAX_RANGE_M} m")
    if m.surface.kind == "wall" and m.surface.quality == "weak":
        why.append(f"the wall {m.surface.id} itself is only weakly supported")
    m.reliable = not why
    m.reasons += why
    return m


def _poly_area(uv: np.ndarray) -> float:
    x, y = uv[:, 0], uv[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def principal_extents(uv: np.ndarray) -> tuple[float, float]:
    """(major, minor) extent of the outline along its principal axes."""
    q = uv - uv.mean(axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    a = q @ vt[0]
    b = q @ vt[1] if len(vt) > 1 else np.zeros(len(q))
    return float(a.max() - a.min()), float(b.max() - b.min())


def metric_extents(m: Mapping, tier: str, quality: str) -> dict:
    """width / height / area / length as Measurements, each with an interval that carries its uncertainty.

    Half-width of an extent L: sqrt( (L*sqrt(scale_rel^2 + range_rel^2))^2 + e_px^2 ), where
      scale_rel  = relative half-width of the surface's own lengths in the property (scale uncertainty for video and photos),
      range_rel  = K * position uncertainty of the surface / range (a plane shifted toward or away from the camera rescales it),
      e_px       = boundary localisation error (2 px) converted to metres at this range and incidence.
    An area uses twice the relative terms (two extents) plus a boundary band of e_px along the outline.
    """
    uv = m.uv
    s = m.surface
    if s.kind == "wall":
        width, height = float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))
    else:
        width, height = principal_extents(uv)
    length, _ = principal_extents(uv)
    area = _poly_area(uv)
    rel = math.hypot(s.scale_rel, K * s.position_sigma_m / max(m.range_m, 1e-6))
    e_px = PIXEL_ERROR * m.px_m
    perim = float(np.sum(np.linalg.norm(np.diff(np.vstack([uv, uv[:1]]), axis=0), axis=1)))

    def length_like(v: float) -> Measurement:
        h = math.hypot(v * rel, e_px)
        return Measurement(v, "m", max(0.0, v - h), v + h, None, quality, tier)

    h_a = math.hypot(area * 2.0 * rel, 0.5 * perim * e_px)
    return {
        "width": length_like(width), "height": length_like(height), "length": length_like(length),
        "area": Measurement(area, "m2", max(0.0, area - h_a), area + h_a, None, quality, tier),
        "extent": {"u_min": float(uv[:, 0].min()), "u_max": float(uv[:, 0].max()),
                   "v_min": float(uv[:, 1].min()), "v_max": float(uv[:, 1].max())},
        "relative_surface_uncertainty": rel, "pixel_error_m": e_px,
    }
