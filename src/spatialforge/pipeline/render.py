"""Product floor plan (plan.png) from the canonical Property. Not a debug view.

Drawn: room polygons (id, area, length x width or "irregular", ceiling height), structural walls (strong and
moderate evidence; weak walls are omitted), accepted openings with their widths, edge dimensions and a scale bar.
NOT drawn: point cloud, ICP/registration diagnostics, candidate or weak walls, rejected or low-confidence openings
(policy: low-confidence openings are kept out of the plan and listed under `unverified_openings` in property.json).
Incomplete results are never presented as complete: walls outside any recovered room are dashed grey, unobserved
gaps in walls that are not verified openings stay dotted, and a PARTIAL badge and note are always shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spatialforge.pipeline.models import Property

BACKGROUND = (255, 255, 255)
WALL_COLOUR = (55, 55, 60)
LOOSE_WALL_COLOUR = (175, 175, 180)
ROOM_FILLS = [(214, 228, 247), (222, 240, 218), (250, 232, 208), (237, 222, 245), (211, 238, 238), (248, 224, 224)]
ROOM_OUTLINE = (110, 120, 135)
TEXT = (30, 30, 35)
MUTED = (110, 110, 120)
PARTIAL = (190, 100, 0)
OPENING_COLOURS = {"door": (25, 85, 200), "window": (0, 140, 150), "opening": (150, 40, 150)}
MARGIN = 90
TOP = 110
FOOTER = 90
MAX_SIDE_PX = 1500
MIN_WIDTH = 1300  # the header note and footer are one line each; narrow plans get white space instead of cut-off text
FOOTERS = {
    "lidar": "Metric LiDAR measurements; uncertainty intervals and quality ratings are in property.json.",
    "video": "Scaled monocular-video measurements: metric scale is estimated, uncertainty is wider than LiDAR. "
             "Intervals and quality ratings are in property.json.",
    "other": "Uncertainty intervals and quality ratings are in property.json.",
}


def _wrap(d, text: str, font, max_w: float) -> list[str]:
    """Greedy word wrap to a pixel width; text that already fits stays one line."""
    if not text:
        return [""]
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}" if cur else word
        if cur and d.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class PlanInfo:
    path: Path
    width_px: int
    height_px: int
    pixels_per_metre: float
    x0: float
    z1: float
    rooms_drawn: int = 0
    walls_drawn: int = 0
    openings_drawn: int = 0
    notes: list[str] = field(default_factory=list)

    def to_px(self, x: float, z: float) -> tuple[float, float]:
        return (MARGIN + (x - self.x0) * self.pixels_per_metre, TOP + (self.z1 - z) * self.pixels_per_metre)


def _extent(prop: Property):
    pts = [(p.x, p.z) for r in prop.rooms for p in r.polygon]
    pts += [(p.x, p.z) for w in prop.walls if w.evidence_quality != "weak" for s in w.segments for p in (s.start, s.end)]
    pts += [(p.x, p.z) for o in prop.openings for p in (o.left_jamb, o.right_jamb)]
    return pts


def _centre_text(draw, xy, text, font, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (x1 - x0) / 2 - x0, xy[1] - (y1 - y0) / 2 - y0), text, font=font, fill=fill)


def _dashed(draw, a, b, colour, width, dash=9, gap=7):
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if length < 1e-6:
        return
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    t = 0.0
    while t < length:
        e = min(t + dash, length)
        draw.line([(a[0] + ux * t, a[1] + uy * t), (a[0] + ux * e, a[1] + uy * e)], fill=colour, width=width)
        t += dash + gap


def _seg_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test (kept here so this module has no sensor-specific imports)."""
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


def label_point(poly: np.ndarray, step: float = 0.2) -> np.ndarray:
    """A point well inside the polygon (grid search for the largest distance to the boundary); deterministic.
    The centroid of an L-shaped room can lie on its boundary, so it is not used."""
    xs = np.arange(poly[:, 0].min(), poly[:, 0].max() + step, step)
    zs = np.arange(poly[:, 1].min(), poly[:, 1].max() + step, step)
    grid = np.array([(x, z) for x in xs for z in zs])
    inside = grid[_inside(grid, poly)]
    if len(inside) == 0:
        return poly.mean(axis=0)
    n = len(poly)
    best = max(range(len(inside)), key=lambda i: (min(_seg_distance(inside[i], poly[j], poly[(j + 1) % n]) for j in range(n)),
                                                  -inside[i][0], -inside[i][1]))
    return inside[best]


def render_plan(prop: Property, path: Path, title: str = "") -> PlanInfo:
    pts = _extent(prop)
    f_title, f_body, f_small = _font(26), _font(17), _font(14)
    if not pts:  # nothing recovered: say so plainly
        img = Image.new("RGB", (900, 360), BACKGROUND)
        d = ImageDraw.Draw(img)
        d.text((40, 40), title or "SpatialForge floor plan", font=f_title, fill=TEXT)
        d.text((40, 90), "NO STRUCTURE RECOVERED", font=f_title, fill=PARTIAL)
        for i, w in enumerate(prop.warnings[:4]):
            d.text((40, 150 + 30 * i), w[:110], font=f_small, fill=MUTED)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return PlanInfo(path, 900, 360, 0.0, 0.0, 0.0, notes=["empty"])

    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, z0, z1 = min(xs) - 0.8, max(xs) + 0.8, min(zs) - 0.8, max(zs) + 0.8
    ppm = float(np.clip(MAX_SIDE_PX / max(x1 - x0, z1 - z0), 25.0, 120.0))
    w, h = max(int((x1 - x0) * ppm) + 2 * MARGIN, MIN_WIDTH), int((z1 - z0) * ppm) + TOP + FOOTER
    img = Image.new("RGB", (w, h), BACKGROUND)
    d = ImageDraw.Draw(img)
    info = PlanInfo(path, w, h, ppm, x0, z1)
    px = info.to_px
    wall_px = int(np.clip(0.11 * ppm, 3, 10))

    # rooms
    in_room = set()
    for k, room in enumerate(prop.rooms):
        poly = [px(p.x, p.z) for p in room.polygon]
        d.polygon(poly, fill=ROOM_FILLS[k % len(ROOM_FILLS)], outline=ROOM_OUTLINE)
        in_room.update(room.wall_ids)
        info.rooms_drawn += 1

    # walls (weak ones are not product geometry)
    for wall in prop.walls:
        if wall.evidence_quality == "weak":
            continue
        loose = wall.id not in in_room
        for s in wall.segments:
            a, b = px(s.start.x, s.start.z), px(s.end.x, s.end.z)
            if loose:
                _dashed(d, a, b, LOOSE_WALL_COLOUR, 2, 10, 8)
            else:
                d.line([a, b], fill=WALL_COLOUR, width=wall_px)
        info.walls_drawn += 1

    # unverified gaps stay dotted (they are unknown, not openings)
    opening_spans = {}
    for o in prop.openings:
        opening_spans.setdefault(o.wall_id, []).append(o)
    for wall in prop.walls:
        if wall.evidence_quality == "weak" or len(wall.segments) < 2:
            continue
        for s0, s1 in zip(wall.segments, wall.segments[1:]):
            mid = ((s0.end.x + s1.start.x) / 2, (s0.end.z + s1.start.z) / 2)
            covered = any(min(o.left_jamb.x, o.right_jamb.x) - 0.3 <= mid[0] <= max(o.left_jamb.x, o.right_jamb.x) + 0.3
                          and min(o.left_jamb.z, o.right_jamb.z) - 0.3 <= mid[1] <= max(o.left_jamb.z, o.right_jamb.z) + 0.3
                          for o in opening_spans.get(wall.id, []))
            if not covered:
                _dashed(d, px(s0.end.x, s0.end.z), px(s1.start.x, s1.start.z), (185, 185, 190), 2, 3, 5)

    # openings: cut the wall open, mark the jambs, add a type glyph and the width
    for o in prop.openings:
        a, b = px(o.left_jamb.x, o.left_jamb.z), px(o.right_jamb.x, o.right_jamb.z)
        colour = OPENING_COLOURS[o.type]
        d.line([a, b], fill=BACKGROUND, width=wall_px + 4)
        n = np.array([-(b[1] - a[1]), b[0] - a[0]], dtype=float)
        n = n / max(np.linalg.norm(n), 1e-9)
        for p in (a, b):
            d.line([(p[0] - n[0] * (wall_px / 2 + 3), p[1] - n[1] * (wall_px / 2 + 3)),
                    (p[0] + n[0] * (wall_px / 2 + 3), p[1] + n[1] * (wall_px / 2 + 3))], fill=WALL_COLOUR, width=3)
        if o.type == "window":
            for s in (-3, 3):
                d.line([(a[0] + n[0] * s, a[1] + n[1] * s), (b[0] + n[0] * s, b[1] + n[1] * s)], fill=colour, width=2)
        elif o.type == "door":
            d.line([a, b], fill=colour, width=2)
        else:
            _dashed(d, a, b, colour, 2, 8, 6)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        label = f"{o.type} {o.width.value:.2f} m"
        _centre_text(d, (mid[0] + n[0] * 26, mid[1] + n[1] * 26), label, f_small, colour)
        info.openings_drawn += 1

    # room labels and dimensions
    labelled_edges = []  # (start, end) of every edge already dimensioned: shared boundaries are labelled once
    for room in prop.rooms:
        poly = np.array([(p.x, p.z) for p in room.polygon])
        c = label_point(poly)
        cx, cy = px(c[0], c[1])
        lines = [f"{room.id.replace('room_', 'R')}   {room.area.value:.1f} m²"]
        lines.append(f"{room.length.value:.2f} × {room.width.value:.2f} m" if room.length is not None else "irregular")
        if room.ceiling.observed:
            lines.append(f"ceiling {room.ceiling.height.value:.2f} m")
        elif room.ceiling.ambiguous:
            lines.append("ceiling ambiguous")
        for i, text in enumerate(lines):
            _centre_text(d, (cx, cy + (i - (len(lines) - 1) / 2) * 22), text, f_body if i == 0 else f_small,
                         TEXT if i == 0 else MUTED)
        n = len(poly)
        seen_pairs = 0
        for i, length in enumerate(room.wall_lengths):
            if length.value < 1.0:
                continue
            if room.length is not None and seen_pairs >= 2:
                break  # a rectangle needs only two distinct sides
            p0, p1 = poly[i], poly[(i + 1) % n]
            mid = (p0 + p1) / 2
            if any(_seg_distance(mid, a, b) < 0.45 for a, b in labelled_edges):
                continue  # this boundary is already dimensioned (by this room's other side or a neighbouring room)
            dx, dz = p1 - p0
            norm = float(np.hypot(dx, dz))
            out_x, out_z = dz / norm, -dx / norm  # outward (polygons are counter-clockwise)
            m_px = px(mid[0], mid[1])
            text = f"{length.value:.2f} m"
            bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=f_small)
            # push the label clear of the wall line: half the text extent along the outward direction + margin
            push = wall_px / 2 + 8 + abs(out_x) * (bx1 - bx0) / 2 + abs(out_z) * (by1 - by0) / 2
            _centre_text(d, (m_px[0] + out_x * push, m_px[1] - out_z * push), text, f_small, MUTED)
            seen_pairs += 1
        labelled_edges += [(poly[i], poly[(i + 1) % n]) for i in range(n)]

    # title, partial badge, scale, notes
    d.text((MARGIN, 24), title or "SpatialForge floor plan", font=f_title, fill=TEXT)
    badge = "COMPLETE" if prop.status == "complete" else "PARTIAL RESULT"
    d.text((MARGIN, 60), badge, font=f_body, fill=PARTIAL if prop.status != "complete" else (0, 120, 60))
    counts = (f"{len(prop.rooms)} rooms · {len(prop.walls)} walls · {len(prop.openings)} openings · "
              f"{prop.capture.get('tier', '?')} · schema {prop.schema_version}")
    d.text((MARGIN + 170, 62), counts, font=f_small, fill=MUTED)
    loose = [w for w in prop.walls if w.evidence_quality != "weak" and w.id not in in_room]
    note = ("Rooms shown are only those whose walls form closed outlines. " +
            (f"{len(loose)} wall(s) outside any recovered room are dashed grey. " if loose else "") +
            "Unverified gaps are dotted. Not a complete property plan.") if prop.status != "complete" else ""
    for i, line in enumerate(_wrap(d, note, f_small, w - MARGIN - 20)):
        d.text((MARGIN, 88 + 16 * i), line, font=f_small, fill=PARTIAL)
    bar_m = 1.0 if (x1 - x0) < 8 else 2.0
    bx, by = MARGIN, h - 55
    d.line([(bx, by), (bx + bar_m * ppm, by)], fill=TEXT, width=4)
    d.text((bx + bar_m * ppm + 10, by - 9), f"{bar_m:g} m", font=f_small, fill=TEXT)
    d.text((bx + bar_m * ppm + 70, by - 9), "+X →   +Z ↑   (capture frame, metres)", font=f_small, fill=MUTED)
    footer = FOOTERS.get(prop.capture.get("tier"), FOOTERS["other"])
    for i, line in enumerate(_wrap(d, footer, f_small, w - MARGIN - 20)):
        d.text((MARGIN, h - 32 + 16 * i), line, font=f_small, fill=MUTED)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return info
