"""Top-down (X-Z) density images for comparing point clouds (Pillow + NumPy only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PIXELS_PER_METRE = 100
MARGIN = 70


def shared_bounds(clouds: list[np.ndarray], pad: float = 0.5) -> tuple[float, float, float, float]:
    """(x_min, x_max, z_min, z_max) in whole metres covering every cloud, so images share one frame."""
    x = np.concatenate([c[:, 0] for c in clouds])
    z = np.concatenate([c[:, 2] for c in clouds])
    return (
        float(np.floor(x.min() - pad)), float(np.ceil(x.max() + pad)),
        float(np.floor(z.min() - pad)), float(np.ceil(z.max() + pad)),
    )


def _plane_bands(analysis) -> list[tuple[float, float, str, tuple]]:
    """(y_low, y_high, label, colour) for the floor and every ceiling level, in metres."""
    bands = []
    if analysis.floor.observed:
        h, t = analysis.floor.height_m, analysis.floor.inlier_threshold_m
        bands.append((h - t, h + t, f"floor y={h:.2f}", (30, 140, 60)))
    for lvl, ht in zip(analysis.ceiling_levels, analysis.ceiling_level_heights):
        t = lvl.inlier_threshold_m
        bands.append((lvl.height_m - t, lvl.height_m + t,
                      f"ceiling y={lvl.height_m:.2f}  height {ht['value_m']:.2f} m", (30, 90, 200)))
    return bands


def render_vertical_profile(analysis, path: Path) -> None:
    """Y-density bar chart (metres) with the detected floor/ceiling bands and the camera height range."""
    centres, counts = analysis.profile_centres, analysis.profile_counts
    share = counts / counts.sum() * 100
    ppm, left, top = 220, 90, 50  # pixels per metre, margins
    y0, y1 = float(np.floor(centres.min())), float(np.ceil(centres.max()))
    h, w = int((y1 - y0) * ppm), 700
    canvas = Image.new("RGB", (left + w + 330, h + top + 50), "white")
    d = ImageDraw.Draw(canvas)

    def py(y):
        return top + h - int((y - y0) * ppm)

    d.rectangle([left, top, left + w, top + h], outline="black")
    # camera path band, then plane bands, then bars on top
    d.rectangle([left, py(analysis.camera_y_max), left + w, py(analysis.camera_y_min)], fill=(255, 235, 200))
    for lo, hi, label, colour in _plane_bands(analysis):
        d.rectangle([left, py(hi), left + w, py(lo)], fill=tuple(int(c * 0.25 + 191) for c in colour))
        d.text((left + w + 10, py((lo + hi) / 2) - 5), label, fill=colour)
    d.text((left + w + 10, py((analysis.camera_y_min + analysis.camera_y_max) / 2) - 5), "camera path range", fill=(200, 120, 0))
    scale = (w - 10) / max(share.max(), 1e-9)
    for c, s in zip(centres, share):
        d.rectangle([left, py(c) - 1, left + int(s * scale), py(c)], fill=(70, 70, 70))
    for y in np.arange(y0, y1 + 0.001, 0.5):
        d.line([left - 5, py(y), left, py(y)], fill="black")
        d.text((left - 45, py(y) - 5), f"{y:.1f}", fill="black")
    d.text((8, 8), f"Vertical profile (world Y, up)   points: {analysis.point_count}", fill="black")
    d.text((8, 24), "bar length = share of points per 2 cm bin; shaded = detected bands; metres", fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_side_projection(points: np.ndarray, analysis, path: Path) -> None:
    """X-Y density view (metres, Y up) with the fitted floor/ceiling heights as lines."""
    x0, x1 = float(np.floor(points[:, 0].min())), float(np.ceil(points[:, 0].max()))
    y0, y1 = float(np.floor(points[:, 1].min())), float(np.ceil(points[:, 1].max()))
    ppm = 80
    w, h = int((x1 - x0) * ppm), int((y1 - y0) * ppm)
    col = ((points[:, 0] - x0) * ppm).astype(int)
    row = h - 1 - ((points[:, 1] - y0) * ppm).astype(int)
    keep = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    counts = np.zeros((h, w), dtype=np.int32)
    np.add.at(counts, (row[keep], col[keep]), 1)
    gray = (255 - 255 * np.clip(np.log1p(counts) / np.log1p(40), 0, 1)).astype(np.uint8)
    canvas = Image.new("RGB", (w + 2 * MARGIN, h + 2 * MARGIN), "white")
    canvas.paste(Image.fromarray(gray).convert("RGB"), (MARGIN, MARGIN))
    d = ImageDraw.Draw(canvas)
    d.rectangle([MARGIN - 1, MARGIN - 1, MARGIN + w, MARGIN + h], outline="black")
    for lo, hi, label, colour in _plane_bands(analysis):
        yy = MARGIN + h - 1 - int(((lo + hi) / 2 - y0) * ppm)
        d.line([MARGIN, yy, MARGIN + w, yy], fill=colour, width=2)
        d.text((MARGIN + 6, yy - 14), label, fill=colour)
    for x in np.arange(x0, x1 + 1, 1.0):
        d.text((MARGIN + int((x - x0) * ppm) - 8, MARGIN + h + 8), f"{x:.0f}", fill="black")
    for y in np.arange(y0, y1 + 1, 1.0):
        d.text((MARGIN - 30, MARGIN + h - 1 - int((y - y0) * ppm) - 5), f"{y:.0f}", fill="black")
    d.text((8, 8), "Side projection X-Y (m), Y up, all points projected", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


WALL_COLOURS = [(220, 30, 30), (30, 90, 220), (20, 150, 60), (200, 20, 200), (0, 150, 160), (110, 60, 190)]
REJECTED_COLOUR = (150, 100, 40)  # brown: never used for accepted walls


def render_walls_topdown(cloud: np.ndarray, analysis, bounds: tuple[float, float, float, float], path: Path) -> None:
    """Light point cloud (X-Z) with accepted wall segments (thick, with IDs and endpoints),
    gaps between segments of one wall (dotted blue) and rejected candidates (thin orange)."""
    x0, x1, z0, z1 = bounds
    w, h = int((x1 - x0) * PIXELS_PER_METRE), int((z1 - z0) * PIXELS_PER_METRE)
    col = ((cloud[:, 0] - x0) * PIXELS_PER_METRE).astype(int)
    row = h - 1 - ((cloud[:, 2] - z0) * PIXELS_PER_METRE).astype(int)
    keep = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    counts = np.zeros((h, w), dtype=np.int32)
    np.add.at(counts, (row[keep], col[keep]), 1)
    gray = (255 - 110 * np.clip(np.log1p(counts) / np.log1p(20), 0, 1)).astype(np.uint8)  # deliberately light
    canvas = Image.new("RGB", (w + 2 * MARGIN, h + 2 * MARGIN), "white")
    canvas.paste(Image.fromarray(gray).convert("RGB"), (MARGIN, MARGIN))
    d = ImageDraw.Draw(canvas)

    def px(p):
        return (MARGIN + (p[0] - x0) * PIXELS_PER_METRE, MARGIN + h - 1 - (p[1] - z0) * PIXELS_PER_METRE)

    d.rectangle([MARGIN - 1, MARGIN - 1, MARGIN + w, MARGIN + h], outline="black")
    for x in np.arange(x0, x1 + 1, 1.0):
        d.text((MARGIN + int((x - x0) * PIXELS_PER_METRE) - 8, MARGIN + h + 8), f"{x:.0f}", fill="black")
    for z in np.arange(z0, z1 + 1, 1.0):
        d.text((MARGIN - 30, MARGIN + h - 1 - int((z - z0) * PIXELS_PER_METRE) - 5), f"{z:.0f}", fill="black")
    for item in analysis.rejected:
        segs = item["segments"] if item["kind"] == "line" and isinstance(item["segments"], list) else [item]
        for s in segs:
            if "start" in s:
                d.line([px(s["start"]), px(s["end"])], fill=REJECTED_COLOUR, width=1)
    for k, wall in enumerate(analysis.walls):
        colour = WALL_COLOURS[k % len(WALL_COLOURS)]
        for s in wall.segments:
            a, b = px(s.start), px(s.end)
            d.line([a, b], fill=colour, width=4)
            for p in (a, b):
                d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=colour, outline="black")
        for g in wall.gaps:
            a, b = px(g["start"]), px(g["end"])
            n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 8))
            for i in range(0, n, 2):
                p0 = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                p1 = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
                d.line([p0, p1], fill=(40, 40, 255), width=2)
        mid = px(((wall.start[0] + wall.end[0]) / 2, (wall.start[1] + wall.end[1]) / 2))
        d.text((mid[0] + 6, mid[1] - 14), f"{wall.id[-3:]} {wall.length_m:.1f}m", fill=colour)
    d.text((8, 8), f"Structural walls (accepted: {len(analysis.walls)} walls, {analysis.segment_count} segments)", fill="black")
    d.text((8, 24), "thick = accepted segment, dotted blue = observed gap, thin brown = rejected candidate, X-Z metres", fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


ROOM_FILLS = [(230, 80, 80), (80, 120, 230), (60, 180, 90), (200, 90, 200), (40, 180, 190), (240, 170, 40)]
TIER_COLOURS = {"strong": (0, 110, 40), "moderate": (30, 80, 200), "weak": (235, 130, 0)}


def _dashed(draw: ImageDraw.ImageDraw, a, b, colour, width=2, dash=8):
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / dash))
    for i in range(0, n, 2):
        p0 = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
        p1 = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
        draw.line([p0, p1], fill=colour, width=width)


def render_rooms_topdown(cloud, topo, wall_inputs, bounds, path: Path) -> None:
    """Room topology diagnostic (X-Z, metres). Faint cloud and observed walls underneath; rooms filled with
    their evidence-coloured boundary (green strong, blue moderate, orange weak-supported, dashed red =
    inferred extension); corners as dots (hollow red = inferred); rejected faces thin purple."""
    x0, x1, z0, z1 = bounds
    w, h = int((x1 - x0) * PIXELS_PER_METRE), int((z1 - z0) * PIXELS_PER_METRE)
    base = Image.new("RGBA", (w + 2 * MARGIN, h + 2 * MARGIN), (255, 255, 255, 255))
    if cloud is not None and len(cloud):
        col = ((cloud[:, 0] - x0) * PIXELS_PER_METRE).astype(int)
        row = h - 1 - ((cloud[:, 2] - z0) * PIXELS_PER_METRE).astype(int)
        keep = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        counts = np.zeros((h, w), dtype=np.int32)
        np.add.at(counts, (row[keep], col[keep]), 1)
        gray = (255 - 60 * np.clip(np.log1p(counts) / np.log1p(20), 0, 1)).astype(np.uint8)
        base.paste(Image.fromarray(gray).convert("RGBA"), (MARGIN, MARGIN))

    def px(p):
        return (MARGIN + (p[0] - x0) * PIXELS_PER_METRE, MARGIN + h - 1 - (p[1] - z0) * PIXELS_PER_METRE)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for k, room in enumerate(topo.rooms):
        od.polygon([px(p) for p in room.polygon], fill=ROOM_FILLS[k % len(ROOM_FILLS)] + (70,))
    img = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(img)
    d.rectangle([MARGIN - 1, MARGIN - 1, MARGIN + w, MARGIN + h], outline="black")
    for x in np.arange(x0, x1 + 1, 1.0):
        d.text((MARGIN + int((x - x0) * PIXELS_PER_METRE) - 8, MARGIN + h + 8), f"{x:.0f}", fill="black")
    for z in np.arange(z0, z1 + 1, 1.0):
        d.text((MARGIN - 30, MARGIN + h - 1 - int((z - z0) * PIXELS_PER_METRE) - 5), f"{z:.0f}", fill="black")

    for wall in wall_inputs:  # observed wall support, lightly
        for a, b in wall.segments:
            d.line([px(a), px(b)], fill=(170, 170, 170), width=3)
    used = set()
    for room in topo.rooms:
        used.update(w for e in room.edges for w in e.wall_ids)
    nodes = topo.graph.nodes
    for e in topo.graph.edges:  # graph edges not on any room boundary stay visible but faint
        if e.wall_id not in used:
            _dashed(d, px(nodes[e.a].point), px(nodes[e.b].point), (140, 150, 190), 1, 6)
    for rej in topo.rejected_faces:
        pts = [px(p) for p in rej["polygon"]]
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=(150, 60, 190), width=1)
    for k, room in enumerate(topo.rooms):
        pts = [px(p) for p in room.polygon]
        for i, e in enumerate(room.edges):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            d.line([a, b], fill=TIER_COLOURS[e.tier], width=4)
            if e.inferred_extension_m > 0.02:
                _dashed(d, a, b, (220, 0, 0), 2, 6)
        c = room.polygon.mean(axis=0)
        cx, cy = px(c)
        d.text((cx - 28, cy - 12), f"{room.id[-3:]}  {room.area_m2:.1f} m2", fill="black")
        dims = f"{room.length_m:.2f}x{room.width_m:.2f}" if room.length_m else "irregular"
        d.text((cx - 28, cy + 2), f"{dims}  {room.topology_quality}", fill=(60, 60, 60))
    for n in nodes:
        p = px(n.point)
        inferred = any(v > 0.02 for v in n.extensions.values())
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(220, 0, 0) if inferred else "black",
                  fill=None if inferred else (30, 30, 30))
    d.text((8, 8), f"Room topology: {len(topo.rooms)} rooms from {topo.wall_count} walls (X-Z metres, +Z up, polygons CCW)", fill="black")
    d.text((8, 24), "boundary: green strong / blue moderate / orange weak-supported / dashed red inferred extension | "
                    "hollow red dot = inferred corner | purple = rejected face | grey = observed walls", fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)


OPENING_COLOURS = {"door": (20, 70, 220), "window": (0, 160, 170), "opening": (200, 20, 200)}


def render_openings_topdown(cloud, topo, wall_inputs, result, bounds, path: Path) -> None:
    """Openings over the walls and room polygons (X-Z, metres). Thick line between the refined jambs:
    blue door, teal window, magenta generic opening; dashed orange = low confidence; small grey x = rejected
    candidate (raw gap midpoint)."""
    x0, x1, z0, z1 = bounds
    w, h = int((x1 - x0) * PIXELS_PER_METRE), int((z1 - z0) * PIXELS_PER_METRE)
    base = Image.new("RGBA", (w + 2 * MARGIN, h + 2 * MARGIN), (255, 255, 255, 255))
    if cloud is not None and len(cloud):
        col = ((cloud[:, 0] - x0) * PIXELS_PER_METRE).astype(int)
        row = h - 1 - ((cloud[:, 2] - z0) * PIXELS_PER_METRE).astype(int)
        keep = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        counts = np.zeros((h, w), dtype=np.int32)
        np.add.at(counts, (row[keep], col[keep]), 1)
        gray = (255 - 45 * np.clip(np.log1p(counts) / np.log1p(20), 0, 1)).astype(np.uint8)
        base.paste(Image.fromarray(gray).convert("RGBA"), (MARGIN, MARGIN))

    def px(p):
        return (MARGIN + (p[0] - x0) * PIXELS_PER_METRE, MARGIN + h - 1 - (p[1] - z0) * PIXELS_PER_METRE)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    if topo is not None:
        for k, room in enumerate(topo.rooms):
            od.polygon([px(p) for p in room.polygon], fill=ROOM_FILLS[k % len(ROOM_FILLS)] + (40,))
    img = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(img)
    d.rectangle([MARGIN - 1, MARGIN - 1, MARGIN + w, MARGIN + h], outline="black")
    for x in np.arange(x0, x1 + 1, 1.0):
        d.text((MARGIN + int((x - x0) * PIXELS_PER_METRE) - 8, MARGIN + h + 8), f"{x:.0f}", fill="black")
    for z in np.arange(z0, z1 + 1, 1.0):
        d.text((MARGIN - 30, MARGIN + h - 1 - int((z - z0) * PIXELS_PER_METRE) - 5), f"{z:.0f}", fill="black")
    for wall in wall_inputs:
        for a, b in wall.segments:
            d.line([px(a), px(b)], fill=(120, 120, 120), width=3)
    if topo is not None:
        for room in topo.rooms:
            pts = [px(p) for p in room.polygon]
            d.line(pts + [pts[0]], fill=(150, 150, 150), width=1)
            c = px(room.polygon.mean(axis=0))
            d.text((c[0] - 16, c[1] - 6), room.id, fill=(110, 110, 110))
    rejected_mid = {r["candidate_id"]: [(r["raw_start"][i] + r["raw_end"][i]) / 2 for i in range(2)] for r in result.rejected}
    for cid, mid in rejected_mid.items():
        p = px(mid)
        d.line([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=(110, 110, 110), width=2)
        d.line([p[0] - 5, p[1] + 5, p[0] + 5, p[1] - 5], fill=(110, 110, 110), width=2)
        d.text((p[0] + 7, p[1] - 6), cid[-3:], fill=(110, 110, 110))
    for o in result.openings:
        a = px((o.left_jamb["x"], o.left_jamb["z"]))
        b = px((o.right_jamb["x"], o.right_jamb["z"]))
        colour = OPENING_COLOURS[o.type]
        if o.status == "accepted":
            d.line([a, b], fill=colour, width=7)
        else:
            _dashed(d, a, b, (235, 120, 0), 5, 7)
        for p in (a, b):
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill="white", outline="black")
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        d.text((mid[0] + 8, mid[1] - 18), f"{o.id[-3:]} {o.type} {o.width_m:.2f} m", fill=(0, 0, 0))
    d.text((8, 8), f"Openings: {len(result.accepted)} accepted, {len(result.low_confidence)} low confidence, "
                   f"{len(result.rejected)} rejected candidates (X-Z metres)", fill="black")
    d.text((8, 24), "blue door / teal window / magenta generic opening, dashed orange = low confidence, grey x = rejected candidate, "
                    "grey lines = walls", fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)


def render_opening_profile(grid, candidate, opening, outcome: str, path: Path) -> None:
    """u-v wall occupancy of one candidate. Dark = solid wall, mid-grey = occupied, light blue = points just in
    front of / behind the wall (possible occlusion). Green dashed = raw gap, red = refined jambs."""
    nv, nu = grid.solid.shape
    cell = 14
    img = Image.new("RGB", (nu * cell + 120, nv * cell + 110), "white")
    d = ImageDraw.Draw(img)
    ox, oy = 60, 50
    for r in range(nv):
        for c in range(nu):
            y = oy + (nv - 1 - r) * cell
            x = ox + c * cell
            colour = (30, 30, 30) if grid.solid[r, c] else (150, 150, 150) if grid.occupied[r, c] else (190, 215, 245) if grid.slab[r, c] else (255, 255, 255)
            d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=colour)
    d.rectangle([ox - 1, oy - 1, ox + nu * cell, oy + nv * cell], outline="black")

    def ux(u):
        return ox + (u - grid.u0) / grid.du * cell

    for u, colour, dash in ((candidate.u0, (0, 150, 0), True), (candidate.u1, (0, 150, 0), True)):
        if dash:
            _dashed(d, (ux(u), oy), (ux(u), oy + nv * cell), colour, 2, 6)
    if opening is not None:
        for jamb in (opening.left_jamb, opening.right_jamb):
            d.line([ux(jamb["u_m"]), oy, ux(jamb["u_m"]), oy + nv * cell], fill=(220, 0, 0), width=2)
    for k in range(0, nv, 2):
        d.text((8, oy + (nv - 1 - k) * cell), f"{grid.v0 + k * grid.dv:.1f}", fill="black")
    d.text((8, 8), f"{candidate.id} on {candidate.wall_id}: {outcome}", fill="black")
    sub = "" if opening is None else f"{opening.type}, width {opening.width_m:.2f} m [{opening.width_interval_m[0]:.2f}, {opening.width_interval_m[1]:.2f}]"
    d.text((8, 24), sub or f"raw gap {candidate.raw_width_m:.2f} m", fill=(60, 60, 60))
    d.text((8, oy + nv * cell + 8), f"u along the wall (m), 5 cm cells, from {grid.u0:.2f};  v height above floor (m), 10 cm rows; "
                                    "dark solid / grey occupied / blue = clutter near wall", fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def render_topdown(
    cloud: np.ndarray,
    bounds: tuple[float, float, float, float],
    title: str,
    path: Path,
    trajectory: np.ndarray | None = None,
) -> None:
    """Dark = many points. +X to the right, +Z up in the image, 1 m grid, fixed 100 px per metre."""
    x0, x1, z0, z1 = bounds
    w, h = int((x1 - x0) * PIXELS_PER_METRE), int((z1 - z0) * PIXELS_PER_METRE)
    col = ((cloud[:, 0] - x0) * PIXELS_PER_METRE).astype(int)
    row = h - 1 - ((cloud[:, 2] - z0) * PIXELS_PER_METRE).astype(int)
    keep = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    counts = np.zeros((h, w), dtype=np.int32)
    np.add.at(counts, (row[keep], col[keep]), 1)
    # Fixed contrast (not scaled to this image's own maximum) so BEFORE and AFTER compare fairly.
    gray = (255 - 255 * np.clip(np.log1p(counts) / np.log1p(20), 0, 1)).astype(np.uint8)

    canvas = Image.new("RGB", (w + 2 * MARGIN, h + 2 * MARGIN), "white")
    canvas.paste(Image.fromarray(gray).convert("RGB"), (MARGIN, MARGIN))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([MARGIN - 1, MARGIN - 1, MARGIN + w, MARGIN + h], outline="black")
    for x in np.arange(x0, x1 + 1, 1.0):
        px = MARGIN + int((x - x0) * PIXELS_PER_METRE)
        draw.line([px, MARGIN, px, MARGIN + h], fill=(215, 225, 255))
        draw.text((px - 8, MARGIN + h + 8), f"{x:.0f}", fill="black")
    for z in np.arange(z0, z1 + 1, 1.0):
        py = MARGIN + h - 1 - int((z - z0) * PIXELS_PER_METRE)
        draw.line([MARGIN, py, MARGIN + w, py], fill=(215, 225, 255))
        draw.text((MARGIN - 30, py - 5), f"{z:.0f}", fill="black")
    if trajectory is not None and len(trajectory) > 1:
        pts = [
            (MARGIN + (t[0] - x0) * PIXELS_PER_METRE, MARGIN + h - 1 - (t[2] - z0) * PIXELS_PER_METRE)
            for t in trajectory
        ]
        draw.line(pts, fill=(220, 40, 40), width=2)
    draw.text((MARGIN + w // 2 - 20, MARGIN + h + 30), "X (m)", fill="black")
    draw.text((8, 8), f"{title}    Z (m) up    {len(cloud)} points", fill="black")
    draw.text((8, 24), "red line = camera path, dark = dense, 1 m grid", fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
