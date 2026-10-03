"""Diagnostic pictures for stitching (not product output): rooms placed in the global frame, the constraint graph on top."""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLOURS = [(214, 228, 247), (222, 240, 218), (250, 232, 208), (237, 222, 245), (211, 238, 238), (248, 224, 224)]
EDGE = {"strong": (0, 140, 60), "moderate": (220, 140, 0), "weak": (200, 40, 40), "rejected": (150, 150, 150)}


def render_stitching_png(path: Path, rooms: list[dict], edges: list[dict], title: str, notes: list[str], size: int = 1100) -> None:
    """rooms: {id, polygon (n,2)|None, walls [(a, b)], cameras (m,2)} in the global frame; edges: {a, b, quality, status}."""
    from PIL import Image, ImageDraw, ImageFont

    pts = []
    for r in rooms:
        if r.get("polygon") is not None:
            pts.append(np.asarray(r["polygon"]))
        for a, b in r.get("walls", []):
            pts.append(np.array([a, b]))
        if len(r.get("cameras", [])):
            pts.append(np.asarray(r["cameras"]))
    try:
        font, small = ImageFont.truetype("arial.ttf", 18), ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((16, 12), title, fill=(30, 30, 35), font=font)
    for i, n in enumerate(notes[:3]):
        d.text((16, 40 + 18 * i), n, fill=(190, 100, 0), font=small)
    if not pts:
        d.text((16, 120), "nothing to draw", fill=(110, 110, 120), font=font)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0) - 0.5, allp.max(axis=0) + 0.5
    scale = (size - 160) / float(max(hi - lo))
    ox, oy = 60, 110

    def px(p):
        return (ox + (p[0] - lo[0]) * scale, oy + (hi[1] - p[1]) * scale)

    centres = {}
    for k, r in enumerate(rooms):
        if r.get("polygon") is not None:
            poly = np.asarray(r["polygon"])
            d.polygon([px(p) for p in poly], fill=COLOURS[k % len(COLOURS)], outline=(110, 120, 135))
            centres[r["id"]] = poly.mean(axis=0)
        elif r.get("walls"):
            centres[r["id"]] = np.mean([np.mean([a, b], axis=0) for a, b in r["walls"]], axis=0)
        for a, b in r.get("walls", []):
            d.line([px(a), px(b)], fill=(55, 55, 60), width=3)
        for c in r.get("cameras", []):
            x, y = px(c)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(40, 90, 200))
        if r["id"] in centres:
            x, y = px(centres[r["id"]])
            d.text((x - 30, y - 8), r["id"].replace("room_", "R"), fill=(30, 30, 35), font=font)
    for e in edges:
        if e["a"] in centres and e["b"] in centres:
            col = EDGE["rejected"] if e.get("status") == "rejected" else EDGE.get(e.get("quality", "weak"), EDGE["weak"])
            d.line([px(centres[e["a"]]), px(centres[e["b"]])], fill=col, width=3)
    d.text((16, size - 24), "blue dots: photo positions · green/orange/red links: strong/moderate/weak stitch constraints "
           "· grey: rejected", fill=(110, 110, 120), font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
