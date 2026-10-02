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
