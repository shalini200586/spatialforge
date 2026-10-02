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
