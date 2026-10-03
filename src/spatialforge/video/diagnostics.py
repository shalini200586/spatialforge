"""Small diagnostic artifacts for the video tier (never committed; written under the run's diagnostics/ folder)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_binary_ply(path: Path, points: np.ndarray, max_points: int = 400_000, comment: str = "") -> int:
    """Binary little-endian XYZ PLY (float32), deterministic stride thinning. Returns the number of points written."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(np.int64)]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ply\nformat binary_little_endian 1.0\n" + (f"comment {comment}\n" if comment else "") + \
        f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(pts.astype("<f4").tobytes())
    return int(len(pts))


def render_trajectory_png(path: Path, cameras: np.ndarray, sparse: np.ndarray | None, metric: bool, title: str,
                          size: int = 900) -> None:
    """Top-down (x, z) view of the camera path and sparse points. Axes are in metres when `metric`, else SfM units."""
    from PIL import Image, ImageDraw

    cams = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((14, 10), title, fill=(30, 30, 30))
    if len(cams) == 0:
        d.text((14, 40), "no registered cameras", fill=(190, 100, 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return
    allp = cams[:, [0, 2]] if sparse is None or not len(sparse) else np.vstack([cams[:, [0, 2]], sparse[:, [0, 2]]])
    lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    span = float(max(hi - lo)) or 1.0
    margin, scale = 60, (size - 120) / span

    def px(p):
        return (margin + (p[0] - lo[0]) * scale, size - margin - (p[1] - lo[1]) * scale)

    if sparse is not None:
        for p in sparse[:: max(1, len(sparse) // 4000)]:
            x, y = px((p[0], p[2]))
            if 0 <= x < size and 0 <= y < size:
                d.point((x, y), fill=(170, 170, 175))
    xy = [px((c[0], c[2])) for c in cams]
    d.line(xy, fill=(120, 150, 200), width=2)
    n = len(xy)
    for i, (x, y) in enumerate(xy):
        t = i / max(1, n - 1)
        col = (int(40 + 200 * t), int(120 - 60 * t), int(220 - 180 * t))
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    d.text((xy[0][0] + 6, xy[0][1] - 12), "start", fill=(40, 120, 220))
    d.text((xy[-1][0] + 6, xy[-1][1] - 12), "end", fill=(240, 60, 40))
    unit = "m" if metric else "SfM units (scale unknown)"
    bar = 1.0 if span > 4 else 0.5
    if not metric:
        bar = float(10 ** np.floor(np.log10(span / 3)))
    x0, y0 = margin, size - 28
    d.line([(x0, y0), (x0 + bar * scale, y0)], fill=(30, 30, 30), width=3)
    d.text((x0, y0 - 14), f"{bar:g} {unit}", fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
