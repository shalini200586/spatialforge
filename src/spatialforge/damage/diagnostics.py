"""Damage diagnostic pictures and the machine-readable damage report (never product output)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from spatialforge.pipeline.serialization import write_json

COLOURS = {"crack": (230, 40, 40), "stain": (240, 150, 0)}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


def annotate_image(rgb: np.ndarray, regions) -> np.ndarray:
    """The evidence image with every detection outlined and labelled (id, class, confidence, surface)."""
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for r in regions:
        col = COLOURS[r.damage_class][::-1]
        poly = np.array(r.image_region["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, col, max(2, img.shape[1] // 400), cv2.LINE_AA)
        x0, y0 = int(r.image_region["bbox"][0]), int(r.image_region["bbox"][1])
        label = f"{r.id if r.id != 'tmp' else ''} {r.damage_class} {r.confidence:.2f}" + (f" {r.surface_id}" if r.surface_id else " (unmapped)")
        if r.width is not None:
            label += f" {r.width.value:.2f}x{r.height.value:.2f}m"
        scale = max(0.5, img.shape[1] / 1600)
        cv2.putText(img, label.strip(), (max(0, x0), max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, label.strip(), (max(0, x0), max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_damage_plan(path: Path, prop, outcome, size: int = 1000) -> None:
    """Top-down: walls, room outlines and one marker per damage that has a metric position on the plan."""
    from PIL import Image, ImageDraw

    pts = [[w.start.x, w.start.z] for w in prop.walls] + [[w.end.x, w.end.z] for w in prop.walls]
    for r in prop.rooms:
        pts += [[p.x, p.z] for p in r.polygon]
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((14, 10), f"Damage on the plan: {len(outcome.plan_points)} of {len(outcome.regions)} detections have a metric position "
           "(crack = red, stain = orange)", fill=(30, 30, 35))
    if not pts:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return
    a = np.array(pts)
    lo, hi = a.min(axis=0) - 0.5, a.max(axis=0) + 0.5
    s = (size - 100) / float(max(hi - lo))

    def px(p):
        return (50 + (p[0] - lo[0]) * s, 50 + (hi[1] - p[1]) * s)

    for r in prop.rooms:
        d.polygon([px((p.x, p.z)) for p in r.polygon], fill=(236, 241, 250), outline=(120, 130, 145))
    for w in prop.walls:
        d.line([px((w.start.x, w.start.z)), px((w.end.x, w.end.z))], fill=(55, 55, 60), width=3)
    for reg in outcome.regions:
        if reg.id in outcome.plan_points:
            x, y = px(outcome.plan_points[reg.id])
            c = COLOURS[reg.damage_class]
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=c, outline=(255, 255, 255))
            d.text((x + 9, y - 7), reg.id.replace("damage_", "D"), fill=(30, 30, 35))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_damage_diagnostics(out_dir: Path, prop, outcome, annotated: list) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "damage_report.json", {
        **outcome.report,
        "regions": [r.to_dict() for r in outcome.regions], "concealed_damage_flags": outcome.flags, "scope_line_items": outcome.items})
    render_damage_plan(out_dir / "damage_plan.png", prop, outcome)
    # annotate the evidence images, now that final ids exist (matched by source image and bbox)
    by_img: dict[str, list] = {}
    for r in outcome.regions:
        by_img.setdefault(r.source["image_id"], []).append(r)
    for image_id, rgb, _drafts in annotated:
        regs = by_img.get(image_id)
        if not regs:
            continue
        ann = annotate_image(rgb, regs)
        (out_dir / "images").mkdir(exist_ok=True)
        cv2.imwrite(str(out_dir / "images" / f"{_safe(image_id)}.jpg"), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
