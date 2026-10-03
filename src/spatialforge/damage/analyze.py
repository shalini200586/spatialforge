"""Damage analysis over RGB evidence: detect -> map to surfaces -> metric extent -> merge views -> rules -> scope items.

The same code serves every tier. An EvidenceImage is an RGB image plus (when known) its camera and pose in the property frame:
LiDAR RGB frames with the production poses, video keyframes with their scaled poses, photos with their placed room poses.
Without geometry a detection is kept, unmapped, with null metric extents and a warning: nothing is guessed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from spatialforge.damage.detect import DetectionOptions, RawDetection, detect_damage, quality_of
from spatialforge.damage.geometry import CameraModel, Surface, build_surfaces, map_polygon, metric_extents
from spatialforge.damage.rules import RuleOptions, evaluate_rules
from spatialforge.damage.scope import build_scope
from spatialforge.pipeline.models import DamageRegion, Property

RANK = {"weak": 1, "moderate": 2, "strong": 3}
NAME = {1: "weak", 2: "moderate", 3: "strong"}


@dataclass
class EvidenceImage:
    id: str  # e.g. frame_000600 / kf_0012.jpg / room_001/img_02.jpg
    tier: str
    load: Callable[[], np.ndarray]  # RGB uint8 at the camera's resolution
    camera: CameraModel | None = None
    T_wc: np.ndarray | None = None  # camera-to-world in the property frame (metres, +Y up); None = no geometry
    floor_y: float | None = None  # height of the floor in that frame (photos: per room)
    room_hint: str | None = None  # photos: the image belongs to this room; only its surfaces are considered
    frame_index: int | None = None


@dataclass
class DamageOptions:
    detection: DetectionOptions = field(default_factory=DetectionOptions)
    rules: RuleOptions = field(default_factory=RuleOptions)
    merge_gap_m: float = 0.15  # two views of one damage: their surface boxes (grown by this) overlap
    max_unmapped: int = 20  # detections that could not be associated with any surface (kept, strongest first)
    write_images: bool = True
    max_annotated_images: int = 40


@dataclass
class DamageOutcome:
    regions: list[DamageRegion]
    flags: list[dict]
    items: list[dict]
    warnings: list[str]
    report: dict
    seconds: float = 0.0
    plan_points: dict = field(default_factory=dict)  # damage id -> (x, z) on the plan, for the diagnostic picture


def _region(d: RawDetection, img: EvidenceImage, size: tuple[int, int], surfaces: list[Surface], room_ids: set[str],
            have_geometry: bool, opts: DamageOptions) -> tuple[DamageRegion, tuple | None]:
    w, h = size
    source = {"image_id": img.id, "tier": img.tier, "frame_index": img.frame_index, "image_size": [w, h]}
    region = {"bbox": [round(v, 1) for v in d.bbox], "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in d.polygon],
              "area_px": round(d.area_px, 1)}
    warns: list[str] = []
    r = DamageRegion("tmp", d.damage_class, round(d.confidence, 3), d.quality, source, region,
                     room_id=img.room_hint if img.room_hint in room_ids else None, warnings=warns)
    plan_pt = None
    if not have_geometry or img.camera is None or img.T_wc is None or img.floor_y is None:
        warns.append("metric extent unavailable: no camera geometry for this image, so the region is not mapped to a surface")
        return r, plan_pt
    m = map_polygon(d.polygon, img.camera, img.T_wc, img.floor_y, surfaces, img.room_hint)
    if m.surface is None or m.agreement < 0.5:
        warns.append("metric extent unavailable: " + ("; ".join(m.reasons) or "no surface association"))
        return r, plan_pt
    s = m.surface
    r.surface_id, r.surface_type = s.id, s.kind
    rooms = [x for x in s.room_ids if x in room_ids]
    r.room_id = (img.room_hint if img.room_hint in rooms else (rooms[0] if rooms else r.room_id))
    if not m.reliable:
        warns.append("metric extent unavailable: " + "; ".join(m.reasons))
        return r, plan_pt
    q = NAME[min(RANK[d.quality], RANK[s.quality])]
    ext = metric_extents(m, img.tier, q)
    r.width, r.height, r.area, r.length = ext["width"], ext["height"], ext["area"], ext["length"]
    r.surface_extent, r.metric_reliable = ext["extent"], True
    uc, vc = (ext["extent"]["u_min"] + ext["extent"]["u_max"]) / 2, (ext["extent"]["v_min"] + ext["extent"]["v_max"]) / 2
    plan_pt = tuple((s.p0 + s.direction * uc).tolist()) if s.kind == "wall" else (uc, vc)
    return r, plan_pt


def _overlap(a: DamageRegion, b: DamageRegion, gap: float) -> bool:
    ea, eb = a.surface_extent, b.surface_extent
    return (ea["u_min"] - gap <= eb["u_max"] and eb["u_min"] - gap <= ea["u_max"]
            and ea["v_min"] - gap <= eb["v_max"] and eb["v_min"] - gap <= ea["v_max"])


def merge_views(regions: list[tuple[DamageRegion, tuple | None]], gap: float) -> list[tuple[DamageRegion, tuple | None]]:
    """The same damage seen from several images becomes ONE region (the strongest view), listing the other sources."""
    metric = [x for x in regions if x[0].metric_reliable]
    rest = [x for x in regions if not x[0].metric_reliable]
    metric.sort(key=lambda x: (-x[0].confidence, x[0].source["image_id"]))
    kept: list[tuple[DamageRegion, tuple | None]] = []
    for r, pt in metric:
        for k, _ in kept:
            if k.surface_id == r.surface_id and k.damage_class == r.damage_class and _overlap(k, r, gap):
                k.other_sources.append(r.source)
                k.confidence = round(min(1.0, k.confidence + 0.05), 3)  # seen in more than one view: a small, heuristic boost
                k.quality = quality_of(k.confidence)
                break
        else:
            kept.append((r, pt))
    return kept + rest


def analyze_damage(prop: Property, images: list[EvidenceImage], out_dir: Path | None = None,
                   opts: DamageOptions | None = None) -> DamageOutcome:
    t0 = time.perf_counter()
    opts = opts or DamageOptions()
    surfaces = build_surfaces(prop)
    room_ids = {r.id for r in prop.rooms}
    drafts: list[tuple[DamageRegion, tuple | None]] = []
    per_image = []
    annotated: list[tuple[str, np.ndarray, list]] = []
    warnings: list[str] = []
    for img in sorted(images, key=lambda i: i.id):
        try:
            rgb = img.load()
        except Exception as exc:
            warnings.append(f"evidence image {img.id} could not be read ({type(exc).__name__}: {exc}); skipped")
            continue
        dets = detect_damage(rgb, opts.detection)
        h, w = rgb.shape[:2]
        mapped = []
        for d in dets:
            mapped.append(_region(d, img, (w, h), surfaces, room_ids, bool(surfaces), opts))
        drafts += mapped
        per_image.append({"image_id": img.id, "tier": img.tier, "size": [w, h], "has_geometry": img.T_wc is not None,
                          "detections": len(dets), "classes": sorted({d.damage_class for d in dets})})
        if dets and opts.write_images and out_dir is not None and len(annotated) < opts.max_annotated_images:
            annotated.append((img.id, rgb, [r for r, _ in mapped]))
    merged = merge_views(drafts, opts.merge_gap_m)
    mapped_regions = [x for x in merged if x[0].surface_id is not None]
    unmapped = sorted((x for x in merged if x[0].surface_id is None), key=lambda x: (-x[0].confidence, x[0].source["image_id"]))
    dropped = max(0, len(unmapped) - opts.max_unmapped)
    kept = mapped_regions + unmapped[:opts.max_unmapped]
    if dropped:
        warnings.append(f"{dropped} further detections that could not be associated with any surface were dropped (weakest first).")

    def key(x):
        r = x[0]
        e = r.surface_extent
        return (r.surface_id or "~", r.damage_class, round(e["u_min"], 2) if e else 0.0, round(e["v_min"], 2) if e else 0.0,
                r.source["image_id"], r.image_region["bbox"][0])

    kept.sort(key=key)
    regions, plan_points = [], {}
    for n, (r, pt) in enumerate(kept, 1):
        r.id = f"damage_{n:03d}"
        regions.append(r)
        if pt is not None:
            plan_points[r.id] = pt
    flags, evaluation = evaluate_rules(regions, surfaces, opts.rules)
    tier = images[0].tier if images else "unknown"
    items = build_scope(regions, flags, tier)
    n_unmetric = sum(1 for r in regions if not r.metric_reliable)
    if n_unmetric:
        warnings.append(f"{n_unmetric} of {len(regions)} damage detections have no reliable metric extent (see each detection's "
                        "warnings); they are kept without width, height or area.")
    report = {
        "method": {"detector": "deterministic OpenCV image processing (no learned model)", "classes": ["crack", "stain"],
                   "crack": "black-hat dark-ridge extraction + elongation/thinness filters, straight lines rejected",
                   "stain": "chroma deviation from a median-filtered Lab background; pure shadows rejected",
                   "mapping": "pixel rays cast onto wall/floor/ceiling planes of the property; nearest valid surface; "
                              "metric extent only if >= 80% of the outline agrees on one non-grazing surface",
                   "confidence": "heuristic score, not a probability"},
        "images_analysed": len(per_image), "per_image": per_image, "detections_raw": len(drafts), "detections_reported": len(regions),
        "detections_with_metric_extent": sum(1 for r in regions if r.metric_reliable), "rule_evaluation": evaluation,
        "warnings": warnings}
    out = DamageOutcome(regions, flags, items, warnings, report, time.perf_counter() - t0, plan_points)
    if out_dir is not None:
        from spatialforge.damage.diagnostics import write_damage_diagnostics

        write_damage_diagnostics(Path(out_dir), prop, out, annotated)
    return out


def apply_to_property(prop: Property, out: DamageOutcome, images_analysed: int) -> None:
    """Attach the analysis to the canonical property and replace the "not implemented" notice with an honest description."""
    prop.damage = list(out.regions)
    prop.concealed_damage_flags = list(out.flags)
    prop.scope_line_items = list(out.items)
    prop.warnings = [w for w in prop.warnings if not w.startswith("Damage detection is not part of this version")]
    prop.warnings.append("Damage analysis is deterministic image processing (cracks, stains) on RGB evidence: detections are "
                         "candidates, not verified defects, and no ground truth was available to measure precision or recall.")
    prop.warnings += [f"damage: {w}" for w in out.warnings]
    if out.flags:
        prop.warnings.append(f"{len(out.flags)} concealed-damage RISK flag(s) were raised by explicit rules from visible evidence; "
                             "none of them asserts that hidden damage exists.")
    prop.provenance["damage_analysis"] = {
        "images_analysed": images_analysed, "detections": len(out.regions), "with_metric_extent": out.report["detections_with_metric_extent"],
        "concealed_flags": len(out.flags), "scope_items": len(out.items), "method": out.report["method"],
        "rules": {rid: v["rule"] for rid, v in out.report["rule_evaluation"]["rules"].items()}}
    prop.validate()
