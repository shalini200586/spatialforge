"""Deterministic visible-damage detectors (OpenCV only; no network, no training, CPU, Windows-friendly).

crack-like damage
    Thin DARK ridges: morphological black-hat isolates dark structures narrower than a kernel, a robust threshold (median +
    k*MAD) adapts to wall texture, nearby fragments are joined, and a component is kept only if it is long, thin and elongated.
    Perfectly straight long lines are rejected (door frames, shelf edges, tile joints are straight; cracks meander).

stain / discoloration
    Colour that deviates from the local wall colour. The background is a large median of the image in Lab; the deviation is
    measured on CHROMA (a, b), so a pure shadow (brightness change, same hue) is not a stain. Dark patches with at least a
    small chroma shift are accepted. Blobs must be reasonably large, compact and smooth inside (textured objects fail).

Both detectors share one prior: damage lies on a PLAIN surface. The area around a candidate (a ring outside it) must be
homogeneous (low local texture). Posters, tile joints, fixtures, hoses, screens and furniture have busy surroundings and are
rejected; cracks and stains in a painted wall or ceiling have plain surroundings. Stains must also have SOFT edges (a screen, a
mug or a dark object has a sharp boundary).

These are candidate detectors, not a trained model: confidence is a heuristic score (not a probability), furniture or art with
stain-like colour can fire, and a dark crack-like object (a cable, a hair) can fire. Mapping to surfaces, metric extents and rules
build on the detections and say so wherever they cannot be sure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionOptions:
    work_max_side: int = 1024
    min_confidence: float = 0.35
    # plain-surface prior (shared)
    plain_window: int = 21  # local-texture window (pixels at working resolution)
    plain_sd: float = 3.5  # local std of the high-pass grey image below this = plain surface
    crack_ring: tuple[int, int] = (6, 28)  # ring around a crack: from 6 to 28 px outside it
    crack_min_plain_ring: float = 0.85  # share of the ring that must be plain
    stain_ring: tuple[int, int] = (12, 45)
    stain_min_plain_ring: float = 0.75
    stain_max_edge_sharpness: float = 0.15  # mean boundary gradient (grey/px) per grey level of contrast; objects are sharper
    stain_min_L: float = 22.0  # a blob darker than this (L*, 0-100) is a black object, not a stain
    max_cracks_per_image: int = 4
    max_stains_per_image: int = 3
    # crack-like
    crack_kernel_frac: float = 0.012  # black-hat kernel, as a fraction of the long side
    crack_min_contrast: float = 12.0  # grey levels
    crack_mad_k: float = 5.0
    crack_min_length_frac: float = 0.09  # of the image diagonal (shorter marks are handles, screws, grout fragments)
    crack_max_thickness_frac: float = 0.012
    crack_min_aspect: float = 7.0
    crack_merge_px: int = 7
    crack_straight_rms_px: float = 1.2  # centre-line deviation below this, over a long component: an edge or joint, not a crack
    crack_straight_min_frac: float = 0.0  # every candidate is tested for straightness (short straight pieces are joints, not cracks)
    hough_min_length_frac: float = 0.12  # straight segments at least this long are removed from the crack mask (grout, edges, frames)
    hough_votes: int = 40
    # stain / discoloration
    stain_bg_width: int = 160
    stain_bg_kernel_frac: float = 0.25
    stain_min_chroma: float = 7.0
    stain_mad_k: float = 4.0
    stain_dark_dl: float = 22.0
    stain_dark_min_chroma: float = 5.0
    stain_min_mean_chroma: float = 5.5  # mean chroma deviation of the blob: below this it is a shadow or lighting
    stain_max_mean_chroma: float = 26.0  # above this the colour is too saturated for a stain (cushions, magnets, labels)
    stain_max_brighter: float = 4.0  # a stain darkens a surface; a clearly brighter patch is a light, a reflection or a window
    stain_max_interior_variation: float = 9.0  # std of the colour deviation INSIDE the blob: a stain is one colour, a pattern is not
    stain_min_yellow_share: float = 0.45  # fraction of the chroma deviation that points toward yellow/brown (+b)
    stain_min_area_frac: float = 0.004
    stain_max_area_frac: float = 0.35
    stain_min_solidity: float = 0.6
    stain_max_edge_density: float = 0.10


@dataclass
class RawDetection:
    damage_class: str
    confidence: float
    quality: str
    polygon: np.ndarray  # (n, 2) pixels of the ORIGINAL image
    area_px: float
    features: dict = field(default_factory=dict)

    @property
    def bbox(self) -> list[float]:
        return [float(self.polygon[:, 0].min()), float(self.polygon[:, 1].min()),
                float(self.polygon[:, 0].max()), float(self.polygon[:, 1].max())]


def quality_of(conf: float) -> str:
    return "strong" if conf >= 0.7 else "moderate" if conf >= 0.5 else "weak"


def _odd(n: int) -> int:
    n = int(n)
    return n + 1 if n % 2 == 0 else n


def _polygon_of(mask_u8: np.ndarray, scale: float, eps_px: float = 1.5) -> np.ndarray | None:
    import cv2

    cs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2).astype(np.float64)
    if len(c) < 3:
        # a one- or two-pixel-wide component: give it a thin box so it is still a polygon
        x0, y0, w, h = cv2.boundingRect(c.astype(np.int32).reshape(-1, 1, 2))
        c = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], dtype=np.float64)
    return c * scale


def _local_texture(gray: np.ndarray, win: int) -> np.ndarray:
    """Local standard deviation of the HIGH-PASS grey image (window `win`): ~noise level on a plain wall even under a lighting
    ramp or a soft shadow, large on posters, tiles, fixtures. (A plain std would count a smooth gradient as texture.)"""
    import cv2

    g = gray.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 4.0)
    mean = cv2.blur(hp, (win, win))
    sq = cv2.blur(hp * hp, (win, win))
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def _ring_plain_fraction(comp: np.ndarray, sd: np.ndarray, inner: int, outer: int, tau: float) -> float:
    """Share of the ring between `inner` and `outer` pixels outside the component whose local texture is below tau."""
    import cv2

    ys, xs = np.nonzero(comp)
    h, w = comp.shape
    m = outer + 2
    x0, x1, y0, y1 = max(0, xs.min() - m), min(w, xs.max() + m + 1), max(0, ys.min() - m), min(h, ys.max() + m + 1)
    c = comp[y0:y1, x0:x1].astype(np.uint8)
    big = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer + 1, 2 * outer + 1)))
    small = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner + 1, 2 * inner + 1)))
    ring = (big > 0) & (small == 0)
    if ring.sum() < 50:
        return 0.0
    return float((sd[y0:y1, x0:x1][ring] < tau).mean())


def _centreline_rms(pts: np.ndarray, bin_px: float = 4.0) -> float:
    """How far the component's CENTRE LINE wanders from a straight line (pixels). The line's own width does not count, so a thick
    straight edge scores ~0 and a meandering crack scores several pixels."""
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    d = vt[0]
    t = q @ d
    s = q @ np.array([-d[1], d[0]])
    idx = np.floor((t - t.min()) / bin_px).astype(int)
    cnt = np.bincount(idx)
    ok = cnt > 0
    centre_s = np.bincount(idx, weights=s)[ok] / cnt[ok]
    centre_t = (np.arange(len(cnt))[ok] + 0.5) * bin_px
    if len(centre_t) < 3:
        return 0.0
    fit = np.polyval(np.polyfit(centre_t, centre_s, 1), centre_t)
    return float(np.sqrt(np.mean((centre_s - fit) ** 2)))


def detect_cracks(rgb: np.ndarray, opts: DetectionOptions, scale: float = 1.0) -> list[RawDetection]:
    import cv2

    h, w = rgb.shape[:2]
    diag = float(np.hypot(h, w))
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    k = _odd(max(5, round(opts.crack_kernel_frac * max(h, w))))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    med = float(np.median(bh))
    mad = 1.4826 * float(np.median(np.abs(bh - med)))
    thr = max(opts.crack_min_contrast, med + opts.crack_mad_k * mad)
    raw = bh > thr
    # long straight dark segments (tile joints, wall/floor lines, door and cabinet edges) are not cracks, and where they cross
    # they would otherwise merge into one wandering-looking component: remove them from the mask first
    segs = cv2.HoughLinesP(raw.astype(np.uint8) * 255, 1, np.pi / 180, opts.hough_votes,
                           minLineLength=int(opts.hough_min_length_frac * diag), maxLineGap=12)
    if segs is not None:
        straight = np.zeros(raw.shape, np.uint8)
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            cv2.line(straight, (int(x1), int(y1)), (int(x2), int(y2)), 255, 7)
        raw = raw & ~(straight > 0)
    sd = _local_texture(g, opts.plain_window)
    joined = cv2.dilate(raw.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opts.crack_merge_px, opts.crack_merge_px)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    min_len, max_thick = opts.crack_min_length_frac * diag, opts.crack_max_thickness_frac * diag
    out = []
    for i in range(1, n):
        bw, bhh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(bw, bhh) < min_len * 0.7:
            continue
        comp = labels == i
        ys, xs = np.nonzero(comp & raw)
        if len(xs) < 20:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        length = float(max(rw, rh))
        if length < min_len:
            continue
        thick = len(xs) / length
        aspect = length / max(thick, 1.0)
        if thick > max_thick or aspect < opts.crack_min_aspect:
            continue
        rms = _centreline_rms(pts)
        if length >= opts.crack_straight_min_frac * diag and rms < opts.crack_straight_rms_px:
            continue  # long and perfectly straight: an edge, a frame or a joint
        plain = _ring_plain_fraction(comp, sd, opts.crack_ring[0], opts.crack_ring[1], opts.plain_sd)
        if plain < opts.crack_min_plain_ring:
            continue  # busy surroundings: an object, a pattern or a joint pattern, not a crack in a plain surface
        contrast = float(bh[ys, xs].mean() / thr)
        conf = float(np.clip(0.30 + 0.30 * min(1.0, (contrast - 1.0) / 2.0) + 0.25 * min(1.0, length / (0.25 * diag))
                             + 0.15 * min(1.0, rms / 6.0), 0, 1))
        if conf < opts.min_confidence:
            continue
        poly = _polygon_of(comp.astype(np.uint8), scale)
        if poly is None:
            continue
        out.append(RawDetection("crack", conf, quality_of(conf), poly, float(len(xs)) * scale * scale,
                                {"length_px": length * scale, "thickness_px": thick * scale, "aspect": aspect, "contrast": contrast,
                                 "straightness_rms_px": rms * scale, "plain_ring": plain}))
    out.sort(key=lambda d: -d.confidence)
    return out[:opts.max_cracks_per_image]


def detect_stains(rgb: np.ndarray, opts: DetectionOptions, scale: float = 1.0) -> list[RawDetection]:
    import cv2

    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)  # L 0-100, a/b about -128..127
    u8 = np.dstack([lab[..., 0] * 2.55, lab[..., 1] + 128.0, lab[..., 2] + 128.0]).clip(0, 255).astype(np.uint8)
    bw = min(opts.stain_bg_width, w)
    bhh = max(8, round(h * bw / w))
    small = cv2.resize(u8, (bw, bhh), interpolation=cv2.INTER_AREA)
    k = _odd(max(5, round(opts.stain_bg_kernel_frac * bw)))
    bg = cv2.resize(np.dstack([cv2.medianBlur(small[..., c], k) for c in range(3)]), (w, h), interpolation=cv2.INTER_LINEAR)
    d = u8.astype(np.float32) - bg.astype(np.float32)
    dL, chroma = d[..., 0] / 2.55, np.hypot(d[..., 1], d[..., 2])
    med = float(np.median(chroma))
    mad = 1.4826 * float(np.median(np.abs(chroma - med)))
    thr = max(opts.stain_min_chroma, med + opts.stain_mad_k * mad)
    mask = ((chroma > thr) | ((dL < -opts.stain_dark_dl) & (chroma > opts.stain_dark_min_chroma))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 40, 120) > 0
    sd = _local_texture(gray, opts.plain_window)
    Lsm = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5)
    gmag = np.hypot(cv2.Sobel(Lsm, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(Lsm, cv2.CV_32F, 0, 1, ksize=3) / 8.0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = float(h * w)
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        frac = area / total
        if not opts.stain_min_area_frac <= frac <= opts.stain_max_area_frac:
            continue
        comp = labels == i
        cu8 = comp.astype(np.uint8)
        cs, _ = cv2.findContours(cu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(c)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < opts.stain_min_solidity:
            continue
        edge_density = float(edges[comp].mean())
        if edge_density > opts.stain_max_edge_density:
            continue  # textured inside: an object, not a smooth discoloration
        mean_chroma = float(chroma[comp].mean())
        if mean_chroma < opts.stain_min_mean_chroma:
            continue  # brightness change without a colour change: shadow or lighting
        if float(lab[..., 0][comp].mean()) < opts.stain_min_L:
            continue  # near-black: a screen, a mug, a dark object
        if float(dL[comp].mean()) > opts.stain_max_brighter:
            continue  # brighter than its surroundings: a light, a reflection, a window
        if mean_chroma > opts.stain_max_mean_chroma:
            continue  # too saturated for a stain: a coloured object
        interior = float(np.hypot(d[..., 1][comp].std(), d[..., 2][comp].std()))
        if interior > opts.stain_max_interior_variation:
            continue  # patterned or multi-coloured inside: a cushion, a label, a picture
        da_m, db_m = float(d[..., 1][comp].mean()), float(d[..., 2][comp].mean())
        yellow_share = max(db_m, 0.0) / max(float(np.hypot(da_m, db_m)), 1e-6)
        if mean_chroma >= 8.0 and yellow_share < opts.stain_min_yellow_share:
            continue  # a strong colour shift that is not toward yellow/brown (water, rust, nicotine, mould): not a typical stain
        plain = _ring_plain_fraction(comp, sd, opts.stain_ring[0], opts.stain_ring[1], opts.plain_sd)
        if plain < opts.stain_min_plain_ring:
            continue  # busy surroundings: not a discoloration of a plain surface
        boundary = comp & ~(cv2.erode(cu8, np.ones((3, 3), np.uint8)) > 0)
        ring_mask = cv2.dilate(cu8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
        ring_mask &= ~(cv2.dilate(cu8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0)
        contrast_L = abs(float(lab[..., 0][comp].mean()) - float(lab[..., 0][ring_mask].mean())) if ring_mask.any() else 0.0
        sharp = float(gmag[boundary].mean()) / max(contrast_L, 1.0) if boundary.any() else 1.0
        if sharp > opts.stain_max_edge_sharpness and mean_chroma < 12.0:
            continue  # a hard-edged patch of weak colour: an object, not a soft stain
        conf = float(np.clip(0.25 + 0.40 * min(1.0, mean_chroma / 20.0) + 0.20 * solidity + 0.15 * min(1.0, frac / 0.03), 0, 1))
        if conf < opts.min_confidence:
            continue
        poly = _polygon_of(cu8, scale, eps_px=2.5)
        if poly is None:
            continue
        out.append(RawDetection("stain", conf, quality_of(conf), poly, area * scale * scale,
                                {"mean_chroma_deviation": mean_chroma, "mean_dL": float(dL[comp].mean()), "solidity": solidity,
                                 "area_fraction": frac, "edge_density": edge_density, "plain_ring": plain, "edge_sharpness": sharp, "yellow_share": yellow_share}))
    out.sort(key=lambda d: -d.confidence)
    return out[:opts.max_stains_per_image]


def detect_damage(rgb: np.ndarray, opts: DetectionOptions | None = None) -> list[RawDetection]:
    """All detections in an RGB uint8 image; polygons are in the pixel coordinates of the INPUT image. Deterministic."""
    import cv2

    opts = opts or DetectionOptions()
    h, w = rgb.shape[:2]
    scale = 1.0
    work = rgb
    if max(h, w) > opts.work_max_side:
        s = opts.work_max_side / max(h, w)
        work = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        scale = 1.0 / s
    found = detect_cracks(work, opts, scale) + detect_stains(work, opts, scale)
    found.sort(key=lambda d: (d.damage_class, round(d.bbox[1]), round(d.bbox[0])))
    return found
