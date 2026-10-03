"""Metric scale for an SfM trajectory, from metric monocular depth.

SfM alone is scale-ambiguous. For each depth keyframe we compare the metric depth the model predicts with the SfM
depth of the sparse points visible in that image (same pixels), giving scale observations
`metric_depth / sfm_depth`. A per-frame scale is the robust (log-domain) median of those observations; the global
scale is the median of the per-frame scales. The spread across frames is reported and drives the uncertainty and the
scale-quality label. If the evidence is too thin or too inconsistent, no scale is returned (`available = False`):
metres are never invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCALE_QUALITIES = ("strong", "moderate", "weak", "unavailable")


@dataclass
class ScaleOptions:
    min_range_m: float = 0.2  # predicted depth outside this range is ignored
    max_range_m: float = 12.0
    window: int = 5  # median window (depth pixels) when sampling the depth map
    edge_threshold: float = 0.15  # relative depth gradient above which a pixel is a depth edge
    ratio_limits: tuple[float, float] = (0.05, 20.0)
    inlier_sigmas: float = 2.5
    min_inlier_floor: float = 0.02  # lower bound on the robust spread used for inlier rejection (log domain)
    min_correspondences: int = 15  # per frame
    min_frames: int = 5
    min_frames_for_moderate: int = 1  # photos: a scale seen by fewer frames than this is at best WEAK
    min_total_correspondences: int = 200
    # Systematic error of an uncalibrated monocular metric-depth model: sigma never goes below this. Measured against
    # ARKit poses on the sample captures, the depth-derived scale was off by +13.9% (oracle poses) and -12.1% (real SfM).
    model_floor_rel: float = 0.10
    # STRONG would mean "validated against a known length". Uncalibrated monocular depth never earns it; a calibrated
    # backend (or a reference object) may set this True.
    allow_strong: bool = False
    # quality rules, on the combined relative sigma and the raw frame-to-frame spread
    strong_sigma: float = 0.08
    moderate_sigma: float = 0.16
    weak_sigma: float = 0.30
    strong_spread: float = 0.30
    moderate_spread: float = 0.50
    max_spread: float = 0.90  # beyond this the scales are not mutually consistent at all


@dataclass
class FrameScale:
    name: str
    correspondences: int  # candidate sparse points visible in the image
    used: int  # after range / edge / outlier rejection
    scale: float | None  # metric depth per SfM unit seen by this frame
    spread_rel: float | None  # robust within-frame relative spread of the observations
    reason: str | None = None  # why the frame gave no scale

    def to_dict(self) -> dict:
        return {"name": self.name, "correspondences": self.correspondences, "used": self.used,
                "scale": None if self.scale is None else round(self.scale, 4),
                "spread_rel": None if self.spread_rel is None else round(self.spread_rel, 4), "reason": self.reason}


@dataclass
class ScaleEstimate:
    available: bool
    quality: str
    scale: float | None  # metres per SfM unit
    frames_used: int
    frames_tried: int
    correspondences: int
    frame_spread_rel: float | None  # robust relative spread (1.4826*MAD in log domain) of the per-frame scales
    standard_error_rel: float | None
    sigma_rel: float | None  # combined relative 1-sigma of the global scale
    min_frame_scale: float | None = None
    max_frame_scale: float | None = None
    reasons: list[str] = field(default_factory=list)
    frames: list[FrameScale] = field(default_factory=list)
    model_floor_rel: float = 0.10

    def to_dict(self) -> dict:
        r = lambda v, n=4: None if v is None else round(float(v), n)  # noqa: E731
        return {"metric_scale_available": self.available, "scale_quality": self.quality, "metres_per_sfm_unit": r(self.scale, 5),
                "frames_used": self.frames_used, "frames_tried": self.frames_tried, "correspondences": self.correspondences,
                "median_scale": r(self.scale, 5), "frame_spread_rel": r(self.frame_spread_rel),
                "standard_error_rel": r(self.standard_error_rel), "model_floor_rel": self.model_floor_rel,
                "sigma_rel": r(self.sigma_rel), "min_frame_scale": r(self.min_frame_scale, 5),
                "max_frame_scale": r(self.max_frame_scale, 5), "reasons": self.reasons,
                "per_frame": [f.to_dict() for f in self.frames]}


def mad_sigma(x: np.ndarray) -> float:
    """Robust standard deviation: 1.4826 * median absolute deviation."""
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def sample_depth(depth: np.ndarray, xy: np.ndarray, image_size: tuple[int, int], opts: ScaleOptions) -> tuple[np.ndarray, np.ndarray]:
    """Predicted depth at image pixels `xy` (N,2; image coordinates of an image of `image_size`=(W,H)).

    Returns (values, ok): median over a small window at depth-map resolution; `ok` is False for invalid values and for
    pixels on depth edges (where a one-pixel misregistration swaps foreground and background depth).
    """
    h, w = depth.shape
    W, H = image_size
    u = np.clip(np.floor(xy[:, 0] * w / W).astype(int), 0, w - 1)
    v = np.clip(np.floor(xy[:, 1] * h / H).astype(int), 0, h - 1)
    gy, gx = np.gradient(depth.astype(np.float64))
    rel_grad = (np.abs(gx) + np.abs(gy)) / np.maximum(depth, 1e-6)
    off = np.arange(-(opts.window // 2), opts.window // 2 + 1)
    vv = np.clip(v[:, None, None] + off[None, :, None], 0, h - 1)  # (N, win, 1)
    uu = np.clip(u[:, None, None] + off[None, None, :], 0, w - 1)  # (N, 1, win)
    vals = np.median(depth[vv, uu].reshape(len(xy), -1), axis=1)
    edge = rel_grad[vv, uu].reshape(len(xy), -1).max(axis=1) > opts.edge_threshold
    ok = np.isfinite(vals) & (vals >= opts.min_range_m) & (vals <= opts.max_range_m) & ~edge
    return vals, ok


def frame_scale(name: str, xy: np.ndarray, sfm_depth: np.ndarray, depth: np.ndarray, image_size: tuple[int, int],
                opts: ScaleOptions) -> FrameScale:
    """Scale observation for one keyframe from its visible sparse points (xy pixels, sfm_depth in SfM units)."""
    n = len(xy)
    if n == 0:
        return FrameScale(name, 0, 0, None, None, "no sparse points visible")
    front = sfm_depth > 1e-9
    pred, ok = sample_depth(depth, xy, image_size, opts)
    keep = ok & front
    ratio = np.full(n, np.nan)
    ratio[keep] = pred[keep] / sfm_depth[keep]
    lo, hi = opts.ratio_limits
    keep &= (ratio >= lo) & (ratio <= hi)
    if keep.sum() < opts.min_correspondences:
        return FrameScale(name, n, int(keep.sum()), None, None, "too few valid correspondences")
    lr = np.log(ratio[keep])
    med = np.median(lr)
    sig = max(mad_sigma(lr), opts.min_inlier_floor)
    inl = np.abs(lr - med) <= opts.inlier_sigmas * sig
    if inl.sum() < opts.min_correspondences:
        return FrameScale(name, n, int(inl.sum()), None, None, "too few inliers after outlier rejection")
    li = lr[inl]
    return FrameScale(name, n, int(inl.sum()), float(np.exp(np.median(li))), float(mad_sigma(li)))


def estimate_metric_scale(frames: list[FrameScale], opts: ScaleOptions | None = None) -> ScaleEstimate:
    """Robust global scale (median of per-frame scales) with explicit spread, uncertainty and quality."""
    opts = opts or ScaleOptions()
    good = [f for f in frames if f.scale is not None]
    total = int(sum(f.used for f in good))
    reasons: list[str] = []
    base = dict(frames_tried=len(frames), frames=list(frames), model_floor_rel=opts.model_floor_rel)
    if len(good) < opts.min_frames:
        reasons.append(f"only {len(good)} keyframes gave a scale observation (need {opts.min_frames})")
    if total < opts.min_total_correspondences:
        reasons.append(f"only {total} usable depth/SfM correspondences (need {opts.min_total_correspondences})")
    if reasons:
        return ScaleEstimate(False, "unavailable", None, len(good), correspondences=total, frame_spread_rel=None,
                             standard_error_rel=None, sigma_rel=None, reasons=reasons, **base)
    ls = np.log([f.scale for f in good])
    g = float(np.exp(np.median(ls)))
    # a MAD over 2-4 values is meaningless: use the plain sample standard deviation there
    spread = mad_sigma(ls) if len(ls) >= 5 else float(np.std(ls, ddof=1))
    se = spread / np.sqrt(len(good))
    sigma = float(np.hypot(se, opts.model_floor_rel))
    reasons.append(f"median of {len(good)} per-frame scales over {total} correspondences")
    reasons.append(f"frame-to-frame spread {spread:.0%}, standard error {se:.1%}, model floor {opts.model_floor_rel:.0%}")
    if spread > opts.max_spread or sigma > opts.weak_sigma:
        reasons.append("per-frame scales are mutually inconsistent; no reliable metric scale")
        return ScaleEstimate(False, "unavailable", None, len(good), correspondences=total, frame_spread_rel=float(spread),
                             standard_error_rel=float(se), sigma_rel=sigma, reasons=reasons,
                             min_frame_scale=float(np.exp(ls.min())), max_frame_scale=float(np.exp(ls.max())), **base)
    if opts.allow_strong and sigma <= opts.strong_sigma and spread <= opts.strong_spread:
        quality = "strong"
    elif sigma <= opts.moderate_sigma and spread <= opts.moderate_spread:
        quality = "moderate"
    else:
        quality = "weak"
        reasons.append("scale varies a lot between keyframes: metric scale is weak")
    if quality == "moderate" and len(good) < opts.min_frames_for_moderate:
        quality = "weak"
        reasons.append(f"only {len(good)} images support the scale (moderate needs {opts.min_frames_for_moderate})")
    return ScaleEstimate(True, quality, g, len(good), correspondences=total, frame_spread_rel=float(spread),
                         standard_error_rel=float(se), sigma_rel=sigma, reasons=reasons,
                         min_frame_scale=float(np.exp(ls.min())), max_frame_scale=float(np.exp(ls.max())), **base)


def frame_alignment_factor(frame: FrameScale, global_scale: float, limits: tuple[float, float] = (0.4, 2.5)) -> float | None:
    """Multiplier that makes this frame's dense depth consistent with the SfM geometry at the global metric scale.

    z_metric = z_sfm * s_global and pred ~ z_sfm * s_frame, so the factor is s_global / s_frame. Frames whose factor
    is extreme are inconsistent with the trajectory and are excluded (None).
    """
    if frame.scale is None or global_scale <= 0:
        return None
    f = global_scale / frame.scale
    return float(f) if limits[0] <= f <= limits[1] else None


def metric_camera_to_world(rotation_cw: np.ndarray, translation_cw: np.ndarray, scale: float) -> np.ndarray:
    """Camera-to-world pose in metres: the camera centre is scaled, orientation is untouched (pure scale, no re-rotation)."""
    R = np.asarray(rotation_cw, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation_cw, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t * scale
    return T
