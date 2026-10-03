"""Back-project metric depth from keyframes with scaled SfM poses and fuse into one metric point cloud.

Deterministic throughout: fixed pixel stride, voxel-hash multi-view check, voxel downsampling that keeps the first point.

Multi-view consistency (kept deliberately simple, no learned MVS): a point survives only if its 10 cm voxel is also
occupied by points of at least `min_views` DIFFERENT keyframes. Surfaces seen by one frame only (depth noise,
floaters, single-view guesses) are dropped. The share removed is reported and feeds the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spatialforge.lidar.geometry import voxel_downsample


@dataclass
class FusionOptions:
    min_range_m: float = 0.3
    max_range_m: float = 8.0
    pixel_stride: int = 3
    edge_threshold: float = 0.10  # relative depth gradient: drop depth-discontinuity pixels
    consistency_voxel_m: float = 0.10
    min_views: int = 2
    voxel_m: float = 0.03
    max_points: int = 3_000_000


@dataclass
class FusionFrame:
    name: str
    depth_m: np.ndarray  # (h, w) metric depth, already aligned to the SfM geometry at the global scale
    fx: float  # intrinsics at DEPTH-MAP resolution
    fy: float
    cx: float
    cy: float
    cam_to_world: np.ndarray  # (4, 4), metres; camera axes x right, y down, z forward
    k1: float = 0.0  # SIMPLE_RADIAL distortion: x_distorted = x_undistorted * (1 + k1 * r_undistorted^2)


@dataclass
class FusionResult:
    points: np.ndarray
    frames_used: int
    candidate_points: int
    after_range_edge: int
    after_consistency: int
    after_voxel: int
    consistency_removed_fraction: float
    warnings: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        return {"frames_used": self.frames_used, "candidate_pixels": self.candidate_points,
                "after_range_and_edge_filter": self.after_range_edge, "after_multiview_consistency": self.after_consistency,
                "after_voxel_downsample": self.after_voxel,
                "multiview_removed_fraction": round(self.consistency_removed_fraction, 4), "warnings": self.warnings}


def backproject_frame(frame: FusionFrame, opts: FusionOptions) -> tuple[np.ndarray, int]:
    """World-frame points (N,3) for one keyframe and the number of pixels considered. Camera model: pinhole."""
    d = np.asarray(frame.depth_m, dtype=np.float64)
    h, w = d.shape
    gy, gx = np.gradient(d)
    rel_grad = (np.abs(gx) + np.abs(gy)) / np.maximum(d, 1e-6)
    s = opts.pixel_stride
    v, u = np.mgrid[0:h:s, 0:w:s]
    z = d[v, u]
    ok = np.isfinite(z) & (z >= opts.min_range_m) & (z <= opts.max_range_m) & (rel_grad[v, u] <= opts.edge_threshold)
    considered = int(z.size)
    u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), z[ok]
    xd, yd = (u + 0.5 - frame.cx) / frame.fx, (v + 0.5 - frame.cy) / frame.fy
    xu, yu = xd, yd
    if frame.k1:  # undo the radial distortion (fixed-point iteration; converges quickly for small k1)
        for _ in range(8):
            f = 1.0 + frame.k1 * (xu * xu + yu * yu)
            xu, yu = xd / f, yd / f
    cam = np.column_stack([xu * z, yu * z, z])
    T = frame.cam_to_world
    return cam @ T[:3, :3].T + T[:3, 3], considered


def multiview_support(points: np.ndarray, frame_ids: np.ndarray, voxel_m: float, min_views: int) -> np.ndarray:
    """Boolean mask: the point's voxel contains points from at least `min_views` distinct frames."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    idx = np.floor(points / voxel_m).astype(np.int64)
    idx -= idx.min(axis=0)
    dims = idx.max(axis=0) + 1
    keys = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    # distinct (voxel, frame) pairs -> number of frames per voxel
    pair = np.unique(np.column_stack([keys, frame_ids]), axis=0)
    uniq, counts = np.unique(pair[:, 0], return_counts=True)
    support = dict(zip(uniq.tolist(), counts.tolist()))
    return np.fromiter((support[k] >= min_views for k in keys.tolist()), dtype=bool, count=len(keys))


def fuse_frames(frames: list[FusionFrame], opts: FusionOptions | None = None) -> FusionResult:
    opts = opts or FusionOptions()
    chunks, ids, considered = [], [], 0
    for i, f in enumerate(frames):
        p, n = backproject_frame(f, opts)
        considered += n
        chunks.append(p)
        ids.append(np.full(len(p), i, dtype=np.int64))
    warnings: list[str] = []
    if not chunks or sum(len(c) for c in chunks) == 0:
        return FusionResult(np.zeros((0, 3)), 0, considered, 0, 0, 0, 0.0, ["no usable depth points"])
    pts, fid = np.vstack(chunks), np.concatenate(ids)
    n0 = len(pts)
    keep = multiview_support(pts, fid, opts.consistency_voxel_m, opts.min_views) if len(frames) >= opts.min_views else \
        np.ones(n0, dtype=bool)
    if len(frames) < opts.min_views:
        warnings.append("fewer keyframes than the multi-view requirement: consistency check skipped")
    pts = pts[keep]
    removed = 1.0 - len(pts) / n0
    if removed > 0.6:
        warnings.append(f"{removed:.0%} of depth points had no support from a second keyframe and were removed")
    n1 = len(pts)
    if len(pts):
        pts = voxel_downsample(pts, opts.voxel_m)
    if len(pts) > opts.max_points:
        pts = pts[np.linspace(0, len(pts) - 1, opts.max_points).astype(np.int64)]
        warnings.append(f"cloud thinned to {opts.max_points} points (deterministic stride)")
    return FusionResult(pts, len(frames), considered, n0, n1, len(pts), float(removed), warnings)
