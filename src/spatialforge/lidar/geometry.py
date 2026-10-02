"""Small, vectorised geometry helpers for LiDAR depth frames (NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Metres per raw uint16 depth unit. This is an ASSUMPTION supported by indoor
# scene scale, not something the PNG files prove (see README, "Depth scale").
DEPTH_SCALE_M_PER_UNIT = 0.001


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def scale_intrinsics(
    intr: Intrinsics, src_width: int, src_height: int, dst_width: int, dst_height: int
) -> Intrinsics:
    """Rescale pinhole intrinsics from one image size to another (no cropping assumed)."""
    sx = dst_width / src_width
    sy = dst_height / src_height
    return Intrinsics(intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy)


def quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """3x3 rotation matrix from a quaternion (x, y, z, w). Normalised first."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("quaternion has zero or non-finite length")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def valid_depth_mask(
    depth_raw: np.ndarray,
    confidence: np.ndarray | None,
    min_confidence: int,
    min_range_m: float,
    max_range_m: float,
    depth_scale: float = DEPTH_SCALE_M_PER_UNIT,
) -> np.ndarray:
    """Boolean mask of usable depth pixels: non-zero, in range, confident enough."""
    metres = depth_raw.astype(np.float32) * depth_scale
    mask = (depth_raw > 0) & (metres >= min_range_m) & (metres <= max_range_m)
    if confidence is not None:
        mask &= confidence >= min_confidence
    return mask


def backproject(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
    depth_scale: float = DEPTH_SCALE_M_PER_UNIT,
) -> np.ndarray:
    """Camera-space XYZ (N, 3) in metres for masked pixels (pinhole model, +Z forward, +Y down).

    X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy,  Z = depth.
    """
    v, u = np.nonzero(mask)  # row = v, column = u
    z = depth_raw[v, u].astype(np.float64) * depth_scale
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.column_stack([x, y, z])


def transform_points(points: np.ndarray, rotation: np.ndarray, translation) -> np.ndarray:
    """world = R @ p + t for every row p."""
    return points @ rotation.T + np.asarray(translation, dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one point per occupied voxel: the first one in input order.

    Output is ordered by voxel index, so the result depends only on the input
    points and their order (no randomness).
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(points) == 0:
        return points
    idx = np.floor(points / voxel_size).astype(np.int64)
    idx -= idx.min(axis=0)
    dims = idx.max(axis=0) + 1
    if int(dims[0]) * int(dims[1]) * int(dims[2]) >= 2**62:
        raise ValueError("point cloud is too large for this voxel size; increase --voxel-size")
    keys = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    # return_index gives the first occurrence of each unique key.
    _, first = np.unique(keys, return_index=True)
    return points[first]


def write_ply(path, points: np.ndarray) -> None:
    """Write an ASCII XYZ-only PLY file."""
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(points)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(fh, points, fmt="%.5f")
