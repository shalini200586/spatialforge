"""Build a metric point cloud (PLY) from a LiDAR capture's depth frames and poses."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from spatialforge.lidar import geometry
from spatialforge.lidar.geometry import DEPTH_SCALE_M_PER_UNIT, Intrinsics
from spatialforge.lidar.models import Status
from spatialforge.lidar.validator import resolve_capture_root, validate_lidar_capture
from spatialforge.lidar.video_info import read_video_size

# Verified on the three sample captures (see README, "Pose convention"):
# poses are camera-to-world, and camera axes are +X right, +Y down, +Z forward.
POSE_CONVENTION = "camera-to-world: world = R(q) @ p_cam + t; camera axes +X right, +Y down, +Z forward"

ASPECT_TOLERANCE = 0.01  # RGB vs depth aspect ratio may differ by at most 1 %
PRINCIPAL_POINT_TOLERANCE = 0.1  # principal point within 10 % of image size from the centre


class ReconstructionError(Exception):
    """A user-facing problem (bad capture, missing data, unwritable output)."""


@dataclass
class ReconstructionOptions:
    frame_step: int = 30
    max_frames: int | None = 60
    min_confidence: int = 2
    voxel_size: float = 0.02
    min_range_m: float = 0.1
    max_range_m: float = 5.0
    source_size: tuple[int, int] | None = None  # override for the intrinsics' image size


@dataclass
class ReconstructionResult:
    capture_path: Path
    output_path: Path
    frames_available: int = 0
    frames_used: int = 0
    depth_scale: float = DEPTH_SCALE_M_PER_UNIT
    pose_convention: str = POSE_CONVENTION
    source_size: tuple[int, int] = (0, 0)
    depth_size: tuple[int, int] = (0, 0)
    scale_factors: tuple[float, float] = (0.0, 0.0)
    example_frame: int = -1
    example_source_intrinsics: Intrinsics | None = None
    example_depth_intrinsics: Intrinsics | None = None
    candidate_points: int = 0  # every pixel of every used frame
    removed_by_depth: int = 0  # zero depth or outside the metric range
    removed_by_confidence: int = 0
    points_after_filter: int = 0
    points_after_downsample: int = 0
    trajectory_min: np.ndarray | None = None
    trajectory_max: np.ndarray | None = None
    cloud_min: np.ndarray | None = None
    cloud_max: np.ndarray | None = None
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ---------- loading ----------


def load_poses(path: Path) -> dict[int, dict]:
    """frame id -> {'t': xyz, 'q': (qx, qy, qz, qw), 'intr': Intrinsics or None}."""
    poses: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = [h.strip().lower() for h in next(reader, [])]
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            try:
                fid = int(row[col["frame"]])
                t = [float(row[col[k]]) for k in ("x", "y", "z")]
                q = [float(row[col[k]]) for k in ("qx", "qy", "qz", "qw")]
            except (ValueError, IndexError, KeyError):
                continue  # the validator already reported malformed rows
            intr = None
            try:
                vals = [float(row[col[k]]) for k in ("fx", "fy", "cx", "cy")]
                intr = Intrinsics(*vals)
            except (ValueError, IndexError, KeyError):
                pass
            poses[fid] = {"t": t, "q": q, "intr": intr}
    return poses


def load_camera_matrix(path: Path) -> Intrinsics | None:
    try:
        with open(path, newline="") as fh:
            rows = [[float(c) for c in r] for r in csv.reader(fh) if r]
        return Intrinsics(rows[0][0], rows[1][1], rows[0][2], rows[1][2])
    except (OSError, ValueError, IndexError):
        return None


def select_frames(ids: list[int], frame_step: int, max_frames: int | None) -> list[int]:
    """Every Nth frame, then (if still too many) evenly spaced down to max_frames. No randomness."""
    if frame_step < 1:
        raise ReconstructionError("--frame-step must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ReconstructionError("--max-frames must be at least 1")
    chosen = sorted(ids)[::frame_step]
    if max_frames is not None and len(chosen) > max_frames:
        if max_frames == 1:
            return chosen[:1]
        picks = [round(i * (len(chosen) - 1) / (max_frames - 1)) for i in range(max_frames)]
        chosen = [chosen[i] for i in picks]
    return chosen


def _read_png(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            return np.array(im)
    except Exception as exc:
        raise ReconstructionError(f"could not read {path.name}: {exc}") from exc


def sanity_flags(result: ReconstructionResult, max_range_m: float) -> list[str]:
    """Simple numeric plausibility checks for an indoor scan. Empty list means nothing suspicious."""
    flags = []
    cloud_extent = result.cloud_max - result.cloud_min
    traj_extent = result.trajectory_max - result.trajectory_min
    if cloud_extent.max() > 50:
        flags.append(f"cloud is {cloud_extent.max():.0f} m across: too large for a room (wrong depth scale?)")
    if cloud_extent.max() < 0.5:
        flags.append(f"cloud is only {cloud_extent.max():.2f} m across: too small for a room (wrong depth scale?)")
    if cloud_extent.max() > 3 * max(traj_extent.max(), max_range_m):
        flags.append("cloud is much larger than the camera path allows (possible duplicated/separated geometry)")
    lo, hi = result.cloud_min - max_range_m, result.cloud_max + max_range_m
    if (result.trajectory_min < lo).any() or (result.trajectory_max > hi).any():
        flags.append("camera path leaves the point cloud's bounding box by more than the max depth range")
    return flags


# ---------- main entry point ----------


def reconstruct_lidar(
    capture: str | Path, output: str | Path, options: ReconstructionOptions | None = None
) -> ReconstructionResult:
    opts = options or ReconstructionOptions()
    started = time.perf_counter()

    validation = validate_lidar_capture(capture)
    if validation.status is Status.INVALID:
        raise ReconstructionError(
            "capture is INVALID; run validate-lidar for details. First error: " + validation.errors[0]
        )
    root = resolve_capture_root(Path(capture))
    result = ReconstructionResult(capture_path=root, output_path=Path(output))
    result.warnings.extend(validation.warnings)

    poses = load_poses(root / "odometry.csv")
    depth_dir, conf_dir = root / "depth", root / "confidence"
    depth_ids = sorted(int(p.stem) for p in depth_dir.glob("*.png") if p.stem.isdigit())
    usable = [i for i in depth_ids if i in poses]
    result.frames_available = len(depth_ids)
    if not usable:
        raise ReconstructionError("none of the depth frames has a pose row in odometry.csv")
    if len(usable) < len(depth_ids):
        result.warnings.append(f"{len(depth_ids) - len(usable)} depth frame(s) have no pose and are skipped")

    frames = select_frames(usable, opts.frame_step, opts.max_frames)
    result.frames_used = len(frames)
    if opts.min_confidence > 0 and not conf_dir.is_dir():
        raise ReconstructionError("confidence/ folder missing; use --min-confidence 0 to ignore confidence")

    # --- intrinsics: source image size, then scale to depth size ---
    depth_h, depth_w = _read_png(depth_dir / f"{frames[0]:06d}.png").shape[:2]
    result.depth_size = (depth_w, depth_h)
    if opts.source_size is not None:
        src_w, src_h = opts.source_size
    else:
        try:
            src_w, src_h = read_video_size(root / "rgb.mp4")
        except ValueError as exc:
            raise ReconstructionError(
                f"cannot determine the image size the intrinsics refer to ({exc}); "
                "pass --source-size WIDTHxHEIGHT"
            ) from exc
    result.source_size = (src_w, src_h)
    if abs((src_w / src_h) / (depth_w / depth_h) - 1) > ASPECT_TOLERANCE:
        raise ReconstructionError(
            f"RGB {src_w}x{src_h} and depth {depth_w}x{depth_h} have different aspect ratios; "
            "proportional intrinsics scaling would be invalid (cropping?)"
        )
    result.scale_factors = (depth_w / src_w, depth_h / src_h)

    fallback = load_camera_matrix(root / "camera_matrix.csv")

    def source_intrinsics(fid: int) -> Intrinsics:
        intr = poses[fid]["intr"] or fallback
        if intr is None:
            raise ReconstructionError(f"no intrinsics for frame {fid} (odometry.csv and camera_matrix.csv)")
        return intr

    check = source_intrinsics(frames[0])
    if abs(check.cx / src_w - 0.5) > PRINCIPAL_POINT_TOLERANCE or abs(check.cy / src_h - 0.5) > PRINCIPAL_POINT_TOLERANCE:
        result.warnings.append(
            f"principal point ({check.cx}, {check.cy}) is far from the centre of a {src_w}x{src_h} image; "
            "source size may be wrong or width/height swapped"
        )

    # --- process frames one at a time ---
    chunks: list[np.ndarray] = []
    positions = []
    for fid in frames:
        depth = _read_png(depth_dir / f"{fid:06d}.png")
        if depth.shape != (depth_h, depth_w):
            raise ReconstructionError(f"depth frame {fid} has shape {depth.shape}, expected {(depth_h, depth_w)}")
        confidence = None
        if opts.min_confidence > 0:
            conf_path = conf_dir / f"{fid:06d}.png"
            if not conf_path.is_file():
                raise ReconstructionError(f"confidence frame {fid} is missing; use --min-confidence 0 to ignore")
            confidence = _read_png(conf_path)

        range_ok = geometry.valid_depth_mask(
            depth, None, 0, opts.min_range_m, opts.max_range_m, result.depth_scale
        )
        mask = range_ok if confidence is None else range_ok & (confidence >= opts.min_confidence)
        result.candidate_points += depth.size
        result.removed_by_depth += int(depth.size - range_ok.sum())
        result.removed_by_confidence += int(range_ok.sum() - mask.sum())

        src_intr = source_intrinsics(fid)
        depth_intr = geometry.scale_intrinsics(src_intr, src_w, src_h, depth_w, depth_h)
        if result.example_frame < 0:
            result.example_frame = fid
            result.example_source_intrinsics = src_intr
            result.example_depth_intrinsics = depth_intr

        pose = poses[fid]
        positions.append(pose["t"])
        try:
            rotation = geometry.quat_to_rotation(*pose["q"])
        except ValueError as exc:
            raise ReconstructionError(f"frame {fid}: bad quaternion ({exc})") from exc
        cam_points = geometry.backproject(depth, mask, depth_intr, result.depth_scale)
        chunks.append(geometry.transform_points(cam_points, rotation, pose["t"]).astype(np.float32))

    cloud = np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float32)
    del chunks
    result.points_after_filter = len(cloud)
    if len(cloud) == 0:
        raise ReconstructionError("no valid depth points remain after filtering; relax --min-confidence or range")
    if not np.isfinite(cloud).all():
        raise ReconstructionError("point cloud contains NaN/inf values (bad pose or intrinsics data)")

    try:
        cloud = geometry.voxel_downsample(cloud, opts.voxel_size)
    except ValueError as exc:
        raise ReconstructionError(str(exc)) from exc
    result.points_after_downsample = len(cloud)

    traj = np.array(positions)
    result.trajectory_min, result.trajectory_max = traj.min(axis=0), traj.max(axis=0)
    result.cloud_min, result.cloud_max = cloud.min(axis=0), cloud.max(axis=0)

    try:
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        geometry.write_ply(result.output_path, cloud)
    except OSError as exc:
        raise ReconstructionError(f"cannot write {result.output_path}: {exc}") from exc

    result.runtime_s = time.perf_counter() - started
    return result
