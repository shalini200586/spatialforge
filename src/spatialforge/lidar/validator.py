"""Validate a LiDAR capture folder (RGB video, odometry, IMU, depth, confidence)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from spatialforge.lidar.models import LidarValidationResult

SAMPLE_FRAME_COUNT = 8  # how many depth/confidence frames to read for statistics
MAX_LISTED_IDS = 20  # cap on frame IDs shown in messages

ODOMETRY_COLUMNS = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"]


def resolve_capture_root(path: Path) -> Path:
    """Use `path` itself, or its only sub-folder if the data sits one level down."""
    if (path / "depth").is_dir() or (path / "odometry.csv").is_file():
        return path
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (
        (subdirs[0] / "depth").is_dir() or (subdirs[0] / "odometry.csv").is_file()
    ):
        return subdirs[0]
    return path


def _short(ids: list[int]) -> str:
    shown = ", ".join(str(i) for i in ids[:MAX_LISTED_IDS])
    more = len(ids) - MAX_LISTED_IDS
    return f"{shown} (+{more} more)" if more > 0 else shown


def _frame_ids_from_folder(folder: Path, result: LidarValidationResult, label: str) -> list[int]:
    """Frame IDs from PNG file names like 000123.png. Non-numeric names are warned about."""
    ids: list[int] = []
    bad: list[str] = []
    for f in folder.iterdir():
        if f.suffix.lower() != ".png":
            continue
        if f.stem.isdigit():
            ids.append(int(f.stem))
        else:
            bad.append(f.name)
    if bad:
        result.warnings.append(
            f"{label}: {len(bad)} PNG file(s) with non-numeric names ignored, e.g. {bad[0]}"
        )
    return sorted(ids)


def _gaps(ids: list[int]) -> list[int]:
    """IDs missing from the range first..last."""
    if not ids:
        return []
    present = set(ids)
    return [i for i in range(ids[0], ids[-1] + 1) if i not in present]


def _read_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im)


def _evenly_spaced(ids: list[int], n: int) -> list[int]:
    if len(ids) <= n:
        return ids
    return [ids[round(i * (len(ids) - 1) / (n - 1))] for i in range(n)]


# ---------- individual checks ----------


def _check_odometry(root: Path, result: LidarValidationResult) -> list[int]:
    """Parse odometry.csv and return the (possibly duplicated) frame IDs found."""
    path = root / "odometry.csv"
    if not path.is_file():
        result.errors.append("odometry.csv is missing (required)")
        return []

    ids: list[int] = []
    malformed: list[int] = []  # line numbers
    fx_values: list[float] = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = [h.strip().lower() for h in next(reader, [])]
            missing_cols = [c for c in ODOMETRY_COLUMNS if c not in header]
            if missing_cols:
                result.errors.append(
                    f"odometry.csv header lacks column(s): {', '.join(missing_cols)}"
                )
                return []
            col = {name: i for i, name in enumerate(header)}
            for line_no, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    continue  # blank line
                try:
                    frame_id = int(row[col["frame"]].strip())
                    for name in ODOMETRY_COLUMNS:
                        if name != "frame":
                            float(row[col[name]])
                    if "fx" in col and row[col["fx"]].strip():
                        fx_values.append(float(row[col["fx"]]))
                except (ValueError, IndexError):
                    malformed.append(line_no)
                else:
                    ids.append(frame_id)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        result.errors.append(f"Could not read odometry.csv: {exc}")
        return []

    if malformed:
        text = f"odometry.csv: {len(malformed)} malformed row(s) skipped (lines {_short(malformed)})"
        # No usable rows at all is fatal; a few bad rows are a warning.
        (result.errors if not ids else result.warnings).append(text)

    dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
    if dupes:
        result.duplicate_pose_frames = dupes
        result.errors.append(f"odometry.csv: {len(dupes)} duplicate frame ID(s): {_short(dupes)}")

    result.pose_frame_count = len(ids)
    if not ids and not malformed:
        result.errors.append("odometry.csv has a header but no data rows")
    if fx_values:
        result.metadata["odometry_fx_range"] = (min(fx_values), max(fx_values))
    return ids


def _check_camera_matrix(root: Path, result: LidarValidationResult) -> None:
    path = root / "camera_matrix.csv"
    if not path.is_file():
        result.warnings.append("camera_matrix.csv is missing (camera intrinsics unavailable)")
        return
    try:
        rows = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.reader(fh):
                if any(c.strip() for c in row):
                    rows.append([float(c) for c in row])
        if len(rows) != 3 or any(len(r) != 3 for r in rows):
            raise ValueError("expected a 3x3 matrix")
    except (OSError, ValueError) as exc:
        result.warnings.append(f"camera_matrix.csv could not be parsed: {exc}")
        return
    result.metadata["intrinsics"] = {
        "fx": rows[0][0],
        "fy": rows[1][1],
        "cx": rows[0][2],
        "cy": rows[1][2],
    }


def _check_imu(root: Path, result: LidarValidationResult) -> None:
    path = root / "imu.csv"
    if not path.is_file():
        result.warnings.append("imu.csv is missing (IMU data unavailable)")
        return
    try:
        with open(path, "rb") as fh:
            lines = sum(1 for line in fh if line.strip())
    except OSError as exc:
        result.warnings.append(f"imu.csv could not be read: {exc}")
        return
    result.metadata["imu_sample_count"] = max(lines - 1, 0)  # minus header


def _check_depth_samples(depth_dir: Path, ids: list[int], result: LidarValidationResult) -> None:
    sample_ids = _evenly_spaced(ids, SAMPLE_FRAME_COUNT)
    valid_chunks: list[np.ndarray] = []
    shapes: set[tuple] = set()
    dtype = None
    for fid in sample_ids:
        path = depth_dir / f"{fid:06d}.png"
        if not path.is_file():
            continue
        try:
            arr = _read_png(path)
        except Exception as exc:  # corrupt image: report, keep going
            result.errors.append(f"Depth frame {path.name} is unreadable: {exc}")
            continue
        shapes.add(arr.shape)
        dtype = arr.dtype
        valid = arr[arr > 0]  # 0 means "no depth measured"
        if valid.size:
            valid_chunks.append(valid)
    if dtype is None:
        return
    shape = sorted(shapes)[0]
    result.metadata["depth_resolution"] = f"{shape[1]}x{shape[0]} (width x height)"
    result.metadata["depth_dtype"] = str(dtype)
    if len(shapes) > 1:
        result.warnings.append(f"Sampled depth frames have differing shapes: {sorted(shapes)}")
    if valid_chunks:
        values = np.concatenate(valid_chunks)
        result.metadata["depth_stats"] = {
            "min": int(values.min()),
            "max": int(values.max()),
            "median": float(np.median(values)),
            "sampled_frames": len(sample_ids),
            "unit": "raw PNG values (unit not verified; often millimetres)",
        }
    else:
        result.warnings.append("Sampled depth frames contain no valid (non-zero) depth values")


def _check_confidence_samples(conf_dir: Path, ids: list[int], result: LidarValidationResult) -> None:
    sample_ids = _evenly_spaced(ids, SAMPLE_FRAME_COUNT)
    hist: Counter = Counter()
    shapes: set[tuple] = set()
    dtype = None
    for fid in sample_ids:
        path = conf_dir / f"{fid:06d}.png"
        if not path.is_file():
            continue
        try:
            arr = _read_png(path)
        except Exception as exc:
            result.errors.append(f"Confidence frame {path.name} is unreadable: {exc}")
            continue
        shapes.add(arr.shape)
        dtype = arr.dtype
        levels, counts = np.unique(arr, return_counts=True)
        for level, count in zip(levels.tolist(), counts.tolist()):
            hist[level] += count
    if dtype is None:
        return
    shape = sorted(shapes)[0]
    result.metadata["confidence_resolution"] = f"{shape[1]}x{shape[0]} (width x height)"
    result.metadata["confidence_dtype"] = str(dtype)
    result.metadata["confidence_histogram"] = dict(sorted(hist.items()))


def _check_alignment(
    result: LidarValidationResult,
    a_ids: list[int],
    b_ids: list[int],
    a_only_label: str,
    b_only_label: str,
    flag: str,
) -> None:
    """Record IDs present in one list but not the other."""
    a_only = sorted(set(a_ids) - set(b_ids))
    b_only = sorted(set(b_ids) - set(a_ids))
    if a_only:
        result.missing_frames[a_only_label] = a_only
        result.warnings.append(f"{len(a_only)} {a_only_label}: {_short(a_only)}")
    if b_only:
        result.missing_frames[b_only_label] = b_only
        result.warnings.append(f"{len(b_only)} {b_only_label}: {_short(b_only)}")
    result.metadata[flag] = not a_only and not b_only


# ---------- main entry point ----------


def validate_lidar_capture(path: str | Path) -> LidarValidationResult:
    """Inspect one LiDAR capture folder. Reports problems instead of raising."""
    given = Path(path)
    result = LidarValidationResult(capture_path=given)

    if not given.is_dir():
        result.errors.append(f"Capture path is not a folder: {given}")
        result.finalize()
        return result

    root = resolve_capture_root(given)
    result.capture_path = root
    if root != given:
        result.metadata["note"] = f"Using nested capture folder: {root.name}"

    depth_dir, conf_dir = root / "depth", root / "confidence"
    result.present = {
        "rgb.mp4": (root / "rgb.mp4").is_file(),
        "odometry.csv": (root / "odometry.csv").is_file(),
        "camera_matrix.csv": (root / "camera_matrix.csv").is_file(),
        "imu.csv": (root / "imu.csv").is_file(),
        "depth/": depth_dir.is_dir(),
        "confidence/": conf_dir.is_dir(),
    }

    # RGB video: only presence and size can be checked without a video library.
    rgb = root / "rgb.mp4"
    if rgb.is_file():
        size = rgb.stat().st_size
        result.metadata["rgb_size_mb"] = round(size / 1e6, 1)
        if size == 0:
            result.warnings.append("rgb.mp4 is empty")
    else:
        result.warnings.append("rgb.mp4 is missing (RGB video unavailable)")

    pose_ids = _check_odometry(root, result)
    _check_camera_matrix(root, result)
    _check_imu(root, result)

    # Depth (required)
    depth_ids: list[int] = []
    if not depth_dir.is_dir():
        result.errors.append("depth/ folder is missing (required)")
    else:
        depth_ids = _frame_ids_from_folder(depth_dir, result, "depth")
        result.depth_frame_count = len(depth_ids)
        if not depth_ids:
            result.errors.append("depth/ folder contains no numbered PNG frames")
        else:
            gaps = _gaps(depth_ids)
            if gaps:
                result.missing_frames["depth IDs missing inside sequence"] = gaps
                result.warnings.append(f"depth/: {len(gaps)} frame ID(s) missing inside sequence: {_short(gaps)}")
            _check_depth_samples(depth_dir, depth_ids, result)

    # Confidence (optional)
    conf_ids: list[int] = []
    if not conf_dir.is_dir():
        result.warnings.append("confidence/ folder is missing (optional)")
    else:
        conf_ids = _frame_ids_from_folder(conf_dir, result, "confidence")
        result.confidence_frame_count = len(conf_ids)
        if not conf_ids:
            result.warnings.append("confidence/ folder contains no numbered PNG frames")
        else:
            _check_confidence_samples(conf_dir, conf_ids, result)

    if depth_ids and conf_ids:
        _check_alignment(
            result, depth_ids, conf_ids,
            "depth frame(s) without confidence frame",
            "confidence frame(s) without depth frame",
            "depth_confidence_aligned",
        )
    if depth_ids and pose_ids:
        _check_alignment(
            result, depth_ids, pose_ids,
            "depth frame(s) without pose row",
            "pose row(s) without depth frame",
            "pose_depth_aligned",
        )

    result.finalize()
    return result
