"""Sparse Structure-from-Motion on the keyframes with PyCOLMAP (CPU, prebuilt wheel, sequential matching).

The result is a trajectory and sparse points in an ARBITRARY scale: SfM alone cannot give metres. Metric scale is
attached later (video.scale) from a metric depth model.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRACKING_QUALITIES = ("strong", "moderate", "weak", "failure")


class SfmError(RuntimeError):
    pass


@dataclass
class SfmOptions:
    camera_model: str = "SIMPLE_RADIAL"  # one shared camera for the whole video, focal length estimated
    max_num_features: int = 4096  # 8192 and a lower peak threshold were ~7x slower to match with no better registration
    peak_threshold: float = 0.0067  # COLMAP's default
    match_overlap: int = 12
    quadratic_overlap: bool = True
    max_ratio: float = 0.85
    min_num_matches: int = 15
    init_min_tri_angle_deg: float = 8.0  # COLMAP default 16: a slow handheld walk gives small baselines
    min_model_size: int = 8
    random_seed: int = 0
    # tracking-quality rules
    strong_registered_ratio: float = 0.80
    moderate_registered_ratio: float = 0.50
    weak_registered_ratio: float = 0.15  # below this (or fewer than min_registered_images) there is no coherent trajectory
    max_reprojection_px_strong: float = 1.0
    max_reprojection_px_moderate: float = 1.5
    min_points_strong: int = 1500
    min_points_moderate: int = 400
    min_points_weak: int = 100
    min_registered_images: int = 10


@dataclass
class SfmCameraInfo:
    model: str
    width: int
    height: int
    params: list[float]
    focal_px: float
    focal_source: str  # "estimated by COLMAP (no metadata prior)"


@dataclass
class SfmImagePose:
    name: str
    image_id: int
    rotation_cw: np.ndarray  # R: x_cam = R x_world + t   (COLMAP convention, arbitrary scale)
    translation_cw: np.ndarray
    n_points3d: int = 0


@dataclass
class SfmResult:
    """Serialisable summary (no arrays)."""

    keyframes_attempted: int
    registered_images: int
    registered_ratio: float
    sparse_points: int
    mean_reprojection_error_px: float | None
    mean_track_length: float | None
    models_found: int
    model_sizes: list[int]
    camera: SfmCameraInfo | None
    unregistered: list[str]
    tracking_quality: str
    quality_reasons: list[str]
    seconds: dict = field(default_factory=dict)
    verified_pairs: int | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["camera"] = None if self.camera is None else dict(self.camera.__dict__)
        return d


@dataclass
class SfmRun:
    result: SfmResult
    poses: dict[str, SfmImagePose]  # image name -> pose
    # image name -> (pixel xy (N,2), world xyz in SfM units (N,3)) of the triangulated points it observes
    observations: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    points_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # all sparse points (SfM units)
    reconstruction: object | None = None  # pycolmap.Reconstruction (heavy; kept only for debugging)


def cam_to_world(rotation_cw: np.ndarray, translation_cw: np.ndarray) -> np.ndarray:
    """4x4 camera-to-world from COLMAP's world-to-camera (x_cam = R x_world + t). Camera axes: x right, y down, z forward."""
    R = np.asarray(rotation_cw, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation_cw, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def quat_xyzw_to_rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def classify_tracking(registered: int, attempted: int, points: int, reproj: float | None, opts: SfmOptions) -> tuple[str, list[str]]:
    """STRONG / MODERATE / WEAK / FAILURE from registration ratio, sparse points and reprojection error."""
    ratio = registered / attempted if attempted else 0.0
    reasons = [f"{registered}/{attempted} keyframes registered ({ratio:.0%})", f"{points} sparse points"]
    if reproj is not None:
        reasons.append(f"mean reprojection error {reproj:.2f} px")
    if registered < opts.min_registered_images or points < opts.min_points_weak or ratio < opts.weak_registered_ratio:
        return "failure", reasons + ["no coherent reconstruction"]
    bad_reproj_m = reproj is not None and reproj > opts.max_reprojection_px_moderate
    if ratio >= opts.strong_registered_ratio and points >= opts.min_points_strong and not (
            reproj is not None and reproj > opts.max_reprojection_px_strong):
        return "strong", reasons
    if ratio >= opts.moderate_registered_ratio and points >= opts.min_points_moderate and not bad_reproj_m:
        return "moderate", reasons
    if ratio >= opts.weak_registered_ratio:
        return "weak", reasons + ["low registration ratio or sparse structure: trajectory may be fragmentary"]
    return "failure", reasons + ["registration ratio too low for a coherent trajectory"]


def run_sfm(image_dir: Path, work_dir: Path, opts: SfmOptions | None = None, log=None) -> SfmRun:
    """Feature extraction, sequential matching and incremental mapping. Keeps the largest model."""
    import pycolmap

    opts = opts or SfmOptions()
    image_dir, work_dir = Path(image_dir), Path(work_dir)
    names = sorted(p.name for p in image_dir.glob("*.jpg"))
    if len(names) < opts.min_registered_images:
        raise SfmError(f"only {len(names)} keyframes available")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    (work_dir / "sparse").mkdir(parents=True)
    db = work_dir / "database.db"
    sec: dict = {}

    t = time.perf_counter()
    fe = pycolmap.FeatureExtractionOptions()
    fe.sift.max_num_features = opts.max_num_features
    fe.sift.peak_threshold = opts.peak_threshold
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = opts.camera_model
    pycolmap.extract_features(str(db), str(image_dir), image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=fe, device=pycolmap.Device.cpu)
    sec["feature_extraction_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = opts.match_overlap
    pairing.quadratic_overlap = opts.quadratic_overlap
    pairing.loop_detection = False  # needs a vocabulary tree file; sequential walkthroughs rarely need it
    matching = pycolmap.FeatureMatchingOptions()
    matching.sift.max_ratio = opts.max_ratio
    pycolmap.match_sequential(str(db), matching_options=matching, pairing_options=pairing, device=pycolmap.Device.cpu)
    sec["matching_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    inc = pycolmap.IncrementalPipelineOptions()
    inc.min_num_matches = opts.min_num_matches
    inc.min_model_size = opts.min_model_size
    inc.mapper.random_seed = opts.random_seed
    inc.triangulation.random_seed = opts.random_seed
    inc.mapper.init_min_tri_angle = opts.init_min_tri_angle_deg
    recs = pycolmap.incremental_mapping(str(db), str(image_dir), str(work_dir / "sparse"), inc)
    sec["mapping_s"] = round(time.perf_counter() - t, 2)

    verified = None
    try:
        import sqlite3

        with sqlite3.connect(str(db)) as con:
            verified = int(con.execute("select count(*) from two_view_geometries where rows >= ?", (opts.min_num_matches,)).fetchone()[0])
    except Exception:
        pass

    sizes = sorted((r.num_reg_images() for r in recs.values()), reverse=True)
    if not recs:
        res = SfmResult(len(names), 0, 0.0, 0, None, None, 0, [], None, names, "failure",
                        ["no reconstruction was produced"], sec, verified)
        return SfmRun(res, {})
    # largest model; ties broken by point count then model id, so the choice is deterministic
    best_id = max(recs, key=lambda k: (recs[k].num_reg_images(), recs[k].num_points3D(), -k))
    rec = recs[best_id]
    poses: dict[str, SfmImagePose] = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        cfw = im.cam_from_world()
        poses[im.name] = SfmImagePose(im.name, int(im.image_id), quat_xyzw_to_rotation(cfw.rotation.quat),
                                      np.asarray(cfw.translation, dtype=np.float64), int(im.num_points3D))
    cam = next(iter(rec.cameras.values()))
    focal = float(cam.params[0]) if len(cam.params) else float("nan")
    cam_info = SfmCameraInfo(str(cam.model).split(".")[-1], int(cam.width), int(cam.height), [float(p) for p in cam.params],
                             focal, "estimated by COLMAP (no focal-length metadata prior used)")
    try:
        reproj = float(rec.compute_mean_reprojection_error())
        track = float(rec.compute_mean_track_length())
    except Exception:
        reproj = track = None
    q, reasons = classify_tracking(len(poses), len(names), int(rec.num_points3D()), reproj, opts)
    if len(recs) > 1:
        reasons.append(f"{len(recs)} disconnected models were found (sizes {sizes[:5]}); only the largest is used")
    res = SfmResult(len(names), len(poses), len(poses) / len(names), int(rec.num_points3D()), reproj, track, len(recs),
                    sizes, cam_info, sorted(set(names) - set(poses)), q, reasons, sec, verified)
    obs = {}
    for im in rec.images.values():
        if im.name in poses:
            xy, xyz = [], []
            for p in im.points2D:
                if p.has_point3D():
                    xy.append(p.xy)
                    xyz.append(rec.points3D[p.point3D_id].xyz)
            obs[im.name] = (np.asarray(xy, dtype=np.float64).reshape(-1, 2), np.asarray(xyz, dtype=np.float64).reshape(-1, 3))
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    return SfmRun(res, poses, obs, pts, rec)
