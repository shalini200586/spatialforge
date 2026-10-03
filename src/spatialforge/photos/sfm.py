"""PyCOLMAP for still photos: ONE database for the whole property, exhaustive matching, per-room and per-room-pair mapping.

Why one database: there are only 2-8 photos per room, so exhaustive matching over every image of the property is cheap
(features are extracted once), and the same match table then serves three purposes: the room's own reconstruction
(mapper restricted to that room's images), the cross-room evidence (verified matches between images of different rooms),
and the pairwise stitching reconstructions (mapper restricted to the images of two rooms).

Like video, still-photo SfM has an ARBITRARY scale; metric scale is attached later from metric monocular depth.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from spatialforge.photos.validator import ImageMeta, RoomFolder, load_oriented_rgb
from spatialforge.video.sfm import SfmOptions, SfmRun, models_to_run


@dataclass
class PhotoSfmOptions(SfmOptions):
    max_num_features: int = 4096
    max_image_side: int = 1280  # working resolution: no need for more, and exhaustive matching stays cheap
    use_exif_prior: bool = True  # only used when EVERY image carries a believable 35 mm focal length
    init_min_tri_angle_deg: float = 4.0  # two or three handheld photos: small baselines are normal
    min_model_size: int = 2
    min_registered_images: int = 2
    image_noun: str = "photos"
    strong_registered_ratio: float = 0.80
    moderate_registered_ratio: float = 0.50
    weak_registered_ratio: float = 0.50  # a room reconstruction must contain most of its photos
    min_points_strong: int = 250
    min_points_moderate: int = 80
    min_points_weak: int = 30
    max_reprojection_px_strong: float = 1.2
    max_reprojection_px_moderate: float = 2.0
    min_num_matches: int = 15
    # COLMAP's defaults (100 inliers for the initial pair, 30 for absolute pose) are tuned for large image sets. Measured on
    # blurry low-texture stills (16-59 verified inliers per pair) they register nothing at all; with a handful of photos per
    # room, these lower thresholds are needed. The tracking-quality rules and reprojection error still guard the result.
    init_min_num_inliers: int = 25
    abs_pose_min_num_inliers: int = 12


@dataclass
class ImageRecord:
    name: str  # COLMAP name: "<room id>/img_NN.jpg"
    room_id: str
    meta: ImageMeta
    work_path: str


class ColmapPhotoBackend:
    """Real SfM backend. Tests inject an object with the same four members (prepare, map_subset, pair_stats, describe)."""

    def __init__(self, work_dir: Path, opts: PhotoSfmOptions | None = None):
        self.work_dir = Path(work_dir)
        self.opts = opts or PhotoSfmOptions()
        self.db = self.work_dir / "database.db"
        self.images_root = self.work_dir / "images"
        self.records: dict[str, ImageRecord] = {}
        self.seconds: dict[str, float] = {}
        self.prior_used = False

    # -- preparation: working images, features, exhaustive matches --
    def prepare(self, rooms: list[RoomFolder]) -> dict[str, ImageRecord]:
        import cv2
        import pycolmap

        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.images_root.mkdir(parents=True)
        t = time.perf_counter()
        for room in rooms:
            for k, meta in enumerate(room.images):
                name = f"{room.canonical_id}/img_{k:02d}.jpg"
                arr = load_oriented_rgb(meta.path, self.opts.max_image_side)  # EXIF orientation applied in memory
                dest = self.images_root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.records[name] = ImageRecord(name, room.canonical_id, meta, str(dest))
        self.seconds["working_images_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        fe = pycolmap.FeatureExtractionOptions()
        fe.sift.max_num_features = self.opts.max_num_features
        fe.sift.peak_threshold = self.opts.peak_threshold
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = self.opts.camera_model
        prior = self._focal_prior()
        if prior is not None:
            reader.camera_params = ",".join(f"{v:.4f}" for v in prior)
            self.prior_used = True
        pycolmap.extract_features(str(self.db), str(self.images_root), image_names=sorted(self.records),
                                  camera_mode=pycolmap.CameraMode.PER_FOLDER, reader_options=reader,
                                  extraction_options=fe, device=pycolmap.Device.cpu)
        self.seconds["feature_extraction_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        matching = pycolmap.FeatureMatchingOptions()
        matching.sift.max_ratio = self.opts.max_ratio
        pycolmap.match_exhaustive(str(self.db), matching_options=matching, device=pycolmap.Device.cpu)
        self.seconds["exhaustive_matching_s"] = round(time.perf_counter() - t, 2)
        return self.records

    def _focal_prior(self) -> list[float] | None:
        """[f, cx, cy, k] in working-image pixels when every image has a believable, mutually consistent 35 mm focal."""
        if not self.opts.use_exif_prior or not self.records:
            return None
        metas = [r.meta for r in self.records.values()]
        if not all(m.focal_prior_reliable for m in metas):
            return None
        sizes = {(m.width, m.height) for m in metas}
        focals = {round(float(m.focal_35mm), 1) for m in metas}
        models = {(m.make, m.model) for m in metas}
        if len(sizes) != 1 or len(focals) != 1 or len(models) != 1:
            return None  # different cameras or crops: let COLMAP estimate each folder's camera
        w, h = next(iter(sizes))
        s = min(1.0, self.opts.max_image_side / max(w, h))
        ww, hh = w * s, h * s
        return [float(metas[0].focal_35mm) * max(ww, hh) / 36.0, ww / 2.0, hh / 2.0, 0.0]

    # -- mapping --
    def map_subset(self, names: list[str], tag: str) -> SfmRun:
        import pycolmap

        out = self.work_dir / "sparse" / tag
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        t = time.perf_counter()
        inc = pycolmap.IncrementalPipelineOptions()
        inc.image_names = list(names)
        inc.min_num_matches = self.opts.min_num_matches
        inc.min_model_size = self.opts.min_model_size
        inc.mapper.random_seed = self.opts.random_seed
        inc.triangulation.random_seed = self.opts.random_seed
        inc.mapper.init_min_tri_angle = self.opts.init_min_tri_angle_deg
        inc.mapper.init_min_num_inliers = self.opts.init_min_num_inliers
        inc.mapper.abs_pose_min_num_inliers = self.opts.abs_pose_min_num_inliers
        recs = pycolmap.incremental_mapping(str(self.db), str(self.images_root), str(out), inc)
        sec = {"mapping_s": round(time.perf_counter() - t, 2), **self.seconds}
        return models_to_run(recs, list(names), self.opts, sec, None,
                             "EXIF 35 mm focal prior, refined by bundle adjustment" if self.prior_used
                             else "estimated by COLMAP (no usable EXIF focal prior)")

    # -- match table --
    def pair_stats(self) -> list[dict]:
        """Every image pair with its raw feature matches and geometrically VERIFIED inliers (COLMAP's two-view geometry)."""
        M = 2147483647
        with sqlite3.connect(str(self.db)) as con:
            names = {i: n for i, n in con.execute("select image_id, name from images")}
            raw = {p: r for p, r in con.execute("select pair_id, rows from matches")}
            ver = {p: (r, c) for p, r, c in con.execute("select pair_id, rows, config from two_view_geometries")}
        out = []
        for pid, r in sorted(raw.items()):
            i2 = pid % M
            i1 = (pid - i2) // M
            if i1 not in names or i2 not in names:
                continue
            inl, cfg = ver.get(pid, (0, 0))
            out.append({"a": names[i1], "b": names[i2], "raw": int(r), "verified": int(inl) if cfg in (2, 3, 4, 5, 6, 8) else 0})
        return out

    def describe(self) -> dict:
        import pycolmap

        return {"backend": "pycolmap", "version": str(pycolmap.__version__), "matching": "exhaustive (all images of the property)",
                "camera_mode": "one camera per room folder", "exif_focal_prior_used": self.prior_used, "seconds": self.seconds}
