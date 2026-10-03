"""Photo-directory validation and image metadata.

Input contract: a directory whose sub-folders are rooms, each holding 2-8 photos (.jpg/.jpeg/.png). Nothing is skipped
silently: every file that is not a supported photo is either an error or listed in a warning. HEIC/HEIF is rejected with
a clear message (no fragile extra dependency); convert to JPEG first. Originals are never modified: EXIF orientation is
applied in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
UNSUPPORTED_BUT_KNOWN = (".heic", ".heif")
IGNORED_SYSTEM_FILES = ("thumbs.db", "desktop.ini", ".ds_store")
MIN_IMAGES, MAX_IMAGES = 2, 8
MIN_SIDE_PX, MAX_SIDE_PX = 320, 20000
MAX_ASPECT = 4.0
ORIENTATION_TAG, MAKE_TAG, MODEL_TAG, EXIF_IFD = 0x0112, 0x010F, 0x0110, 0x8769
FOCAL_TAG, FOCAL35_TAG, DATETIME_TAG = 0x920A, 0xA405, 0x9003
# 35 mm equivalent focal lengths outside this range are not believable for a phone/camera photo
FOCAL35_RANGE = (10.0, 200.0)


@dataclass
class ImageMeta:
    path: str
    name: str
    width: int  # after orientation correction
    height: int
    raw_width: int
    raw_height: int
    exif_orientation: int | None
    make: str | None = None
    model: str | None = None
    focal_mm: float | None = None
    focal_35mm: float | None = None
    timestamp: str | None = None
    has_exif: bool = False

    @property
    def focal_prior_reliable(self) -> bool:
        return self.focal_35mm is not None and FOCAL35_RANGE[0] <= self.focal_35mm <= FOCAL35_RANGE[1]

    def focal_px(self) -> float | None:
        """Focal length in pixels of the (oriented) image from the 35 mm equivalent: long side = 36 mm."""
        if not self.focal_prior_reliable:
            return None
        return float(self.focal_35mm) * max(self.width, self.height) / 36.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RoomFolder:
    folder: str
    source_label: str
    canonical_id: str
    images: list[ImageMeta] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PhotoValidation:
    root: str
    rooms: list[RoomFolder] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.all_errors()

    def all_errors(self) -> list[str]:
        return list(self.errors) + [f"{r.source_label}: {e}" for r in self.rooms for e in r.errors]

    def all_warnings(self) -> list[str]:
        return list(self.warnings) + [f"{r.source_label}: {w}" for r in self.rooms for w in r.warnings]

    def to_dict(self) -> dict:
        return {"root": self.root, "ok": self.ok, "errors": self.all_errors(), "warnings": self.all_warnings(),
                "rooms": [{"folder": r.folder, "source_label": r.source_label, "canonical_id": r.canonical_id,
                           "images": [i.to_dict() for i in r.images]} for r in self.rooms]}


def _rational(v) -> float | None:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_metadata(path: Path) -> ImageMeta:
    """Decode the image header + EXIF. Raises ValueError if the file cannot be decoded as an image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            im.load()  # a truncated file fails here, not later in the pipeline
            raw_w, raw_h = im.size
            exif = im.getexif()
            orientation = exif.get(ORIENTATION_TAG)
            ifd = {}
            try:
                ifd = exif.get_ifd(EXIF_IFD)
            except Exception:
                pass
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc
    swap = orientation in (5, 6, 7, 8)  # these EXIF orientations rotate the image by 90 degrees
    w, h = (raw_h, raw_w) if swap else (raw_w, raw_h)
    f35 = ifd.get(FOCAL35_TAG)
    return ImageMeta(
        path=str(path), name=path.name, width=w, height=h, raw_width=raw_w, raw_height=raw_h,
        exif_orientation=int(orientation) if orientation else None,
        make=(str(exif.get(MAKE_TAG)).strip() or None) if exif.get(MAKE_TAG) else None,
        model=(str(exif.get(MODEL_TAG)).strip() or None) if exif.get(MODEL_TAG) else None,
        focal_mm=_rational(ifd.get(FOCAL_TAG)), focal_35mm=_rational(f35) if f35 else None,
        timestamp=str(ifd.get(DATETIME_TAG)) if ifd.get(DATETIME_TAG) else None, has_exif=len(exif) > 0)


def load_oriented_rgb(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """RGB uint8 array with EXIF orientation applied (in memory only), optionally downscaled so the long side <= max_side."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        return np.asarray(im)


def validate_photo_dir(path: str | Path) -> PhotoValidation:
    root = Path(path)
    res = PhotoValidation(str(root))
    if not root.exists():
        res.errors.append(f"photo directory does not exist: {root}")
        return res
    if not root.is_dir():
        res.errors.append(f"not a directory: {root} (photo input is a directory of room folders)")
        return res
    entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    folders = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
    loose = [p.name for p in entries if p.is_file() and p.name.lower() not in IGNORED_SYSTEM_FILES]
    if not folders:
        res.errors.append("the photo directory contains no room sub-folders (expected one folder per room, 2-8 photos each)")
        return res
    if loose:
        res.warnings.append(f"{len(loose)} file(s) directly in the photo directory are not in a room folder and are NOT used: "
                            + ", ".join(loose[:10]))
    for k, folder in enumerate(folders, 1):
        room = RoomFolder(str(folder), folder.name, f"room_{k:03d}")  # deterministic ids from the sorted folder names
        res.rooms.append(room)
        files = sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name.lower())
        subdirs = [p.name for p in folder.iterdir() if p.is_dir()]
        if subdirs:
            room.warnings.append("nested folders are not used: " + ", ".join(subdirs))
        photos = []
        for f in files:
            ext = f.suffix.lower()
            if f.name.lower() in IGNORED_SYSTEM_FILES:
                room.warnings.append(f"ignored system file {f.name}")
            elif ext in SUPPORTED_EXTENSIONS:
                photos.append(f)
            elif ext in UNSUPPORTED_BUT_KNOWN:
                room.errors.append(f"{f.name}: HEIC/HEIF is not supported in this environment; convert it to JPEG first")
            else:
                room.errors.append(f"{f.name}: unsupported file type {ext or '(no extension)'!r} "
                                   f"(supported: {', '.join(SUPPORTED_EXTENSIONS)})")
        if len(photos) < MIN_IMAGES:
            room.errors.append(f"has {len(photos)} supported photo(s); {MIN_IMAGES}-{MAX_IMAGES} are required per room")
        elif len(photos) > MAX_IMAGES:
            room.errors.append(f"has {len(photos)} photos; at most {MAX_IMAGES} are allowed per room")
        for f in photos:
            try:
                meta = read_metadata(f)
            except ValueError as exc:
                room.errors.append(f"{f.name}: {exc}")
                continue
            if min(meta.width, meta.height) < MIN_SIDE_PX:
                room.errors.append(f"{f.name}: {meta.width}x{meta.height} is too small (minimum side {MIN_SIDE_PX} px)")
            elif max(meta.width, meta.height) > MAX_SIDE_PX or max(meta.width, meta.height) / min(meta.width, meta.height) > MAX_ASPECT:
                room.errors.append(f"{f.name}: {meta.width}x{meta.height} is not a sensible photo size")
            elif meta.exif_orientation not in (None, 1, 2, 3, 4, 5, 6, 7, 8):
                room.errors.append(f"{f.name}: invalid EXIF orientation {meta.exif_orientation}")
            else:
                room.images.append(meta)
        if room.images and not any(i.has_exif for i in room.images):
            room.warnings.append("no EXIF metadata: intrinsics will be estimated from the images alone")
    labels = [r.source_label.lower() for r in res.rooms]
    if len(set(labels)) != len(labels):
        res.errors.append("room folder names differ only by case")
    return res
