"""Photo-directory validation, EXIF metadata and orientation."""

import numpy as np
import pytest
from helpers_photo import make_property_dir, write_photo

from spatialforge.photos.validator import MAX_IMAGES, load_oriented_rgb, read_metadata, validate_photo_dir


def test_valid_photo_folder_structure(tmp_path):
    make_property_dir(tmp_path, {"room_01": 3, "room_02": 4, "hallway": 2})
    v = validate_photo_dir(tmp_path)
    assert v.ok, v.all_errors()
    assert [r.source_label for r in v.rooms] == ["hallway", "room_01", "room_02"]  # sorted, deterministic
    assert [r.canonical_id for r in v.rooms] == ["room_001", "room_002", "room_003"]  # ids never come from folder meaning
    assert [len(r.images) for r in v.rooms] == [2, 3, 4]
    assert v.to_dict()["rooms"][0]["images"][0]["width"] == 640


def test_room_ids_do_not_depend_on_folder_semantics(tmp_path):
    make_property_dir(tmp_path, {"bedroom": 2, "kitchen": 2, "bathroom": 2})
    v = validate_photo_dir(tmp_path)
    assert [r.canonical_id for r in v.rooms] == ["room_001", "room_002", "room_003"]
    assert all("bed" not in r.canonical_id and "kitchen" not in r.canonical_id for r in v.rooms)


def test_a_room_with_one_photo_is_rejected(tmp_path):
    make_property_dir(tmp_path, {"a": 3, "b": 1})
    v = validate_photo_dir(tmp_path)
    assert not v.ok and any("b: has 1 supported photo(s)" in e for e in v.all_errors())


def test_a_room_with_two_photos_is_allowed(tmp_path):
    make_property_dir(tmp_path, {"a": 2})
    assert validate_photo_dir(tmp_path).ok


def test_a_room_with_too_many_photos_is_rejected_clearly(tmp_path):
    make_property_dir(tmp_path, {"a": MAX_IMAGES + 1, "b": MAX_IMAGES})
    v = validate_photo_dir(tmp_path)
    errs = v.all_errors()
    assert not v.ok and any("a: has 9 photos; at most 8 are allowed" in e for e in errs)
    assert not any(e.startswith("b:") for e in errs)


def test_unsupported_files_are_reported_not_skipped(tmp_path):
    make_property_dir(tmp_path, {"a": 3})
    (tmp_path / "a" / "notes.txt").write_text("hello")
    (tmp_path / "a" / "IMG_900.HEIC").write_bytes(b"\x00" * 64)
    (tmp_path / "a" / "Thumbs.db").write_bytes(b"x")
    (tmp_path / "stray.jpg").write_bytes(b"x")
    v = validate_photo_dir(tmp_path)
    errs = " | ".join(v.all_errors())
    assert not v.ok
    assert "notes.txt: unsupported file type '.txt'" in errs and "IMG_900.HEIC: HEIC/HEIF is not supported" in errs
    warns = " | ".join(v.all_warnings())
    assert "ignored system file Thumbs.db" in warns and "stray.jpg" in warns and "NOT used" in warns  # listed, never silent


def test_structure_errors(tmp_path):
    assert "does not exist" in validate_photo_dir(tmp_path / "nope").errors[0]
    f = tmp_path / "file.jpg"
    write_photo(f)
    assert "not a directory" in validate_photo_dir(f).errors[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no room sub-folders" in validate_photo_dir(empty).errors[0]


def test_corrupt_and_tiny_images_are_rejected(tmp_path):
    make_property_dir(tmp_path, {"a": 2})
    (tmp_path / "a" / "broken.jpg").write_bytes(b"not an image" * 20)
    write_photo(tmp_path / "a" / "tiny.jpg", size=(100, 80))
    errs = " | ".join(validate_photo_dir(tmp_path).all_errors())
    assert "broken.jpg: cannot decode image" in errs and "tiny.jpg: 100x80 is too small" in errs


def test_png_is_supported(tmp_path):
    write_photo(tmp_path / "a" / "x.png", fmt="PNG")
    write_photo(tmp_path / "a" / "y.png", fmt="PNG", seed=2)
    assert validate_photo_dir(tmp_path).ok


def test_exif_orientation_is_applied_in_memory_only(tmp_path):
    p = write_photo(tmp_path / "r.jpg", size=(640, 480), orientation=6, seed=3)  # needs a 90 degree rotation
    before = p.read_bytes()
    meta = read_metadata(p)
    assert (meta.raw_width, meta.raw_height) == (640, 480) and (meta.width, meta.height) == (480, 640)
    assert meta.exif_orientation == 6
    arr = load_oriented_rgb(p)
    assert arr.shape[:2] == (640, 480)
    assert p.read_bytes() == before  # the original file is untouched
    upright = write_photo(tmp_path / "u.jpg", size=(640, 480), orientation=1, seed=3)
    assert load_oriented_rgb(upright).shape[:2] == (480, 640)
    small = load_oriented_rgb(p, max_side=320)
    assert max(small.shape[:2]) == 320 and small.shape[0] > small.shape[1]


def test_exif_focal_length_metadata(tmp_path):
    p = write_photo(tmp_path / "f.jpg", size=(1600, 1200), make="Apple", model="iPhone 14", focal35=26, focal=5.7,
                    timestamp="2026:05:01 10:20:30")
    m = read_metadata(p)
    assert (m.make, m.model, m.focal_35mm, m.timestamp) == ("Apple", "iPhone 14", 26.0, "2026:05:01 10:20:30")
    assert m.focal_mm == pytest.approx(5.7, rel=1e-3) and m.has_exif and m.focal_prior_reliable
    assert m.focal_px() == pytest.approx(26 * 1600 / 36.0)  # 35 mm equivalent: long side = 36 mm
    none = read_metadata(write_photo(tmp_path / "n.jpg"))
    assert none.focal_35mm is None and not none.focal_prior_reliable and none.focal_px() is None and not none.has_exif
    silly = read_metadata(write_photo(tmp_path / "s.jpg", focal35=900))
    assert not silly.focal_prior_reliable and silly.focal_px() is None  # an implausible tag is not used as a prior


def test_no_exif_is_a_warning_not_an_error(tmp_path):
    make_property_dir(tmp_path, {"a": 2})
    v = validate_photo_dir(tmp_path)
    assert v.ok and any("no EXIF metadata" in w for w in v.all_warnings())
