"""Damage analysis inside the main process pipeline, and the per-tier evidence providers."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from helpers_damage import CAM, add_crack, add_stain, finish, wall_image  # noqa: E402
from helpers_pipeline import synthetic_outputs  # noqa: E402
from helpers_video import camera_pose, make_test_video  # noqa: E402

from spatialforge.cli import main  # noqa: E402
from spatialforge.damage.analyze import EvidenceImage  # noqa: E402
from spatialforge.damage.evidence import (  # noqa: E402
    evenly_spaced, lidar_evidence_fn, photo_evidence_fn, planar_to_matrix, video_evidence_fn,
)
from spatialforge.pipeline import lidar_pipeline  # noqa: E402
from spatialforge.pipeline.lidar_pipeline import process_capture  # noqa: E402
from spatialforge.pipeline.schema import build_schema, validate_against_schema  # noqa: E402
from spatialforge.pipeline.serialization import strip_volatile  # noqa: E402
from spatialforge.photos.stitch import apply  # noqa: E402
from spatialforge.video.sfm import SfmCameraInfo  # noqa: E402


def damaged_wall(low=False):
    img = wall_image()
    add_stain(img, cx=640, cy=480, rx=100, ry=60)
    if not low:
        add_crack(img, x0=300, y0=150, length=300)
    return img


def outputs_with_evidence(low=False, fail=False):
    """The synthetic two-room LiDAR outputs (10 x 5 m, floor at y = -1.4) plus one view of the north wall from inside room A."""
    so = synthetic_outputs()

    def evidence():
        if fail:
            raise RuntimeError("video could not be opened")
        T = camera_pose([2.5, 0.0, 1.5], 0.0, -0.31 if low else 0.0)  # 3.5 m from the wall at z = 5, 1.4 m above the floor
        return [EvidenceImage("frame_000100", "lidar", lambda: finish(damaged_wall(low)), CAM, T, so.floor_y_m, None, 100)]

    so.evidence_fn = evidence
    return so


def run(tmp_path, name="out", so=None, **kw):
    so = so or outputs_with_evidence()
    return process_capture("C:/data/cap", tmp_path / name, stages_fn=lambda c, o: so, **kw)


def test_damage_flows_through_the_main_pipeline(tmp_path):
    r = run(tmp_path)
    d = r.property
    assert r.exit_code == 0 and validate_against_schema(d, build_schema()) == []
    classes = sorted(x["class"] for x in d["damage_regions"])
    assert classes == ["crack", "stain"] and d["damage"] == d["damage_regions"]
    for x in d["damage_regions"]:
        assert x["surface_type"] == "wall" and x["surface_id"] in {w["id"] for w in d["walls"]} and x["room_id"] in {r_["id"] for r_ in d["rooms"]}
        assert x["metric_reliable"] and x["width_m"]["interval"]["low"] < x["width_m"]["value"] < x["width_m"]["interval"]["high"]
        assert x["source"] == {"image_id": "frame_000100", "tier": "lidar", "frame_index": 100, "image_size": [1280, 960]}
    assert {i["action"] for i in d["scope_line_items"]} == {"seal_and_repair_crack", "treat_stain_and_repaint"}
    assert all(i["surface_id"] and i["damage_id"] for i in d["scope_line_items"])
    stage = next(s for s in r.run_report["stages"] if s["name"] == "damage_analysis")
    assert stage["status"] in ("ok", "warning") and stage["details"]["detections"] == 2 and stage["seconds"] > 0
    assert d["timing"]["damage_analysis_s"] > 0 and d["provenance"]["damage_analysis"]["images_analysed"] == 1
    diag = tmp_path / "out" / "diagnostics" / "damage"
    assert (diag / "damage_report.json").exists() and (diag / "damage_plan.png").exists() and (diag / "images" / "frame_000100.jpg").exists()
    assert not any("Damage detection is not part of this version" in w for w in d["warnings"])


def test_a_low_stain_raises_a_concealed_risk_flag_and_an_inspection_item(tmp_path):
    d = run(tmp_path, so=outputs_with_evidence(low=True)).property
    flags = d["concealed_damage_flags"]
    assert [f["rule_id"] for f in flags] == ["CDR-001"] and flags[0]["claims_presence"] is False
    assert flags[0]["damage_ids"] and flags[0]["damage_ids"][0] in {x["damage_id"] for x in d["damage_regions"]}
    assert any(i["flag_id"] == flags[0]["flag_id"] and i["action"] == "inspect_for_concealed_moisture" for i in d["scope_line_items"])
    assert any("RISK flag" in w and "none of them asserts" in w for w in d["warnings"])


def test_damage_can_be_switched_off_and_nothing_else_changes(tmp_path):
    off = run(tmp_path, "off", damage=False).property
    assert off["damage_regions"] == off["damage"] == [] and off["concealed_damage_flags"] == [] and off["scope_line_items"] == []
    assert any("Damage detection is not part of this version" in w for w in off["warnings"])  # the old notice stays when it did not run
    on = run(tmp_path, "on").property
    strip = lambda p: {k: v for k, v in p.items() if k not in ("damage", "damage_regions", "concealed_damage_flags", "scope_line_items", "warnings", "provenance", "timing")}  # noqa: E731
    assert strip(on) == strip(off)  # rooms, walls, openings and footprint are untouched by the damage layer


def test_a_failing_damage_analysis_is_a_warning_not_a_failure(tmp_path):
    r = run(tmp_path, so=outputs_with_evidence(fail=True))
    d = r.property
    assert r.exit_code == 0 and d["damage_regions"] == [] and (tmp_path / "out" / "property.json").exists()
    assert any("Damage analysis failed and was skipped" in w and "video could not be opened" in w for w in d["warnings"])
    stage = next(s for s in r.run_report["stages"] if s["name"] == "damage_analysis")
    assert stage["status"] == "failed" and "video could not be opened" in stage["error"]


def test_no_rgb_evidence_skips_the_analysis_cleanly(tmp_path):
    so = synthetic_outputs()
    so.evidence_fn = lambda: []
    r = run(tmp_path, so=so)
    stage = next(s for s in r.run_report["stages"] if s["name"] == "damage_analysis")
    assert stage["status"] == "skipped" and r.property["damage_regions"] == []
    plain = run(tmp_path, "plain", so=synthetic_outputs())  # a tier without any evidence provider: no stage at all
    assert all(s["name"] != "damage_analysis" for s in plain.run_report["stages"])


def test_damage_output_through_the_pipeline_is_deterministic(tmp_path):
    a, b = run(tmp_path, "a"), run(tmp_path, "b")
    ja, jb = json.loads((a.output_dir / "property.json").read_text()), json.loads((b.output_dir / "property.json").read_text())
    assert strip_volatile(ja) == strip_volatile(jb) and ja["damage_regions"]
    assert (a.output_dir / "plan.png").read_bytes() == (b.output_dir / "plan.png").read_bytes()
    assert (a.output_dir / "diagnostics" / "damage" / "damage_plan.png").read_bytes() == (b.output_dir / "diagnostics" / "damage" / "damage_plan.png").read_bytes()


def test_cli_no_damage_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: outputs_with_evidence())
    assert main(["process", "C:/data/cap", "--tier", "lidar", "--output", str(tmp_path / "on")]) == 0
    assert main(["process", "C:/data/cap", "--tier", "lidar", "--output", str(tmp_path / "off"), "--no-damage"]) == 0
    capsys.readouterr()
    on = json.loads((tmp_path / "on" / "property.json").read_text())
    off = json.loads((tmp_path / "off" / "property.json").read_text())
    assert len(on["damage_regions"]) == 2 and off["damage_regions"] == []


# ---------------- evidence providers ----------------


def test_evenly_spaced_is_deterministic_and_bounded():
    assert evenly_spaced(list(range(10)), 20) == list(range(10))
    pick = evenly_spaced(list(range(100)), 8)
    assert len(pick) == 8 and pick[0] == 0 and pick[-1] == 99 and pick == evenly_spaced(list(range(100)), 8)


def test_photo_evidence_composes_the_room_placement_with_the_local_pose():
    pose = (0.6, 2.0, -1.5)  # room frame -> property frame
    local = camera_pose([1.0, 1.4, 2.0], 0.4)
    cam = SfmCameraInfo("SIMPLE_RADIAL", 1280, 960, [1000.0, 640.0, 480.0, 0.02], 1000.0, "x")
    placed = [{"room_id": "room_002", "pose": pose, "floor_y": -0.2, "poses": {"room_002/img_00.jpg": local}, "cameras": {"room_002/img_00.jpg": cam}}]
    img = photo_evidence_fn(placed, lambda n: np.zeros((960, 1280, 3), np.uint8))()[0]
    assert img.tier == "photos" and img.room_hint == "room_002" and img.floor_y == -0.2 and img.camera.k1 == 0.02
    assert img.T_wc[:3, 3][[0, 2]] == pytest.approx(apply(pose, np.array([[1.0, 2.0]]))[0])  # the planar convention of stitching
    assert img.T_wc[1, 3] == pytest.approx(1.4)  # height is unchanged
    assert img.T_wc[:3, :3] @ img.T_wc[:3, :3].T == pytest.approx(np.eye(3), abs=1e-9)
    assert img.load().shape == (960, 1280, 3)
    M = planar_to_matrix((math.pi / 2, 0, 0))
    assert M[:3, :3] @ np.array([1.0, 0, 0]) == pytest.approx([0, 0, 1.0])  # x' = c x - s z, z' = s x + c z


def test_video_evidence_uses_the_scaled_keyframe_poses(tmp_path):
    kdir = tmp_path / "kf"
    kdir.mkdir()
    for i in range(6):
        cv2.imwrite(str(kdir / f"kf_{i:04d}.jpg"), np.full((96, 128, 3), 40 + i, np.uint8))
    names = [f"kf_{i:04d}.jpg" for i in range(6)]
    T = {n: camera_pose([float(i), 1.4, 0.0], 0.1 * i) for i, n in enumerate(names)}
    cam = SfmCameraInfo("SIMPLE_RADIAL", 128, 96, [100.0, 64.0, 48.0, 0.03], 100.0, "x")
    out = SimpleNamespace(floor_y_m=-1.4)
    imgs = video_evidence_fn(kdir, names, T, cam, out, {names[2]: 1234}, max_images=4)()
    assert len(imgs) == 4 and all(i.tier == "video" and i.floor_y == -1.4 and i.camera.k1 == 0.03 for i in imgs)
    assert imgs[0].id == "kf_0000.jpg" and imgs[-1].id == "kf_0005.jpg" and imgs[0].load().shape == (96, 128, 3)
    assert np.allclose(imgs[1].T_wc, T[imgs[1].id])
    assert video_evidence_fn(kdir, names, T, cam, SimpleNamespace(floor_y_m=None), None)() == []  # no floor, no evidence


def test_lidar_evidence_uses_the_production_poses_and_rgb_intrinsics(tmp_path):
    root = tmp_path / "cap"
    root.mkdir()
    make_test_video(root / "rgb.mp4", seconds=6, fps=20, size=(640, 480))
    frames = [0, 20, 40, 60, 80, 100]
    poses = [camera_pose([0.1 * i, 1.4, 0.0], 0.05 * i) for i in range(6)]
    intr = SimpleNamespace(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    comp = SimpleNamespace(frames=frames, production_poses=poses, ctx=SimpleNamespace(
        source_size=(640, 480), poses={f: {"intr": intr} for f in frames}, fallback_intrinsics=None))
    out = SimpleNamespace(floor_y_m=-1.4)
    imgs = lidar_evidence_fn(root, comp, out, max_images=4)()
    assert [i.frame_index for i in imgs] == [0, 40, 60, 100]  # 4 evenly spaced of the 6 frames (indices 0, 2, 3, 5)
    assert all(i.tier == "lidar" and i.floor_y == -1.4 for i in imgs) and imgs[0].id == "frame_000000"
    assert imgs[0].camera.fx == 800.0 and (imgs[0].camera.width, imgs[0].camera.height) == (640, 480)  # no resize below 1280 px
    assert np.allclose(imgs[1].T_wc, poses[frames.index(imgs[1].frame_index)])
    assert imgs[1].load().shape == (480, 640, 3)
    assert lidar_evidence_fn(root, comp, SimpleNamespace(floor_y_m=None))() == []
    assert lidar_evidence_fn(tmp_path / "nowhere", comp, out)() == []
