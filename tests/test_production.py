"""The product contract: one CLI, one output contract, one status model, structured warnings/errors, safe output, tier detection."""

import hashlib
import json
import os
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from helpers_damage import CAM  # noqa: E402,F401
from helpers_photo import SyntheticPhotoSfm, synthetic_property_dir  # noqa: E402
from helpers_pipeline import synthetic_outputs  # noqa: E402
from helpers_video import SyntheticWalkthrough, make_test_video  # noqa: E402
from test_damage_pipeline import outputs_with_evidence  # noqa: E402

import spatialforge  # noqa: E402
import spatialforge.config as cfgmod  # noqa: E402
from spatialforge.cli import main  # noqa: E402
from spatialforge.config import ProcessConfig  # noqa: E402
from spatialforge.photos.pipeline import PhotoOptions  # noqa: E402
from spatialforge.pipeline import lidar_pipeline, outputs  # noqa: E402
from spatialforge.pipeline.lidar_adapter import StageRecord  # noqa: E402
from spatialforge.pipeline.messages import ErrorRecord, UserError, classify_error, classify_warning  # noqa: E402
from spatialforge.pipeline.product import run_product  # noqa: E402
from spatialforge.pipeline.provenance import redact_text, scrub  # noqa: E402
from spatialforge.pipeline.report import check_report_shape  # noqa: E402
from spatialforge.pipeline.schema import build_schema, validate_against_schema  # noqa: E402
from spatialforge.pipeline.serialization import strip_volatile  # noqa: E402
from spatialforge.pipeline.stage_model import STANDARD_STAGES, fold_stages, standard_name  # noqa: E402
from spatialforge.pipeline.stages import StageTimer  # noqa: E402
from spatialforge.pipeline.tier_detect import detect_tier  # noqa: E402
from spatialforge.video.fusion import FusionOptions  # noqa: E402
from spatialforge.video.pipeline import VideoOptions  # noqa: E402
from spatialforge.video.sfm import SfmResult, SfmRun  # noqa: E402

HOME_NAME = Path.home().name


# ---------------- helpers ----------------


def lidar_dir(root: Path) -> Path:
    """A folder with the LiDAR capture layout (names only: the stage functions are mocked where data is needed)."""
    (root / "depth").mkdir(parents=True)
    (root / "depth" / "000000.png").write_bytes(b"x")
    (root / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n")
    return root


@pytest.fixture
def mocked_lidar(monkeypatch):
    def install(so=None):
        monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: so or synthetic_outputs())

    install()
    return install


def video_options(walk=None, **kw):
    walk = walk or SyntheticWalkthrough()
    return VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=kw.pop("sfm_fn", walk.sfm_fn), depth_backend=walk.depth_backend(),
                        fusion=FusionOptions(pixel_stride=1), **kw)


def photo_options(sfm=None):
    sfm = sfm or SyntheticPhotoSfm()
    o = PhotoOptions(sfm_backend=sfm, depth_backend=sfm.depth_backend())
    o.room.fusion = FusionOptions(pixel_stride=1, min_views=1)
    return o


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    return make_test_video(tmp_path_factory.mktemp("vid") / "walk.mp4", seconds=6)


def load(result, name="property.json"):
    return json.loads((result.output_dir / name).read_text())


def weak_sfm(walk):
    base = walk.sfm_fn

    def weak(kdir, work, opts):
        run = base(kdir, work, opts)
        run.result.tracking_quality, run.result.registered_ratio, run.result.model_sizes = "weak", 0.3, [24, 12]
        return run

    return weak


def partial_dirs(parent: Path):
    return [p for p in parent.iterdir() if ".partial-" in p.name]


# ---------------- the output contract, per tier ----------------


def test_lidar_success_or_partial_run_has_the_standard_outputs(tmp_path, mocked_lidar):
    r = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "out", ProcessConfig(tier="lidar"))
    assert r.status in ("success", "partial") and r.exit_code == 0
    for name in ("property.json", "plan.png", "run_report.json"):
        assert (tmp_path / "out" / name).stat().st_size > 0, name
    assert (tmp_path / "out" / "diagnostics").is_dir() and partial_dirs(tmp_path) == []  # the staging folder is gone
    d = load(r)
    assert validate_against_schema(d, build_schema()) == []
    expected = "success" if d["property"]["status"] == "complete" else "partial"
    assert r.status == expected == load(r, "run_report.json")["status"]
    assert check_report_shape(load(r, "run_report.json")) == []


def test_video_partial_with_weak_tracking_has_structured_warnings(tmp_path, video_file):
    # Real weak tracking caps the geometry quality, and this small synthetic room then yields no walls (a failure, below). To get
    # "weak tracking AND usable geometry" the real stages run on a strong track and the weak-tracking blocker is added to the outputs.
    from spatialforge.video.pipeline import run_video_stages

    def weak_tracking(capture, options):
        so = run_video_stages(capture, options)
        so.status_blockers.append("Tracking is weak, so the property is reported as partial.")
        return so

    r = run_product(video_file, tmp_path / "out", ProcessConfig(tier="video", damage=False), stages_fn=weak_tracking, options=video_options())
    assert r.status == "partial" and r.exit_code == 0 and load(r)["rooms"]
    rep = load(r, "run_report.json")
    codes = {w["code"]: w for w in rep["warnings"]}
    w = codes["VIDEO_TRACKING_WEAK"]
    assert (w["stage"], w["severity"], w["recoverable"]) == ("frontend", "warning", True) and "weak" in w["message"].lower()
    assert rep["tier"] == "video" and load(r)["capture"]["tier"] == "video"
    assert check_report_shape(rep) == []


def test_video_weak_tracking_without_geometry_is_not_hidden_as_partial(tmp_path, video_file):
    walk = SyntheticWalkthrough()
    o = VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=weak_sfm(walk), depth_backend=walk.depth_backend())  # default (coarse) fusion
    r = run_product(video_file, tmp_path / "out", ProcessConfig(tier="video", damage=False), options=o)
    assert r.status == "failure" and r.exit_code == 1 and not (tmp_path / "out" / "property.json").exists()
    rep = load(r, "run_report.json")
    assert rep["errors"][0]["code"] == "NO_GEOMETRY" and "VIDEO_TRACKING_WEAK" in [w["code"] for w in rep["warnings"]]


def test_video_fatal_sfm_failure_is_a_failure_not_a_partial(tmp_path, video_file):
    walk = SyntheticWalkthrough()

    def failing(kdir, work, opts):
        res = SfmResult(24, 3, 0.125, 40, 0.9, 3.0, 4, [3, 3, 3, 3], None, ["x"], "failure", ["no coherent reconstruction"])
        return SfmRun(res, {})

    r = run_product(video_file, tmp_path / "out", ProcessConfig(tier="video"), options=video_options(walk, sfm_fn=failing))
    assert r.status == "failure" and r.exit_code == 1 and r.property is None
    e = load(r, "run_report.json")["errors"][0]
    assert (e["code"], e["stage"]) == ("SFM_FAILED", "frontend") and e["hint"]
    assert not (tmp_path / "out" / "property.json").exists() and not (tmp_path / "out" / "plan.png").exists()


def test_video_without_a_metric_scale_is_a_failure_with_the_reason(tmp_path, video_file):
    walk = SyntheticWalkthrough(depth_noise=1.4)  # per-frame scales that disagree wildly: no metric scale can be trusted
    r = run_product(video_file, tmp_path / "out", ProcessConfig(tier="video"), options=video_options(walk))
    assert r.status == "failure" and r.exit_code == 1 and not (tmp_path / "out" / "property.json").exists()
    rep = load(r, "run_report.json")
    assert rep["errors"][0]["code"] == "METRIC_SCALE_UNAVAILABLE" and rep["errors"][0]["stage"] == "metric_scale"
    assert next(s for s in rep["stages"] if s["stage"] == "metric_scale")["status"] in ("failed", "warning")
    assert (tmp_path / "out" / "diagnostics" / "scale_report.json").exists()  # the evidence for the failure is kept


def test_photos_connected_property(tmp_path):
    root = synthetic_property_dir(tmp_path / "photos", ["a", "b", "c"])
    r = run_product(root, tmp_path / "out", ProcessConfig(tier="photos", damage=False), options=photo_options())
    assert r.status in ("success", "partial") and r.exit_code == 0
    d = load(r)
    assert [x["id"] for x in d["rooms"]] == ["room_001", "room_002", "room_003"] and d["capture"]["tier"] == "photos"
    rep = load(r, "run_report.json")
    assert next(s for s in rep["stages"] if s["stage"] == "stitching")["status"] in ("ok", "warning")
    assert next(s for s in rep["stages"] if s["stage"] == "metric_scale")["duration_seconds"] > 0  # the rooms' own stages are folded in
    assert check_report_shape(rep) == []


def test_photos_disconnected_property_is_partial(tmp_path):
    root = synthetic_property_dir(tmp_path / "photos", ["a", "b", "e"])
    sfm = SyntheticPhotoSfm(label_to_layout={"e": "room_005"})  # the corridor shares nothing with the others
    r = run_product(root, tmp_path / "out", ProcessConfig(tier="photos", damage=False), options=photo_options(sfm))
    assert r.status == "partial" and r.exit_code == 0
    codes = [w["code"] for w in load(r, "run_report.json")["warnings"]]
    assert "PHOTO_ROOM_UNPLACED" in codes
    assert [x["id"] for x in load(r)["rooms"]] == ["room_001", "room_002"]


def test_photos_where_every_room_fails_is_a_failure(tmp_path):
    root = synthetic_property_dir(tmp_path / "photos", ["a", "b"])
    sfm = SyntheticPhotoSfm(fail_rooms={"room_001", "room_002"})
    r = run_product(root, tmp_path / "out", ProcessConfig(tier="photos"), options=photo_options(sfm))
    assert r.status == "failure" and r.exit_code == 1 and not (tmp_path / "out" / "property.json").exists()
    e = load(r, "run_report.json")["errors"][0]
    assert e["code"] == "SFM_FAILED" and e["stage"] == "frontend"


def test_photos_without_any_room_scale_is_a_failure_not_an_empty_partial(tmp_path):
    root = synthetic_property_dir(tmp_path / "photos", ["a", "b"])
    sfm = SyntheticPhotoSfm()
    from spatialforge.video.depth import DepthUnavailable

    def factory():
        raise DepthUnavailable("model weights not found")

    o = PhotoOptions(sfm_backend=sfm, depth_backend_factory=factory)
    r = run_product(root, tmp_path / "out", ProcessConfig(tier="photos"), options=o)
    assert r.status == "failure" and load(r, "run_report.json")["errors"][0]["code"] == "METRIC_SCALE_UNAVAILABLE"


def test_lidar_with_walls_but_no_rooms_is_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: synthetic_outputs(rooms=False, ceiling=False))
    r = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    d = load(r)
    assert r.status == "partial" and d["rooms"] == [] and len(d["walls"]) >= 1
    assert "STRUCTURE_NO_ROOMS" in [w["code"] for w in d["warning_records"]]


def test_lidar_with_no_geometry_at_all_is_a_failure(tmp_path, monkeypatch):
    so = synthetic_outputs()
    so.wall_dicts, so.topology, so.openings = [], None, None  # nothing was recovered
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: so)
    r = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "out", ProcessConfig(tier="lidar"))
    assert r.status == "failure" and load(r, "run_report.json")["errors"][0]["code"] == "NO_GEOMETRY"
    assert not (tmp_path / "out" / "property.json").exists()


def test_invalid_lidar_capture_fails_cleanly_without_a_traceback(tmp_path, capsys):
    bad = lidar_dir(tmp_path / "cap")  # right names, no real data
    code = main(["process", str(bad), "--tier", "lidar", "--output", str(tmp_path / "out")])
    out = capsys.readouterr()
    assert code == 1
    assert "ERROR [INPUT_INVALID]" in out.out and "Traceback" not in out.out + out.err
    rep = json.loads((tmp_path / "out" / "run_report.json").read_text())
    assert rep["status"] == "failure" and rep["errors"][0]["code"] == "INPUT_INVALID" and rep["errors"][0]["stage"] == "validation"
    assert not (tmp_path / "out" / "property.json").exists()


# ---------------- damage ----------------


def test_damage_enabled_disabled_and_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: outputs_with_evidence())
    cap = lidar_dir(tmp_path / "cap")
    on = run_product(cap, tmp_path / "on", ProcessConfig(tier="lidar"))
    assert len(load(on)["damage_regions"]) == 2
    assert next(s for s in load(on, "run_report.json")["stages"] if s["stage"] == "damage")["status"] in ("ok", "warning")
    off = run_product(cap, tmp_path / "off", ProcessConfig(tier="lidar", damage=False))
    rep = load(off, "run_report.json")
    assert load(off)["damage_regions"] == [] and next(s for s in rep["stages"] if s["stage"] == "damage")["status"] == "skipped"
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: outputs_with_evidence(fail=True))
    bad = run_product(cap, tmp_path / "bad", ProcessConfig(tier="lidar"))
    assert bad.status == off.status and bad.exit_code == 0  # an optional layer failing warns and continues; it cannot downgrade the run
    codes = [w["code"] for w in load(bad, "run_report.json")["warnings"]]
    assert "DAMAGE_FAILED" in codes and load(bad)["damage_regions"] == []
    assert next(s for s in load(bad, "run_report.json")["stages"] if s["stage"] == "damage")["status"] == "failed"


# ---------------- output safety ----------------


def test_an_existing_result_is_never_overwritten_by_default(tmp_path, mocked_lidar):
    cap = lidar_dir(tmp_path / "cap")
    first = run_product(cap, tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    assert first.exit_code == 0
    digest = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "out").iterdir() if p.is_file()}  # noqa: E731
    before = digest()
    again = run_product(cap, tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    assert again.status == "failure" and again.exit_code == 1
    assert again.run_report["errors"][0]["code"] == "OUTPUT_EXISTS" and "--overwrite" in again.run_report["errors"][0]["hint"]
    assert digest() == before and partial_dirs(tmp_path) == []  # nothing was touched, not even the old report


def test_overwrite_replaces_a_previous_result_only_when_asked(tmp_path, mocked_lidar):
    cap = lidar_dir(tmp_path / "cap")
    run_product(cap, tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    (tmp_path / "out" / "stale.txt").write_text("from the previous run")
    r = run_product(cap, tmp_path / "out", ProcessConfig(tier="lidar", damage=False, overwrite=True))
    assert r.exit_code == 0 and not (tmp_path / "out" / "stale.txt").exists() and (tmp_path / "out" / "property.json").exists()
    # a non-empty folder that is not a SpatialForge result is never deleted, even with --overwrite
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "thesis.docx").write_text("do not delete")
    refused = run_product(cap, precious, ProcessConfig(tier="lidar", overwrite=True))
    assert refused.status == "failure" and refused.run_report["errors"][0]["code"] == "OUTPUT_NOT_A_RESULT"
    assert (precious / "thesis.docx").read_text() == "do not delete" and not (precious / "run_report.json").exists()


def test_a_crash_leaves_a_marked_incomplete_folder_and_no_valid_looking_result(tmp_path):
    cap = lidar_dir(tmp_path / "cap")

    def interrupted(capture, options):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_product(cap, tmp_path / "out", ProcessConfig(tier="lidar"), stages_fn=interrupted)
    assert not (tmp_path / "out").exists()
    left = partial_dirs(tmp_path)
    assert len(left) == 1 and (left[0] / outputs.INCOMPLETE_MARKER).exists() and not (left[0] / "property.json").exists()


def test_the_overwrite_flag_works_from_the_command_line(tmp_path, mocked_lidar, capsys):
    cap = lidar_dir(tmp_path / "cap")
    args = ["process", str(cap), "--tier", "lidar", "--output", str(tmp_path / "out"), "--no-damage"]
    assert main(args) == 0
    assert main(args) == 1 and "ERROR [OUTPUT_EXISTS]" in capsys.readouterr().out
    assert main(args + ["--overwrite"]) == 0


# ---------------- tier detection ----------------


def test_auto_tier_lidar(tmp_path, mocked_lidar):
    cap = lidar_dir(tmp_path / "cap")
    assert detect_tier(cap).tier == "lidar"
    r = run_product(cap, tmp_path / "out", ProcessConfig(tier="auto", damage=False))
    rep = load(r, "run_report.json")
    assert rep["tier"] == "lidar" and rep["tier_requested"] == "auto" and "odometry.csv" in rep["tier_detection"]
    nested = tmp_path / "outer"
    lidar_dir(nested / "capture123")  # the layout with a single sub-folder
    assert detect_tier(nested).tier == "lidar"


def test_auto_tier_video(tmp_path, video_file):
    assert detect_tier(video_file).tier == "video"
    r = run_product(video_file, tmp_path / "out", ProcessConfig(tier="auto", damage=False), options=video_options())
    assert r.run_report["tier"] == "video" and r.run_report["tier_requested"] == "auto" and r.exit_code == 0


def test_auto_tier_photos(tmp_path):
    root = synthetic_property_dir(tmp_path / "photos", ["a", "b"])
    assert detect_tier(root).tier == "photos"
    r = run_product(root, tmp_path / "out", ProcessConfig(tier="auto", damage=False), options=photo_options())
    assert r.run_report["tier"] == "photos" and r.run_report["tier_requested"] == "auto"


def test_ambiguous_or_unknown_input_is_refused_not_guessed(tmp_path, capsys):
    both = lidar_dir(tmp_path / "both")
    (both / "room_a").mkdir()
    (both / "room_a" / "1.jpg").write_bytes(b"x")  # LiDAR files AND an image folder: ambiguous
    empty = tmp_path / "empty"
    empty.mkdir()
    with_video = tmp_path / "withvideo"
    with_video.mkdir()
    (with_video / "walk.mp4").write_bytes(b"x")
    image = tmp_path / "pic.jpg"
    image.write_bytes(b"x")
    other = tmp_path / "notes.txt"
    other.write_text("x")
    stray = tmp_path / "stray"
    (stray / "docs").mkdir(parents=True)
    (stray / "docs" / "a.txt").write_text("x")
    for target in (both, empty, with_video, image, other, stray):
        with pytest.raises(UserError) as ei:
            detect_tier(target)
        assert ei.value.code == "TIER_AMBIGUOUS" and "--tier" in ei.value.hint, target.name
    code = main(["process", str(empty), "--tier", "auto", "--output", str(tmp_path / "out")])
    assert code == 1 and "ERROR [TIER_AMBIGUOUS]" in capsys.readouterr().out
    with pytest.raises(UserError) as ei:
        detect_tier(tmp_path / "nope")
    assert ei.value.code == "INPUT_NOT_FOUND"


# ---------------- structured warnings and errors ----------------


def test_warnings_are_structured_and_keep_their_readable_message(tmp_path, mocked_lidar):
    r = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    d = load(r)
    assert len(d["warning_records"]) == len(d["warnings"]) and [w["message"] for w in d["warning_records"]] == d["warnings"]
    for w in d["warning_records"]:
        assert set(w) == {"code", "stage", "severity", "message", "recoverable"} and w["severity"] in ("info", "warning", "error")
        assert w["stage"] in STANDARD_STAGES or w["stage"] == "pipeline"
    by_code = {w["code"]: w for w in d["warning_records"]}
    assert by_code["NO_GROUND_TRUTH"]["severity"] == "info" and by_code["OPENINGS_RECALL_UNVERIFIED"]["stage"] == "openings"
    assert validate_against_schema(d, build_schema()) == []
    rep_codes = {w["code"] for w in load(r, "run_report.json")["warnings"]}
    assert rep_codes == {w["code"] for w in d["warning_records"]}


def test_warning_classification_rules():
    assert classify_warning("SfM tracking quality is WEAK: 25/145 keyframes registered").code == "VIDEO_TRACKING_WEAK"
    assert classify_warning("Metric scale is NOT available, so no dimensions are reported").severity == "error"
    assert classify_warning("room_003 (e) is not part of the property plan: no stitch evidence").code == "PHOTO_ROOM_UNPLACED"
    assert classify_warning("damage: 3 of 5 damage detections have no reliable metric extent").stage == "damage"
    odd = classify_warning("something no rule knows about")
    assert (odd.code, odd.stage, odd.severity, odd.recoverable) == ("GENERAL_WARNING", "pipeline", "warning", True)
    assert odd.message == "something no rule knows about"  # never dropped, never rewritten


def test_errors_are_structured_with_hints():
    e = classify_error("video file does not exist: x.mp4", "validation", "video_validation")
    assert (e.code, e.stage, e.raw_stage) == ("INPUT_NOT_FOUND", "validation", "video_validation") and e.hint
    assert classify_error("no coherent reconstruction: 12/150", "frontend").code == "SFM_FAILED"
    assert classify_error("cannot decode image: truncated", "validation").code == "INPUT_UNREADABLE"
    assert classify_error("???", "frontend").code == "PIPELINE_ERROR"
    assert set(ErrorRecord("A", "frontend", "m").to_dict()) == {"code", "stage", "message", "hint", "recoverable", "raw_stage"}
    assert UserError("X", "boom", "fix it").record().hint == "fix it"


def test_unexpected_exceptions_are_clean_failures_and_debug_keeps_the_trace(tmp_path, capsys):
    cap = lidar_dir(tmp_path / "cap")

    def crash(capture, options):
        raise RuntimeError("something unforeseen")

    r = run_product(cap, tmp_path / "out", ProcessConfig(tier="lidar"), stages_fn=crash)
    e = r.run_report["errors"][0]
    assert r.status == "failure" and e["code"] == "UNEXPECTED_ERROR" and "something unforeseen" in e["message"] and "--debug" in e["hint"]
    assert "traceback" not in r.run_report  # no raw traceback in the product output by default
    d = run_product(cap, tmp_path / "dbg", ProcessConfig(tier="lidar", debug=True), stages_fn=crash)
    assert "RuntimeError" in d.run_report["traceback"]
    assert "something unforeseen" in (tmp_path / "out" / "diagnostics" / "run.log").read_text()  # the details are in the run log


# ---------------- stages and timing ----------------


def test_run_report_stage_and_timing_schema(tmp_path, mocked_lidar):
    r = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    rep = load(r, "run_report.json")
    assert [s["stage"] for s in rep["stages"]] == list(STANDARD_STAGES) and len(STANDARD_STAGES) == 12
    for s in rep["stages"]:
        assert set(s) >= {"stage", "status", "duration_seconds", "warnings", "error"}
        assert isinstance(s["duration_seconds"], float) and s["duration_seconds"] >= 0 and s["status"] in ("ok", "warning", "failed", "skipped")
    assert rep["total_duration_seconds"] >= 0
    by = {s["stage"]: s for s in rep["stages"]}
    assert by["stitching"]["status"] == "skipped" and by["metric_scale"]["status"] == "skipped"  # not used by LiDAR
    assert by["rendering"]["status"] == "ok" and by["serialization"]["status"] == "ok" and by["validation"]["status"] in ("ok", "warning", "skipped")
    assert "runtime_s" not in rep and rep["substages"]
    assert load(r)["timing"]["stages"].keys() == set(STANDARD_STAGES) and "total_duration_seconds" in load(r)["timing"]


def test_the_tiers_share_one_timing_schema(tmp_path, video_file):
    cap = lidar_dir(tmp_path / "cap")
    reps = []
    import spatialforge.pipeline.lidar_pipeline as lp

    old = lp.run_lidar_stages
    lp.run_lidar_stages = lambda c, o: synthetic_outputs()
    try:
        reps.append(run_product(cap, tmp_path / "l", ProcessConfig(tier="lidar", damage=False)).run_report)
    finally:
        lp.run_lidar_stages = old
    reps.append(run_product(video_file, tmp_path / "v", ProcessConfig(tier="video", damage=False), options=video_options()).run_report)
    reps.append(run_product(synthetic_property_dir(tmp_path / "p", ["a", "b"]), tmp_path / "ph", ProcessConfig(tier="photos", damage=False),
                            options=photo_options()).run_report)
    for rep in reps:
        assert check_report_shape(rep) == [] and [s["stage"] for s in rep["stages"]] == list(STANDARD_STAGES)
    assert {tuple(sorted(r.keys() & {"stages", "total_duration_seconds", "warnings", "errors", "status", "tier"})) for r in reps}.pop()


def test_raw_stage_records_fold_into_the_standard_stages():
    recs = [StageRecord("capture_validation", "ok", 0.5), StageRecord("sfm_reconstruction", "warning", 10.0, ["w"]),
            StageRecord("keyframe_extraction", "ok", 2.0), StageRecord("room_001_reconstruction", "ok", 3.0),
            StageRecord("room_001/metric_depth_and_scale", "ok", 4.0), StageRecord("room_002/structural_walls", "failed", 1.0, [], "boom"),
            StageRecord("cross_room_stitching", "ok", 0.2), StageRecord("global_placement", "ok", 0.1)]
    out = {s["stage"]: s for s in fold_stages(recs, lambda std, m: [])}
    assert out["validation"]["duration_seconds"] == 0.5 and out["frontend"]["duration_seconds"] == 15.0  # sfm + keyframes + room mapping
    assert out["frontend"]["status"] == "warning" and out["metric_scale"]["duration_seconds"] == 4.0
    assert out["structure"]["status"] == "failed" and out["structure"]["error"]["message"] == "boom"
    assert out["stitching"]["duration_seconds"] == pytest.approx(0.3) and out["damage"]["status"] == "skipped"
    assert standard_name("room_007/openings") == "openings" and standard_name("nonsense") is None


def test_console_progress_and_the_run_log(tmp_path, monkeypatch, capsys):
    def stages(capture, options):
        so = synthetic_outputs()
        for name in ("capture_validation", "metric_reconstruction_and_pose_check", "floor_and_ceiling_planes"):
            with StageTimer(so.stages, name):
                pass
        return so

    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", stages)
    cap = lidar_dir(tmp_path / "cap")
    assert main(["process", str(cap), "--tier", "lidar", "--output", str(tmp_path / "out"), "--no-damage"]) == 0
    err = capsys.readouterr().err
    assert "[1/8] Validate input" in err and "Reconstruct metric geometry" in err and "Render plan" in err and "Write results" in err
    assert "Traceback" not in err
    log = (tmp_path / "out" / "diagnostics" / "run.log").read_text()
    assert "stage start: capture_validation" in log and "spatialforge process: tier=lidar" in log


# ---------------- version, provenance, paths ----------------


def test_package_version_has_one_source_and_is_everywhere(tmp_path, mocked_lidar, capsys):
    assert spatialforge.__version__ == "0.1.0"
    toml = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert "version" in toml["project"]["dynamic"] and toml["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "spatialforge.__version__"}
    assert "version" not in toml["project"]  # no second copy of the number
    r = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    assert load(r, "run_report.json")["spatialforge_version"] == "0.1.0"
    prov = load(r)["provenance"]
    assert prov["pipeline"]["spatialforge_version"] == "0.1.0" and prov["runtime"]["spatialforge_version"] == "0.1.0"
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0 and capsys.readouterr().out.strip() == "spatialforge 0.1.0"


def test_provenance_is_standardised_and_contains_nothing_personal(tmp_path, mocked_lidar, monkeypatch):
    r = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "out", ProcessConfig(tier="lidar", damage=False))
    p = load(r)["provenance"]
    rt = p["runtime"]
    assert rt["tier"] == "lidar" and rt["python"].count(".") == 2 and rt["platform"] and "numpy" in rt["libraries"]
    assert rt["damage_method"] is None and "not_bit_reproducible" in rt["determinism"] and p["fallback_decisions"]
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: outputs_with_evidence())
    r2 = run_product(lidar_dir(tmp_path / "cap2"), tmp_path / "out2", ProcessConfig(tier="lidar"))
    assert load(r2)["provenance"]["runtime"]["damage_method"]["name"] == "deterministic-opencv"
    for name in ("property.json", "run_report.json"):
        text = (tmp_path / "out" / name).read_text()
        assert str(Path.home()) not in text and str(Path.home()).replace("\\", "\\\\") not in text, name  # redacted to "~"
    assert load(r, "run_report.json")["input"]["name"] == "cap" and load(r, "run_report.json")["input"]["path"].startswith("~")


def test_path_redaction():
    home = str(Path.home())
    assert redact_text(home + "\\data\\cap") == "~\\data\\cap" and redact_text(home.replace("\\", "/") + "/x") == "~/x"
    assert redact_text(home.upper() + "\\x") == "~\\x"  # Windows paths are case-insensitive
    assert redact_text("C:\\captures\\single_room") == "C:\\captures\\single_room"  # not personal: unchanged
    assert scrub({"a": [home + "\\x", {"b": home}], "n": 3}) == {"a": ["~\\x", {"b": "~"}], "n": 3}


# ---------------- model paths and dependencies ----------------


def test_model_path_helpers(tmp_path, monkeypatch):
    monkeypatch.setenv(cfgmod.MODELS_ENV, str(tmp_path / "models"))
    assert cfgmod.models_dir() == tmp_path / "models"
    monkeypatch.undo()  # drop the autouse "weights present" patch too
    monkeypatch.setenv(cfgmod.MODELS_ENV, str(tmp_path / "models"))
    assert cfgmod.depth_weights_present() is False
    snap = tmp_path / "models" / "hf" / ("models--" + cfgmod.DEPTH_MODEL_REPO.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x")
    assert cfgmod.depth_weights_present() is True
    hint = cfgmod.model_setup_hint()
    assert "python scripts/download_models.py" in hint and cfgmod.MODELS_ENV in hint and str(tmp_path / "models") in hint
    monkeypatch.delenv(cfgmod.MODELS_ENV)
    assert cfgmod.models_dir() == Path.home() / ".spatialforge" / "models"


def test_missing_model_weights_is_a_clear_early_failure(tmp_path, video_file, monkeypatch, capsys):
    monkeypatch.setattr(cfgmod, "depth_weights_present", lambda: False)
    code = main(["process", str(video_file), "--tier", "video", "--output", str(tmp_path / "out")])
    out = capsys.readouterr()
    assert code == 1
    assert "ERROR [MODEL_WEIGHTS_MISSING]" in out.out and "python scripts/download_models.py" in out.out and "Traceback" not in out.out + out.err
    rep = json.loads((tmp_path / "out" / "run_report.json").read_text())
    e = rep["errors"][0]
    assert (e["code"], e["stage"]) == ("MODEL_WEIGHTS_MISSING", "metric_scale") and "python scripts/download_models.py" in e["hint"]
    assert rep["status"] == "failure" and rep["total_duration_seconds"] == 0.0 and partial_dirs(tmp_path) == []  # failed before any work
    # LiDAR needs no weights
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: synthetic_outputs())
    assert main(["process", str(lidar_dir(tmp_path / "cap")), "--tier", "lidar", "--output", str(tmp_path / "ok"), "--no-damage"]) == 0


def test_a_missing_python_dependency_names_the_fix(tmp_path, video_file, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: None if n == "pycolmap" else real(n, *a, **k))
    r = run_product(video_file, tmp_path / "out", ProcessConfig(tier="video"))
    e = r.run_report["errors"][0]
    assert e["code"] == "DEPENDENCY_MISSING" and "pycolmap" in e["message"] and "pip install -e .[video]" in e["hint"]


# ---------------- diagnostics policy and determinism ----------------


def test_normal_runs_keep_the_output_small_and_diagnostics_keeps_the_clouds(tmp_path, video_file):
    normal = run_product(video_file, tmp_path / "n", ProcessConfig(tier="video", damage=False), options=video_options())
    full = run_product(video_file, tmp_path / "f", ProcessConfig(tier="video", damage=False, diagnostics=True), options=video_options())
    nd, fd = tmp_path / "n" / "diagnostics", tmp_path / "f" / "diagnostics"
    assert not (nd / "keyframes").exists() and not list(nd.rglob("*.ply"))
    assert (fd / "keyframes").is_dir() and (fd / "metric_cloud.ply").exists() and (fd / "sparse_sfm.ply").exists()
    assert (nd / "scale_report.json").exists() and (nd / "trajectory.png").exists() and (nd / "run.log").exists()  # the small ones stay
    size = lambda d: sum(p.stat().st_size for p in d.rglob("*") if p.is_file())  # noqa: E731
    assert size(nd) < size(fd) and normal.exit_code == full.exit_code == 0


def test_canonical_json_is_deterministic_across_runs(tmp_path, mocked_lidar):
    cap = lidar_dir(tmp_path / "cap")
    a = run_product(cap, tmp_path / "a", ProcessConfig(tier="lidar"))
    b = run_product(cap, tmp_path / "b", ProcessConfig(tier="lidar"))
    ja, jb = load(a), load(b)
    assert strip_volatile(ja) == strip_volatile(jb)
    assert (tmp_path / "a" / "plan.png").read_bytes() == (tmp_path / "b" / "plan.png").read_bytes()
    ra, rb = load(a, "run_report.json"), load(b, "run_report.json")
    volatile = ("started_at", "ended_at", "total_duration_seconds", "stages", "substages", "output")
    assert {k: v for k, v in ra.items() if k not in volatile} == {k: v for k, v in rb.items() if k not in volatile}


def test_the_schema_is_identical_across_tiers(tmp_path, video_file, mocked_lidar):
    l = run_product(lidar_dir(tmp_path / "cap"), tmp_path / "l", ProcessConfig(tier="lidar", damage=False))
    v = run_product(video_file, tmp_path / "v", ProcessConfig(tier="video", damage=False), options=video_options())
    p = run_product(synthetic_property_dir(tmp_path / "p", ["a", "b"]), tmp_path / "ph", ProcessConfig(tier="photos", damage=False),
                    options=photo_options())
    docs = [load(x) for x in (l, v, p)]
    keys = [tuple(d) for d in docs]
    assert keys[0] == keys[1] == keys[2]  # the same top-level structure, in the same order
    required = {"schema_version", "capture", "property", "rooms", "walls", "openings", "unverified_openings", "damage_regions",
                "concealed_damage_flags", "scope_line_items", "warnings", "warning_records", "provenance", "timing"}
    for d in docs:
        assert set(d) == required and validate_against_schema(d, build_schema()) == [] and d["schema_version"] == "1.0"
        assert d["capture"]["tier"] in ("lidar", "video", "photos") and d["provenance"]["pipeline"]["tier"] == d["capture"]["tier"]
        assert d["provenance"]["runtime"]["tier"] == d["capture"]["tier"]
    assert [d["capture"]["tier"] for d in docs] == ["lidar", "video", "photos"]
    schema_file = json.loads((Path(__file__).resolve().parent.parent / "schema" / "property.schema.json").read_text())
    assert schema_file == build_schema()  # the committed schema file is in sync with the code
