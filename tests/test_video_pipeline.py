"""Video pipeline end to end with a mocked SfM and a mocked metric-depth model (no weights, no pycolmap run)."""

import json

import numpy as np
import pytest
from helpers_video import SyntheticWalkthrough, make_test_video

pytest.importorskip("cv2")  # the video tier is an optional install (pip install -e .[video])

from spatialforge.cli import main
from spatialforge.pipeline.lidar_pipeline import process_capture
from spatialforge.pipeline.schema import build_schema, validate_against_schema
from spatialforge.pipeline.serialization import strip_volatile
from spatialforge.video.depth import DepthUnavailable, FunctionDepthBackend
from spatialforge.video.fusion import FusionOptions
from spatialforge.video.pipeline import VideoOptions, run_video_stages
from spatialforge.video.sfm import SfmResult, SfmRun


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_test_video(tmp_path_factory.mktemp("vid") / "walk.mp4", seconds=6)


def run(video, tmp_path, name="out", walk=None, **over):
    walk = walk or SyntheticWalkthrough()
    opts = VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=walk.sfm_fn, depth_backend=walk.depth_backend(),
                        fusion=FusionOptions(pixel_stride=1), **over)  # stride 1: the synthetic maps are small
    return process_capture(video, tmp_path / name, opts, stages_fn=run_video_stages, tier="video"), walk


def load(result, name="property.json"):
    return json.loads((result.output_dir / name).read_text())


def all_values(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from all_values(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_values(v, key)


def test_mocked_video_pipeline_runs_end_to_end(video, tmp_path):
    r, walk = run(video, tmp_path)
    assert r.exit_code == 0 and r.status in ("success", "partial")
    for f in ("property.json", "plan.png", "run_report.json"):
        assert (r.output_dir / f).stat().st_size > 0, f
    diag = r.output_dir / "diagnostics"
    for f in ("scale_report.json", "video_frontend.json", "trajectory.png", "trajectory.json", "sparse_sfm.ply", "metric_cloud.ply"):
        assert (diag / f).exists(), f
    assert (diag / "keyframes").is_dir() and len(list((diag / "keyframes").glob("*.jpg"))) == len(walk.names)
    stages = [s["name"] for s in r.run_report["stages"]]
    assert stages[:6] == ["video_validation", "keyframe_extraction", "sfm_reconstruction", "metric_depth_and_scale",
                          "depth_fusion", "gravity_alignment"]
    assert {"floor_and_ceiling_planes", "structural_walls"} <= set(stages)  # the shared structural backend ran
    assert validate_against_schema(r.property, build_schema()) == []  # the same canonical schema as LiDAR
    assert r.run_report["metric_scale_available"] and r.run_report["tracking_quality"] == "strong"
    assert r.run_report["production_pose_source"] == "sfm_scaled"


def test_metric_scale_is_estimated_not_assumed(video, tmp_path):
    r, walk = run(video, tmp_path)
    ms = r.run_report["video"]["metric_scale"]
    assert ms["metric_scale_available"] and ms["metres_per_sfm_unit"] == pytest.approx(walk.true_metres_per_sfm_unit, rel=0.12)
    assert ms["frames_used"] >= 20 and ms["correspondences"] > 1000 and ms["frame_spread_rel"] > 0.03
    report = json.loads((r.output_dir / "diagnostics" / "scale_report.json").read_text())
    assert len(report["per_frame"]) == len(walk.names) and report["depth_backend"]["model"] == "mock-metric-depth"
    g = r.run_report["video"]["gravity"]
    assert abs(np.array(g["up"]) @ (walk.Q @ np.array([0, 1.0, 0]))) > 0.99  # gravity recovered in the SfM frame


def test_recovered_geometry_is_metric_and_lies_in_the_true_room(video, tmp_path):
    r, walk = run(video, tmp_path)
    d = r.property
    assert len(d["walls"]) >= 3
    lengths = sorted(w["length"]["value"] for w in d["walls"] if w["evidence_quality"] in ("strong", "moderate"))
    assert lengths and 2.5 < max(lengths) < 6.5  # walls of a 5 x 4 m room, not SfM-unit sized
    for room in d["rooms"]:
        assert 8.0 < room["area"]["value"] < 32.0  # true area is 20 m2
        assert room["area"]["interval"]["low"] < room["area"]["value"] < room["area"]["interval"]["high"]


def test_canonical_property_is_tagged_as_video(video, tmp_path):
    r, _ = run(video, tmp_path)
    d = r.property
    assert d["capture"]["tier"] == "video" and r.run_report["tier"] == "video"
    tiers = set(all_values(d, "source_tier"))
    assert tiers == {"video"}  # every measurement says where it came from
    assert d["schema_version"] == "1.0" and d["capture"]["video"]["registered_images"] == 24
    prov = d["provenance"]
    assert prov["pipeline"]["tier"] == "video" and prov["sfm"]["backend"] == "pycolmap" and prov["sfm"]["version"]
    assert prov["metric_depth"]["model"] == "mock-metric-depth" and prov["metric_scale"]["metric_scale_available"]
    assert prov["keyframe_strategy"]["count"] == 24 and prov["sfm"]["registered_ratio"] == 1.0
    assert "gravity" in r.run_report["video"] and r.run_report["video"]["gravity"]["method"]
    assert any("less certain than LiDAR" in w for w in d["warnings"])
    assert any("estimated by COLMAP" in w for w in d["warnings"])


def test_video_intervals_are_wider_than_lidar_style_intervals_and_grow_with_scale_variance(video, tmp_path):
    steady, _ = run(video, tmp_path, "steady", walk=SyntheticWalkthrough(depth_noise=0.04))
    noisy, _ = run(video, tmp_path, "noisy", walk=SyntheticWalkthrough(depth_noise=0.30))
    su, nu = steady.run_report["video"]["uncertainty"], noisy.run_report["video"]["uncertainty"]
    assert nu["metric_scale_relative_sigma"] > su["metric_scale_relative_sigma"]
    assert nu["total_relative_sigma"] > su["total_relative_sigma"]

    def rel_width(d):
        ws = [w for w in d["walls"] if w["evidence_quality"] in ("strong", "moderate")]
        return float(np.median([(w["length"]["interval"]["high"] - w["length"]["interval"]["low"]) / w["length"]["value"]
                                for w in ws]))

    assert rel_width(noisy.property) > rel_width(steady.property) > 0.08  # even the steady run is wider than a bare LiDAR wall


def test_failed_sfm_is_a_clean_failure(video, tmp_path):
    walk = SyntheticWalkthrough()

    def failing(kdir, work, opts):
        res = SfmResult(24, 3, 0.125, 40, 0.9, 3.0, 4, [3, 3, 3, 3], None, ["x"], "failure", ["no coherent reconstruction"])
        return SfmRun(res, {})

    opts = VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=failing, depth_backend=walk.depth_backend())
    r = process_capture(video, tmp_path / "out", opts, stages_fn=run_video_stages, tier="video")
    assert r.status == "failure" and r.exit_code == 1 and r.property is None
    assert r.run_report["errors"][0]["stage"] == "sfm_reconstruction" and "no coherent" in r.run_report["errors"][0]["message"]
    assert not (tmp_path / "out" / "property.json").exists() and not (tmp_path / "out" / "plan.png").exists()
    assert r.run_report["stages"][-1]["name"] == "sfm_reconstruction" and r.run_report["metric_scale_available"] is False

    def crashing(kdir, work, opts):
        raise RuntimeError("colmap exploded")

    opts.sfm_fn = crashing
    r2 = process_capture(video, tmp_path / "out2", opts, stages_fn=run_video_stages, tier="video")
    assert r2.status == "failure" and "colmap exploded" in r2.run_report["errors"][0]["message"]


@pytest.mark.parametrize("why", ["wild_scales", "not_metric", "no_backend"])
def test_without_a_reliable_metric_scale_no_metres_are_invented(video, tmp_path, why):
    walk = SyntheticWalkthrough(depth_noise=1.4 if why == "wild_scales" else 0.05)
    backend = FunctionDepthBackend(walk.depth_fn, "relative-depth", metric=(why != "not_metric"))
    over = {}
    if why == "no_backend":
        def factory():
            raise DepthUnavailable("model weights not found")

        over = {"depth_backend_factory": factory}
        opts = VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=walk.sfm_fn, **over)
    else:
        opts = VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=walk.sfm_fn, depth_backend=backend)
    r = process_capture(video, tmp_path / "out", opts, stages_fn=run_video_stages, tier="video")
    d = r.property
    assert r.exit_code == 0 and r.status == "partial" and d["property"]["status"] == "partial"
    assert d["rooms"] == [] and d["walls"] == [] and d["openings"] == []  # no geometry without metres
    assert d["property"]["footprint"] is None
    assert any("Metric scale is NOT available" in w for w in d["warnings"])
    assert r.run_report["metric_scale_available"] is False
    assert d["provenance"]["metric_scale"]["metric_scale_available"] is False
    assert d["provenance"]["metric_scale"].get("metres_per_sfm_unit") in (None, 0)
    names = {s["name"]: s["status"] for s in r.run_report["stages"]}
    assert names["gravity_alignment"] == "skipped" and names["structural_walls"] == "skipped"
    assert not (r.output_dir / "diagnostics" / "metric_cloud.ply").exists()
    assert (r.output_dir / "diagnostics" / "scale_report.json").exists() and (r.output_dir / "plan.png").exists()
    rep = json.loads((r.output_dir / "diagnostics" / "scale_report.json").read_text())
    assert rep["metric_scale_available"] is False and rep["metres_per_sfm_unit"] is None


def test_weak_tracking_continues_but_is_reported_as_partial(video, tmp_path):
    walk = SyntheticWalkthrough()
    base = walk.sfm_fn

    def weak(kdir, work, opts):
        run = base(kdir, work, opts)
        s = run.result
        s.tracking_quality, s.registered_ratio = "weak", 0.3
        s.model_sizes = [24, 20, 12]
        return run

    opts = VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=weak, depth_backend=walk.depth_backend())
    r = process_capture(video, tmp_path / "out", opts, stages_fn=run_video_stages, tier="video")
    d = r.property
    assert r.status == "partial" and d["property"]["status"] == "partial"
    assert any("tracking quality is WEAK" in w for w in d["warnings"]) and any("disconnected models" in w for w in d["warnings"])
    assert r.run_report["video"]["uncertainty"]["sfm_relative_sigma"] > 0.05  # fewer registered frames -> wider


def test_depth_inference_crash_degrades_to_no_scale_not_to_invented_metres(video, tmp_path):
    walk = SyntheticWalkthrough()

    def boom(rgb):
        raise MemoryError("out of memory")

    opts = VideoOptions(keyframes_fn=walk.keyframes_fn, sfm_fn=walk.sfm_fn, depth_backend=FunctionDepthBackend(boom, "m", True))
    r = process_capture(video, tmp_path / "out", opts, stages_fn=run_video_stages, tier="video")
    assert r.status == "partial" and r.property["walls"] == [] and r.run_report["metric_scale_available"] is False
    stage = next(s for s in r.run_report["stages"] if s["name"] == "metric_depth_and_scale")
    assert stage["status"] == "warning" and any("inference failed" in w for w in stage["warnings"])


def test_a_failed_denser_retry_keeps_the_first_sfm_result(video, tmp_path):
    walk = SyntheticWalkthrough()
    calls = []

    def kfn(v, d, o):
        calls.append(d)
        if len(calls) > 1:
            raise RuntimeError("decode failed on retry")
        return walk.keyframes_fn(v, d, o)

    base = walk.sfm_fn

    def weak(kdir, work, opts):
        run = base(kdir, work, opts)
        run.result.tracking_quality = "weak"
        return run

    opts = VideoOptions(keyframes_fn=kfn, sfm_fn=weak, depth_backend=walk.depth_backend(), fusion=FusionOptions(pixel_stride=1))
    r = process_capture(video, tmp_path / "out", opts, stages_fn=run_video_stages, tier="video")
    assert len(calls) == 2 and r.status == "partial" and r.exit_code == 0
    sfm_stage = next(s for s in r.run_report["stages"] if s["name"] == "sfm_reconstruction")
    assert any("retry failed" in w and "kept the first attempt" in w for w in sfm_stage["warnings"])
    assert r.run_report["metric_scale_available"] is True
    assert not (r.output_dir / "diagnostics" / "keyframes_dense").exists()


def test_invalid_video_input_fails_before_any_heavy_work(tmp_path):
    r = process_capture(tmp_path / "missing.mp4", tmp_path / "out", VideoOptions(), stages_fn=run_video_stages, tier="video")
    assert r.status == "failure" and r.run_report["errors"][0]["stage"] == "video_validation"
    assert "does not exist" in r.run_report["errors"][0]["message"] and r.run_report["metric_scale_available"] is False
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 50)
    assert process_capture(bad, tmp_path / "out2", VideoOptions(), stages_fn=run_video_stages, tier="video").status == "failure"


def test_video_output_is_deterministic(video, tmp_path):
    a, _ = run(video, tmp_path, "a", walk=SyntheticWalkthrough(seed=5))
    b, _ = run(video, tmp_path, "b", walk=SyntheticWalkthrough(seed=5))
    ja, jb = load(a), load(b)
    assert strip_volatile(ja) == strip_volatile(jb)
    assert (a.output_dir / "plan.png").read_bytes() == (b.output_dir / "plan.png").read_bytes()
    assert (a.output_dir / "diagnostics" / "metric_cloud.ply").read_bytes() == (b.output_dir / "diagnostics" / "metric_cloud.ply").read_bytes()
    assert a.run_report["video"]["metric_scale"] == b.run_report["video"]["metric_scale"]


def test_cli_video_tier(video, tmp_path, monkeypatch, capsys):
    walk = SyntheticWalkthrough()
    real = run_video_stages

    def patched(v, opts):
        opts.keyframes_fn, opts.sfm_fn, opts.depth_backend = walk.keyframes_fn, walk.sfm_fn, walk.depth_backend()
        return real(v, opts)

    monkeypatch.setattr("spatialforge.video.pipeline.run_video_stages", patched)
    code = main(["process", str(video), "--tier", "video", "--output", str(tmp_path / "ok")])
    out = capsys.readouterr().out
    assert code == 0 and "tier: video" in out and "metric scale available: True" in out
    assert (tmp_path / "ok" / "property.json").exists()
    assert main(["process", str(tmp_path / "nope.mp4"), "--tier", "video", "--output", str(tmp_path / "bad")]) == 1
    assert "ERROR [video_validation]" in capsys.readouterr().out
    assert main(["process", str(tmp_path / "no_photos"), "--tier", "photos", "--output", str(tmp_path / "p")]) == 1  # photos exist now
