"""Pipeline orchestration and packaging with synthetic stage outputs (real room/opening modules, no real captures)."""

import json
from pathlib import Path

import pytest
from helpers_pipeline import synthetic_outputs

from spatialforge.cli import main
from spatialforge.pipeline import lidar_pipeline
from spatialforge.pipeline.lidar_adapter import assemble_property
from spatialforge.pipeline.lidar_pipeline import PipelineFailure, process_capture
from spatialforge.pipeline.schema import build_schema, validate_against_schema
from spatialforge.pipeline.serialization import property_to_dict, strip_volatile


def run(tmp_path, name="out", **kw):
    so = synthetic_outputs(**kw)
    return process_capture("C:/data/cap", tmp_path / name, stages_fn=lambda c, o: so)


def load(result, file="property.json"):
    return json.loads((result.output_dir / file).read_text())


def test_complete_synthetic_property(tmp_path):
    r = run(tmp_path)
    d = r.property
    assert r.exit_code == 0 and r.status in ("success", "partial")
    assert len(d["rooms"]) == 2 and len(d["openings"]) == 1
    assert validate_against_schema(d, build_schema()) == []
    room_ids = {x["id"] for x in d["rooms"]}
    op = d["openings"][0]
    assert op["type"] == "door" and set(op["room_ids"]) == room_ids and set(op["connected_room_ids"]) == room_ids
    assert all(x["connected_room_ids"] for x in d["rooms"])  # connectivity propagated, separate from adjacency
    assert all(x["adjacent_room_ids"] for x in d["rooms"])
    assert d["damage"] == [] and d["concealed_damage_flags"] == [] and d["scope_line_items"] == []
    room = d["rooms"][0]
    assert room["area"]["interval"]["low"] <= room["area"]["value"] <= room["area"]["interval"]["high"]  # intervals kept
    assert op["width"]["interval"] is not None and op["width"]["quality"] in ("strong", "moderate", "weak")


def test_partial_property_with_no_rooms_is_reported_honestly(tmp_path):
    r = run(tmp_path, rooms=False, ceiling=False)
    d = r.property
    assert r.status == "partial" and d["property"]["status"] == "partial"
    assert d["rooms"] == [] and len(d["walls"]) >= 1
    assert any("No closed rooms were recovered" in w for w in d["warnings"])
    assert any("Opening recall has not been verified" in w for w in d["warnings"])
    assert any("No physical ground truth" in w for w in d["warnings"])
    assert d["property"]["footprint"] is None or d["property"]["footprint"]["complete"] is False


def test_property_without_ceiling_reports_unobserved_ceilings(tmp_path):
    d = run(tmp_path, ceiling=False).property
    assert all(not r["ceiling"]["observed"] and r["ceiling"]["height"] is None for r in d["rooms"])
    assert any("Ceiling height was not observed" in w for w in d["warnings"])
    with_ceiling = run(tmp_path, "withc", ceiling=True).property
    assert any(r["ceiling"]["observed"] and r["ceiling"]["height"]["value"] == pytest.approx(2.4) for r in with_ceiling["rooms"])
    assert any(r["ceiling"]["height"]["confidence"] == pytest.approx(0.6) for r in with_ceiling["rooms"] if r["ceiling"]["observed"])


def test_property_without_openings(tmp_path):
    d = run(tmp_path, openings=False).property
    assert d["openings"] == [] and len(d["rooms"]) == 2
    assert all(r["connected_room_ids"] == [] for r in d["rooms"])  # adjacent, but no verified opening connects them
    assert all(r["adjacent_room_ids"] for r in d["rooms"])


def test_stage_warnings_propagate_to_property_and_run_report(tmp_path):
    r = run(tmp_path, stage_warnings=["2 separate ceiling levels found (3.02 m, 2.41 m)"])
    assert "2 separate ceiling levels found (3.02 m, 2.41 m)" in r.property["warnings"]
    assert "2 separate ceiling levels found (3.02 m, 2.41 m)" in load(r, "run_report.json")["warnings"]


def test_fatal_validation_error_fails_cleanly(tmp_path):
    def boom(capture, options):
        raise PipelineFailure("capture_validation", "depth/ folder is missing (required)")

    r = process_capture("C:/data/bad", tmp_path / "out", stages_fn=boom)
    assert r.status == "failure" and r.exit_code == 1 and r.property is None
    report = load(r, "run_report.json")
    assert report["status"] == "failure" and report["errors"][0]["stage"] == "capture_validation"
    assert not (tmp_path / "out" / "property.json").exists() and not (tmp_path / "out" / "plan.png").exists()


def test_real_capture_validation_failure_on_an_empty_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    r = process_capture(tmp_path / "empty", tmp_path / "out")
    assert r.status == "failure" and r.exit_code == 1
    assert any("depth" in e["message"] for e in r.run_report["errors"])


def test_unexpected_exception_is_a_clean_failure(tmp_path):
    def crash(capture, options):
        raise RuntimeError("something unforeseen")

    r = process_capture("C:/data/x", tmp_path / "out", stages_fn=crash)
    assert r.status == "failure" and "something unforeseen" in r.run_report["errors"][0]["message"]


def test_canonical_output_files_are_created(tmp_path):
    r = run(tmp_path)
    for name in ("property.json", "plan.png", "run_report.json"):
        assert (r.output_dir / name).stat().st_size > 0, name
    assert (r.output_dir / "diagnostics").is_dir()
    report = load(r, "run_report.json")
    assert report["status"] == r.status and report["tier"] == "lidar"
    assert report["counts"]["rooms"] == 2 and report["counts"]["openings"] == 1
    assert {s["name"] for s in report["stages"]} >= {"capture_validation", "openings"}
    assert report["production_pose_source"] == "original" and report["fallback_decisions"][0]["decision"].startswith("kept")
    assert report["plan"]["rooms_drawn"] == 2 and report["plan"]["openings_drawn"] == 1
    assert report["runtime_s"] >= 0 and "started_at" in report and "ended_at" in report


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", lambda c, o: synthetic_outputs())
    code = main(["process", "C:/data/cap", "--tier", "lidar", "--output", str(tmp_path / "ok")])
    out = capsys.readouterr().out
    assert code == 0 and "Status:" in out and "Rooms 2" in out and (tmp_path / "ok" / "property.json").exists()

    def fail(c, o):
        raise PipelineFailure("capture_validation", "not a capture")

    monkeypatch.setattr(lidar_pipeline, "run_lidar_stages", fail)
    assert main(["process", "C:/data/bad", "--tier", "lidar", "--output", str(tmp_path / "bad")]) == 1
    assert "ERROR [capture_validation]" in capsys.readouterr().out
    assert main(["process", "C:/data/cap", "--tier", "video", "--output", str(tmp_path / "v")]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_repeated_run_gives_the_same_json_and_plan(tmp_path):
    a, b = run(tmp_path, "a"), run(tmp_path, "b")
    ja, jb = load(a), load(b)
    assert strip_volatile(ja) == strip_volatile(jb)  # only the explicit timing section may differ
    assert (a.output_dir / "plan.png").read_bytes() == (b.output_dir / "plan.png").read_bytes()
    assert "timing" in ja and "stages" in ja["timing"]


def test_low_confidence_openings_are_kept_out_of_the_plan_but_listed(tmp_path):
    so = synthetic_outputs()
    so.openings.openings[0].status = "low_confidence"  # demote the only opening
    r = process_capture("C:/data/cap", tmp_path / "out", stages_fn=lambda c, o: so)
    d = r.property
    assert d["openings"] == [] and len(d["unverified_openings"]) == 1
    assert d["unverified_openings"][0]["id"].startswith("unverified_")
    assert any("low-confidence" in w for w in d["warnings"])
    assert r.run_report["plan"]["openings_drawn"] == 0


def test_a_failed_nonessential_stage_gives_partial_not_failure(tmp_path):
    so = synthetic_outputs()
    so.openings = None
    so.stages[-1].status, so.stages[-1].error = "failed", "RuntimeError: boom"
    r = process_capture("C:/data/cap", tmp_path / "out", stages_fn=lambda c, o: so)
    assert r.status == "partial" and r.exit_code == 0 and r.property["openings"] == []
    assert any("failed and were skipped" in w for w in r.property["warnings"])
    assert any(s["status"] == "failed" for s in r.run_report["stages"])


def test_assembly_never_invents_values(tmp_path):
    prop = assemble_property(synthetic_outputs())
    d = property_to_dict(prop)
    for w in d["walls"]:
        assert w["length"]["interval"] is None  # the wall stage gives no length interval, so none is invented
        assert w["length"]["confidence"] is None
    assert all(r["area"]["confidence"] is None for r in d["rooms"])  # only ceiling heights carry a stage score
    assert d["capture"]["device"]["model"] is None
