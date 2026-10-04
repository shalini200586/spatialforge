"""The benchmark harness: ground truth, matching, metrics, gates, repeatability, incumbent comparison, fix-loop records, doctor."""

import copy
import json
from pathlib import Path

import pytest

from spatialforge.cli import main
from spatialforge.doctor import format_doctor, run_doctor
from spatialforge.evaluation import incumbent
from spatialforge.evaluation.benchmark import evaluate, evaluate_repeat, room_overlap_count
from spatialforge.evaluation.fix_loop import compare_runs, freeze_baseline, render_comparison
from spatialforge.evaluation.metrics import abs_error, aggregate, covered, interval_stats, pct_error, spread
from spatialforge.evaluation.models import (FAIL, NOT_EVALUATED, NOT_PHYSICAL_BANNER, PASS, GateConfig, load_gate_config, load_ground_truth,
                                            parse_ground_truth, validate_ground_truth)

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"


def m(value, half=None, unit="m"):
    return {"value": value, "unit": unit, "interval": None if half is None else {"low": value - half, "high": value + half},
            "confidence": None, "quality": "moderate", "source_tier": "lidar"}


def square(x0, z0, side):
    return [{"x": x0, "z": z0}, {"x": x0 + side, "z": z0}, {"x": x0 + side, "z": z0 + side}, {"x": x0, "z": z0 + side}]


def room(rid, area, walls, ceiling=None, polygon=None, adjacent=(), connected=()):
    return {"id": rid, "area": m(area, 0.5, "m2"), "wall_ids": [f"{rid}_w{i}" for i in range(len(walls))], "wall_lengths": [m(w, 0.05) for w in walls],
            "ceiling": {"observed": ceiling is not None, "height": m(ceiling, 0.01) if ceiling is not None else None}, "polygon": polygon or [],
            "adjacent_room_ids": list(adjacent), "connected_room_ids": list(connected)}


def prop(tier="lidar", **over):
    p = {"capture": {"tier": tier}, "property": {"footprint": {"area": m(41.2, 1.0, "m2")}},
         "rooms": [room("room_001", 20.1, [5.02, 3.98, 5.01, 4.03], 2.452, square(0, 0, 4.5), connected=["room_003"]),
                   room("room_002", 12.1, [4.01, 3.0, 3.99, 3.02], 2.46, square(5, 0, 3.5), connected=["room_003"]),
                   room("room_003", 8.8, [], 2.45, square(0, 5, 3), adjacent=["room_001", "room_002"])],
         "openings": [{"id": "opening_001", "type": "door", "width": m(0.91, 0.05), "room_ids": ["room_001"]},
                      {"id": "opening_002", "type": "door", "width": m(0.86, 0.05), "room_ids": ["room_002"]},
                      {"id": "opening_003", "type": "opening", "width": m(2.5, 0.05), "room_ids": []}],  # nobody measured this one: a phantom
         "unverified_openings": [{"id": "u1"}]}
    p.update(over)
    return p


@pytest.fixture
def gt():
    g = load_ground_truth(BENCH / "example_ground_truth.json")
    return g


def gate(res, name):
    return next(g for g in res["gates"] if g["gate"] == name)


# ---------------- ground truth ----------------


def test_the_example_ground_truth_and_gate_file_are_valid():
    g = load_ground_truth(BENCH / "example_ground_truth.json")
    assert g.property_id.startswith("EXAMPLE") and g.physical_measurement is False and len(g.rooms) == 3 and len(g.openings) == 3
    assert load_gate_config(BENCH / "gates.json") == GateConfig()  # the file and the defaults are the same numbers
    json.loads((BENCH / "ground_truth.schema.json").read_text())


def test_invalid_ground_truth_is_refused_with_reasons():
    raw = json.loads((BENCH / "example_ground_truth.json").read_text())
    bad = copy.deepcopy(raw)
    del bad["rooms"][0]["id"]
    assert any("missing required key 'id'" in p for p in validate_ground_truth(bad))
    bad = copy.deepcopy(raw)
    bad["adjacency"] = [["living_room", "attic"]]
    assert any("unknown room 'attic'" in p for p in validate_ground_truth(bad))
    bad = copy.deepcopy(raw)
    bad["openings"][0]["room_id"] = "attic"
    assert any("unknown room" in p for p in validate_ground_truth(bad))
    bad = copy.deepcopy(raw)
    bad["rooms"].append(copy.deepcopy(bad["rooms"][0]))
    assert any("duplicate" in p for p in validate_ground_truth(bad))
    with pytest.raises(ValueError, match="invalid ground truth"):
        parse_ground_truth({"schema": "wrong", "property_id": "x", "rooms": []})
    with pytest.raises(ValueError, match="unknown gate setting"):
        p = Path(__file__).parent / "_bad_gates.json"
        try:
            p.write_text('{"opening_width_abs": 0.02}')
            load_gate_config(p)
        finally:
            p.unlink()


# ---------------- metrics ----------------


def test_metric_helpers():
    assert abs_error(5.02, 5.0) == pytest.approx(0.02) and pct_error(5.02, 5.0) == pytest.approx(0.4) and pct_error(1, 0) is None
    assert covered(5.0, m(5.02, 0.05)) is True and covered(5.0, m(5.2, 0.05)) is False and covered(5.0, m(5.0)) is None
    assert spread([2.45, 2.46, 2.455]) == pytest.approx(0.01) and spread([1.0]) is None
    assert aggregate([1, 2, 6], "mean") == 3 and aggregate([1, 2, 6], "median") == 2 and aggregate([1, 2, 6], "max") == 6 and aggregate([], "mean") is None
    with pytest.raises(ValueError):
        aggregate([1], "mode")
    st = interval_stats([(5.0, m(5.02, 0.05)), (4.0, m(4.5, 0.1)), (3.0, m(3.0))])
    assert st["measurements"] == 3 and st["with_interval"] == 2 and st["without_interval"] == 1 and st["coverage_rate"] == 0.5
    assert interval_stats([])["coverage_rate"] is None


# ---------------- evaluate ----------------


def test_evaluate_matches_rooms_walls_and_openings_and_reports_errors(gt):
    res = evaluate(prop(), gt)
    assert res["banner"] == NOT_PHYSICAL_BANNER  # the example is not a physical measurement
    assert [(x["gt"], x["pred"], x["matched_by"]) for x in res["rooms"]["matched"]] == [
        ("living_room", "room_001", "geometry"), ("bedroom", "room_002", "geometry"), ("hallway", "room_003", "geometry")]
    walls = {(i["room"], i["gt_wall"]): i for i in res["walls"]["items"]}
    assert len(walls) == 8 and walls[("living_room", "north")]["abs_error_m"] <= 0.02 and res["walls"]["abs_error_m"]["max"] <= 0.03
    assert walls[("living_room", "north")]["covered"] is True
    o = res["openings"]
    assert o["counts"] == {"ground_truth": 3, "matched": 2, "missed": 1, "phantom": 1} and o["missed"] == ["window_living"] and o["phantom"] == ["opening_003"]
    assert o["unverified_not_counted"] == 1 and o["within_target"]["of"] == 2
    assert res["ceiling"]["abs_error_m"]["max"] == pytest.approx(0.01, abs=1e-3)
    assert res["property"]["footprint"]["pct_error"] == pytest.approx(0.488, abs=1e-2) and res["property"]["room_overlap_count"] == 0
    adj = res["property"]["adjacency"]
    assert adj["recall"] == 1.0 and adj["extra"] == []
    assert res["confidence"]["with_interval"] > 10 and res["confidence"]["coverage_rate"] is not None
    json.dumps(res)  # serialisable: no internal fields leak
    assert all("_m" not in i for i in res["walls"]["items"])


def test_gates_pass_fail_and_not_evaluated_are_distinct(gt):
    res = evaluate(prop(), gt)
    ow = gate(res, "opening_width")  # 0.91 vs 0.90 (1 cm) and 0.86 vs 0.80 (6 cm): one of two within 2 cm
    assert ow["status"] == FAIL and ow["value"] == 0.5
    assert gate(res, "ceiling_height")["status"] == PASS  # 2.452 / 2.46 / 2.45 against 2.45: worst 1 cm <= 1.5 cm
    for name in ("photo_wall_length", "video_wall_length", "photo_footprint", "photo_stitching"):
        assert gate(res, name)["status"] == NOT_EVALUATED  # not this tier
    assert gate(res, "opening_detection")["status"] == NOT_EVALUATED  # no threshold was supplied: numbers only
    assert gate(res, "confidence_calibration")["status"] == NOT_EVALUATED and gate(res, "confidence_calibration")["value"] is not None
    assert gate(res, "repeat_wall")["status"] == NOT_EVALUATED and gate(res, "repeat_ceiling")["status"] == NOT_EVALUATED
    assert gate(res, "drift_accountability")["status"] == NOT_EVALUATED  # no run report
    tight = evaluate(prop(), gt, GateConfig(ceiling_abs_m=0.005, opening_detection_min_recall=0.9, opening_detection_max_phantoms=0,
                                            confidence_min_coverage=0.5))
    assert gate(tight, "ceiling_height")["status"] == FAIL and gate(tight, "opening_detection")["status"] == FAIL
    assert gate(tight, "confidence_calibration")["status"] in (PASS, FAIL)


def test_missing_data_is_never_a_pass(gt):
    # ground truth that exists but has no prediction is a FAIL, not a quiet skip
    no_ceiling = prop()
    for r in no_ceiling["rooms"]:
        r["ceiling"] = {"observed": False, "height": None}
    res = evaluate(no_ceiling, gt)
    assert gate(res, "ceiling_height")["status"] == FAIL and "failing or missing" in gate(res, "ceiling_height")["detail"]
    no_openings = evaluate(prop(openings=[]), gt)
    assert gate(no_openings, "opening_width")["status"] == FAIL and no_openings["openings"]["counts"]["missed"] == 3
    # no ground truth for a quantity: NOT EVALUATED
    bare = parse_ground_truth({"schema": "spatialforge-ground-truth/1", "property_id": "p", "physical_measurement": True,
                               "rooms": [{"id": "a", "floor_area_m2": 20.0}]})
    res = evaluate(prop(), bare)
    assert res["banner"] is None
    for name in ("opening_width", "opening_detection", "ceiling_height"):
        assert gate(res, name)["status"] == NOT_EVALUATED, name
    nothing = evaluate({"rooms": [], "openings": [], "capture": {"tier": "photos"}}, gt)
    assert gate(nothing, "ceiling_height")["status"] == FAIL and gate(nothing, "photo_footprint")["status"] == FAIL
    assert gate(nothing, "photo_wall_length")["status"] == FAIL and gate(nothing, "photo_stitching")["status"] == FAIL
    assert not any(g["status"] == PASS for g in nothing["gates"])


def test_explicit_ids_win_over_geometry_and_unmatched_stay_unmatched():
    raw = {"schema": "spatialforge-ground-truth/1", "property_id": "p", "physical_measurement": True,
           "rooms": [{"id": "kitchen", "predicted_id": "room_002", "floor_area_m2": 12.0, "walls": [{"id": "n", "predicted_id": "room_002_w1", "length_m": 3.0}]},
                     {"id": "attic", "floor_area_m2": 60.0}],  # nothing in the prediction is anywhere near 60 m2
           "openings": [{"id": "d", "predicted_id": "opening_002", "width_m": 0.80}]}
    res = evaluate(prop(), parse_ground_truth(raw))
    assert (res["rooms"]["matched"][0]["pred"], res["rooms"]["matched"][0]["matched_by"]) == ("room_002", "id")
    assert res["rooms"]["matched"][1]["pred"] is None and res["rooms"]["unmatched_gt"] == ["attic"]
    assert res["walls"]["items"][0]["pred_wall"] == "room_002_w1" and res["walls"]["items"][0]["matched_by"] == "id"
    assert res["openings"]["matched"][0]["pred_opening"] == "opening_002" and res["openings"]["matched"][0]["matched_by"] == "id"
    assert res["rooms"]["unmatched_pred"] == ["room_001", "room_003"]


def test_photo_and_video_wall_gates_use_their_own_thresholds(gt):
    # every predicted wall 5% long: fails the video 3% gate, passes the photo 8% gate
    p = prop("photos")
    for r in p["rooms"]:
        r["wall_lengths"] = [m(w["value"] * 1.05, 0.5) for w in r["wall_lengths"]]
    photos = evaluate(p, gt)
    assert gate(photos, "photo_wall_length")["status"] == PASS and gate(photos, "video_wall_length")["status"] == NOT_EVALUATED
    pv = copy.deepcopy(p)
    pv["capture"]["tier"] = "video"
    video = evaluate(pv, gt)
    assert gate(video, "video_wall_length")["status"] == FAIL and gate(video, "photo_wall_length")["status"] == NOT_EVALUATED
    assert gate(evaluate(pv, gt, GateConfig(video_wall_pct=6.0)), "video_wall_length")["status"] == PASS  # thresholds are configuration
    assert gate(evaluate(pv, gt, tier="photos"), "photo_wall_length")["status"] == PASS  # --tier overrides the recorded tier


def test_photo_footprint_and_stitching_gates(gt):
    ok = evaluate(prop("photos"), gt)
    assert gate(ok, "photo_footprint")["status"] == PASS and gate(ok, "photo_stitching")["status"] == PASS
    off = prop("photos")
    off["property"]["footprint"]["area"] = m(48.0, 1.0, "m2")  # 17% too large
    assert gate(evaluate(off, gt), "photo_footprint")["status"] == FAIL
    unplaced = prop("photos")
    unplaced["rooms"] = unplaced["rooms"][:2]  # the hallway was not placed
    res = evaluate(unplaced, gt)
    assert gate(res, "photo_stitching")["status"] == FAIL and "hallway" in gate(res, "photo_stitching")["detail"]
    overlapping = prop("photos")
    overlapping["rooms"][1]["polygon"] = square(2, 0, 3.5)  # on top of room_001
    res = evaluate(overlapping, gt)
    assert res["property"]["room_overlap_count"] >= 1 and gate(res, "photo_stitching")["status"] == FAIL


def test_room_overlap_count():
    a, b, c = {"id": "a", "polygon": square(0, 0, 4)}, {"id": "b", "polygon": square(4, 0, 4)}, {"id": "c", "polygon": square(3, 1, 4)}
    assert room_overlap_count([a, b]) == 0  # touching edges are not an overlap
    assert room_overlap_count([a, c]) == 1 and room_overlap_count([a, b, c]) == 2 and room_overlap_count([{"id": "x", "polygon": []}, a]) == 0


def test_drift_accountability_gate_needs_a_recorded_pose_decision(gt):
    rep = {"status": "partial", "production_pose_source": "original", "fallback_decisions": [{"stage": "pose_refinement", "decision": "kept the supplied poses"}],
           "total_duration_seconds": 12.5, "stages": [{"stage": "frontend", "duration_seconds": 10.0, "status": "ok"}, {"stage": "stitching", "duration_seconds": 0.0, "status": "skipped"}]}
    res = evaluate(prop(), gt, run_report=rep)
    assert gate(res, "drift_accountability")["status"] == PASS and res["status"] == "partial"
    assert res["runtime"] == {"total_duration_seconds": 12.5, "stages": {"frontend": 10.0}} and res["metrics"]["runtime.stage.frontend_s"] == 10.0
    rep["fallback_decisions"] = []
    assert gate(evaluate(prop(), gt, run_report=rep), "drift_accountability")["status"] == FAIL


# ---------------- repeatability ----------------


def test_repeatability_gates():
    a, b = prop(), prop()
    b["rooms"][0]["wall_lengths"][0] = m(5.025, 0.05)  # 5 mm
    b["rooms"][1]["ceiling"]["height"] = m(2.465, 0.01)  # 5 mm
    res = evaluate_repeat(a, b)
    assert res["rooms_paired"] == 3 and {g["gate"]: g["status"] for g in res["gates"]} == {"repeat_wall": PASS, "repeat_ceiling": PASS}
    assert res["summary"]["wall_diff_m"]["max"] == pytest.approx(0.005, abs=1e-3)
    b["rooms"][0]["wall_lengths"][1] = m(4.10, 0.05)  # 12 cm and 3%: outside both 1 cm and 0.5%
    b["rooms"][2]["ceiling"]["height"] = m(2.50, 0.01)
    res = evaluate_repeat(a, b)
    assert {g["gate"]: g["status"] for g in res["gates"]} == {"repeat_wall": FAIL, "repeat_ceiling": FAIL}
    assert evaluate_repeat(a, b, GateConfig(repeat_wall_pct=5.0, repeat_ceiling_spread_m=0.1))["gates"][0]["status"] == PASS  # 0.5% is configuration
    empty = evaluate_repeat({"rooms": []}, a)
    assert [g["status"] for g in empty["gates"]] == [NOT_EVALUATED, NOT_EVALUATED]


# ---------------- incumbent ----------------


def test_incumbent_placeholders_are_pending_not_wins():
    rows = incumbent.read_rows(BENCH / "incumbent_template.csv")
    assert len(rows) >= 8 and {r["room_id"] for r in rows} == {"room_A", "room_B"}
    res = incumbent.compare(rows)
    assert res["counts"]["compared"] == 0 and res["pending_rows"] == len(rows) and res["beat_or_tie_pct"] is None
    assert "placeholders only" in res["notes"][0] and "Incumbent comparison" in incumbent.render_markdown(res)


def test_incumbent_winner_tie_and_beat_or_tie_percentage(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("room_id,dimension_id,ground_truth_m,spatialforge_m,incumbent_m,notes\n"
                   "A,w1,4.000,4.010,4.100,app v?\n"       # SpatialForge better
                   "A,w2,3.000,3.050,3.002,\n"             # incumbent better
                   "B,w1,5.000,5.020,5.021,\n"             # within 5 mm of each other: tie
                   "B,w2,2.500,,2.51,not exported yet\n",  # pending
                   encoding="utf-8")
    res = incumbent.compare(incumbent.read_rows(csv), 0.005)
    assert [d["winner"] for d in res["rows"]] == ["spatialforge", "incumbent", "tie"] and res["pending_rows"] == 1
    assert res["beat_or_tie_pct"] == pytest.approx(66.7, abs=0.1) and res["counts"] == {"compared": 3, "spatialforge_wins": 1, "ties": 1, "incumbent_wins": 1}
    assert res["rooms"] == ["A", "B"] and res["notes"] == []
    one_room = incumbent.compare(incumbent.read_rows(csv)[:2])
    assert "1 room(s)" in one_room["notes"][0]
    (tmp_path / "bad.csv").write_text("room_id,dimension_id\nA,w\n")
    with pytest.raises(ValueError, match="missing column"):
        incumbent.read_rows(tmp_path / "bad.csv")
    (tmp_path / "bad2.csv").write_text("room_id,dimension_id,ground_truth_m,spatialforge_m,incumbent_m,notes\nA,w,abc,1,1,\n")
    with pytest.raises(ValueError, match="line 2"):
        incumbent.read_rows(tmp_path / "bad2.csv")


# ---------------- fix loop ----------------


def make_run(folder: Path, p: dict):
    folder.mkdir(parents=True)
    (folder / "property.json").write_text(json.dumps(p))
    (folder / "run_report.json").write_text(json.dumps({"status": "partial", "total_duration_seconds": 5.0, "stages": [], "fallback_decisions": [],
                                                       "config": {"tier": "lidar"}, "input": {"path": str(Path.home() / "captures" / "room")},
                                                       "spatialforge_version": "0.1.0"}))
    return folder


def test_freeze_and_compare_a_fix(tmp_path):
    gt_path = BENCH / "example_ground_truth.json"
    before_run = make_run(tmp_path / "run_before", prop())
    worse = prop()
    for r in worse["rooms"]:
        r["wall_lengths"] = [m(w["value"] * 1.04, 0.5) for w in r["wall_lengths"]]
    after_run = make_run(tmp_path / "run_after", worse)
    root = tmp_path / "evaluation_runs"
    b = freeze_baseline(before_run, gt_path, "room1", "before", root, command="python -m spatialforge process <capture> --tier lidar --output <out>")
    a = freeze_baseline(after_run, gt_path, "room1", "after", root, command="python -m spatialforge process <capture> --tier lidar --output <out>")
    man = json.loads((b / "manifest.json").read_text())
    assert {"git_sha", "capture", "tier", "command", "ground_truth_file", "config", "property_output", "metrics", "timestamp"} <= set(man)
    assert man["ground_truth_file"] == "example_ground_truth.json" and man["tier"] == "lidar" and man["ground_truth_physical_measurement"] is False
    assert str(Path.home()) not in (b / "manifest.json").read_text() and man["capture"].startswith("~")  # no personal path
    assert (b / "property.json").exists() and (b / "evaluation.json").exists() and (b / "evaluation.md").exists()
    with pytest.raises(FileExistsError):
        freeze_baseline(before_run, gt_path, "room1", "before", root)  # a frozen baseline is never overwritten
    with pytest.raises(FileNotFoundError):
        freeze_baseline(tmp_path / "nothing", gt_path, "x", "before", root)
    with pytest.raises(ValueError):
        freeze_baseline(before_run, gt_path, "x", "middle", root)
    cmp = compare_runs(b, a)
    row = next(r for r in cmp["metrics"] if r["metric"] == "walls.mean_pct_error")
    assert row["after"] > row["before"] and row["abs_change"] == pytest.approx(row["after"] - row["before"], abs=1e-4) and row["pct_change"] > 0
    assert cmp["same_ground_truth"] and cmp["banner"] == NOT_PHYSICAL_BANNER and {g["gate"] for g in cmp["gates"]} >= {"opening_width", "ceiling_height"}
    text = render_comparison(cmp)
    assert "Before / after" in text and NOT_PHYSICAL_BANNER in text and "walls.mean_pct_error" in text


# ---------------- CLI ----------------


def test_cli_evaluate_writes_reports_and_strict_exit(tmp_path, capsys):
    run = make_run(tmp_path / "run", prop())
    out = tmp_path / "eval"
    code = main(["evaluate", "--property", str(run), "--ground-truth", str(BENCH / "example_ground_truth.json"), "--output", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and (out / "evaluation.json").exists() and (out / "evaluation.md").exists()
    assert NOT_PHYSICAL_BANNER in text and "## Gates" in text and "opening_width" in text
    assert json.loads((out / "evaluation.json").read_text())["banner"] == NOT_PHYSICAL_BANNER
    assert main(["evaluate", "--property", str(run), "--ground-truth", str(BENCH / "example_ground_truth.json"), "--strict"]) == 1  # opening_width FAILS
    capsys.readouterr()
    other = make_run(tmp_path / "run2", prop())
    assert main(["evaluate", "--property", str(run), "--ground-truth", str(BENCH / "example_ground_truth.json"), "--repeat", str(other), "--output", str(tmp_path / "e2")]) == 0
    rep_text = capsys.readouterr().out
    assert "Repeatability" in rep_text and (tmp_path / "e2" / "repeatability.json").exists()
    assert gate(json.loads((tmp_path / "e2" / "evaluation.json").read_text()), "repeat_wall")["status"] == PASS  # the headline table carries it


def test_cli_evaluate_errors_are_clear(tmp_path, capsys):
    assert main(["evaluate", "--property", str(tmp_path / "missing"), "--ground-truth", str(BENCH / "example_ground_truth.json")]) == 1
    assert "ERROR [EVALUATION_INPUT]" in capsys.readouterr().err
    (tmp_path / "gt.json").write_text('{"schema": "x"}')
    assert main(["evaluate", "--property", str(make_run(tmp_path / "r", prop())), "--ground-truth", str(tmp_path / "gt.json")]) == 1
    assert "invalid ground truth" in capsys.readouterr().err


def test_cli_compare_incumbent(tmp_path, capsys):
    assert main(["compare-incumbent", "--csv", str(BENCH / "incumbent_template.csv"), "--output", str(tmp_path)]) == 0
    assert "placeholder" in capsys.readouterr().out and (tmp_path / "incumbent_comparison.json").exists()
    assert main(["compare-incumbent", "--csv", str(tmp_path / "nope.csv")]) == 1
    assert "ERROR [INCUMBENT_INPUT]" in capsys.readouterr().err


# ---------------- doctor ----------------


def test_doctor_reports_versions_models_and_tiers(monkeypatch, capsys, tmp_path):
    import spatialforge.config as cfgmod

    monkeypatch.setattr(cfgmod, "depth_weights_present", lambda: False)
    monkeypatch.setenv(cfgmod.MODELS_ENV, str(tmp_path / "models"))
    d = run_doctor()
    assert d["spatialforge_version"] == "0.1.0" and set(d["tiers"]) == {"LiDAR", "Video", "Photos", "Damage"} and d["packages"]["numpy"]["present"]
    assert d["tiers"]["Video"]["ready"] is False and any("download_models" in r for r in d["tiers"]["Video"]["reasons"])
    assert d["models"]["from_environment"] is True and d["models"]["depth_weights_present"] is False
    text = format_doctor(d)
    assert "Tiers:" in text and "NOT READY" in text and "python scripts/download_models.py" in text
    monkeypatch.setattr(cfgmod, "depth_weights_present", lambda: True)
    assert run_doctor()["tiers"]["Photos"]["ready"] == all(run_doctor()["packages"][p]["present"] for p in ("numpy", "PIL", "cv2", "pycolmap", "torch", "transformers"))
    assert main(["doctor"]) in (0, 1) and "SpatialForge 0.1.0" in capsys.readouterr().out


def test_doctor_needs_no_capture_and_marks_missing_packages(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: None if n == "pycolmap" else real(n, *a, **k))
    d = run_doctor()
    assert d["tiers"]["Video"]["ready"] is False and "pycolmap" in d["tiers"]["Video"]["reasons"][0] and d["tiers"]["LiDAR"]["ready"] is True
