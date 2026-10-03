"""Photo tier end to end with a mocked SfM backend and a mocked metric-depth model (ray-cast apartment, no weights)."""

import json

import numpy as np
import pytest

pytest.importorskip("cv2")  # the photo tier is an optional install (pip install -e .[video])

from helpers_pipeline import simple_opening, simple_wall, square_room  # noqa: E402
from helpers_photo import SyntheticPhotoSfm, synthetic_property_dir  # noqa: E402

from spatialforge.cli import main  # noqa: E402
from spatialforge.photos.merge import PlacedRoom, dedupe_openings, dedupe_walls, scale_offsets, transform_room, union_footprint  # noqa: E402
from spatialforge.photos.pipeline import PhotoOptions, run_photo_stages  # noqa: E402
from spatialforge.photos import room as room_mod  # noqa: E402
from spatialforge.photos.sfm import ImageRecord, PhotoSfmOptions  # noqa: E402
from spatialforge.photos.validator import validate_photo_dir  # noqa: E402
from spatialforge.pipeline.lidar_pipeline import process_capture  # noqa: E402
from spatialforge.pipeline.scene import MetricScene  # noqa: E402
from spatialforge.pipeline.schema import build_schema, validate_against_schema  # noqa: E402
from spatialforge.pipeline.serialization import strip_volatile  # noqa: E402
from spatialforge.video.fusion import FusionOptions  # noqa: E402
from spatialforge.video.sfm import SfmResult  # noqa: E402

TRUE_AREA = {"room_001": 20.0, "room_002": 16.0, "room_003": 15.0, "room_004": 12.0}


def opts_for(sfm):
    o = PhotoOptions(sfm_backend=sfm, depth_backend=sfm.depth_backend())
    o.room.fusion = FusionOptions(pixel_stride=1, min_views=1)  # the synthetic depth maps are small
    return o


def run(tmp, name="out", labels=("a", "b", "c"), sfm=None, **kw):
    root = synthetic_property_dir(tmp / f"in_{name}" / "photos", list(labels))
    sfm = sfm or SyntheticPhotoSfm()
    r = process_capture(root, tmp / name, opts_for(sfm), stages_fn=run_photo_stages, tier="photos")
    return r, sfm


def all_values(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from all_values(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_values(v, key)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("photo_base")
    r, sfm = run(tmp)
    return r, sfm, tmp


# ---------------- the product run ----------------


def test_three_room_property_is_stitched_into_one(base):
    r, sfm, tmp = base
    d = r.property
    assert r.exit_code == 0 and r.status in ("success", "partial")
    assert [x["id"] for x in d["rooms"]] == ["room_001", "room_002", "room_003"]  # deterministic ids, not folder semantics
    assert validate_against_schema(d, build_schema()) == []  # the same canonical schema as LiDAR and video
    for x in d["rooms"]:
        truth = TRUE_AREA[x["id"]]
        assert 0.7 * truth < x["area"]["value"] < 1.3 * truth, (x["id"], x["area"]["value"])  # metric, not SfM units
        iv = x["area"]["interval"]
        assert iv["low"] < x["area"]["value"] < iv["high"]
    ph = d["provenance"]["photos"]
    assert ph["placed_rooms"] == ["room_001", "room_002", "room_003"] and ph["unplaced_rooms"] == []
    for f in ("property.json", "plan.png", "run_report.json"):
        assert (r.output_dir / f).stat().st_size > 0, f


def test_source_tier_is_photos_everywhere(base):
    r, _, _ = base
    d = r.property
    assert d["capture"]["tier"] == "photos" and r.run_report["tier"] == "photos"
    assert set(all_values(d, "source_tier")) == {"photos"}
    assert d["schema_version"] == "1.0"
    prov = d["provenance"]
    assert prov["pipeline"]["tier"] == "photos" and prov["photos"]["folders"] == 3
    assert prov["photos"]["images_per_folder"] == {"room_001 (a)": 8, "room_002 (b)": 8, "room_003 (c)": 8}
    assert prov["photos"]["metric_depth"]["model"] == "mock-metric-depth" and prov["photos"]["sfm"]["backend"] == "synthetic"
    assert prov["photos"]["source_labels"] == {"room_001": "a", "room_002": "b", "room_003": "c"}
    assert prov["photos"]["metadata_availability"]["with_exif"] == 0
    assert any("less certain than video or LiDAR" in w for w in d["warnings"])


def test_per_room_scale_is_estimated_and_reported(base):
    r, _, _ = base
    rooms = r.run_report["photos"]["rooms"]
    assert len(rooms) == 3
    for rm in rooms:
        s = rm["scale"]
        assert rm["status"] == "reconstructed" and s["metric_scale_available"] and s["frames_used"] == 8
        assert s["scale_quality"] in ("moderate", "weak") and s["sigma_rel"] >= 0.10  # never below the model floor, never "strong"
        assert s["correspondences"] > 1000 and s["frame_spread_rel"] is not None
        assert rm["sfm"]["registered_images"] == 8 and rm["sfm"]["tracking_quality"] in ("strong", "moderate")
        assert rm["gravity"]["quality"] in ("strong", "moderate", "weak") and rm["local_area_m2"] > 5
    cons = r.run_report["photos"]["scale_consistency"]
    assert set(cons["rooms"]) == {"room_001", "room_002", "room_003"} and cons["regularisation_applied"] is False
    assert cons["scale_inconsistent"] is False


def test_stitch_graph_constraints_and_loop_free_chain(base):
    r, _, tmp = base
    cons = r.run_report["photos"]["constraints"]
    kinds = {(c["source_room_id"], c["target_room_id"]): c for c in cons}
    assert ("room_001", "room_002") in kinds and ("room_001", "room_003") in kinds
    c12 = kinds[("room_001", "room_002")]
    assert c12["evidence_type"] in ("visual", "visual+doorway") and c12["verified_match_count"] >= 50 and c12["inlier_ratio"] > 0.25
    assert c12["translation_uncertainty_m"] > 0 and c12["rotation_uncertainty_deg"] > 0 and c12["quality"] in ("moderate", "strong")
    assert all(c["status"] == "active" for c in cons)
    d = tmp / "out" / "diagnostics" / "stitching"
    for f in ("room_graph.json", "stitch_constraints.json", "overlap_report.json", "stitching_initial.png", "stitching_final.png"):
        assert (d / f).stat().st_size > 0, f
    graph = json.loads((d / "room_graph.json").read_text())
    assert len(graph["nodes"]) == 3 and graph["components"][0]["rooms"] == ["room_001", "room_002", "room_003"]
    rooms_dir = tmp / "out" / "diagnostics" / "rooms" / "room_001"
    for f in ("reconstruction.json", "sparse_sfm.ply", "metric_cloud.ply", "local_plan.png"):
        assert (rooms_dir / f).exists(), f
    assert (tmp / "out" / "diagnostics" / "photo_frontend.json").exists()


def test_placed_rooms_do_not_overlap_and_doorways_connect_them(base):
    r, _, _ = base
    ov = r.run_report["photos"]["overlaps"]
    assert len(ov) == 3 and max(o["fraction_of_smaller"] for o in ov) < 0.08  # valid adjacent rooms: no significant overlap
    d = r.property
    rooms = {x["id"]: x for x in d["rooms"]}
    assert "room_002" in rooms["room_001"]["connected_room_ids"] and "room_003" in rooms["room_001"]["connected_room_ids"]
    assert rooms["room_002"]["connected_room_ids"] == ["room_001"] and rooms["room_003"]["connected_room_ids"] == ["room_001"]
    verified = [a for a in d["provenance"]["photos"]["adjacency"] if a["kind"] == "verified_connection"]
    assert len(verified) == 2 and all(a["via_openings"] for a in verified)
    via = {o for a in verified for o in a["via_openings"]}
    ops = {o["id"]: o for o in d["openings"]}
    for oid in via:
        assert len(ops[oid]["connected_room_ids"]) == 2 and ops[oid]["type"] in ("door", "opening")
    assert max(a["quality"] == "strong" for a in verified)  # doorway + visual evidence agree


def test_adjacency_outputs_and_visual_only_adjacency_is_lower_quality(base):
    r, _, _ = base
    adj = {tuple(a["rooms"]): a for a in r.property["provenance"]["photos"]["adjacency"]}
    assert adj[("room_001", "room_002")]["kind"] == "verified_connection"
    # rooms 002 and 003 only touch at a corner: placed by visual evidence, not geometrically adjacent, not connected
    a23 = adj[("room_002", "room_003")]
    assert a23["kind"] == "probable_adjacency_visual_only" and a23["quality"] in ("weak", "moderate") and a23["via_openings"] == []
    rooms = {x["id"]: x for x in r.property["rooms"]}
    assert "room_003" not in rooms["room_002"]["adjacent_room_ids"] and "room_003" not in rooms["room_002"]["connected_room_ids"]
    assert rooms["room_001"]["adjacent_room_ids"] == ["room_002", "room_003"]


def test_footprint_covers_the_placed_rooms(base):
    r, _, _ = base
    fp = r.property["property"]["footprint"]
    total = sum(x["area"]["value"] for x in r.property["rooms"])
    assert fp is not None and fp["kind"] == "placed_rooms_union" and fp["complete"] is False
    assert 0.9 * total < fp["area"]["value"] < 1.15 * total  # union of rooms that share walls
    assert fp["area"]["interval"]["low"] < fp["area"]["value"] < fp["area"]["interval"]["high"] and fp["area"]["quality"] == "weak"
    assert "NOT a verified property footprint" in fp["note"]


def test_the_plan_is_rendered_with_the_photo_footer(base):
    r, _, _ = base
    from PIL import Image

    img = Image.open(r.output_dir / "plan.png")
    assert img.size[0] >= 1300 and r.run_report["plan"]["rooms_drawn"] == 3 and r.run_report["plan"]["walls_drawn"] >= 4


def test_photo_output_is_deterministic(tmp_path):
    root = synthetic_property_dir(tmp_path / "in" / "photos", ["a", "b", "c"])  # one input, two runs
    outs = []
    for name in ("a", "b"):
        sfm = SyntheticPhotoSfm()
        outs.append(process_capture(root, tmp_path / name, opts_for(sfm), stages_fn=run_photo_stages, tier="photos"))
    a, b = outs
    ja, jb = json.loads((a.output_dir / "property.json").read_text()), json.loads((b.output_dir / "property.json").read_text())
    assert strip_volatile(ja) == strip_volatile(jb)  # only the explicit timing section may differ
    assert "timing" in ja and "stages" in ja["timing"]
    assert (a.output_dir / "plan.png").read_bytes() == (b.output_dir / "plan.png").read_bytes()


# ---------------- failure and partial policies ----------------


def test_a_failed_room_is_unreconstructed_and_the_property_is_partial(tmp_path):
    r, _ = run(tmp_path, sfm=SyntheticPhotoSfm(fail_rooms={"room_002"}))
    d = r.property
    assert r.exit_code == 0 and r.status == "partial" and d["property"]["status"] == "partial"
    assert [x["id"] for x in d["rooms"]] == ["room_001", "room_003"]
    un = d["provenance"]["photos"]["unplaced_rooms"]
    assert [u["canonical_id"] for u in un] == ["room_002"] and un[0]["status"] == "unreconstructed"
    assert any("room_002 (b) is not part of the property plan" in w for w in d["warnings"])
    rep = {x["canonical_id"]: x for x in r.run_report["photos"]["rooms"]}
    assert rep["room_002"]["status"] == "unreconstructed" and rep["room_002"]["quality"] == "failure"
    assert rep["room_002"]["local_counts"] is None  # no fake geometry from a failed SfM
    assert "room_002" not in [c for con in r.run_report["photos"]["constraints"] for c in (con["source_room_id"], con["target_room_id"])]
    fp = d["property"]["footprint"]
    assert fp is not None and fp["area"]["value"] < 40  # room_002 is excluded


def test_without_metric_depth_no_room_gets_metric_dimensions(tmp_path):
    sfm = SyntheticPhotoSfm()
    root = synthetic_property_dir(tmp_path / "p", ["a", "b"])

    def factory():
        from spatialforge.video.depth import DepthUnavailable

        raise DepthUnavailable("model weights not found")

    o = PhotoOptions(sfm_backend=sfm, depth_backend_factory=factory)
    r = process_capture(root, tmp_path / "out", o, stages_fn=run_photo_stages, tier="photos")
    d = r.property
    assert r.status == "partial" and d["rooms"] == [] and d["walls"] == [] and d["property"]["footprint"] is None
    assert [u["status"] for u in d["provenance"]["photos"]["unplaced_rooms"]] == ["unscaled", "unscaled"]
    assert any("Metric depth is unavailable" in w for w in d["warnings"])
    assert r.run_report["metric_scale_available"] is False


def test_a_disconnected_room_stays_unplaced_and_is_not_in_the_footprint(tmp_path):
    sfm = SyntheticPhotoSfm(label_to_layout={"e": "room_005"})  # the corridor has no door and shares no view with the others
    r, _ = run(tmp_path, labels=("a", "b", "e"), sfm=sfm)
    d = r.property
    assert d["property"]["status"] == "partial" and [x["id"] for x in d["rooms"]] == ["room_001", "room_002"]
    un = d["provenance"]["photos"]["unplaced_rooms"]
    assert [u["canonical_id"] for u in un] == ["room_003"] and "no stitch evidence" in un[0]["reason"]
    assert any("room_003" in w and "not part of the property plan" in w for w in d["warnings"])
    assert all("room_003" not in c["source_room_id"] + c["target_room_id"] for c in r.run_report["photos"]["constraints"])
    total = sum(x["area"]["value"] for x in d["rooms"])
    assert d["property"]["footprint"]["area"]["value"] < 1.2 * total  # the 14 m2 corridor is not counted
    assert all(not wid.startswith("room_003") for x in d["rooms"] for wid in x["wall_ids"])
    assert all(not w["id"].startswith("room_003") for w in d["walls"])  # nothing of it is drawn beside the property


def test_four_room_loop_closes(tmp_path):
    r, _ = run(tmp_path, labels=("a", "b", "c", "d"))
    d = r.property
    loops = r.run_report["photos"]["loops"]
    assert len(d["rooms"]) == 4 and len(loops) >= 1
    assert all(l["translation_residual_m"] < 0.6 and l["rotation_residual_deg"] < 4.0 for l in loops)
    assert all(c["status"] == "active" for c in r.run_report["photos"]["constraints"])
    assert max(o["fraction_of_smaller"] for o in r.run_report["photos"]["overlaps"]) < 0.08
    for x in d["rooms"]:
        assert 0.7 * TRUE_AREA[x["id"]] < x["area"]["value"] < 1.3 * TRUE_AREA[x["id"]]


def test_room_folder_order_does_not_change_the_result(tmp_path):
    a, _ = run(tmp_path, "a")
    # same three rooms, folder names chosen so that the sorted order (and therefore the canonical ids) is reversed
    sfm = SyntheticPhotoSfm(label_to_layout={"z1": "room_001", "y2": "room_002", "x3": "room_003"})
    b, _ = run(tmp_path, "b", labels=("z1", "y2", "x3"), sfm=sfm)
    da, db = a.property, b.property
    assert len(da["rooms"]) == len(db["rooms"]) == 3
    ar_a = sorted(x["area"]["value"] for x in da["rooms"])
    ar_b = sorted(x["area"]["value"] for x in db["rooms"])
    assert np.allclose(ar_a, ar_b, rtol=0.12)
    fa, fb = da["property"]["footprint"]["area"]["value"], db["property"]["footprint"]["area"]["value"]
    assert abs(fa - fb) / fa < 0.1
    kinds = lambda d: sorted(x["kind"] for x in d["provenance"]["photos"]["adjacency"])  # noqa: E731
    assert kinds(da) == kinds(db)
    assert db["provenance"]["photos"]["source_labels"] == {"room_001": "x3", "room_002": "y2", "room_003": "z1"}


# ---------------- uncertainty ----------------


def test_uncertainty_widens_when_room_scales_disagree(tmp_path, base):
    steady = base[0].property
    r, _ = run(tmp_path, sfm=SyntheticPhotoSfm(room_depth_bias={"room_002": 1.45}))
    d = r.property
    cons = r.run_report["photos"]["scale_consistency"]
    assert cons["scale_inconsistent"] is True and cons["regularisation_applied"] is False  # flagged, never forced equal
    assert any("scale_inconsistent" in w for w in d["warnings"]) and d["property"]["status"] == "partial"
    devs = {k: v["deviation_vs_property_median"] for k, v in cons["rooms"].items()}
    assert max(devs.values()) > 0.2

    def rel_width(prop, rid):
        a = next(x for x in prop["rooms"] if x["id"] == rid)["area"]
        return (a["interval"]["high"] - a["interval"]["low"]) / a["value"]

    worst = max(devs, key=devs.get)
    assert rel_width(d, worst) > rel_width(steady, worst)  # the outlier room gets a wider interval, not a corrected value
    unc = next(x for x in r.run_report["photos"]["rooms"] if x["canonical_id"] == worst)["uncertainty"]
    assert unc["scale_inconsistency_term"] > 0 and unc["extra_relative_sigma"] > 0


@pytest.fixture(scope="module")
def local(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("photo_local")
    root = synthetic_property_dir(tmp / "p", ["a", "b", "c"])
    val = validate_photo_dir(root)
    sfm = SyntheticPhotoSfm()
    recs = sfm.prepare(val.rooms)
    be = sfm.depth_backend()
    cache = {}

    def depth_for(n):
        cache.setdefault(n, be.predict(sfm.load_rgb(n)))
        return cache[n]

    opts = room_mod.PhotoRoomOptions()
    opts.fusion = FusionOptions(pixel_stride=1, min_views=1)
    room = val.rooms[0]
    names = sorted(n for n in recs if recs[n].room_id == room.canonical_id)
    run_ = sfm.map_subset(names, room.canonical_id)
    res = room_mod.reconstruct_room(room, 0, run_, depth_for, opts)
    return res, room, run_, depth_for, opts, sfm, recs


def test_room_local_result_and_metric_scene(local):
    res, room, run_, *_ = local
    assert res.status == "reconstructed" and res.usable and res.canonical_id == "room_001" and res.source_label == "a"
    assert res.quality in ("moderate", "strong") and res.primary_room_id is not None and len(res.poses) == 8
    area = next(r.area.value for r in res.prop.rooms if r.id == res.primary_room_id)
    assert 14 < area < 26 and res.prop.capture["tier"] == "photos"
    assert set(all_values(res.prop.to_dict(), "source_tier")) == {"photos"}
    scene = MetricScene(res.cloud, "photos", "moderate", res.scale["scale_quality"],
                        camera_poses=np.array(list(res.poses.values())), frame_refs=list(res.poses))
    s = scene.summary()
    assert s["source_tier"] == "photos" and s["points"] > 5000 and s["cameras"] == 8
    up = res.cloud[:, 1]
    assert np.percentile(up, 99.5) - np.percentile(up, 0.5) == pytest.approx(2.6, abs=0.5)  # gravity-aligned: a room 2.6 m tall
    rep = res.report()
    assert rep["sfm"]["registered_images"] == 8 and json.loads(json.dumps(rep))["canonical_id"] == "room_001"


def test_room_sfm_result_serialises(local):
    res, room, run_, *_ = local
    d = json.loads(json.dumps(run_.result.to_dict()))
    assert d["registered_images"] == 8 and d["keyframes_attempted"] == 8 and d["camera"]["model"] == "SIMPLE_RADIAL"
    rec = ImageRecord("room_001/img_00.jpg", "room_001", room.images[0], "x.jpg")
    assert rec.room_id == "room_001" and rec.meta.width == 640
    o = PhotoSfmOptions()
    assert o.min_registered_images == 2 and o.min_model_size == 2  # a two-photo room is allowed


def test_a_two_photo_room_is_supported_but_never_rated_strong(tmp_path):
    root = synthetic_property_dir(tmp_path / "p", ["a"], per_room=2)
    val = validate_photo_dir(root)
    assert val.ok
    sfm = SyntheticPhotoSfm(per_room=2)
    recs = sfm.prepare(val.rooms)
    be = sfm.depth_backend()
    opts = room_mod.PhotoRoomOptions()
    opts.fusion = FusionOptions(pixel_stride=1, min_views=1)
    run_ = sfm.map_subset(sorted(recs), "room_001")
    res = room_mod.reconstruct_room(val.rooms[0], 0, run_, lambda n: be.predict(sfm.load_rgb(n)), opts)
    assert res.sfm["registered_images"] == 2 and res.sfm["tracking_quality"] != "strong"
    assert res.quality in ("weak", "failure", "moderate") and res.quality != "strong"
    if res.scale:
        assert res.scale["scale_quality"] in ("weak", "unavailable")  # moderate scale needs at least three supporting photos


def test_sfm_failure_for_a_room_gives_no_geometry(local):
    res, room, run_, depth_for, opts, sfm, recs = local
    failed = SfmResult(8, 3, 0.375, 40, 0.9, 3.0, 3, [3, 3, 2], None, ["x"], "failure", ["no coherent reconstruction"])
    run_.result = failed
    out = room_mod.reconstruct_room(room, 0, run_, depth_for, opts)
    assert out.status == "unreconstructed" and out.quality == "failure" and out.prop is None and out.cloud is None
    assert any("SfM failed" in r for r in out.reasons) and not out.poses


# ---------------- merge helpers ----------------


def test_stitching_uncertainty_widens_the_placed_geometry(local):
    res = local[0]
    base_room, base_walls, *_ = transform_room(PlacedRoom("room_001", "a", res.prop, res.primary_room_id, (0.3, 2.0, -1.0)))
    wide_room, wide_walls, *_ = transform_room(PlacedRoom("room_001", "a", res.prop, res.primary_room_id, (0.3, 2.0, -1.0),
                                                           extra_rel_sigma=0.15, position_sigma_m=0.4))
    a, b = base_room.area, wide_room.area
    assert a.value == b.value and (b.high - b.low) > (a.high - a.low)  # the value is untouched, only the interval grows
    assert (wide_room.length.high - wide_room.length.low) > (base_room.length.high - base_room.length.low)
    pa, pb = base_walls[0].position_uncertainty, wide_walls[0].position_uncertainty
    assert (pb.high - pb.low) > (pa.high - pa.low)
    assert all(m.source_tier == "photos" for m in [a, b, pa, pb])
    # placing a room rotates it rigidly: lengths and area are unchanged, orientations are shifted
    assert base_room.perimeter.value == pytest.approx(next(r.perimeter.value for r in res.prop.rooms if r.id == res.primary_room_id))
    assert base_walls[0].orientation.value == pytest.approx((res.prop.walls[0].orientation.value + np.degrees(0.3)) % 180.0)


def test_scale_offsets_least_squares():
    obs = [("a", "b", np.log(1.2)), ("b", "c", np.log(0.9)), ("a", "c", np.log(1.2 * 0.9))]
    u = scale_offsets(obs)
    assert abs(sum(u.values())) < 1e-9 and u["a"] - u["b"] == pytest.approx(np.log(1.2)) and u["b"] - u["c"] == pytest.approx(np.log(0.9))
    assert scale_offsets([]) == {}


def test_duplicate_walls_and_openings_across_rooms_are_merged():
    from spatialforge.pipeline.models import Point2D

    w1 = simple_wall("room_001_wall_001", (0, 0), (5, 0), "strong")
    w2 = simple_wall("room_002_wall_003", (0.2, 0.15), (4.9, 0.15), "moderate")  # the same wall seen from the other room
    w3 = simple_wall("room_002_wall_004", (0, 3), (5, 3), "strong")  # a different wall
    w1.room_ids, w2.room_ids = ["room_001"], ["room_002"]
    r1, r2 = square_room("room_001", wall_ids=("room_001_wall_001",)), square_room("room_002", wall_ids=("room_002_wall_003", "room_002_wall_004"))
    o1 = simple_opening("room_001_opening_001", "room_001_wall_001")
    o2 = simple_opening("room_002_opening_001", "room_002_wall_003")
    o2.left_jamb, o2.right_jamb, o2.position = Point2D(1.6, 0.15), Point2D(2.5, 0.15), Point2D(2.05, 0.15)
    walls, removed = dedupe_walls([r1, r2], [w1, w2, w3], [o1, o2], [])
    assert removed == {"room_002_wall_003": "room_001_wall_001"} and [w.id for w in walls] == ["room_001_wall_001", "room_002_wall_004"]
    assert r2.wall_ids == ["room_001_wall_001", "room_002_wall_004"] and set(walls[0].room_ids) == {"room_001", "room_002"}
    assert o2.wall_id == "room_001_wall_001"
    ops, merged = dedupe_openings([r1, r2], walls, [o1, o2], [])
    assert len(ops) == 1 and merged and list(merged.values()) == [ops[0].id]


def test_union_footprint_area_and_outline():
    a = np.array([[0, 0], [5, 0], [5, 4], [0, 4]], float)
    b = np.array([[5, 0], [9, 0], [9, 4], [5, 4]], float)
    far = np.array([[20, 0], [22, 0], [22, 2], [20, 2]], float)
    pts, area = union_footprint([a, b])
    assert area == pytest.approx(36.0, rel=0.03) and len(pts) >= 4
    _, area_far = union_footprint([a, b, far])
    assert area_far == pytest.approx(40.0, rel=0.03)  # a disconnected polygon only counts if it is passed in
    assert union_footprint([]) is None


# ---------------- validation failures and the CLI ----------------


def test_validation_failure_is_a_clean_failure(tmp_path):
    root = synthetic_property_dir(tmp_path / "p", ["a"], per_room=1)
    r = process_capture(root, tmp_path / "out", PhotoOptions(), stages_fn=run_photo_stages, tier="photos")
    assert r.status == "failure" and r.exit_code == 1 and r.property is None
    assert r.run_report["errors"][0]["stage"] == "photo_validation" and "has 1 supported photo(s)" in r.run_report["errors"][0]["message"]
    assert not (tmp_path / "out" / "property.json").exists() and r.run_report["stages"][0]["name"] == "photo_validation"


def test_cli_photo_tier(tmp_path, monkeypatch, capsys):
    root = synthetic_property_dir(tmp_path / "p", ["a", "b"])
    sfm = SyntheticPhotoSfm()
    real = run_photo_stages

    def patched(capture, opts):
        opts.sfm_backend, opts.depth_backend = sfm, sfm.depth_backend()
        opts.room.fusion = FusionOptions(pixel_stride=1, min_views=1)
        return real(capture, opts)

    monkeypatch.setattr("spatialforge.photos.pipeline.run_photo_stages", patched)
    code = main(["process", str(root), "--tier", "photos", "--output", str(tmp_path / "ok")])
    out = capsys.readouterr().out
    assert code == 0 and "tier: photos" in out and (tmp_path / "ok" / "property.json").exists()
    assert main(["process", str(tmp_path / "missing"), "--tier", "photos", "--output", str(tmp_path / "bad")]) == 1
    assert "ERROR [photo_validation]" in capsys.readouterr().out
