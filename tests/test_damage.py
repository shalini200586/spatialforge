"""Damage analysis: classes, detectors, surface mapping, metric extents, concealed-risk rules, scope items, determinism."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # the damage analysis uses OpenCV (pip install -e .[damage])

from helpers_damage import (  # noqa: E402
    CAM, add_bright_patch, add_black_object, add_crack, add_patterned_patch, add_shadow, add_stain, add_straight_line,
    add_tile_grid, finish, room_property, wall_image, wall_view,
)
from helpers_video import camera_pose  # noqa: E402

from spatialforge.damage.analyze import DamageOptions, EvidenceImage, analyze_damage, apply_to_property  # noqa: E402
from spatialforge.damage.classes import normalize_damage_class, require_damage_class  # noqa: E402
from spatialforge.damage.detect import detect_damage  # noqa: E402
from spatialforge.damage.geometry import CameraModel, build_surfaces, map_polygon  # noqa: E402
from spatialforge.damage.rules import RuleOptions, evaluate_rules  # noqa: E402
from spatialforge.damage.scope import build_scope  # noqa: E402
from spatialforge.pipeline.models import DamageRegion, Measurement, ModelError  # noqa: E402
from spatialforge.pipeline.schema import build_schema, validate_against_schema  # noqa: E402
from spatialforge.pipeline.serialization import property_to_dict  # noqa: E402


def stain_wall(**kw):
    img = wall_image()
    add_stain(img, **kw)
    return img


def run(prop, images, tmp_path=None, **kw):
    return analyze_damage(prop, images, tmp_path, DamageOptions(**kw))


# ---------------- classes ----------------


def test_class_normalization():
    for label in ("crack", "Cracks", "Hairline-Crack", "crack_like damage", "FISSURE", " wall crack "):
        assert normalize_damage_class(label) == "crack", label
    for label in ("stain", "Water Stain", "discolouration", "damp-patch", "Mold", "mildew", "tide mark"):
        assert normalize_damage_class(label) == "stain", label
    assert normalize_damage_class("spalling") is None and normalize_damage_class("") is None
    assert require_damage_class("stains") == "stain"
    with pytest.raises(ValueError, match="unknown damage class"):
        require_damage_class("graffiti")


# ---------------- detectors ----------------


def test_detector_finds_a_crack_and_a_stain_and_ignores_lookalikes():
    img = wall_image()
    add_crack(img)
    add_stain(img)
    add_straight_line(img)  # a door frame / joint: straight
    add_shadow(img)  # brightness only
    dets = detect_damage(finish(img))
    assert sorted(d.damage_class for d in dets) == ["crack", "stain"]
    crack = next(d for d in dets if d.damage_class == "crack")
    x0, y0, x1, y1 = crack.bbox
    assert 250 < x0 < 330 and 150 < y0 < 230 and y1 > 560 and crack.confidence > 0.5
    stain = next(d for d in dets if d.damage_class == "stain")
    sx0, sy0, sx1, sy1 = stain.bbox
    assert 740 < sx0 < 840 and 560 < sy0 < 650 and sx1 > 980 and stain.confidence > 0.5


def test_detector_has_no_false_positives_on_clean_walls():
    assert detect_damage(finish(wall_image())) == []
    assert detect_damage(finish(wall_image(noise=8.0, seed=5))) == []
    only_shadow = wall_image()
    add_shadow(only_shadow)
    assert detect_damage(finish(only_shadow)) == []
    straight = wall_image()
    add_straight_line(straight)
    assert detect_damage(finish(straight)) == []


def test_detector_rejects_the_lookalikes_found_in_real_rooms():
    """Tile joints, patterned objects, lights and dark objects fired in real footage; none of them is wall damage."""
    cases = {
        "tile grid": lambda i: add_tile_grid(i),
        "patterned patch": lambda i: add_patterned_patch(i),
        "bright patch": lambda i: add_bright_patch(i),
        "black object": lambda i: add_black_object(i),
    }
    for name, add in cases.items():
        img = wall_image()
        add(img)
        assert detect_damage(finish(img)) == [], name
    # ...and real damage on the same wall is still found next to them
    img = wall_image()
    add_tile_grid(img, step=420)
    add_crack(img, x0=700, y0=220, length=200)  # a crack on open tile surface, clear of the joints (a crack that touches a joint is dropped)
    add_patterned_patch(img, x0=900, y0=700, w=200, h=150)
    found = detect_damage(finish(img))
    assert [d.damage_class for d in found] == ["crack"]


def test_detector_is_deterministic_and_scales_back_to_the_input_size():
    img = finish(stain_wall())
    a, b = detect_damage(img), detect_damage(img)
    assert [(d.damage_class, d.confidence) for d in a] == [(d.damage_class, d.confidence) for d in b]
    assert all(np.array_equal(x.polygon, y.polygon) for x, y in zip(a, b))
    big = cv2.resize(img, (2560, 1920), interpolation=cv2.INTER_LINEAR)  # analysed at <= 1024 px, reported in input pixels
    d_big = detect_damage(big)
    assert len(d_big) == 1 and d_big[0].bbox[0] == pytest.approx(2 * a[0].bbox[0], rel=0.08)


# ---------------- serialization ----------------


def test_detection_serialization_and_schema():
    prop = room_property()
    out = run(prop, [wall_view(stain_wall())])
    apply_to_property(prop, out, 1)
    d = property_to_dict(prop)  # validated against the published schema
    assert validate_against_schema(d, build_schema()) == []
    r = d["damage_regions"][0]
    assert "damage" not in d  # the deprecated alias was removed: damage_regions is the one canonical key
    assert set(r) >= {"damage_id", "class", "confidence", "quality", "source", "image_region", "room_id", "surface_id", "width_m",
                      "height_m", "area_m2", "length_m", "metric_reliable"}
    assert r["damage_id"] == "damage_001" and r["class"] == "stain" and r["source"]["image_id"] == "frame_000001"
    assert r["source"]["tier"] == "lidar" and r["source"]["image_size"] == [1280, 960] and len(r["image_region"]["bbox"]) == 4
    assert json.loads(json.dumps(r)) == r
    assert any("Damage analysis is deterministic" in w for w in d["warnings"])
    assert not any("Damage detection is not part of this version" in w for w in d["warnings"])
    assert d["provenance"]["damage_analysis"]["detections"] == 1


def test_damage_region_model_rules():
    src = {"image_id": "a", "tier": "lidar", "frame_index": None, "image_size": [10, 10]}
    reg = {"bbox": [0, 0, 1, 1], "polygon": [[0, 0], [1, 0], [1, 1]], "area_px": 1.0}
    with pytest.raises(ModelError):
        DamageRegion("d", "graffiti", 0.5, "weak", src, reg)
    with pytest.raises(ModelError):
        DamageRegion("d", "crack", 1.5, "weak", src, reg)
    with pytest.raises(ModelError, match="reliable projection"):
        DamageRegion("d", "crack", 0.5, "weak", src, reg, width=Measurement(1.0, "m"))  # a metric extent needs a surface
    prop = room_property()
    prop.damage = [DamageRegion("d1", "crack", 0.5, "weak", src, reg, room_id="room_999")]
    with pytest.raises(ModelError, match="unknown room"):
        prop.validate()
    prop.damage = [DamageRegion("d1", "crack", 0.5, "weak", src, reg, surface_id="nope")]
    with pytest.raises(ModelError, match="unknown surface"):
        prop.validate()


# ---------------- surface and room mapping ----------------


def test_wall_mapping():
    prop = room_property()
    out = run(prop, [wall_view(stain_wall())])
    r = out.regions[0]
    assert (r.surface_id, r.surface_type, r.room_id) == ("w3", "wall", "room_001")  # the wall the camera faces
    assert r.metric_reliable and 0.0 < r.surface_extent["v_min"] < r.surface_extent["v_max"] < 2.0
    # the stain centre is at pixel x = 900, 260 px right of the image centre; looking along +z the camera's right is world -x, so it
    # lies 0.88 m toward smaller x (x = 1.12) and wall w3 runs from x = 4 to x = 0, giving u = 4 - 1.12 = 2.88
    assert (r.surface_extent["u_min"] + r.surface_extent["u_max"]) / 2 == pytest.approx(2.88, abs=0.1)
    assert out.plan_points[r.id][0] == pytest.approx(1.12, abs=0.1) and out.plan_points[r.id][1] == pytest.approx(4.0, abs=0.01)


def test_room_mapping_floor_and_unknown_surface():
    prop = room_property()
    floor_img = wall_image(colour=(150, 140, 120))
    add_stain(floor_img, cx=640, cy=480)
    out = run(prop, [wall_view(floor_img, pitch=-0.95)])  # looking down at the floor of room_001
    r = out.regions[0]
    assert (r.surface_id, r.surface_type, r.room_id) == ("floor:room_001", "floor", "room_001")
    assert r.metric_reliable and r.area.value > 0
    # a camera outside every room: the rays hit nothing, so the damage is kept but belongs to no surface and no room
    out = run(prop, [wall_view(stain_wall(), pos=(30.0, 1.4, 30.0))])
    assert len(out.regions) == 1 and out.regions[0].surface_id is None and out.regions[0].room_id is None
    assert out.regions[0].width is None and any("hit no modelled surface" in w for w in out.regions[0].warnings)


def test_photo_style_room_hint_restricts_surfaces_and_sets_the_room():
    prop = room_property()
    img = wall_view(stain_wall(), room="room_001")
    out = run(prop, [img])
    assert out.regions[0].room_id == "room_001" and out.regions[0].surface_id == "w3"
    other = wall_view(stain_wall(), room="room_999")  # a hint for a room that has no surfaces here
    out = run(prop, [other])
    assert out.regions[0].surface_id is None  # only the hinted room's surfaces are considered


# ---------------- no-geometry fallback ----------------


def test_no_geometry_fallback_keeps_the_detection_without_metric_extent():
    prop = room_property()
    img = EvidenceImage("frame_000007", "video", lambda: finish(stain_wall()), CAM, None, None, None, 7)  # no pose
    out = run(prop, [img])
    r = out.regions[0]
    assert r.damage_class == "stain" and r.surface_id is None and r.room_id is None
    assert (r.width, r.height, r.area, r.length, r.surface_extent) == (None, None, None, None, None) and r.metric_reliable is False
    assert any("no camera geometry" in w for w in r.warnings)
    assert any("no reliable metric extent" in w for w in out.warnings)
    d = r.to_dict()
    assert d["width_m"] is None and d["area_m2"] is None and d["image_region"]["bbox"][2] > d["image_region"]["bbox"][0]
    apply_to_property(prop, out, 1)
    assert property_to_dict(prop)["damage_regions"][0]["width_m"] is None  # still valid canonical output
    prop2 = room_property()
    prop2.rooms, prop2.walls, prop2.openings = [], [], []  # a property with no geometry at all
    out2 = run(prop2, [wall_view(stain_wall())])
    assert out2.regions[0].surface_id is None and out2.regions[0].width is None


def test_grazing_or_mixed_views_do_not_get_a_metric_extent():
    prop = room_property()
    cam_pos = camera_pose([3.9, 1.4, 0.5], np.radians(84.0))  # looking almost along wall w3 (z = 4)
    surfaces = build_surfaces(prop)
    poly = np.array([[600, 400], [700, 400], [700, 560], [600, 560]], dtype=float)
    m = map_polygon(poly, CAM, cam_pos, 0.0, surfaces)
    assert not m.reliable or m.cos_incidence >= 0.25
    far = camera_pose([2.0, 1.4, -8.0], 0.0)  # 12 m from wall w3: beyond the range gate
    m2 = map_polygon(np.array([[600, 400], [700, 400], [700, 560], [600, 560]], dtype=float), CAM, far, 0.0, surfaces)
    assert not m2.reliable and any("range" in r for r in m2.reasons)
    # a region straddling the wall and the ceiling edge: rays disagree on the surface, so no metric extent
    wide = np.array([[100, 0], [1180, 0], [1180, 960], [100, 960]], dtype=float)
    m3 = map_polygon(wide, CAM, camera_pose([2.0, 1.4, 0.6], 0.0, 0.1), 0.0, surfaces)
    assert m3.agreement < 0.8 and not m3.reliable


# ---------------- metric extent ----------------


def test_metric_extent_matches_the_projected_geometry():
    prop = room_property()
    img = wall_image()
    add_stain(img, cx=640, cy=480, rx=110, ry=80)  # 220 x 160 px at 3.4 m with f = 1000: ~0.75 x 0.54 m
    out = run(prop, [wall_view(img)])
    r = out.regions[0]
    px_m = 3.4 / 1000
    assert r.metric_reliable and r.surface_id == "w3"
    assert r.width.value == pytest.approx(240 * px_m, rel=0.2) and r.height.value == pytest.approx(190 * px_m, rel=0.25)
    assert r.area.value == pytest.approx(np.pi * 120 * 95 * px_m ** 2, rel=0.3) and r.area.unit == "m2"
    for m in (r.width, r.height, r.area):
        assert m.low < m.value < m.high and m.source_tier == "lidar" and m.quality in ("weak", "moderate", "strong")
    cr = wall_image()
    add_crack(cr, x0=640, y0=200, length=420)
    c = run(prop, [wall_view(cr)]).regions[0]
    assert c.damage_class == "crack" and c.length.value == pytest.approx(440 * px_m, rel=0.25)  # a ~1.5 m crack
    assert c.length.value > c.width.value * 0.9 and c.surface_type == "wall"


# ---------------- uncertainty ----------------


def test_uncertainty_propagates_from_surface_scale_and_range():
    img = stain_wall()
    lidar = run(room_property(), [wall_view(img)]).regions[0]
    video = run(room_property(scale_rel=0.15), [wall_view(img)]).regions[0]
    assert video.width.value == pytest.approx(lidar.width.value)  # the value is the same
    rel = lambda m: (m.high - m.low) / m.value  # noqa: E731
    assert rel(video.width) > rel(lidar.width) * 2 and rel(video.area) > rel(lidar.area) * 2  # scale uncertainty carried through
    assert rel(lidar.width) > 0.02  # even LiDAR keeps pixel and plane-position terms
    near = run(room_property(), [wall_view(img, pos=(2.0, 1.4, 1.6))]).regions[0]  # 2.4 m from the wall instead of 3.4 m
    far = lidar
    assert near.width.value < far.width.value  # nearer: the same pixels cover less wall
    assert rel(near.area) != rel(far.area)
    wide_pos = room_property()
    for w in wide_pos.walls:
        w.position_uncertainty = Measurement(0.30, "m")  # a poorly located wall plane
    loose = run(wide_pos, [wall_view(img)]).regions[0]
    assert rel(loose.width) > rel(lidar.width) * 1.5  # a plane that may be 0.3 m off rescales the projection


# ---------------- concealed-damage rules ----------------


def regions_for(prop, img, pitch):
    return run(prop, [wall_view(img, pitch=pitch)]).regions


def test_concealed_rule_triggers_for_a_stain_at_the_floor_wall_junction():
    prop = room_property()
    out = run(prop, [wall_view(stain_wall(cx=640, cy=480, rx=100, ry=60), pitch=-0.31)])
    r = out.regions[0]
    assert r.surface_extent["v_min"] <= 0.30, r.surface_extent
    assert [f["rule_id"] for f in out.flags] == ["CDR-001"]
    f = out.flags[0]
    assert f["rule_name"] == "stain_near_floor_wall_junction" and f["risk"] == "moisture" and f["damage_ids"] == [r.id]
    assert f["surface_id"] == "w3" and f["room_id"] == "room_001" and f["flag_id"] == "concealed_001"
    assert "lowest edge <= 0.30 m" in f["explanation"] and "commonly associated" in f["explanation"]
    assert f["claims_presence"] is False and "does NOT assert" in f["note"] and f["quality"] in ("weak", "moderate")
    assert "is present" not in f["explanation"] and "confirmed" not in f["explanation"].lower()


def test_concealed_rule_does_not_trigger_mid_wall_or_without_a_metric_position():
    prop = room_property()
    mid = run(prop, [wall_view(stain_wall(cx=640, cy=480, rx=100, ry=60), pitch=0.0)])  # a stain at about 1.4 m height
    assert mid.regions[0].surface_extent["v_min"] > 1.0 and mid.flags == []
    assert mid.report["rule_evaluation"]["rules"]["CDR-001"]["regions_evaluated"] == 1
    assert mid.report["rule_evaluation"]["rules"]["CDR-001"]["triggered"] == 0
    # same low stain, but the camera has no pose: the rule cannot be evaluated, so it can never fire on a guess
    img = EvidenceImage("frame_1", "video", lambda: finish(stain_wall(cx=640, cy=480, rx=100, ry=60)), CAM, None, None, None, 1)
    blind = run(prop, [img])
    assert blind.flags == [] and blind.regions[0].id in blind.report["rule_evaluation"]["regions_without_metric_position_not_evaluated"]
    assert all(v["triggered"] == 0 for v in blind.report["rule_evaluation"]["rules"].values())


def test_other_rules_trigger_on_their_stated_conditions():
    prop = room_property()
    surfaces = build_surfaces(prop)
    src = {"image_id": "a", "tier": "lidar", "frame_index": None, "image_size": [10, 10]}
    reg = {"bbox": [0, 0, 1, 1], "polygon": [[0, 0], [1, 0], [1, 1]], "area_px": 1.0}

    def region(i, cls, u0, u1, v0, v1, surface="w3", length=None):
        m = lambda v: Measurement(v, "m", v * 0.9, v * 1.1, None, "moderate", "lidar")  # noqa: E731
        return DamageRegion(f"damage_{i:03d}", cls, 0.8, "moderate", src, reg, "room_001", surface, "wall", m(u1 - u0), m(v1 - v0),
                            m(0.1), None if length is None else m(length), {"u_min": u0, "u_max": u1, "v_min": v0, "v_max": v1}, True)

    ceil = region(1, "stain", 1.0, 1.4, 2.0, 2.3)  # top edge 0.1 m below the 2.4 m ceiling
    flags, _ = evaluate_rules([ceil], surfaces)
    assert [f["rule_id"] for f in flags] == ["CDR-002"] and "0.10 m below the ceiling" in flags[0]["explanation"]
    pair = [region(1, "stain", 0.5, 0.9, 1.0, 1.4), region(2, "stain", 1.6, 2.0, 1.2, 1.6)]
    flags, _ = evaluate_rules(pair, surfaces)
    assert [f["rule_id"] for f in flags] == ["CDR-003"] and flags[0]["damage_ids"] == ["damage_001", "damage_002"]
    apart = [region(1, "stain", 0.1, 0.4, 1.0, 1.4), region(2, "stain", 3.0, 3.4, 1.0, 1.4)]
    assert evaluate_rules(apart, surfaces)[0] == []  # 2.8 m apart: not a cluster
    long_crack = region(1, "crack", 1.0, 1.2, 0.5, 1.7, length=1.2)
    assert [f["rule_id"] for f in evaluate_rules([long_crack], surfaces)[0]] == ["CDR-005"]
    short = region(1, "crack", 1.0, 1.1, 0.5, 0.9, length=0.4)
    assert evaluate_rules([short], surfaces)[0] == []
    both = [region(1, "crack", 1.0, 1.1, 1.0, 1.4, length=0.4), region(2, "stain", 1.2, 1.6, 1.0, 1.5)]
    assert [f["rule_id"] for f in evaluate_rules(both, surfaces)[0]] == ["CDR-006"]
    # CDR-004 needs a wall with an opening of known height: w1 has the door (jambs at u = 1.55 and 2.45, 2.05 m high)
    corner = region(1, "crack", 1.3, 1.5, 1.9, 2.2, surface="w1", length=0.4)
    assert [f["rule_id"] for f in evaluate_rules([corner], surfaces)[0]] == ["CDR-004"]
    far_from_door = region(1, "crack", 3.2, 3.4, 1.9, 2.2, surface="w1", length=0.4)
    assert evaluate_rules([far_from_door], surfaces)[0] == []
    opts = RuleOptions(junction_m=0.5)
    low = region(1, "stain", 1.0, 1.4, 0.45, 0.8)
    assert evaluate_rules([low], surfaces)[0] == [] and [f["rule_id"] for f in evaluate_rules([low], surfaces, opts)[0]] == ["CDR-001"]


# ---------------- scope items ----------------


def test_scope_items_are_keyed_to_surfaces_and_quantified_only_when_justified():
    prop = room_property()
    cr, st = wall_image(), wall_image()
    add_crack(cr, x0=500)
    out = run(prop, [wall_view(cr, image_id="f1"), wall_view(stain_wall(cx=640, cy=480, rx=100, ry=60), pitch=-0.31, image_id="f2")])
    by_action = {i["action"]: i for i in out.items}
    crack_item, stain_item = by_action["seal_and_repair_crack"], by_action["treat_stain_and_repaint"]
    cdmg = next(r for r in out.regions if r.damage_class == "crack")
    sdmg = next(r for r in out.regions if r.damage_class == "stain")
    assert crack_item["surface_id"] == "w3" and crack_item["damage_id"] == cdmg.id and crack_item["room_id"] == "room_001"
    assert crack_item["quantity"]["unit"] == "m" and crack_item["quantity"]["value"] == pytest.approx(cdmg.length.value)
    assert stain_item["quantity"]["unit"] == "m2" and stain_item["quantity"]["value"] == pytest.approx(sdmg.area.value)
    assert crack_item["quality"] in ("weak", "moderate", "strong") and "measured" in crack_item["basis"]
    inspect = by_action["inspect_for_concealed_moisture"]  # the low stain raised CDR-001: an INSPECTION item, quantity 1
    assert inspect["flag_id"] == out.flags[0]["flag_id"] and inspect["quantity"] == {**inspect["quantity"], "value": 1.0, "unit": "each"}
    assert inspect["quality"] == "weak" and "hidden damage is not assumed" in inspect["basis"]
    assert [i["item_id"] for i in out.items] == [f"scope_{n:03d}" for n in range(1, len(out.items) + 1)]
    # a detection without a metric extent still gets an item, but with no invented quantity
    blind = EvidenceImage("f9", "video", lambda: finish(stain_wall()), CAM, None, None, None, 9)
    items = run(prop, [blind]).items
    assert len(items) == 1 and items[0]["quantity"] is None and items[0]["quality"] == "weak" and items[0]["surface_id"] is None
    assert "no reliable metric extent" in items[0]["basis"]
    assert build_scope([], [], "lidar") == []


# ---------------- merging, determinism, diagnostics ----------------


def test_two_views_of_the_same_damage_become_one_region():
    prop = room_property()
    img = stain_wall(cx=640, cy=480)
    a = wall_view(img, image_id="frame_000001")
    b = wall_view(img, image_id="frame_000002", pos=(2.1, 1.4, 0.8))  # a slightly different viewpoint of the same stain
    out = run(prop, [a, b])
    assert len(out.regions) == 1 and out.regions[0].other_sources and out.regions[0].other_sources[0]["image_id"] in ("frame_000001", "frame_000002")
    single = run(prop, [a]).regions[0]
    assert out.regions[0].confidence >= single.confidence  # a second view can only raise the heuristic score


def test_damage_output_is_deterministic_and_writes_diagnostics(tmp_path):
    imgs = lambda: [wall_view(stain_wall(cx=1150, cy=700, rx=80, ry=60), image_id="f2"),  # noqa: E731
                    wall_view(stain_wall(cx=640, cy=480), pitch=-0.31, image_id="f1")]
    a = run(room_property(), imgs(), tmp_path / "a")
    b = run(room_property(), imgs(), tmp_path / "b")
    assert [r.to_dict() for r in a.regions] == [r.to_dict() for r in b.regions]
    assert a.flags == b.flags and a.items == b.items
    for f in ("damage_report.json", "damage_plan.png"):
        assert (tmp_path / "a" / f).stat().st_size > 0, f
    assert (tmp_path / "a" / "damage_plan.png").read_bytes() == (tmp_path / "b" / "damage_plan.png").read_bytes()
    jpgs = sorted(p.name for p in (tmp_path / "a" / "images").glob("*.jpg"))
    assert jpgs == ["f1.jpg", "f2.jpg"]
    assert (tmp_path / "a" / "images" / "f1.jpg").read_bytes() == (tmp_path / "b" / "images" / "f1.jpg").read_bytes()
    rep = json.loads((tmp_path / "a" / "damage_report.json").read_text())
    assert rep["images_analysed"] == 2 and rep["detections_reported"] == len(a.regions) and rep["method"]["classes"] == ["crack", "stain"]
    assert rep["regions"][0]["damage_id"] == "damage_001" and "rule_evaluation" in rep
    # evidence order does not matter
    c = run(room_property(), list(reversed(imgs())))
    assert [r.to_dict() for r in c.regions] == [r.to_dict() for r in a.regions]


def test_unreadable_evidence_is_skipped_with_a_warning():
    def boom():
        raise OSError("cannot decode frame 3")

    bad = EvidenceImage("frame_3", "lidar", boom, CAM, camera_pose([2, 1.4, 0.6], 0.0), 0.0, None, 3)
    out = run(room_property(), [bad, wall_view(stain_wall())])
    assert len(out.regions) == 1 and any("frame_3 could not be read" in w for w in out.warnings)
