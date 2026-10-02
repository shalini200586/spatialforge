"""Product plan rendering."""

import numpy as np
from helpers_pipeline import simple_opening, simple_property, simple_wall, square_room
from PIL import Image

from spatialforge.lidar.rooms import point_in_polygon
from spatialforge.pipeline.render import BACKGROUND, LOOSE_WALL_COLOUR, ROOM_FILLS, WALL_COLOUR, label_point, render_plan


def pixel(img, info, x, z):
    px, py = info.to_px(x, z)
    return img.getpixel((int(round(px)), int(round(py))))


def test_plan_png_is_written_with_dimensions(tmp_path):
    info = render_plan(simple_property(), tmp_path / "plan.png", "Test")
    assert (tmp_path / "plan.png").stat().st_size > 1000
    img = Image.open(tmp_path / "plan.png")
    assert img.size == (info.width_px, info.height_px) and min(img.size) > 100
    assert info.rooms_drawn == 1 and info.walls_drawn == 4 and info.openings_drawn == 1


def test_room_wall_and_opening_pixels(tmp_path):
    info = render_plan(simple_property(), tmp_path / "plan.png")
    img = Image.open(tmp_path / "plan.png").convert("RGB")
    assert pixel(img, info, 0.8, 1.0) == ROOM_FILLS[0]  # inside the room, away from labels
    assert pixel(img, info, 4.0, 3.0) == WALL_COLOUR  # on the east wall
    assert pixel(img, info, 3.0, 0.0) == WALL_COLOUR  # on the south wall, outside the opening
    assert pixel(img, info, 2.0, 0.0) != WALL_COLOUR  # the opening (jambs at x = 1.55 and 2.45) cuts the wall open
    assert pixel(img, info, 1.55, 0.05) == WALL_COLOUR  # a jamb tick is drawn across the wall at each jamb
    assert pixel(img, info, -0.7, -0.7) == BACKGROUND  # outside everything


def test_metric_aspect_ratio_is_preserved(tmp_path):
    info = render_plan(simple_property(), tmp_path / "plan.png")
    (ax, ay), (bx, by) = info.to_px(0, 0), info.to_px(4, 0)
    (cx, cy) = info.to_px(0, 4)
    assert abs((bx - ax) - (ay - cy)) < 1e-6  # 4 m in x spans the same number of pixels as 4 m in z


def test_weak_walls_and_unverified_openings_are_not_drawn(tmp_path):
    prop = simple_property(unverified_openings=[simple_opening("unverified_001", "w1")])
    prop.walls.append(simple_wall("w9", (6, 0), (8, 0), "weak"))
    info = render_plan(prop, tmp_path / "plan.png")
    assert info.walls_drawn == 4 and info.openings_drawn == 1  # the weak wall and the unverified opening are skipped


def test_walls_outside_any_room_are_drawn_dashed_and_light(tmp_path):
    prop = simple_property()
    prop.walls.append(simple_wall("w9", (6, 0), (10, 0), "strong"))  # not part of any room
    info = render_plan(prop, tmp_path / "plan.png")
    img = Image.open(tmp_path / "plan.png").convert("RGB")
    colours = {pixel(img, info, 6.0 + 0.05 * i, 0.0) for i in range(60)}
    assert LOOSE_WALL_COLOUR in colours and WALL_COLOUR not in colours  # light, never the solid wall colour
    assert BACKGROUND in colours  # dashed, so not a continuous line


def test_partial_result_is_visibly_flagged(tmp_path):
    render_plan(simple_property(status="partial"), tmp_path / "p.png")
    render_plan(simple_property(status="complete"), tmp_path / "c.png")
    a = np.asarray(Image.open(tmp_path / "p.png").convert("RGB")).astype(int)
    b = np.asarray(Image.open(tmp_path / "c.png").convert("RGB")).astype(int)
    assert a.shape == b.shape and (a != b).any()  # badge and note differ
    assert ((a[:110] == (190, 100, 0)).all(axis=2)).any()  # the partial colour is used in the header
    assert not ((b[:110] == (190, 100, 0)).all(axis=2)).any()


def test_plan_render_is_deterministic(tmp_path):
    render_plan(simple_property(), tmp_path / "a.png", "T")
    render_plan(simple_property(), tmp_path / "b.png", "T")
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()


def test_empty_property_renders_an_explicit_message(tmp_path):
    prop = simple_property(rooms=[], walls=[], openings=[], warnings=["No closed rooms were recovered."])
    info = render_plan(prop, tmp_path / "plan.png")
    assert (tmp_path / "plan.png").stat().st_size > 0 and info.notes == ["empty"]


def test_label_point_is_inside_an_l_shaped_room():
    l_shape = np.array([[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]], dtype=float)
    p = label_point(l_shape)
    assert point_in_polygon(p[None], l_shape)[0]
    # it is well inside: further from every edge than a point hugging the boundary would be
    assert p[0] > 0.3 and p[1] > 0.3


def test_regular_and_irregular_rooms_are_both_drawn(tmp_path):
    irregular = square_room("room_002", 3.0, x0=6.0)
    irregular.length = irregular.width = irregular.dimension_method = None
    prop = simple_property()
    prop.rooms.append(irregular)
    info = render_plan(prop, tmp_path / "plan.png")
    img = Image.open(tmp_path / "plan.png").convert("RGB")
    assert info.rooms_drawn == 2
    assert pixel(img, info, 6.4, 0.5) == ROOM_FILLS[1]  # the second room has its own fill
