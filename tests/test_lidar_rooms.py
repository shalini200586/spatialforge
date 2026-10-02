"""Room topology tests on synthetic wall sets (no real captures)."""

import json

import numpy as np

from spatialforge.lidar import rooms as rm
from spatialforge.lidar.rooms import RoomOptions, WallInput, build_topology


def W(wid, a, b, tier="strong", unc=0.03, trim=(0.0, 0.0), segments=None):
    """A wall from a to b; `trim` shortens the observed segment at (start, end) by that many metres."""
    a, b = np.array(a, float), np.array(b, float)
    length = np.linalg.norm(b - a)
    d = (b - a) / length
    segs = segments if segments is not None else [(tuple(a + trim[0] * d), tuple(b - trim[1] * d))]
    return WallInput(wid, tier, a, d, unc, segs)


def rect(x0, z0, x1, z1, tier="strong", prefix="w", trim=(0.0, 0.0), unc=0.03):
    return [
        W(f"{prefix}1", (x0, z0), (x1, z0), tier, unc, trim),
        W(f"{prefix}2", (x1, z0), (x1, z1), tier, unc, trim),
        W(f"{prefix}3", (x1, z1), (x0, z1), tier, unc, trim),
        W(f"{prefix}4", (x0, z1), (x0, z0), tier, unc, trim),
    ]


def rotate(walls_pts, deg, origin=(0, 0)):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    R = np.array([[c, -s], [s, c]])
    return [(R @ (np.array(p) - origin) + origin) for p in walls_pts]


# ---------- closure and corners ----------


def test_four_perfect_walls_make_one_rectangular_room():
    t = build_topology(rect(0, 0, 6, 5))
    assert len(t.rooms) == 1
    r = t.rooms[0]
    assert r.id == "room_001"
    assert abs(r.area_m2 - 30.0) < 1e-6
    assert len(r.polygon) == 4
    assert rm.signed_area(r.polygon) > 0  # counter-clockwise
    assert t.graph.corners and all(c.ext_a < 1e-6 and c.ext_b < 1e-6 for c in t.graph.corners)


def test_endpoints_short_by_10cm_infer_corners_and_close():
    t = build_topology(rect(0, 0, 6, 5, trim=(0.1, 0.1)))
    assert len(t.rooms) == 1
    assert abs(t.rooms[0].area_m2 - 30.0) < 0.05
    inferred = [n for n in t.graph.nodes if any(e > 0.05 for e in n.extensions.values())]
    assert len(inferred) == 4
    assert all(abs(max(n.extensions.values()) - 0.1) < 0.02 for n in inferred)
    assert t.rooms[0].inferred_extension_total_m > 0.5


def test_gap_beyond_maximum_extension_does_not_close():
    t = build_topology(rect(0, 0, 6, 5, trim=(0.9, 0.9)))  # every wall stops 0.9 m short of each corner
    assert t.rooms == []
    t = build_topology(rect(0, 0, 6, 5, trim=(0.6, 0.6)))  # 0.6 m is beyond the default 0.5 m cap too
    assert t.rooms == []
    # raising the allowed extension is what would change that, not the data
    loose = build_topology(rect(0, 0, 6, 5, trim=(0.6, 0.6)), RoomOptions(max_corner_extension_m=0.8))
    assert len(loose.rooms) == 1
    # ...and even with a huge cap the 'mostly inferred boundary' filter still refuses a mostly invented room
    huge = build_topology(rect(0, 0, 6, 5, trim=(0.9, 0.9)), RoomOptions(max_corner_extension_m=1.2))
    assert huge.rooms == [] and any("inferred" in r["reason"] for r in huge.rejected_faces)


def test_near_parallel_walls_make_no_corner():
    walls = [W("a", (0, 0), (6, 0)), W("b", (0, 0.5), (6, 0.5 + 6 * np.tan(np.radians(10))))]
    assert rm.find_corners(walls, RoomOptions()) == []


def test_door_sized_collinear_gap_is_preserved_inside_the_room_edge():
    walls = rect(0, 0, 6, 5)
    walls[0] = W("w1", (0, 0), (6, 0), segments=[((0, 0), (2.5, 0)), ((3.5, 0), (6, 0))])  # 1.0 m gap
    t = build_topology(walls)
    assert len(t.rooms) == 1  # the conceptual wall continues across the gap, so the room closes
    edge = next(e for e in t.rooms[0].edges if e.wall_ids == ["w1"])
    assert abs(edge.gap_m - 1.0) < 1e-6
    assert abs(edge.observed_support_fraction - 5.0 / 6.0) < 1e-6  # observed support is kept separate
    graph_edge = next(e for e in t.graph.edges if e.wall_id == "w1")
    assert abs(graph_edge.gap_m - 1.0) < 1e-6


# ---------- weak walls ----------


def test_weak_isolated_wall_and_weak_only_rectangle_do_not_create_rooms():
    assert build_topology([W("x", (0, 0), (4, 0), "weak")]).rooms == []
    assert build_topology(rect(0, 0, 6, 5, tier="weak")).rooms == []  # four weak walls cannot invent a room


def test_weak_wall_can_close_a_room_whose_other_edges_are_strong():
    walls = rect(0, 0, 6, 5)
    walls[2] = W("w3", (6, 5), (0, 5), "weak")
    t = build_topology(walls)
    assert len(t.rooms) == 1
    r = t.rooms[0]
    assert r.uses_weak_walls and r.topology_quality != "strong"
    assert any(e.tier == "weak" for e in r.edges)
    # two weak walls are not allowed to close it
    walls[1] = W("w2", (6, 0), (6, 5), "weak")
    assert build_topology(walls).rooms == []


def test_weak_wall_cannot_subdivide_a_supported_room():
    walls = rect(0, 0, 10, 5) + [W("partition", (5, 0), (5, 5), "weak")]
    t = build_topology(walls)
    assert len(t.rooms) == 1 and abs(t.rooms[0].area_m2 - 50.0) < 0.1
    assert any("subdivide" in r["reason"] for r in t.rejected_faces)


# ---------- several rooms, adjacency, footprint ----------


def two_rooms():
    return rect(0, 0, 10, 5) + [W("partition", (5, 0), (5, 5))]


def test_two_adjacent_rooms_make_two_polygons():
    t = build_topology(two_rooms())
    assert len(t.rooms) == 2
    assert sorted(round(r.area_m2) for r in t.rooms) == [25, 25]
    assert [r.id for r in t.rooms] == ["room_001", "room_002"]  # ordered by centroid X


def test_shared_boundary_gives_geometric_adjacency():
    a, b = build_topology(two_rooms()).rooms
    assert [x["room_id"] for x in a.adjacent] == ["room_002"]
    assert [x["room_id"] for x in b.adjacent] == ["room_001"]
    assert abs(a.adjacent[0]["shared_boundary_m"] - 5.0) < 0.1


def test_outer_footprint_does_not_replace_interior_faces():
    t = build_topology(two_rooms())
    assert all(r.area_m2 < 30 for r in t.rooms)  # no 50 m2 'room'
    assert t.outer_boundary is not None and abs(t.outer_boundary["area_m2"] - 50.0) < 0.1


def test_outline_that_encloses_interior_walls_is_not_a_room():
    # Outer rectangle plus partitions that do not close any inner room: the outline would otherwise be accepted.
    # (p1 stops 1 m short of the top wall, p2 floats 1+ m from everything: neither forms a corner)
    walls = rect(0, 0, 10, 5) + [W("p1", (5, 0), (5, 4.0)), W("p2", (1.5, 2.0), (4.0, 2.0))]
    t = build_topology(walls, RoomOptions(footprint_interior_fraction=0.2))
    assert t.rooms == []
    assert any("property footprint" in r["reason"] for r in t.rejected_faces)


def test_tiny_furniture_rectangle_is_rejected():
    t = build_topology(rect(0, 0, 0.8, 0.8))
    assert t.rooms == []
    assert any("area" in r["reason"] for r in t.rejected_faces)


def test_self_intersecting_polygon_is_rejected():
    bow = np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    assert not rm.is_simple(bow)
    assert rm.is_simple(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]))
    walls = {f"s{i}": W(f"s{i}", bow[i], bow[(i + 1) % 4]) for i in range(4)}
    nodes = [rm.Node(i, bow[i]) for i in range(4)]
    edges = [rm.Edge(i, i, (i + 1) % 4, f"s{i}", "strong", float(np.linalg.norm(bow[(i + 1) % 4] - bow[i])), 1.0, 0.0, 0.0, 0.03)
             for i in range(4)]
    g = rm.Graph(nodes, edges, [], [])
    face = {"nodes": [0, 1, 2, 3], "edges": edges, "area": 0.0}
    info, reason = rm._evaluate_face(face, g, walls, RoomOptions())
    assert info is None and "self-intersecting" in reason


def test_duplicate_room_polygons_are_suppressed():
    a = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 5.0], [0.0, 5.0]])
    assert rm._same_polygon(a, a + 0.05)
    assert not rm._same_polygon(a, a + [0.0, 1.0])
    twin = rect(0.05, 0.05, 6.05, 5.05, prefix="d", tier="moderate")  # the same room measured twice, 5 cm apart
    t = build_topology(rect(0, 0, 6, 5) + twin)
    assert len(t.rooms) == 1
    assert len(t.suppressed_walls) == 4


# ---------- measurements ----------


def test_area_perimeter_and_rectangular_dimensions():
    r = build_topology(rect(0, 0, 6, 5)).rooms[0]
    assert abs(r.area_m2 - 30.0) < 1e-6
    assert abs(r.perimeter_m - 22.0) < 1e-6
    assert sorted(round(x, 3) for x in r.wall_lengths_m) == [5.0, 5.0, 6.0, 6.0]
    assert r.dimension_method == "opposite_wall_distances"
    assert abs(r.length_m - 6.0) < 1e-6 and abs(r.width_m - 5.0) < 1e-6


def test_rotated_rectangle_dimensions_do_not_depend_on_orientation():
    pts = rotate([(0, 0), (6, 0), (6, 5), (0, 5)], 30)
    walls = [W(f"r{i}", pts[i], pts[(i + 1) % 4]) for i in range(4)]
    r = build_topology(walls).rooms[0]
    assert abs(r.area_m2 - 30.0) < 1e-6
    assert abs(r.length_m - 6.0) < 1e-6 and abs(r.width_m - 5.0) < 1e-6


def test_opposite_wall_distance_uses_fitted_lines_not_vertex_noise():
    # Top wall tilted by 1 degree: the distance between the (almost parallel) lines is taken at their middles.
    top_end = (0.0, 5.0 + 6 * np.tan(np.radians(1.0)))
    walls = [W("a", (0, 0), (6, 0)), W("b", (6, 0), (6, 5.1)), W("c", (6, 5.0 + 6 * np.tan(np.radians(1.0))), (0, 5.0)),
             W("d", (0, 5.0), (0, 0))]
    r = build_topology(walls).rooms[0]
    assert r.dimension_method == "opposite_wall_distances"
    assert abs(r.width_m - 5.05) < 0.1


def test_irregular_polygon_reports_area_and_perimeter_without_rectangular_dimensions():
    trap = [(0, 0), (6, 0), (4, 4), (0, 4)]  # one slanted side
    walls = [W(f"t{i}", trap[i], trap[(i + 1) % 4]) for i in range(4)]
    r = build_topology(walls).rooms[0]
    assert abs(r.area_m2 - 20.0) < 1e-6
    assert abs(r.perimeter_m - (6 + np.hypot(2, 4) + 4 + 4)) < 1e-6
    assert r.length_m is None and r.width_m is None and r.dimension_method is None


# ---------- ceilings ----------


def grid(x0, x1, z0, z1, step=0.03):
    x, z = np.meshgrid(np.arange(x0, x1, step), np.arange(z0, z1, step))
    return np.column_stack([x.ravel(), z.ravel()])


def test_ceiling_level_is_assigned_by_spatial_overlap():
    t = build_topology(two_rooms(), ceiling_levels=[
        {"height_m": 2.4, "interval_m": [2.35, 2.45], "points_xz": grid(0, 5, 0, 5)},
        {"height_m": 3.0, "interval_m": [2.9, 3.1], "points_xz": grid(5, 10, 0, 5)},
    ])
    a, b = t.rooms
    assert a.ceiling["ceiling_observed"] and a.ceiling["ceiling_height_m"] == 2.4
    assert b.ceiling["ceiling_observed"] and b.ceiling["ceiling_height_m"] == 3.0
    assert a.ceiling["coverage"] > 0.9


def test_ambiguous_ceiling_overlap_is_reported_not_guessed():
    t = build_topology(rect(0, 0, 6, 5), ceiling_levels=[
        {"height_m": 2.4, "interval_m": None, "points_xz": grid(0, 3.2, 0, 5)},
        {"height_m": 3.0, "interval_m": None, "points_xz": grid(2.8, 6, 0, 5)},
    ])
    c = t.rooms[0].ceiling
    assert c["ambiguous"] and not c["ceiling_observed"] and "ceiling_height_m" not in c
    assert len(c["levels"]) == 2


def test_room_without_ceiling_overlap_reports_not_observed():
    t = build_topology(rect(0, 0, 6, 5), ceiling_levels=[
        {"height_m": 2.4, "interval_m": None, "points_xz": grid(20, 25, 20, 25)}])
    assert t.rooms[0].ceiling["ceiling_observed"] is False and not t.rooms[0].ceiling["ambiguous"]


# ---------- uncertainty, determinism, serialisation ----------


def test_intervals_widen_with_wall_position_uncertainty():
    tight = build_topology(rect(0, 0, 6, 5, unc=0.01)).rooms[0]
    loose = build_topology(rect(0, 0, 6, 5, unc=0.08)).rooms[0]
    width = lambda iv: iv[1] - iv[0]
    assert width(loose.area_interval_m2) > 3 * width(tight.area_interval_m2)
    assert width(loose.length_interval_m) > width(tight.length_interval_m)
    assert tight.area_interval_m2[0] < tight.area_m2 < tight.area_interval_m2[1]
    weak_unc = build_topology(rect(0, 0, 6, 5, unc=0.03, trim=(0.3, 0.3))).rooms[0]  # inferred corners add uncertainty
    assert width(weak_unc.area_interval_m2) > width(build_topology(rect(0, 0, 6, 5, unc=0.03)).rooms[0].area_interval_m2)


def test_topology_is_deterministic():
    walls = two_rooms() + [W("stub", (2, 5), (2, 3.5), "weak")]
    a = rm.topology_to_dict(build_topology(walls))
    b = rm.topology_to_dict(build_topology(list(reversed(walls))))
    assert a == b


def test_results_serialise_to_json_and_geojson():
    t = build_topology(two_rooms(), ceiling_levels=[{"height_m": 2.4, "interval_m": [2.3, 2.5], "points_xz": grid(0, 10, 0, 5)}])
    d = rm.topology_to_dict(t)
    assert json.loads(json.dumps(d)) == d
    r = d["rooms"][0]
    for key in ("id", "polygon", "wall_ids", "area_m2", "area_interval_m2", "perimeter_m", "wall_lengths_m", "length_m",
                "width_m", "ceiling", "adjacent_room_ids", "topology_quality"):
        assert key in r
    assert d["accepted_rooms"] == 2 and d["corner_count"] >= 6
    graph = json.loads(json.dumps(rm.graph_to_dict(t)))
    assert graph["edges"] and graph["corners"]
    gj = json.loads(json.dumps(rm.rooms_to_geojson(t)))
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 2
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
