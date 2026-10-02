"""Shared builders: realistic synthetic stage outputs made with the real room/opening modules."""

import numpy as np
from test_lidar_openings import FLOOR, DOOR, cloud_on, make_wall, subsets_of

from spatialforge.lidar.openings import OpeningOptions, detect_openings
from spatialforge.lidar.rooms import build_topology
from spatialforge.pipeline.lidar_adapter import StageOutputs, StageRecord
from spatialforge.pipeline.models import (
    Ceiling, Measurement, Opening, Point2D, Property, Room, Segment, Wall,
)


def wall_dict(w):
    """A Ticket 5 wall record for a WallInput."""
    segs = [{"start": list(a), "end": list(b), "length_m": float(np.hypot(b[0] - a[0], b[1] - a[1]))} for a, b in w.segments]
    start, end = segs[0]["start"], segs[-1]["end"]
    n = w.normal
    return {
        "id": w.id, "start": start, "end": end, "length_m": float(np.hypot(end[0] - start[0], end[1] - start[1])),
        "observed_length_m": float(sum(s["length_m"] for s in segs)),
        "orientation_deg": float(np.degrees(np.arctan2(w.direction[1], w.direction[0])) % 180),
        "orientation_uncertainty_deg": 0.15, "evidence": w.evidence, "position_uncertainty_m": w.position_uncertainty_m,
        "segments": segs, "gaps": w.gaps, "plane": [float(n[0]), 0.0, float(n[1]), float(-w.offset)],
    }


def two_room_walls(partition_door=True, tier="strong"):
    walls = [
        make_wall("south", (0, 0), (10, 0), ((0, 10),), tier), make_wall("east", (10, 0), (10, 5), ((0, 5),), tier),
        make_wall("north", (10, 5), (0, 5), ((0, 10),), tier), make_wall("west", (0, 5), (0, 0), ((0, 5),), tier),
    ]
    walls.append(make_wall("partition", (5, 0), (5, 5), ((0, 2.0), (2.9, 5.0)) if partition_door else ((0, 5),), tier))
    clouds = [cloud_on(walls[i], extent=(0, e)) for i, e in enumerate((10, 5, 10, 5))]
    clouds.append(cloud_on(walls[4], extent=(0, 5), holes=[(2.0, 2.9, 0.0, 2.05)] if partition_door else ()))
    return walls, np.vstack(clouds)


def synthetic_outputs(rooms=True, ceiling=True, openings=True, stage_warnings=(), name="synthetic_capture"):
    if rooms:
        walls, cloud = two_room_walls(partition_door=openings)
    else:  # a lone wall: no closed room can form
        walls = [make_wall("lone", (0, 0), (6, 0), ((0, 2.5), (3.4, 6.0)))]
        cloud = cloud_on(walls[0], holes=[DOOR] if openings else ())
    levels = [{"height_m": 2.4, "interval_m": [2.33, 2.47], "points_xz": _grid(0, 10, 0, 5)}] if ceiling else []
    topo = build_topology(walls, ceiling_levels=levels if rooms else None)
    res = detect_openings(cloud, FLOOR, walls, topo, subsets_of(cloud), OpeningOptions())
    return StageOutputs(
        capture_path=f"C:/data/{name}", capture_name=name,
        device={"model": None, "depth_resolution": "256x192", "rgb_resolution": "1920x1440"},
        validation_warnings=[], frames={"used": 400, "available": 5000, "frame_step": 1, "max_frames": 400},
        pose_source="original",
        drift={"correction_accepted": False, "fallback_reason": "metrics got worse: floor_band_thickness_m",
               "mean_translation_m": 0.05, "max_translation_m": 0.09, "max_rotation_deg": 1.0},
        floor_y_m=-1.4,
        ceiling_levels=[{"height_m": 2.4, "interval_m": [2.33, 2.47], "confidence": 0.6}] if ceiling else [],
        wall_dicts=[wall_dict(w) for w in walls], topology=topo, openings=res, stage_warnings=list(stage_warnings),
        stages=[StageRecord("capture_validation", "ok", 0.1), StageRecord("structural_walls", "ok", 0.2),
                StageRecord("room_topology", "ok", 0.01), StageRecord("openings", "ok", 0.3)],
        parameters={"min_confidence": 2, "voxel_size_m": 0.02},
    )


def _grid(x0, x1, z0, z1, step=0.03):
    x, z = np.meshgrid(np.arange(x0, x1, step), np.arange(z0, z1, step))
    return np.column_stack([x.ravel(), z.ravel()])


def meas(v, unit="m", low=None, high=None, q=None):
    return Measurement(v, unit, low, high, None, q)


def square_room(rid="room_001", size=4.0, x0=0.0, z0=0.0, wall_ids=("w1", "w2", "w3", "w4")):
    poly = [Point2D(x0, z0), Point2D(x0 + size, z0), Point2D(x0 + size, z0 + size), Point2D(x0, z0 + size)]
    return Room(rid, poly, meas(size * size, "m2", size * size - 1, size * size + 1, "strong"),
                meas(4 * size, "m", 4 * size - 1, 4 * size + 1, "strong"), list(wall_ids),
                [meas(size) for _ in range(4)], Ceiling(True, meas(2.4, "m", 2.3, 2.5), False, 0.8), "strong",
                length=meas(size), width=meas(size), dimension_method="opposite_wall_distances")


def simple_wall(wid="w1", a=(0, 0), b=(4, 0), tier="strong"):
    L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    return Wall(wid, Point2D(*a), Point2D(*b), meas(L, q=tier), meas(L, q=tier), meas(0.0, "deg", -0.1, 0.1, tier), tier,
                meas(0.03), [Segment(Point2D(*a), Point2D(*b), meas(L, q=tier))])


def simple_opening(oid="opening_001", wall="w1", otype="door"):
    return Opening(oid, wall, otype, meas(0.9, "m", 0.85, 0.95, "moderate"), Point2D(2, 0), Point2D(1.55, 0), Point2D(2.45, 0),
                   "moderate", "moderate", "strong", height=meas(2.05), sill_height=meas(0.0), room_ids=["room_001"],
                   connects=["room_001", "unmodelled"])


def simple_property(**kw):
    walls = [simple_wall(f"w{i + 1}", a, b) for i, (a, b) in enumerate([((0, 0), (4, 0)), ((4, 0), (4, 4)), ((4, 4), (0, 4)), ((0, 4), (0, 0))])]
    base = dict(capture={"tier": "lidar", "source": {"name": "x"}}, status="partial", rooms=[square_room()], walls=walls,
                openings=[simple_opening()], warnings=["a warning"])
    base.update(kw)
    return Property(**base)
