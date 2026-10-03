"""Exact synthetic geometry for stitching tests: an apartment of rectangular rooms with known true poses."""

import math

import numpy as np

from spatialforge.photos.stitch import (
    OpeningGeom, RoomGeom, RoomStitchConstraint, apply, compose, invert, rot,
)


def rect(x0, z0, x1, z1):
    return np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], dtype=float)


# true global layout (metres): A | B side by side, C above A, D above B, a corridor E to the right
LAYOUT = {
    "room_001": rect(0, 0, 5, 4),
    "room_002": rect(5, 0, 9, 4),
    "room_003": rect(0, 4, 5, 7),
    "room_004": rect(5, 4, 9, 7),
    "room_005": rect(9, 0, 11, 7),
}
# doorways in the true global frame: (room_a, room_b, left jamb, right jamb)
DOORS = [
    ("room_001", "room_002", (5.0, 1.6), (5.0, 2.5)),
    ("room_001", "room_003", (2.0, 4.0), (2.9, 4.0)),
    ("room_002", "room_004", (6.5, 4.0), (7.4, 4.0)),
    ("room_003", "room_004", (5.0, 5.2), (5.0, 6.1)),
]


def true_pose(i: int):
    """Each room's local frame is a deterministic arbitrary rotation + offset of the global frame."""
    return (math.radians(37.0 * (i + 1) % 360 - 180), 1.7 * i - 3.0, 2.3 - 0.9 * i)  # local -> global


def to_local(pose, pts):
    return apply(invert(pose), pts)


def make_geoms(layout=LAYOUT, doors=DOORS, with_polygons=True):
    geoms = {}
    ids = sorted(layout)
    for i, rid in enumerate(ids):
        P = true_pose(i)
        ops = []
        for k, (ra, rb, l, r) in enumerate(doors):
            if rid in (ra, rb):
                lj, rj = to_local(P, np.array([l, r]))
                ops.append(OpeningGeom(f"opening_{k + 1:03d}", f"wall_{k + 1}", lj, rj, float(np.linalg.norm(rj - lj))))
        poly = to_local(P, layout[rid]) if with_polygons else None
        geoms[rid] = RoomGeom(rid, poly, ops, rank=(3, len(layout) - i))
    return geoms


def true_constraint(a: str, b: str, ids, noise=(0.0, 0.0), quality="strong", evidence="visual", verified=200, ratio=0.5):
    """T_AB = P_A^-1 ∘ P_B maps B's local frame into A's, optionally perturbed (degrees, metres)."""
    Pa, Pb = true_pose(ids.index(a)), true_pose(ids.index(b))
    t = compose(invert(Pa), Pb)
    return RoomStitchConstraint(a, b, math.degrees(t[0]) + noise[0], t[1] + noise[1], t[2], evidence, verified, ratio, None, None,
                                0.10, 1.5, quality)


def global_error(poses: dict, ids, anchor):
    """Max position / rotation error of the solved poses, after expressing the truth relative to the anchor."""
    Pa = true_pose(ids.index(anchor))
    worst_t = worst_r = 0.0
    for r, p in poses.items():
        truth = compose(invert(Pa), true_pose(ids.index(r)))
        worst_t = max(worst_t, math.hypot(p[1] - truth[1], p[2] - truth[2]))
        worst_r = max(worst_r, abs(math.degrees((p[0] - truth[0] + math.pi) % (2 * math.pi) - math.pi)))
    return worst_t, worst_r
