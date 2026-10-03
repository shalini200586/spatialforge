"""Synthetic wall images with known damage, and cameras/properties to project it onto."""

import cv2
import numpy as np


def wall_image(w=1280, h=960, seed=0, colour=(222, 216, 204), noise=3.0):
    """A lightly textured, evenly lit wall."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = colour
    tex = cv2.GaussianBlur(rng.normal(0, noise, (h, w)).astype(np.float32), (0, 0), 1.5)
    img += tex[..., None]
    return img


def finish(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def add_crack(img, x0=300, y0=200, length=420, seed=1, thickness=2, depth=95):
    """A meandering dark crack running roughly downward from (x0, y0)."""
    rng = np.random.default_rng(seed)
    pts, x, y = [(x0, y0)], float(x0), float(y0)
    while y < y0 + length:
        y += rng.uniform(12, 22)
        x += rng.normal(0, 9)
        pts.append((int(x), int(y)))
    layer = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.polylines(layer, [np.array(pts, dtype=np.int32)], False, 255, thickness, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8).astype(np.float32) / 255.0
    img -= layer[..., None] * depth
    return pts


def add_straight_line(img, p0=(700, 100), p1=(700, 860), thickness=3, depth=90):
    """A perfectly straight dark line: a door frame, a joint, a shelf edge."""
    layer = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.line(layer, p0, p1, 255, thickness, cv2.LINE_AA)
    img -= (cv2.GaussianBlur(layer, (0, 0), 0.8).astype(np.float32) / 255.0)[..., None] * depth


def add_stain(img, cx=900, cy=700, rx=110, ry=80, delta=(-18, -4, -52), soft=18):
    """A soft-edged brownish-yellow patch (water stain): darker, and shifted toward yellow/brown."""
    layer = np.zeros(img.shape[:2], dtype=np.float32)
    cv2.ellipse(layer, (cx, cy), (rx, ry), 20, 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), soft)
    for c in range(3):
        img[..., c] += layer * delta[c]


def add_shadow(img, x0=100, x1=500, y0=120, y1=520, depth=0.55):
    """A soft brightness drop with the SAME colour (a shadow)."""
    layer = np.zeros(img.shape[:2], dtype=np.float32)
    cv2.rectangle(layer, (x0, y0), (x1, y1), 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 40)
    img *= (1.0 - depth * layer)[..., None]


def add_tile_grid(img, step=210, thickness=3, depth=70):
    """Dark straight tile joints crossing each other: a grid, which merges into one component where the lines meet."""
    layer = np.zeros(img.shape[:2], dtype=np.uint8)
    for x in range(100, img.shape[1], step):
        cv2.line(layer, (x, 0), (x, img.shape[0]), 255, thickness, cv2.LINE_AA)
    for y in range(80, img.shape[0], step):
        cv2.line(layer, (0, y), (img.shape[1], y), 255, thickness, cv2.LINE_AA)
    img -= (cv2.GaussianBlur(layer, (0, 0), 0.8).astype(np.float32) / 255.0)[..., None] * depth


def add_patterned_patch(img, x0=800, y0=500, w=220, h=170, seed=3):
    """A multi-coloured, busy patch: a cushion, a poster, a picture."""
    rng = np.random.default_rng(seed)
    block = (rng.random((h // 12 + 1, w // 12 + 1, 3)) * 255).astype(np.float32)
    img[y0:y0 + h, x0:x0 + w] = cv2.resize(block, (w, h), interpolation=cv2.INTER_NEAREST)[: h, : w]


def add_bright_patch(img, cx=900, cy=500, rx=100, ry=70):
    """A soft, warm, BRIGHTER patch: a lamp, a reflection, a lit window."""
    layer = np.zeros(img.shape[:2], dtype=np.float32)
    cv2.ellipse(layer, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 14)
    img[..., 0] += layer * 25
    img[..., 1] += layer * 20
    img[..., 2] -= layer * 25


def add_black_object(img, x0=820, y0=560, w=200, h=140):
    """A near-black object with sharp edges: a screen, a mug, a bag."""
    img[y0:y0 + h, x0:x0 + w] = (18, 18, 20)


# ---------------- a known room and cameras looking into it ----------------

from helpers_pipeline import simple_property  # noqa: E402
from helpers_video import camera_pose  # noqa: E402

from spatialforge.damage.analyze import EvidenceImage  # noqa: E402
from spatialforge.damage.geometry import CameraModel  # noqa: E402
from spatialforge.pipeline.models import Measurement  # noqa: E402

CAM = CameraModel(1000.0, 1000.0, 640.0, 480.0, 1280, 960)


def room_property(scale_rel=0.0):
    """The 4 x 4 m room of helpers_pipeline (walls w1..w4, a door in w1, ceiling 2.4 m); floor at y = 0.

    scale_rel > 0 widens every wall length to +-scale_rel (what a video or photo property carries for its scale uncertainty)."""
    prop = simple_property()
    for w in prop.walls:
        w.room_ids = ["room_001"]
        if scale_rel:
            L = w.length.value
            w.length = Measurement(L, "m", L * (1 - scale_rel), L * (1 + scale_rel), None, "strong", "video")
    return prop


def wall_view(img, pitch=0.0, pos=(2.0, 1.4, 0.6), yaw=0.0, image_id="frame_000001", tier="lidar", cam=CAM, floor_y=0.0, room=None):
    """An evidence image looking at wall w3 (the wall at z = 4) from (2, 1.4, 0.6): 3.4 m away, centre of the wall at u = 2."""
    return EvidenceImage(image_id, tier, lambda: finish(img.copy()), cam, camera_pose(list(pos), yaw, pitch), floor_y, room, 1)
