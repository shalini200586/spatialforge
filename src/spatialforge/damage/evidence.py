"""RGB evidence per tier, with the camera and pose the property geometry itself was built from.

LiDAR  : frames of rgb.mp4 with the PRODUCTION poses (original or drift-corrected, whichever the pipeline used) and the capture's
         RGB intrinsics.
Video  : the SfM keyframes with their metric, gravity-aligned poses and the COLMAP camera (focal length + radial term).
Photos : each room's registered photos; the room-local poses are composed with the room's placement in the property frame.

All providers are lazy: images are only decoded when the analysis asks for them. Frames or photos are sampled evenly and
deterministically, so a long capture costs a bounded amount of time.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from spatialforge.damage.analyze import EvidenceImage
from spatialforge.damage.geometry import CameraModel

MAX_SIDE = 1280


def evenly_spaced(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    return [items[i] for i in sorted({int(round(k)) for k in np.linspace(0, len(items) - 1, n)})]


def _read_scaled(path: str | Path, max_side: int = MAX_SIDE):
    """RGB uint8 of an image file, shrunk so the long side <= max_side. Returns (array, scale)."""
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise OSError(f"cannot read {path}")
    s = min(1.0, max_side / max(bgr.shape[:2]))
    if s < 1.0:
        bgr = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), s


def planar_to_matrix(pose) -> np.ndarray:
    """4x4 for a planar pose (theta, tx, tz): x' = c x - s z + tx, z' = s x + c z + tz, y unchanged (the plan convention)."""
    th, tx, tz = pose
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, -s, tx], [0, 1, 0, 0], [s, 0, c, tz], [0, 0, 0, 1.0]])


# ---------------- LiDAR ----------------


def lidar_evidence_fn(root: Path, comp, out, max_images: int = 24):
    """`comp` is the metric-reconstruction result (frames, production poses, capture context); `out` supplies the floor height."""

    def build() -> list[EvidenceImage]:
        import cv2

        video = Path(root) / "rgb.mp4"
        if out.floor_y_m is None or not video.exists():
            return []
        ctx = comp.ctx
        poses = dict(zip(comp.frames, comp.production_poses))
        w0, h0 = ctx.source_size
        s = min(1.0, MAX_SIDE / max(w0, h0))
        images = []
        frames = list(comp.frames)
        if len(frames) > 12:
            frames = frames[:-3]  # the last frames of a video often cannot be decoded by seeking
        for fid in evenly_spaced(frames, max_images):
            intr = ctx.poses[fid]["intr"] or ctx.fallback_intrinsics
            if intr is None:
                continue
            cam = CameraModel(intr.fx * s, intr.fy * s, intr.cx * s, intr.cy * s, round(w0 * s), round(h0 * s))

            def load(fid=fid):
                cap = cv2.VideoCapture(str(video))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ok, bgr = cap.read()
                finally:
                    cap.release()
                if not ok:
                    raise OSError(f"cannot decode frame {fid}")
                if s < 1.0:
                    bgr = cv2.resize(bgr, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            images.append(EvidenceImage(f"frame_{fid:06d}", "lidar", load, cam, np.asarray(poses[fid], dtype=np.float64),
                                        float(out.floor_y_m), None, int(fid)))
        return images

    return build


# ---------------- video ----------------


def video_evidence_fn(kdir: Path, names: list[str], T_world: dict, camera, out, frame_index: dict | None = None, max_images: int = 40):
    """`camera`: the SfM camera (model, width, height, params); `T_world`: image name -> 4x4 camera-to-property pose (metres)."""

    def build() -> list[EvidenceImage]:
        if out.floor_y_m is None:
            return []
        f, cx, cy = camera.params[0], camera.params[1], camera.params[2]
        k1 = camera.params[3] if camera.model in ("SIMPLE_RADIAL", "RADIAL") and len(camera.params) > 3 else 0.0
        cam = CameraModel(f, f, cx, cy, camera.width, camera.height, float(k1))
        images = []
        for n in evenly_spaced(sorted(names), max_images):
            images.append(EvidenceImage(n, "video", lambda n=n: _read_scaled(Path(kdir) / n, 10 ** 6)[0], cam,
                                        np.asarray(T_world[n], dtype=np.float64), float(out.floor_y_m), None,
                                        (frame_index or {}).get(n)))
        return images

    return build


# ---------------- photos ----------------


def photo_evidence_fn(placed: list[dict], load_rgb, max_per_room: int = 8):
    """`placed`: one dict per PLACED room: {room_id, pose (theta, tx, tz), floor_y, poses {name: 4x4 local}, cameras {name: camera info}}."""

    def build() -> list[EvidenceImage]:
        images = []
        for room in sorted(placed, key=lambda r: r["room_id"]):
            M = planar_to_matrix(room["pose"])
            for n in evenly_spaced(sorted(room["poses"]), max_per_room):
                c = room["cameras"][n]
                k1 = c.params[3] if c.model in ("SIMPLE_RADIAL", "RADIAL") and len(c.params) > 3 else 0.0
                cam = CameraModel(c.params[0], c.params[0], c.params[1], c.params[2], c.width, c.height, float(k1))
                images.append(EvidenceImage(n, "photos", lambda n=n: load_rgb(n), cam, M @ np.asarray(room["poses"][n], dtype=np.float64),
                                            float(room["floor_y"]), room["room_id"], None))
        return images

    return build
