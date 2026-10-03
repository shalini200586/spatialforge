"""Synthetic walkthrough for the video tier: a ray-cast box room, a camera path, an SfM stand-in under an unknown
similarity, and a mock metric-depth model. No weights, no pycolmap, no real video needed."""

from pathlib import Path

import numpy as np

from spatialforge.video.depth import FunctionDepthBackend
from spatialforge.video.keyframes import Keyframe, KeyframeResult
from spatialforge.video.sfm import SfmCameraInfo, SfmImagePose, SfmResult, SfmRun

ROOM = (5.0, 2.6, 4.0)  # x, y (height), z extents in metres; floor at y = 0, one corner at the origin
DEPTH_SHAPE = (120, 160)
IMAGE_SIZE = (1280, 960)
F_PX = 1000.0  # focal length at IMAGE_SIZE


def camera_pose(position, yaw, pitch=0.0):
    """Camera-to-world (x right, y down, z forward) looking along `yaw` (about +Y) with a little pitch."""
    f = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    up = np.array([0.0, 1.0, 0.0])
    down = -(up - (up @ f) * f)
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, f])
    T[:3, 3] = position
    return T


def walk_path(n=24, seed=3):
    """Cameras 1.4 m above the floor wandering inside the room, turning as they go."""
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = np.array([2.5 + 1.2 * np.cos(a), 1.4 + 0.05 * rng.normal(), 2.0 + 0.9 * np.sin(a)])
        yaw = a * 1.7 + 0.4
        pitch = 0.40 * np.sin(3.1 * a + 0.5) + 0.04 * rng.normal()  # a person looks down at the floor and up at the ceiling
        poses.append(camera_pose(pos, yaw, pitch))
    return poses


def raycast_depth(T_wc, room=ROOM, shape=DEPTH_SHAPE, image_size=IMAGE_SIZE, f=F_PX):
    """Exact z-depth (m) of the inside of an axis-aligned box seen from camera T_wc (pinhole, no distortion)."""
    h, w = shape
    W, H = image_size
    v, u = np.mgrid[0:h, 0:w]
    xn = ((u + 0.5) * W / w - W / 2) / f
    yn = ((v + 0.5) * H / h - H / 2) / f
    dirs = np.stack([xn, yn, np.ones_like(xn)], axis=-1) @ T_wc[:3, :3].T
    c = T_wc[:3, 3]
    lo, hi = np.zeros(3), np.array(room)
    best = np.full((h, w), np.inf)
    for ax in range(3):
        for bound in (lo[ax], hi[ax]):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (bound - c[ax]) / dirs[..., ax]
            p = c + t[..., None] * dirs
            other = [a for a in range(3) if a != ax]
            ok = (t > 0) & np.all((p[..., other] >= lo[other] - 1e-6) & (p[..., other] <= hi[other] + 1e-6), axis=-1)
            best = np.where(ok & (t < best), t, best)
    return best.astype(np.float32)  # direction z-component is 1, so t is the z-depth


class SyntheticWalkthrough:
    """Ground truth + the stand-ins the pipeline injects."""

    def __init__(self, n=24, a=0.37, seed=5, depth_noise=0.12, depth_bias=1.0, room=ROOM):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.Q, self.a, self.room, self.depth_bias = Q, a, room, depth_bias
        self.poses = walk_path(n)
        self.depth_gt = [raycast_depth(T, room) for T in self.poses]
        # per-frame multiplicative error of the mock depth model (the real model varies by tens of percent between frames)
        self.frame_factor = np.exp(rng.normal(0, depth_noise, size=n)) * depth_bias
        self.names = [f"kf_{i:04d}.jpg" for i in range(n)]
        self.rng = rng

    @property
    def true_metres_per_sfm_unit(self):
        return 1.0 / self.a

    def keyframes_fn(self, video, out_dir, opts):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import cv2

        for i, n in enumerate(self.names):
            img = np.full((120, 160, 3), 40 + i, dtype=np.uint8)  # content is irrelevant: depth is mocked per image index
            cv2.imwrite(str(out_dir / n), img)
        keys = [Keyframe(i * 10, i * 0.4, 100.0, 0.05, n) for i, n in enumerate(self.names)]
        return KeyframeResult(keys, len(keys), video_facts={"work_size": IMAGE_SIZE, "fps": 30.0})

    def depth_fn(self, rgb):
        i = int(rgb[0, 0, 0]) - 40  # the image index is stored in the pixel value
        return self.depth_gt[i] * self.frame_factor[i]

    def depth_backend(self):
        return FunctionDepthBackend(self.depth_fn, "mock-metric-depth", metric=True)

    def sfm_fn(self, kdir, work, opts):
        a, Q = self.a, self.Q
        poses, obs, allpts = {}, {}, []
        for name, T in zip(self.names, self.poses):
            R, t = T[:3, :3], T[:3, 3]
            Rcw = R.T @ Q.T
            C = a * Q @ t
            poses[name] = SfmImagePose(name, 0, Rcw, -Rcw @ C)
            d = self.depth_gt[self.names.index(name)]
            h, w = d.shape
            v, u = np.mgrid[0:h:3, 0:w:3]
            z = d[v, u].ravel()
            keep = np.isfinite(z)
            uu, vv, z = (u.ravel()[keep] + 0.5), (v.ravel()[keep] + 0.5), z[keep]
            Xc = np.column_stack([(uu * IMAGE_SIZE[0] / w - IMAGE_SIZE[0] / 2) / F_PX * z,
                                  (vv * IMAGE_SIZE[1] / h - IMAGE_SIZE[1] / 2) / F_PX * z, z])
            Xw = Xc @ R.T + t
            Xs = a * (Xw @ Q.T)
            xy = np.column_stack([uu * IMAGE_SIZE[0] / w, vv * IMAGE_SIZE[1] / h]) + self.rng.normal(0, 0.3, (len(z), 2))
            obs[name] = (xy, Xs)
            allpts.append(Xs)
        cam = SfmCameraInfo("SIMPLE_RADIAL", *IMAGE_SIZE, [F_PX, IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2, 0.0], F_PX, "synthetic")
        n = len(self.names)
        res = SfmResult(n, n, 1.0, int(sum(len(o[0]) for o in obs.values())), 0.5, 8.0, 1, [n], cam, [], "strong",
                        ["synthetic stand-in"])
        return SfmRun(res, poses, obs, np.vstack(allpts))


def make_test_video(path: Path, seconds=6, fps=20, size=(640, 480), moving=True):
    """A small real MP4 (mp4v) of moving texture, enough for the validator and keyframe tests."""
    import cv2

    rng = np.random.default_rng(0)
    base = (rng.random((size[1], size[0] * 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        off = int(i * 3) if moving else 0
        frame = np.dstack([base[:, off:off + size[0]]] * 3)
        vw.write(frame)
    vw.release()
    return path
