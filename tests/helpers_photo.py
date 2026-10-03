"""Photo-tier test helpers: image files with EXIF, plus (added below) a synthetic multi-room apartment."""

from pathlib import Path

import numpy as np
from PIL import Image


def write_photo(path: Path, size=(640, 480), orientation=None, make=None, model=None, focal35=None, focal=None,
                timestamp=None, seed=0, fmt="JPEG"):
    """A textured test image, optionally with EXIF tags."""
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    ifd = {}
    if focal35 is not None:
        ifd[0xA405] = int(focal35)
    if focal is not None:
        ifd[0x920A] = float(focal)
    if timestamp:
        ifd[0x9003] = timestamp
    if ifd:
        exif[0x8769] = ifd
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "PNG":
        im.save(path, "PNG")
    else:
        if len(exif):
            im.save(path, "JPEG", exif=exif)
        else:
            im.save(path, "JPEG")
    return path


def make_property_dir(root: Path, rooms: dict[str, int], **kw) -> Path:
    """root/<room>/IMG_xxx.jpg with the given number of photos per room."""
    k = 0
    for name, n in rooms.items():
        for i in range(n):
            write_photo(root / name / f"IMG_{k:03d}.JPG", seed=k, **kw)
            k += 1
    return root


# ===================== synthetic apartment (ray-cast) + mock SfM / depth backends =====================

import math  # noqa: E402

from helpers_stitch import DOORS, LAYOUT  # noqa: E402
from helpers_video import camera_pose  # noqa: E402

from spatialforge.photos.sfm import ImageRecord, PhotoSfmOptions  # noqa: E402
from spatialforge.photos.validator import ImageMeta  # noqa: E402
from spatialforge.video.depth import FunctionDepthBackend  # noqa: E402
from spatialforge.video.sfm import (  # noqa: E402
    SfmCameraInfo, SfmImagePose, SfmResult, SfmRun, classify_tracking,
)

HEIGHT, DOOR_H = 2.6, 2.05
SHAPE = (150, 200)  # depth map; the "photo" is 1280x960
IMAGE_SIZE = (1280, 960)
F_PX = 1000.0


class Apartment:
    """Rectangular rooms with doorway holes cut in the walls; depth is exact, rays pass through doorways into the next room."""

    def __init__(self, layout=LAYOUT, doors=DOORS):
        self.layout, self.doors = layout, doors
        self.rects = {k: (v[0, 0], v[0, 1], v[2, 0], v[2, 1]) for k, v in layout.items()}
        walls = []
        for (x0, z0, x1, z1) in self.rects.values():
            walls += [("z", z0, x0, x1), ("z", z1, x0, x1), ("x", x0, z0, z1), ("x", x1, z0, z1)]
        self.walls = []
        for axis, c, lo, hi in walls:
            holes = []
            for (_, _, l, r) in doors:
                if axis == "x" and abs(l[0] - c) < 1e-6 and abs(r[0] - c) < 1e-6:
                    a, b = sorted([l[1], r[1]])
                elif axis == "z" and abs(l[1] - c) < 1e-6 and abs(r[1] - c) < 1e-6:
                    a, b = sorted([l[0], r[0]])
                else:
                    continue
                if a >= lo - 1e-6 and b <= hi + 1e-6:
                    holes.append((a, b))
            self.walls.append((axis, c, lo, hi, holes))

    def inside_any(self, x, z):
        ok = np.zeros(np.shape(x), dtype=bool)
        for (x0, z0, x1, z1) in self.rects.values():
            ok |= (x >= x0 - 1e-6) & (x <= x1 + 1e-6) & (z >= z0 - 1e-6) & (z <= z1 + 1e-6)
        return ok

    def depth(self, T_wc, shape=SHAPE, image_size=IMAGE_SIZE, f=F_PX):
        h, w = shape
        W, H = image_size
        v, u = np.mgrid[0:h, 0:w]
        xn = ((u + 0.5) * W / w - W / 2) / f
        yn = ((v + 0.5) * H / h - H / 2) / f
        d = np.stack([xn, yn, np.ones_like(xn)], axis=-1) @ T_wc[:3, :3].T
        c = T_wc[:3, 3]
        best = np.full((h, w), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for yy in (0.0, HEIGHT):
                t = (yy - c[1]) / d[..., 1]
                px, pz = c[0] + t * d[..., 0], c[2] + t * d[..., 2]
                ok = (t > 0) & self.inside_any(px, pz)
                best = np.where(ok & (t < best), t, best)
            for axis, coord, lo, hi, holes in self.walls:
                i, j = (0, 2) if axis == "x" else (2, 0)
                t = (coord - c[i]) / d[..., i]
                a = c[j] + t * d[..., j]
                y = c[1] + t * d[..., 1]
                ok = (t > 0) & (a >= lo - 1e-6) & (a <= hi + 1e-6) & (y >= 0) & (y <= HEIGHT)
                for (ha, hb) in holes:
                    ok &= ~((a > ha) & (a < hb) & (y < DOOR_H))
                best = np.where(ok & (t < best), t, best)
        return best.astype(np.float32)

    def surface_points(self, step=0.4):
        """Points on every surface, and each point's surface normal axis (0: x-wall, 1: floor/ceiling, 2: z-wall)."""
        pts, axes = [], []
        for (x0, z0, x1, z1) in self.rects.values():
            xs, zs = np.arange(x0 + step / 2, x1, step), np.arange(z0 + step / 2, z1, step)
            gx, gz = np.meshgrid(xs, zs)
            for yy in (0.0, HEIGHT):
                pts.append(np.column_stack([gx.ravel(), np.full(gx.size, yy), gz.ravel()]))
                axes.append(np.full(gx.size, 1))
        for axis, coord, lo, hi, holes in self.walls:
            a, y = np.meshgrid(np.arange(lo + step / 2, hi, step), np.arange(step / 2, HEIGHT, step))
            a, y = a.ravel(), y.ravel()
            keep = np.ones(a.shape, dtype=bool)
            for (ha, hb) in holes:
                keep &= ~((a > ha) & (a < hb) & (y < DOOR_H))
            a, y = a[keep], y[keep]
            pts.append(np.column_stack([np.full(a.shape, coord), y, a]) if axis == "x" else np.column_stack([a, y, np.full(a.shape, coord)]))
            axes.append(np.full(a.shape, 0 if axis == "x" else 2))
        return np.vstack(pts), np.concatenate(axes)


def room_cameras(apt: Apartment, rid: str, n=8, seed=0):
    """n cameras 1.4 m up, as a careful photographer shoots a room: from near each corner toward the opposite corner (so
    every wall is seen whole), one view aimed at every doorway (so the next room is visible), the rest filling gaps."""
    x0, z0, x1, z1 = apt.rects[rid]
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    inset = min(0.9, 0.22 * (x1 - x0), 0.22 * (z1 - z0))
    corners = [(x0 + inset, z0 + inset), (x1 - inset, z0 + inset), (x1 - inset, z1 - inset), (x0 + inset, z1 - inset)]
    doors = [(np.array(l) + np.array(r)) / 2 for (ra, rb, l, r) in apt.doors if rid in (ra, rb)]
    views = []
    for (px, pz) in corners:
        views.append(((px, pz), math.atan2(cx - px + (cx - px) * 0.0, cz - pz)))  # toward the room centre = the opposite corner's side
    for d in doors:
        px, pz = cx + 0.18 * (x1 - x0), cz - 0.18 * (z1 - z0)
        views.append(((px, pz), math.atan2(d[0] - px, d[1] - pz)))
    k = 0
    while len(views) < n:
        a = 2 * math.pi * k / 4 + 0.7
        views.append(((cx + 0.15 * (x1 - x0) * math.cos(a), cz + 0.15 * (z1 - z0) * math.sin(a)), a + math.pi / 2))
        k += 1
    out = []
    for i, ((px, pz), yaw) in enumerate(views[:n]):
        out.append(camera_pose([px, 1.4, pz], yaw, 0.30 * math.sin(2.3 * i + 0.4) - 0.05))
    return out


class SyntheticPhotoSfm:
    """Stand-in for the COLMAP backend. Each room has its OWN arbitrary SfM frame; two rooms mapped together share a joint
    frame only if their images have verified cross-room matches (derived from true visibility), exactly as COLMAP would."""

    def __init__(self, apt: Apartment | None = None, per_room=8, seed=3, fail_rooms=(), room_depth_bias=None, depth_noise=0.06,
                 label_to_layout=None):
        self.label_map = label_to_layout or {}  # folder name -> layout room id (default: the room's canonical id)
        self.apt = apt or Apartment()
        self.per_room, self.seed = per_room, seed
        self.fail_rooms = set(fail_rooms)
        self.seconds = {}
        self.rng = np.random.default_rng(seed)
        self.bias = room_depth_bias or {}
        self.noise = depth_noise
        self.cams, self.names, self.room_of, self.index = {}, [], {}, {}
        self.depth_gt = {}
        self.frames_noise = {}

    # -- backend protocol --
    def prepare(self, rooms):
        records = {}
        for room in rooms:
            rid = room.canonical_id
            lay = self.label_map.get(room.source_label, rid)
            self.cams[rid] = room_cameras(self.apt, lay, len(room.images), self.seed)
            for k, meta in enumerate(room.images):
                name = f"{rid}/img_{k:02d}.jpg"
                self.index[name] = len(self.names)
                self.names.append(name)
                self.room_of[name] = rid
                self.depth_gt[name] = self.apt.depth(self.cams[rid][k])
                records[name] = ImageRecord(name, rid, meta, "<synthetic>")
        for name in self.names:
            rid = self.room_of[name]
            self.frames_noise[name] = float(np.exp(self.rng.normal(0, self.noise)) * self.bias.get(rid, 1.0))
        self.records = records
        self._visible = None
        return records

    def load_rgb(self, name):
        return np.full((SHAPE[0], SHAPE[1], 3), self.index[name], dtype=np.uint8)

    def depth_backend(self):
        def fn(rgb):
            i = int(rgb[0, 0, 0])
            name = self.names[i]
            return self.depth_gt[name] * self.frames_noise[name]

        return FunctionDepthBackend(fn, "mock-metric-depth", True)

    def _T(self, name):
        rid = self.room_of[name]
        return self.cams[rid][int(name.split("img_")[1][:2])]

    def _visibility(self):
        if self._visible is None:
            P, axes = self.apt.surface_points(0.4)
            self._points = P
            vis = {}
            for n in self.names:
                T = self._T(n)
                Xc = (P - T[:3, 3]) @ T[:3, :3]
                z = Xc[:, 2]
                ok = z > 0.3
                u = F_PX * Xc[:, 0] / np.where(ok, z, 1) + IMAGE_SIZE[0] / 2
                v = F_PX * Xc[:, 1] / np.where(ok, z, 1) + IMAGE_SIZE[1] / 2
                ok &= (u >= 0) & (u < IMAGE_SIZE[0]) & (v >= 0) & (v < IMAGE_SIZE[1])
                gt = self.depth_gt[n]
                ui = np.clip((u * SHAPE[1] / IMAGE_SIZE[0]).astype(int), 0, SHAPE[1] - 1)
                vi = np.clip((v * SHAPE[0] / IMAGE_SIZE[1]).astype(int), 0, SHAPE[0] - 1)
                ok &= np.abs(gt[vi, ui] - z) < 0.10 + 0.03 * z
                # a wall looks different from its two sides: a point seen from the other side is a different feature
                side = np.where(axes == 1, 1, (T[:3, 3][np.where(axes == 0, 0, 2)] - P[np.arange(len(P)), np.where(axes == 0, 0, 2)]) > 0)
                key = np.arange(len(P)) * 2 + side.astype(int)
                vis[n] = set(key[ok].tolist())
            self._visible = vis
        return self._visible

    def pair_stats(self):
        vis = self._visibility()
        out = []
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                shared = len(vis[a] & vis[b])
                if shared == 0:
                    continue
                raw = int(shared * 0.8)
                verified = int(raw * 0.85) if shared >= 12 else int(raw * 0.1)
                out.append({"a": a, "b": b, "raw": raw, "verified": verified})
        return out

    def _frame(self, tag, rids):
        r = np.random.default_rng(sum(map(ord, tag)) * 7919 + self.seed)
        Q = np.linalg.qr(r.normal(size=(3, 3)))[0]
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return float(r.uniform(0.2, 0.9)), Q, r.uniform(-3, 3, 3)

    def describe(self):
        return {"backend": "synthetic", "version": "test"}

    def map_subset(self, names, tag):
        names = list(names)
        rooms = sorted({self.room_of[n] for n in names})
        opts = PhotoSfmOptions()
        if any(r in self.fail_rooms for r in rooms):
            res = SfmResult(len(names), 0, 0.0, 0, None, None, 0, [], None, names, "failure", ["no coherent reconstruction"])
            return SfmRun(res, {})
        stats = {frozenset((p["a"], p["b"])): p for p in self.pair_stats()}
        # which rooms end up in ONE model: connected through verified cross-room pairs
        parent = {r: r for r in rooms}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for (a, b), p in [(tuple(k), v) for k, v in stats.items()]:
            if a in names and b in names and self.room_of[a] != self.room_of[b] and p["verified"] >= 25:
                parent[find(self.room_of[a])] = find(self.room_of[b])
        groups = {}
        for r in rooms:
            groups.setdefault(find(r), []).append(r)
        best = max(groups.values(), key=lambda g: (sum(1 for n in names if self.room_of[n] in g), g[0]))
        keep = [n for n in names if self.room_of[n] in best]
        a, Q, t0 = self._frame(tag, best)
        poses, obs, allpts = {}, {}, []
        for n in keep:
            T = self._T(n)
            R, C = T[:3, :3], T[:3, 3]
            Rcw = R.T @ Q.T
            Cs = a * Q @ (C - t0)
            poses[n] = SfmImagePose(n, 0, Rcw, -Rcw @ Cs)
            d = self.depth_gt[n]
            h, w = d.shape
            vv, uu = np.mgrid[0:h:3, 0:w:3]
            z = d[vv, uu].ravel()
            ok = np.isfinite(z)
            uu, vv, z = uu.ravel()[ok] + 0.5, vv.ravel()[ok] + 0.5, z[ok]
            Xc = np.column_stack([(uu * IMAGE_SIZE[0] / w - IMAGE_SIZE[0] / 2) / F_PX * z, (vv * IMAGE_SIZE[1] / h - IMAGE_SIZE[1] / 2) / F_PX * z, z])
            Xs = a * ((Xc @ R.T + C - t0) @ Q.T)
            xy = np.column_stack([uu * IMAGE_SIZE[0] / w, vv * IMAGE_SIZE[1] / h]) + self.rng.normal(0, 0.3, (len(z), 2))
            obs[n] = (xy, Xs)
            allpts.append(Xs)
        cam = SfmCameraInfo("SIMPLE_RADIAL", *IMAGE_SIZE, [F_PX, IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2, 0.0], F_PX, "synthetic")
        npts = int(sum(len(o[0]) for o in obs.values()))
        q, reasons = classify_tracking(len(keep), len(names), npts, 0.5, opts)
        res = SfmResult(len(names), len(keep), len(keep) / len(names), npts, 0.5, 6.0, len(groups), sorted([sum(1 for n in names if self.room_of[n] in g) for g in groups.values()], reverse=True),
                        cam, sorted(set(names) - set(keep)), q, reasons)
        return SfmRun(res, poses, obs, np.vstack(allpts), {n: cam for n in keep})


def synthetic_property_dir(root: Path, rooms: list[str], per_room=8, size=(640, 480)):
    """Photo folders with the right counts (the pixels are irrelevant: the mock backends use ids, not image content)."""
    return make_property_dir(root, {r: per_room for r in rooms}, size=size)
