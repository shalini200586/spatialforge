"""Read the pixel size of an MP4 video using only the standard library."""

from __future__ import annotations

import struct
from pathlib import Path


def _child_boxes(fh, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        size, kind = struct.unpack(">I4s", fh.read(8))
        header = 8
        if size == 1:  # 64-bit size follows
            size = struct.unpack(">Q", fh.read(8))[0]
            header = 16
        elif size == 0:  # box runs to the end of its parent
            size = end - pos
        if size < header:
            return
        yield kind, pos + header, pos + size
        pos += size


def _find(fh, start: int, end: int, path: list[bytes]):
    for kind, s, e in _child_boxes(fh, start, end):
        if kind == path[0]:
            if len(path) == 1:
                yield s, e
            else:
                yield from _find(fh, s, e, path[1:])


def read_video_size(path: Path) -> tuple[int, int]:
    """Return (width, height) from the video track header. Raises ValueError if unreadable."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            for start, end in _find(fh, 0, size, [b"moov", b"trak", b"tkhd"]):
                # tkhd ends with: matrix (36 bytes), width and height (16.16 fixed point each).
                fh.seek(end - 8)
                w, h = struct.unpack(">II", fh.read(8))
                w, h = w >> 16, h >> 16
                if w > 0 and h > 0:  # audio tracks have 0x0
                    return w, h
    except (OSError, struct.error) as exc:
        raise ValueError(f"could not read {path.name}: {exc}") from exc
    raise ValueError(f"no video track found in {path.name}")
