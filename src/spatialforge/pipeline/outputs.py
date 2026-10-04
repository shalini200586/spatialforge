"""Output-folder safety: no accidental overwrite, no half-written result presented as complete.

A run writes into a hidden staging folder next to the final one and swaps it into place only when it is finished, so
  * an existing result is never touched unless --overwrite is given, and even then only if it LOOKS like a SpatialForge result
    (a run_report.json, property.json or diagnostics folder) or is empty: `--output C:\\Users\\me` is refused, not deleted;
  * a crash leaves a clearly named, marked `.partial` staging folder, never a property.json that claims success.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from spatialforge.pipeline.messages import UserError

INCOMPLETE_MARKER = "INCOMPLETE.txt"
RESULT_SIGNS = ("run_report.json", "property.json", "plan.png", "diagnostics")


def looks_like_result(d: Path) -> bool:
    return any((d / s).exists() for s in RESULT_SIGNS)


def check_output(out: Path, overwrite: bool) -> None:
    """Raise UserError if `out` may not be used. Does not create or delete anything."""
    if out.exists() and not out.is_dir():
        raise UserError("OUTPUT_NOT_A_DIRECTORY", f"output path exists and is a file: {out}", "Choose another --output folder.")
    if not out.exists() or not any(out.iterdir()):
        return
    if not overwrite:
        raise UserError("OUTPUT_EXISTS", f"output folder is not empty: {out}",
                        "Use --overwrite to replace a previous result, or choose another --output folder.")
    if not looks_like_result(out):
        raise UserError("OUTPUT_NOT_A_RESULT", f"refusing to overwrite {out}: it is not empty and does not look like a SpatialForge "
                        "result (no run_report.json, property.json or diagnostics folder)", "Choose an empty or new --output folder.")


def staging_dir(out: Path) -> Path:
    s = out.parent / f".{out.name}.partial-{os.getpid()}"
    if s.exists():
        shutil.rmtree(s, ignore_errors=True)
    s.mkdir(parents=True)
    return s


def mark_incomplete(staging: Path, reason: str) -> None:
    try:
        (staging / INCOMPLETE_MARKER).write_text(f"This run did not finish: {reason}\nNothing in this folder is a valid result.\n",
                                                 encoding="utf-8")
    except OSError:
        pass


def _replace(src: Path, dst: Path) -> None:
    """os.replace, retried briefly: on Windows an indexer or antivirus can hold a freshly written folder for a moment."""
    for attempt in range(8):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25)


def commit(staging: Path, out: Path) -> None:
    """Swap the finished staging folder into place (replacing an existing result that check_output already approved).
    The old result is moved aside first and only deleted once the new one is in place."""
    out.parent.mkdir(parents=True, exist_ok=True)
    aside = None
    if out.exists():
        aside = out.with_name(f"{out.name}.old-{os.getpid()}")
        _replace(out, aside)
    _replace(staging, out)
    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)
