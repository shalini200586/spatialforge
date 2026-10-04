"""Stage bookkeeping shared by every tier's orchestration."""

from __future__ import annotations

import time
from typing import Callable

from spatialforge.pipeline.lidar_adapter import StageRecord

_listener: Callable[[str], None] | None = None


def set_stage_listener(fn: Callable[[str], None] | None) -> None:
    """Register a callback told the (raw) name of every stage as it starts: used for console progress."""
    global _listener
    _listener = fn


def notify_stage(name: str) -> None:
    if _listener is not None:
        try:
            _listener(name)
        except Exception:  # progress output must never break a run
            pass


class PipelineFailure(Exception):
    """The run cannot produce a property (invalid capture, no metric reconstruction)."""

    def __init__(self, stage: str, message: str, stages: list | None = None, extra: dict | None = None, code: str | None = None,
                 hint: str | None = None):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message
        self.stages = stages  # StageRecords completed so far, for the run report
        self.extra = extra or {}  # additional run_report entries (e.g. SfM statistics)
        self.code = code  # an explicit error code; otherwise it is derived from the message
        self.hint = hint


class StageTimer:
    def __init__(self, out_stages: list[StageRecord], name: str):
        self.rec = StageRecord(name, "ok")
        out_stages.append(self.rec)

    def __enter__(self):
        notify_stage(self.rec.name)
        self.t0 = time.perf_counter()
        return self.rec

    def __exit__(self, exc_type, exc, tb):
        self.rec.seconds = time.perf_counter() - self.t0
        if exc is not None:
            self.rec.status, self.rec.error = "failed", f"{type(exc).__name__}: {exc}"
        elif self.rec.warnings and self.rec.status == "ok":
            self.rec.status = "warning"
        return False  # never swallow here; callers decide whether a failure is fatal
