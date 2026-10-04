"""The product entry point: `python -m spatialforge process INPUT --tier lidar|video|photos|auto --output OUT`.

run_product() = resolve the tier -> check the output folder -> preflight (dependencies, model weights) -> run the tier into a
staging folder with console progress and a run log -> swap the finished folder into place. Expected problems (bad input, missing
weights, an output folder in the way) become a clear message, a non-zero exit code and, where the output folder can safely be
created, a run_report.json: never a raw traceback (use --debug for one).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from spatialforge import config as cfgmod
from spatialforge.config import ProcessConfig, TIER_CHOICES
from spatialforge.pipeline import logs, outputs
from spatialforge.pipeline.lidar_pipeline import PipelineOptions, PipelineResult, process_capture
from spatialforge.pipeline.messages import ErrorRecord, UserError
from spatialforge.pipeline.provenance import scrub
from spatialforge.pipeline.report import build_report, exit_code
from spatialforge.pipeline.serialization import write_json
from spatialforge.pipeline.tier_detect import detect_tier

LOG = logging.getLogger("spatialforge.pipeline")

_MODULES = {"lidar": ["numpy", "PIL", "open3d"], "video": ["numpy", "PIL", "cv2", "pycolmap", "torch", "transformers"],
            "photos": ["numpy", "PIL", "cv2", "pycolmap", "torch", "transformers"]}
_EXTRA = {"lidar": "pip install -e .", "video": "pip install -e .[video]", "photos": "pip install -e .[video]"}


def resolve_tier(capture, requested: str) -> tuple[str, str | None]:
    """(tier, how it was decided). `auto` detects from the input and refuses to guess."""
    if requested not in TIER_CHOICES:
        raise UserError("TIER_UNKNOWN", f"unknown tier {requested!r}; choose one of {', '.join(TIER_CHOICES)}")
    if requested != "auto":
        return requested, None
    d = detect_tier(capture)
    return d.tier, d.reason


def preflight(tier: str, damage: bool) -> None:
    """Fail early, with the exact fix, if the tier cannot run here: before minutes of work are spent."""
    missing = [m for m in _MODULES[tier] if importlib.util.find_spec(m) is None]
    if missing:
        raise UserError("DEPENDENCY_MISSING", f"the {tier} tier needs Python packages that are not installed: {', '.join(missing)}",
                        f"Install them with:  {_EXTRA[tier]}")
    if tier in ("video", "photos") and not cfgmod.depth_weights_present():
        raise UserError("MODEL_WEIGHTS_MISSING", "the metric-depth model weights are not installed, so metric dimensions cannot be "
                        "estimated for this tier.", cfgmod.model_setup_hint(), stage="metric_scale")


def build_tier(tier: str, cfg: ProcessConfig):
    """(options, stages_fn) for a tier. Imports are local: a tier's heavy libraries load only when it runs."""
    if tier == "photos":
        from spatialforge.photos.pipeline import PhotoOptions, run_photo_stages

        return PhotoOptions(), run_photo_stages
    if tier == "video":
        from spatialforge.video.pipeline import VideoOptions, run_video_stages

        opts = VideoOptions()
        opts.keyframes.target_max = cfg.video_max_keyframes
        opts.max_depth_frames = cfg.video_max_depth_frames
        return opts, run_video_stages
    return PipelineOptions(frame_step=cfg.lidar_frame_step, max_frames=cfg.lidar_max_frames), None  # None: the LiDAR default


def _config_dict(cfg: ProcessConfig, tier: str) -> dict:
    return {"tier": tier, "damage": cfg.damage, "diagnostics": cfg.diagnostics, "overwrite": cfg.overwrite, "debug": cfg.debug,
            "lidar_max_frames": cfg.lidar_max_frames, "lidar_frame_step": cfg.lidar_frame_step,
            "video_max_keyframes": cfg.video_max_keyframes, "video_max_depth_frames": cfg.video_max_depth_frames,
            "models_dir": f"${cfgmod.MODELS_ENV}" if cfgmod.MODELS_ENV in os.environ else "~/.spatialforge/models"}


def _early_failure(capture, out: Path, cfg: ProcessConfig, tier: str | None, detection, err: ErrorRecord, write: bool) -> PipelineResult:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = build_report(tier=tier or cfg.tier, tier_requested=cfg.tier, detection=detection, status="failure", capture=capture,
                          display_output=out, started=now, ended=now, duration_s=0.0, raw_stages=[], property_warning_records=[],
                          errors=[err], counts=None, decisions=[], config=_config_dict(cfg, tier or cfg.tier),
                          files=["run_report.json"] if write else [])
    report = scrub(report)
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "run_report.json", report)
    return PipelineResult("failure", exit_code("failure"), out, None, report, {"run_report": out / "run_report.json"} if write else {})


def run_product(capture: str | Path, output: str | Path, cfg: ProcessConfig | None = None, *, stages_fn=None, options=None) -> PipelineResult:
    """Process one input end to end. Never raises for an expected problem: it returns a failure result with a structured error."""
    cfg = cfg or ProcessConfig()
    out = Path(output)
    tier = detection = None
    try:
        tier, detection = resolve_tier(capture, cfg.tier)
        outputs.check_output(out, cfg.overwrite)
        preflight(tier, cfg.damage)
    except UserError as exc:
        # An output-folder problem must not drop a report into the folder that is in the way. Any other early failure writes a
        # run_report.json when the folder is new or empty (or an approved --overwrite of a previous result).
        folder_free = not out.exists() or (out.is_dir() and not any(out.iterdir()))
        replace_old = cfg.overwrite and out.is_dir() and outputs.looks_like_result(out)
        write = not exc.code.startswith("OUTPUT_") and (folder_free or replace_old)
        if write and not folder_free:
            shutil.rmtree(out)
        return _early_failure(capture, out, cfg, tier, detection, exc.record(), write)

    opts, tier_stages = build_tier(tier, cfg)
    if options is not None:
        opts = options
    stages_fn = stages_fn or tier_stages
    staging = outputs.staging_dir(out)
    cleanup = logs.setup(tier, cfg.damage, staging / "diagnostics" / "run.log", debug=cfg.debug)
    LOG.info("spatialforge process: tier=%s damage=%s diagnostics=%s", tier, cfg.damage, cfg.diagnostics)
    try:
        result = process_capture(capture, staging, opts, stages_fn=stages_fn, tier=tier, damage=cfg.damage, diagnostics=cfg.diagnostics,
                                 display_output=out, tier_requested=cfg.tier, detection=detection, debug=cfg.debug)
    except BaseException as exc:  # Ctrl-C, out of memory ...: leave an unmistakably incomplete folder, never a valid-looking result
        outputs.mark_incomplete(staging, f"{type(exc).__name__}")
        LOG.error("run aborted; incomplete results left in %s", staging)
        raise
    finally:
        cleanup()
    outputs.commit(staging, out)
    result.output_dir = out
    result.files = {k: out / v.name for k, v in result.files.items()}
    return result
