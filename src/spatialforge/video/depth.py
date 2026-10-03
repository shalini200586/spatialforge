"""Metric monocular depth backends.

Chosen backend: Depth Anything V2 *Metric-Indoor-Small* (Hypersim fine-tune). Its config declares
`depth_estimation_type: "metric"` with `max_depth: 20`, i.e. the output is in metres, not relative disparity.
It is NOT intrinsics-aware, so its metric scale is only approximately right on an arbitrary phone; that is why the
pipeline never trusts one frame: scale is estimated robustly against SfM over many frames (video.scale) and its
spread is carried into the output intervals.

Weights live outside Git: `python scripts/download_models.py` (cache: %SPATIALFORGE_MODELS% or ~/.spatialforge/models).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

MODEL_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
MODEL_NAME = "Depth Anything V2 Metric-Indoor-Small"


class DepthUnavailable(RuntimeError):
    """The metric depth backend cannot be used (missing dependency or weights)."""


class DepthBackend(Protocol):
    name: str
    metric: bool  # True only when the output is documented as absolute depth in metres

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (h, w) float32 depth in metres (resolution may differ from the input)."""

    def describe(self) -> dict: ...


def models_dir() -> Path:
    env = os.environ.get("SPATIALFORGE_MODELS")
    return Path(env) if env else Path.home() / ".spatialforge" / "models"


class DepthAnythingMetric:
    name = MODEL_NAME
    metric = True

    def __init__(self, cache_dir: Path | None = None, repo: str = MODEL_REPO):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DepthUnavailable("torch and transformers are required for the video tier "
                                   "(pip install -e .[video])") from exc
        self.repo = repo
        self.cache_dir = Path(cache_dir) if cache_dir else models_dir() / "hf"
        try:
            self.processor = AutoImageProcessor.from_pretrained(repo, cache_dir=str(self.cache_dir), local_files_only=True)
            self.model = AutoModelForDepthEstimation.from_pretrained(repo, cache_dir=str(self.cache_dir),
                                                                     local_files_only=True).eval()
        except Exception as exc:
            raise DepthUnavailable(f"model weights not found in {self.cache_dir}. Run: python scripts/download_models.py "
                                   f"({type(exc).__name__}: {exc})") from exc
        self._torch = torch
        cfg = self.model.config
        self.max_depth_m = float(getattr(cfg, "max_depth", 0) or 0)
        if getattr(cfg, "depth_estimation_type", None) != "metric":
            raise DepthUnavailable("the loaded checkpoint is not a metric-depth model (depth_estimation_type != 'metric')")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            out = self.model(**self.processor(images=rgb, return_tensors="pt"))
        return out.predicted_depth[0].numpy().astype(np.float32)

    def describe(self) -> dict:
        snaps = sorted((self.cache_dir / ("models--" + self.repo.replace("/", "--")) / "snapshots").glob("*"))
        return {"model": self.name, "repo": self.repo, "revision": snaps[-1].name if snaps else None,
                "metric": True, "max_depth_m": self.max_depth_m, "parameters_m": round(
                    sum(p.numel() for p in self.model.parameters()) / 1e6, 1), "device": "cpu",
                "license_note": "upstream Depth-Anything-V2 repository: Small models Apache-2.0 (the Base/Large/Giant "
                                "checkpoints are CC-BY-NC-4.0); the Hugging Face card carries no licence field - verify "
                                "before redistribution"}


class FunctionDepthBackend:
    """Wraps any callable rgb -> depth(m). Used by tests and for fixtures; declares itself metric explicitly."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = "function", metric: bool = True):
        self.fn, self.name, self.metric = fn, name, metric

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(rgb), dtype=np.float32)

    def describe(self) -> dict:
        return {"model": self.name, "metric": self.metric}
