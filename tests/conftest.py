"""Shared fixtures."""

import pytest

import spatialforge.config as cfg


@pytest.fixture(autouse=True)
def model_weights_are_installed(monkeypatch):
    """The preflight checks that the metric-depth weights are cached. Tests mock the depth model, so by default pretend they are;
    a test about the missing-weights error sets this to False itself."""
    monkeypatch.setattr(cfg, "depth_weights_present", lambda: True)
