"""Small, pure metric helpers. Lengths in metres, areas in square metres, percentages in percent."""

from __future__ import annotations

import statistics


def abs_error(pred: float, truth: float) -> float:
    return abs(pred - truth)


def pct_error(pred: float, truth: float) -> float | None:
    """Percent error relative to the truth; None when the truth is zero (no meaningful percentage)."""
    return None if truth == 0 else abs(pred - truth) / abs(truth) * 100.0


def interval_of(measurement: dict | None) -> tuple[float, float] | None:
    """(low, high) of a Measurement dict, or None when the stage reported no interval."""
    if not measurement:
        return None
    iv = measurement.get("interval")
    if not iv or iv.get("low") is None or iv.get("high") is None:
        return None
    return float(iv["low"]), float(iv["high"])


def covered(truth: float, measurement: dict | None) -> bool | None:
    """Does the ground truth fall inside the predicted interval? None when there is no interval to test."""
    iv = interval_of(measurement)
    return None if iv is None else iv[0] <= truth <= iv[1]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {"n": len(values), "mean": round(statistics.fmean(values), 6), "median": round(statistics.median(values), 6),
            "max": round(max(values), 6)}


def aggregate(values: list[float], how: str) -> float | None:
    if how not in ("mean", "median", "max"):
        raise ValueError(f"unknown aggregate {how!r}: use mean, median or max")
    if not values:
        return None
    return max(values) if how == "max" else statistics.median(values) if how == "median" else statistics.fmean(values)


def spread(values: list[float]) -> float | None:
    """max - min across repeat captures; None with fewer than two values."""
    return None if len(values) < 2 else max(values) - min(values)


def interval_stats(pairs: list[tuple[float, dict | None]]) -> dict:
    """Coverage of the ground truth by the predicted intervals, over the measurements that carry an interval."""
    with_iv = [(t, m) for t, m in pairs if interval_of(m) is not None]
    hits = sum(1 for t, m in with_iv if covered(t, m))
    widths = [interval_of(m)[1] - interval_of(m)[0] for _, m in with_iv]
    return {"measurements": len(pairs), "with_interval": len(with_iv), "without_interval": len(pairs) - len(with_iv),
            "covered": hits, "coverage_rate": (hits / len(with_iv)) if with_iv else None,
            "mean_interval_width": round(statistics.fmean(widths), 6) if widths else None}
