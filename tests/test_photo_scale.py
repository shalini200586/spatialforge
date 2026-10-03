"""Per-room metric scale with only 2-8 photos."""

import numpy as np
import pytest

from spatialforge.photos.room import PhotoRoomOptions
from spatialforge.video.scale import FrameScale, estimate_metric_scale

OPTS = PhotoRoomOptions().scale


def frames(scales, used=200):
    return [FrameScale(f"img_{i:02d}.jpg", used + 20, used, s, 0.05) for i, s in enumerate(scales)]


def test_photo_scale_options_are_for_few_images():
    assert OPTS.min_frames == 2 and OPTS.min_frames_for_moderate == 3 and OPTS.allow_strong is False
    assert OPTS.model_floor_rel >= 0.10  # never below the unvalidated-model floor


def test_two_agreeing_photos_give_a_scale_but_only_a_weak_one():
    e = estimate_metric_scale(frames([1.20, 1.23]), OPTS)
    assert e.available and e.scale == pytest.approx(1.215, rel=0.02) and e.quality == "weak"
    assert any("only 2 images support the scale" in r for r in e.reasons)
    assert e.frames_used == 2 and e.sigma_rel >= 0.10


def test_three_agreeing_photos_can_be_moderate():
    e = estimate_metric_scale(frames([1.20, 1.23, 1.18]), OPTS)
    assert e.available and e.quality == "moderate" and e.scale == pytest.approx(1.20, rel=0.02)


def test_a_single_photo_or_too_few_correspondences_gives_no_scale():
    assert not estimate_metric_scale(frames([1.2]), OPTS).available
    thin = estimate_metric_scale(frames([1.2, 1.2, 1.2], used=20), OPTS)  # 60 correspondences in total
    assert not thin.available and thin.scale is None and thin.quality == "unavailable"


def test_small_samples_use_the_sample_spread_not_a_meaningless_mad():
    two = estimate_metric_scale(frames([1.0, 1.5]), OPTS)
    assert two.frame_spread_rel == pytest.approx(np.std(np.log([1.0, 1.5]), ddof=1), rel=1e-6)
    assert two.sigma_rel > estimate_metric_scale(frames([1.0, 1.05]), OPTS).sigma_rel  # disagreement shows up in the sigma


def test_one_outlier_photo_does_not_move_a_robust_scale():
    e = estimate_metric_scale(frames([1.20, 1.22, 1.18, 1.21, 1.19, 1.23, 3.4, 1.20]), OPTS)  # 8 photos, one wild
    assert e.available and e.scale == pytest.approx(1.205, rel=0.03)
    assert e.max_frame_scale == pytest.approx(3.4)  # the outlier is still reported, just not trusted


def test_photos_that_disagree_wildly_give_no_metric_scale():
    e = estimate_metric_scale(frames([0.4, 1.9, 0.7, 3.0]), OPTS)
    assert not e.available and e.scale is None and any("inconsistent" in r for r in e.reasons)
