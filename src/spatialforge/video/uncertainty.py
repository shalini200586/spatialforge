"""Video measurement uncertainty: how much wider than LiDAR, and why.

Lengths measured from a scaled monocular reconstruction carry a RELATIVE error from three independent sources, combined
in quadrature (a length L then has sigma = sigma_rel * L, and an area A has 2 * sigma_rel * A to first order):

  scale   - the global metric scale: robust frame-to-frame spread / sqrt(frames), plus a floor for the unvalidated
            depth model's systematic error (video.scale);
  sfm     - trajectory quality: a base term plus penalties for reprojection error above 0.5 px and for unregistered
            keyframes (an incomplete trajectory drifts more);
  depth   - dense-depth inconsistency: base term plus the share of depth points that no second keyframe confirms.

Wall-fit residual and room-topology quality are NOT repeated here: the shared structural stages already put them in
the per-measurement intervals, and the adapter widens THOSE intervals by this relative sigma (in quadrature), so
evidence-poor geometry stays wide. The coefficients below are engineering judgements, documented, not calibrated
against ground truth; the output says so.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UncertaintyModel:
    sfm_base: float = 0.01
    sfm_reproj_coeff: float = 0.02  # per pixel above 0.5 px
    sfm_unregistered_coeff: float = 0.10  # times the unregistered fraction
    depth_base: float = 0.02
    depth_removed_coeff: float = 0.10  # times the multi-view-removed fraction
    k: float = 2.0  # half-width in sigmas


@dataclass
class VideoUncertainty:
    scale_rel_sigma: float
    sfm_rel_sigma: float
    depth_rel_sigma: float
    total_rel_sigma: float
    k: float

    def to_dict(self) -> dict:
        return {"metric_scale_relative_sigma": round(self.scale_rel_sigma, 4), "sfm_relative_sigma": round(self.sfm_rel_sigma, 4),
                "depth_relative_sigma": round(self.depth_rel_sigma, 4), "total_relative_sigma": round(self.total_rel_sigma, 4),
                "interval_k_sigma": self.k,
                "note": "relative 1-sigma on lengths (areas: x2); intervals are +/-k sigma added in quadrature to each "
                        "stage's own interval. Engineering estimates, not calibrated against ground truth."}


def propagate(scale_sigma_rel: float, reprojection_px: float | None, registered_ratio: float,
              multiview_removed_fraction: float, model: UncertaintyModel | None = None) -> VideoUncertainty:
    m = model or UncertaintyModel()
    sfm = m.sfm_base + m.sfm_reproj_coeff * max(0.0, (reprojection_px or 0.0) - 0.5) + \
        m.sfm_unregistered_coeff * max(0.0, 1.0 - registered_ratio)
    depth = m.depth_base + m.depth_removed_coeff * max(0.0, multiview_removed_fraction)
    total = (scale_sigma_rel ** 2 + sfm ** 2 + depth ** 2) ** 0.5
    return VideoUncertainty(scale_sigma_rel, sfm, depth, total, m.k)
