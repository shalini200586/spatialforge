"""Benchmark and evaluation tooling: compare a SpatialForge result with hand-measured ground truth, gate it, report it.

Nothing here runs a reconstruction. It reads `property.json` / `run_report.json` and a ground-truth file. A metric with no
ground truth behind it is reported as NOT EVALUATED, never as a pass.
"""

from spatialforge.evaluation.benchmark import evaluate, evaluate_repeat  # noqa: F401
from spatialforge.evaluation.models import GateConfig, GroundTruth, load_ground_truth  # noqa: F401
