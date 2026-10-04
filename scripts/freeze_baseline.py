"""Freeze a run as a before/after record for the fix loop.

    python scripts/freeze_baseline.py --run-dir C:\results\room1 --ground-truth C:\benchmark\room1.json --name room1 --slot before \
        --command "python -m spatialforge process C:\captures\room1 --tier lidar --output C:\results\room1"

Writes evaluation_runs/<slot>/<name>/{manifest.json, evaluation.json, evaluation.md, property.json}. The manifest holds the git SHA,
capture, tier, command, ground-truth file, config, property output, metrics and timestamp. An existing record is never overwritten.
After shipping the fix, re-run `process`, then freeze again with --slot after and use scripts/compare_runs.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spatialforge.evaluation.fix_loop import freeze_baseline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="a `process` output folder (needs property.json)")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--name", required=True, help="a short id for this capture, e.g. room1")
    ap.add_argument("--slot", choices=("before", "after"), default="before")
    ap.add_argument("--root", default="evaluation_runs")
    ap.add_argument("--command", help="the exact process command that produced the run (recorded verbatim)")
    ap.add_argument("--capture", help="the capture path or description to record")
    ap.add_argument("--gates", help="gate thresholds JSON (default: the assignment thresholds)")
    args = ap.parse_args()
    try:
        dest = freeze_baseline(args.run_dir, args.ground_truth, args.name, args.slot, args.root, args.command, args.gates, args.capture, REPO)
    except (OSError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    print(f"Frozen {args.slot} record: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
