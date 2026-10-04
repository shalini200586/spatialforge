"""Compare two frozen runs (before / after a fix): metric, before, after, absolute change, percent change, gate before / after.

    python scripts/compare_runs.py --before evaluation_runs/before/room1 --after evaluation_runs/after/room1 [--output fix_report.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spatialforge.evaluation.fix_loop import compare_runs, render_comparison  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--output", help="write the Markdown table here as well")
    args = ap.parse_args()
    try:
        text = render_comparison(compare_runs(args.before, args.after))
    except (OSError, KeyError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
