# Frozen fix-loop runs

`before/<name>/` and `after/<name>/` hold frozen records written by `scripts/freeze_baseline.py`: `manifest.json` (git SHA, capture, tier, command, ground-truth file, config, metrics, timestamp), `evaluation.json`, `evaluation.md` and `property.json`. A record is never overwritten. Compare two with `scripts/compare_runs.py`. See [../docs/fix_loop.md](../docs/fix_loop.md).

**Currently empty: STATUS PENDING REAL BENCHMARK.** No physical ground truth exists yet, so no real before/after record has been frozen.
