# Fix declaration

Write this **before** changing code. Commit it with, or just before, the fix.

| Field | Entry |
|---|---|
| Date | |
| Author | |
| Benchmark property / capture(s) | |
| Tier(s) | lidar / video / photos |
| Ground truth file (and whether it is a physical measurement) | |
| **Worst gate** (name, threshold, current value, how far it misses) | |
| Evidence that it is the worst gate (link to `evaluation.md`) | |
| **Hypothesis** (why does it fail? one cause, falsifiable) | |
| How the hypothesis was checked before coding (diagnostic, plot, ablation) | |
| **Predicted number** after the fix (metric, value, and why that value) | |
| **The fix** (what changes, which files; keep it to one cause) | |
| What could get worse (other gates, runtime, other tiers) and how it is guarded | |
| Tests added or changed | |
| Frozen BEFORE record | `evaluation_runs/before/<name>` (git SHA: ) |

## After the fix

| Field | Entry |
|---|---|
| Fix commit (SHA, readable diff) | |
| Frozen AFTER record | `evaluation_runs/after/<name>` |
| **Result** (metric before -> after, absolute and percent change) | |
| Prediction vs result (hit, partial, miss) and why | |
| Gate before -> after | |
| Other metrics that moved (from `compare_runs.py`) | |
| Decision (keep, iterate, revert) | |
