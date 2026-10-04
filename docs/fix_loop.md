# Fix loop

**STATUS: PENDING REAL BENCHMARK.** The infrastructure below is built and tested. No fix has been shipped *because of a measured benchmark failure*, because no tape/laser ground truth and no real benchmark captures exist yet. A before/after result is not invented here; the assignment's full fix-loop credit requires a real one.

## The loop

1. **Benchmark** every capture with `evaluate` (below). Read the gate table; pick the **worst gate** (the largest relative miss of its threshold).
2. **Declare the fix before coding it** with [templates/fix_declaration.md](templates/fix_declaration.md): the gate, the hypothesis for why it fails, the **predicted number** after the fix, and what will be changed.
3. **Freeze the baseline** (`before`) so the starting point can never be edited.
4. **Ship the fix** as one focused commit (readable diff, tests).
5. **Re-run `process`, freeze again** (`after`), and **compare**. Report the prediction versus the result honestly, including a miss.

## Commands (once real data exists)

    # 0. process the capture and evaluate it
    python -m spatialforge process C:\captures\roomA\lidar --tier lidar --output C:\results\roomA_before
    python -m spatialforge evaluate --property C:\results\roomA_before --ground-truth C:\benchmark\roomA\ground_truth.json --output C:\results\roomA_before\evaluation

    # 1. freeze the BEFORE record (git SHA, command, config, metrics are stored)
    python scripts\freeze_baseline.py --run-dir C:\results\roomA_before --ground-truth C:\benchmark\roomA\ground_truth.json ^
        --name roomA --slot before --command "python -m spatialforge process C:\captures\roomA\lidar --tier lidar --output C:\results\roomA_before"

    # 2. ... commit the fix on its own, run the tests ...

    # 3. process again, freeze the AFTER record
    python -m spatialforge process C:\captures\roomA\lidar --tier lidar --output C:\results\roomA_after
    python scripts\freeze_baseline.py --run-dir C:\results\roomA_after --ground-truth C:\benchmark\roomA\ground_truth.json ^
        --name roomA --slot after --command "python -m spatialforge process C:\captures\roomA\lidar --tier lidar --output C:\results\roomA_after"

    # 4. compare: metric, before, after, absolute change, percent change, gate before / after
    python scripts\compare_runs.py --before evaluation_runs\before\roomA --after evaluation_runs\after\roomA --output fix_report.md

    # 5. the readable diff of the fix
    git show <fix-commit>

## What is stored

    evaluation_runs\
        before\<name>\  manifest.json  evaluation.json  evaluation.md  property.json
        after\<name>\   (same)

`manifest.json`: git SHA, capture, tier, command, ground-truth file name, whether that ground truth was a physical measurement, config (gate thresholds and the run's settings), property output, metrics, gate statuses, timestamp, SpatialForge version. Home-directory paths are redacted. A frozen record is **never overwritten** (`FileExistsError`); pick a new `--name`.

If the `before` and `after` runs used different ground-truth files the comparison prints a warning, and a comparison against placeholder ground truth carries the banner **NOT PHYSICAL GROUND TRUTH BENCHMARK**.

## Candidate fixes already visible in the sample evidence (hypotheses, not results)

These come from internal sample runs, not from a benchmark, and are listed only as places to look once real numbers exist:

- LiDAR openings: sample runs report few verified openings and wide width intervals (e.g. one 1.15 m opening with a 0.75-1.55 m interval) - the opening gate is the likeliest worst gate.
- LiDAR rooms: the three samples recovered only 1-3 closed rooms from 9-26 walls each, so room-level area and footprint metrics may be weak until more wall loops close.
- Video/photo scale: monocular scale is typically 10-15% off in earlier checks; the video and photo wall gates (3% and 8%) are the likeliest to fail.
