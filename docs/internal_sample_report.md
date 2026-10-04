# Internal sample report

> **NOT PHYSICAL GROUND TRUTH BENCHMARK.** These are runs on the three sample LiDAR captures and one video. They support statements about runtime, determinism, drift behaviour and pipeline health. They **cannot** establish dimensional accuracy: no tape or laser measurement exists for any of them.

Artifacts: [evidence/](evidence/README.md). Hardware: a Windows laptop CPU, no GPU. All runs used the one-command interface (`process`), default options, Python 3.11.

## Results

| Run | Status | Total | Rooms | Walls (strong/moderate) | Openings (+unverified) | Ceiling levels / rooms with height | Damage detections | Production poses |
|---|---|---|---|---|---|---|---|---|
| LiDAR `single_room` | PARTIAL | 92 s | 1 | 9 (7) | 0 (+0) | 0 / 0 | 3 | corrected |
| LiDAR `single_scan_floor_only` | PARTIAL | 137 s | 2 | 26 (18) | 0 (+1) | 0 / 0 | 1 | original |
| LiDAR `single_scan_with_ceiling` | PARTIAL | 177 s | 3 | 26 (23) | 1 (+1) | 2 / 2 | 1 | original |
| Video, `single_room` `rgb.mp4` | PARTIAL | 414 s | 0 | 1 (1) | 0 (+0) | 0 / 0 | 1 | sfm_scaled |

Every run is PARTIAL: the structure backend closes only some wall loops (for example 3 of 7 recovered walls are in no room on `single_room`), ceilings are observed only where captured, and recall of openings is unverified. `run_report.json` lists the reason for each in structured warnings.

**Where the time goes** (LiDAR, seconds): frontend (reconstruction and drift stage) 45-71, openings 11-16, damage 23-28, geometry 2-49 and structure 1-10 depending on cloud size, rendering and serialization 1-4. Video: SfM and keyframes 252, metric scale (depth model on CPU) 125, the rest under 15.

## Determinism

- Two complete `process` runs on `single_room` (92 s and 114 s wall time) produced **identical** rooms, walls, openings, damage regions, scope items and provenance, and a **byte-identical** `plan.png`. Only the `timing` section and the classification of warning messages (a rule was added between the runs) differ.
- The three drift ablations were run twice; every reported number was reproduced exactly ([drift_ablation.md](drift_ablation.md)).
- The video and photo tiers use COLMAP, which is multi-threaded and **not bit-reproducible**; this is recorded in each run's provenance. Not tested on real data for repeatability here.

## Drift

The correction was accepted on `single_room` and rejected on the two larger scans (floor flatness got worse), so those used the original poses and say so in their reports. No universal improvement is claimed: [drift_ablation.md](drift_ablation.md).

## Real video, honestly

The video was the capture's own `rgb.mp4`: a LiDAR scanning pass (fast sweeps, bare walls, close range), not a walkthrough. SfM registered 24 of 145 keyframes in the largest of six disconnected models (tracking WEAK); metric scale was available but only `moderate` (0.207 m per SfM unit); gravity confidence was low (the vertical axis differs from the camera-path normal by 39 degrees). One wall and no room were recovered, the run is PARTIAL with `VIDEO_TRACKING_WEAK`, and no dimension should be trusted. It says little about how the tier performs on a proper walkthrough, which has **not** been run.

## What these runs do show

- The pipeline is robust on real data: no crash, structured warnings, a valid canonical `property.json` (schema checked), a readable plan, a `run_report.json` with the standard 12 stages and timings.
- The product behaviour is as designed: partial results are labelled, nothing is invented (no room is created from open walls, no ceiling height without an observed ceiling, no metres without a scale).
- Intervals exist but are coarse where uncertainty is large: the single verified opening (a 1.15 m door) has a 0.75-1.55 m interval. Every top-level wall length in the LiDAR runs (9 to 26 per capture) has `interval: null`, although the room-level wall lengths carry intervals; this is a known gap in the "interval on every measurement" requirement.
- Damage detection is a candidate generator: the samples produced 1-3 detections each, 0 concealed-risk flags, and several detections without a metric extent (kept with null width and a warning). Whether they are real damage is NOT MEASURED.

## What they do not show

Dimensional accuracy, opening-width accuracy, ceiling-height accuracy, repeatability across captures, confidence-interval calibration, photo stitching on real rooms, video performance on a walkthrough, damage precision or recall. All are **NOT VALIDATED**; see [benchmark_report.md](benchmark_report.md) and [submission_status.md](submission_status.md).
