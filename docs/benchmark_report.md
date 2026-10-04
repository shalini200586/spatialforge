# Benchmark report

**Status: REAL BENCHMARK NOT YET MEASURED.** No tape/laser ground truth, no real photo set, no real walkthrough video, no repeat capture and no consumer-app export exist yet. This report is the structure the real results go into; sections without real data say so and nothing is filled with an estimate. Everything labelled *internal sample* is **NOT PHYSICAL GROUND TRUTH BENCHMARK** (see [internal_sample_report.md](internal_sample_report.md)).

## Dataset

| Item | Required by the assignment | Status |
|---|---|---|
| Multi-room property (3+ rooms plus a connector) | yes | NOT YET MEASURED (not captured) |
| Furnished room with real damage | yes | NOT YET MEASURED (not captured) |
| Same property captured with all three tiers | yes | NOT YET MEASURED |
| Repeat capture of the same room/property | yes | NOT YET MEASURED |
| Consumer-app scan of two rooms | yes | NOT YET MEASURED |
| Internal sample data (not a benchmark) | no | 3 LiDAR captures (1-3 rooms recovered each) and the `rgb.mp4` of one of them: [evidence/](evidence/README.md) |

## Capture hardware

NOT YET MEASURED for the benchmark. Intended: iPhone 15 or newer (photos, video) and a LiDAR-equipped iPhone/iPad Pro with a stock capture app (LiDAR), per [capture_protocol.md](capture_protocol.md) and [device_matrix.md](device_matrix.md). The device model of the sample LiDAR captures is not recorded in the files SpatialForge reads (depth maps are 256 x 192, RGB 1920 x 1440).

## Ground-truth procedure

Procedure defined, not yet executed. Laser distance meter or steel tape, millimetre reading, every wall, ceiling height, floor area, opening width and room adjacency recorded in `ground_truth.json` ([../benchmark/README.md](../benchmark/README.md)) with `physical_measurement: true`. The placeholder file `benchmark/example_ground_truth.json` is labelled NOT PHYSICAL GROUND TRUTH BENCHMARK in every report built from it.

## Tier results

### LiDAR

Accuracy: **NOT YET MEASURED.** Internal sample (execution only): 3 captures, all PARTIAL, 92-177 s each, deterministic; 1-3 rooms and 9-26 walls recovered; one verified opening (1.15 m, interval 0.75-1.55 m) on one capture; ceilings observed on one capture (2.41 m and 3.02 m, two levels). These numbers have no ground truth behind them.

### Video

Accuracy: **NOT YET MEASURED.** Internal sample: one real run on a LiDAR scanning-pass video gave PARTIAL with weak tracking (24 of 145 keyframes), 1 wall, 0 rooms, metric scale `moderate`. Not a walkthrough. Earlier checks of the depth-derived scale against ARKit poses on sample footage: +12% to +16% (oracle poses), -12% and +5% (two real-SfM runs): scale quality, not a wall-length measurement.

### Photos

**NOT YET MEASURED.** No real photo set has been run. Software behaviour is covered by synthetic tests only.

## Gates

Evaluated with `python -m spatialforge evaluate` against real ground truth only. Current state of every gate:

| Gate | Threshold | Result |
|---|---|---|
| Opening detection | recall/phantom limits not supplied (reported only) | NOT EVALUATED |
| Opening width | <= 0.02 m on >= 85% of matched openings | NOT EVALUATED (no ground truth) |
| Ceiling height | <= 0.015 m per room | NOT EVALUATED |
| Repeatability: ceiling spread | <= 0.01 m | NOT EVALUATED |
| Repeatability: wall | <= 0.01 m or <= 0.5% | NOT EVALUATED |
| Photo wall length | <= 8% | NOT EVALUATED |
| Video wall length | <= 3% | NOT EVALUATED |
| Photo footprint | <= 8% | NOT EVALUATED |
| Photo whole-property stitching | all rooms placed, no overlap | NOT EVALUATED |
| Drift accountability | pose decision recorded | evidence available (see Drift ablation) |
| Confidence calibration | coverage limit not supplied (reported only) | NOT EVALUATED |

### Opening gate

NOT YET MEASURED. Expectation (not a result): the one verified sample opening has a 0.8 m wide interval, so a 2 cm gate would not be met without a tighter jamb measurement.

### Ceiling gate

NOT YET MEASURED.

### Repeatability

NOT YET MEASURED (no repeat capture). Within a single capture, repeated `process` runs on `single_room` give identical geometry (internal determinism check), which is not the same as repeat-capture repeatability.

### Confidence calibration

NOT YET MEASURED. The metric exists: share of ground-truth values falling inside the predicted interval, plus mean interval width; measurements with no interval are counted separately. Intervals and `confidence` scores are engineering estimates, not calibrated probabilities.

### Photo whole-property stitch

NOT YET MEASURED (synthetic tests only).

### Drift ablation

Available (internal sample): accepted on 1 of 3 captures, original poses kept on 2 because floor flatness got worse. See [drift_ablation.md](drift_ablation.md). Accountability only, not accuracy.

### Runtime

Internal sample, CPU laptop: LiDAR 92-177 s per capture; video 414 s for a 145-keyframe video. Stage timings are in every `run_report.json`. Real-benchmark runtimes NOT YET MEASURED.

### Damage

NOT YET MEASURED against labelled or measured damage. Internal sample: 1-3 candidate detections per LiDAR capture, 0 concealed-risk flags, some detections without a reliable metric extent. Precision and recall unknown.

### Consumer app head-to-head

NOT YET MEASURED. `python -m spatialforge compare-incumbent` and `benchmark/incumbent_template.csv` are ready (two rooms, dimension by dimension, SpatialForge error vs incumbent error, win/tie/loss, beat-or-tie percentage). No app is named and no export has been supplied.

## Known failures

- Video on LiDAR scanning-pass footage: weak tracking, one wall (above).
- Opening intervals too wide for a 2 cm claim; opening recall unmeasured.
- Only partial room topology on all samples.
- Top-level wall lengths carry no interval on the LiDAR samples (room-level ones do).
- Drift correction not usable on 2 of 3 samples (safe fallback kept the original poses).
- Anything on real photos and real walkthroughs: unknown.
