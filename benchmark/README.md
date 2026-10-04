# Benchmark

Tools to measure SpatialForge against **hand-measured ground truth** and apply the assignment gates. Nothing in this folder contains real measurements: `example_ground_truth.json` is a format example with placeholder numbers and is labelled **NOT PHYSICAL GROUND TRUTH BENCHMARK** in every report built from it.

| File | Purpose |
|---|---|
| `ground_truth.schema.json` | schema of the ground-truth file (checked by `evaluate`) |
| `example_ground_truth.json` | a complete example: 3 rooms, walls, ceilings, areas, openings, adjacency, damage |
| `gates.json` | the assignment thresholds: the **one place** they are configured |
| `incumbent_template.csv` | two-room, dimension-by-dimension template for the consumer-app comparison |
| `raw/README.md` | where the real captures go (outside Git) |

## Ground truth (what to measure)

One JSON file per property (`schema: "spatialforge-ground-truth/1"`). Measure with a laser distance meter or steel tape, to the millimetre, and set `"physical_measurement": true` only then.

- every wall length of every room (`rooms[].walls[]`), floor area (or length x width), ceiling height (several points, record the mean);
- every opening width (door/window), the room it is in, and which rooms it connects;
- which rooms are adjacent; the property footprint if measurable;
- `repeat_group` and `capture_id` on each file when the same room was captured more than once;
- measured damage (`damage[]`: class, length, width or area) if you have it.

Everything but `schema`, `property_id` and `rooms` is optional. Anything you leave out is reported **NOT EVALUATED**, never PASS. Add `predicted_id` (`room_001`, `wall_003`, `opening_002`) to pin a ground-truth item to a prediction; without it rooms are matched by floor area (within 25%), walls by length (within 25%), and openings by width (within 0.5 m) and room, all greedily and one-to-one. Items with no acceptable partner stay unmatched (a miss or a phantom), never forced into a pairing.

## Commands

    python -m spatialforge evaluate --property C:\results\roomA --ground-truth C:\benchmark\roomA\ground_truth.json --output C:\results\roomA\evaluation
    python -m spatialforge evaluate ... --repeat C:\results\roomA_capture2          # adds repeatability (A vs B)
    python -m spatialforge evaluate ... --gates my_gates.json --strict               # --strict: exit 1 if any gate FAILS
    python -m spatialforge compare-incumbent --csv C:\benchmark\incumbent.csv --output C:\results\incumbent

`--property` takes a `process` result folder (or its `property.json`; `run_report.json` next to it supplies runtime and drift evidence). `scripts\reproduce.ps1` runs process + evaluate for every tier you supply.

## Metrics

Walls: absolute (m) and percent error. Openings: matched / missed / phantom counts, width error, share within 2 cm (unverified candidates are not counted as detections). Ceiling: absolute error per room. Area and footprint: absolute and percent error. Property: overlapping room pairs, adjacency recall and precision. Repeatability: wall differences, ceiling and area spread between two captures. Confidence: whether the truth falls inside each reported interval, coverage rate, mean interval width (measurements with no interval are counted separately, never as covered). Runtime: total and per stage.

## Gates (statuses: PASS, FAIL, NOT EVALUATED)

| Gate | Threshold (`gates.json`) | Notes |
|---|---|---|
| opening_width | <= 0.02 m on >= 85% of matched openings | ground-truth openings that exist but are unmatched = FAIL |
| ceiling_height | <= 0.015 m in every room | a room with truth but no predicted height = FAIL |
| repeat_ceiling | spread <= 0.01 m | needs `--repeat` |
| repeat_wall | <= 0.01 m **or** <= 0.5% | needs `--repeat` |
| photo_wall_length | <= 8% (aggregate: mean, configurable) | photos tier only |
| video_wall_length | <= 3% (aggregate: mean, configurable) | video tier only |
| photo_footprint | <= 8% | photos tier only |
| photo_stitching | every ground-truth room placed, no room overlap | photos tier only |
| drift_accountability | pose source and pose-refinement decision recorded | LiDAR; an evidence gate, not a measurement |
| opening_detection | recall / phantom limits are `null` in `gates.json` | no number was supplied, so the counts are reported and the gate stays NOT EVALUATED |
| confidence_calibration | coverage limit is `null` | same: coverage and interval width are reported, not gated |

A gate applying to another tier, or without ground truth, is NOT EVALUATED. Set a `null` threshold in a copy of `gates.json` to turn that gate on.
