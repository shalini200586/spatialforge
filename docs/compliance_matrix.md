# Assignment compliance matrix

Statuses: **PASS** (implemented and backed by evidence listed), **PARTIAL** (implemented, but incomplete or only partly evidenced), **NOT VALIDATED** (implemented or measurable, but no real benchmark data exists, so no claim), **MISSING** (not present; needs external data or work).

Evidence paths are relative to the repository. `evidence/` means [docs/evidence/](evidence/README.md): internal sample runs on three LiDAR captures and one video, **not** a ground-truth benchmark. "Tests" means `python -m pytest -q` (see [submission_status.md](submission_status.md) for the latest result). No row is PASS on the strength of an accuracy claim: no tape/laser ground truth exists yet.

## Capture

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Stock capture protocol | One-page protocol for all three tiers, naming the stock LiDAR route (Stray Scanner-format export) | `docs/capture_protocol.md` | the document | PARTIAL | Written, not yet followed in the field by a non-engineer. The LiDAR app route is inferred from the sample's file layout; confirm against the app version used. |
| Device matrix | Tier / hardware / data / accuracy status / limitations | `docs/device_matrix.md` | the document | PASS | Document complete; the accuracy column honestly says NOT VALIDATED for every tier. |

## Input tiers

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Photos: 2-8 stills per room, no poses or depth | Room folders, joint COLMAP SfM, per-room metric scale from a monocular depth model, cross-room stitching | `src/spatialforge/photos/` | `tests/test_photo_pipeline.py` (synthetic SfM and depth) | PARTIAL | Software path only, on synthetic data. **No real photo set has been run.** |
| Video: handheld walkthrough | Keyframes, PyCOLMAP SfM, monocular metric depth for scale, shared structure backend | `src/spatialforge/video/` | `tests/test_video_pipeline.py` (mocked); `evidence/video_single_room_rgb/` (one real run) | PARTIAL | The one real run used a LiDAR scan-pass video, not a walkthrough: PARTIAL result, weak tracking (24/145 keyframes), 1 wall, 0 rooms. |
| LiDAR: depth + poses + intrinsics | Validation, metric reconstruction, drift correction with safe fallback, planes, walls, rooms, openings | `src/spatialforge/lidar/`, `pipeline/lidar_adapter.py` | `evidence/lidar_*` (3 real captures); tests | PASS | Executes end to end on real captures (PARTIAL results by design). Dimensional accuracy NOT VALIDATED. |

## Output contract

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Per-room walls | `rooms[].wall_ids` and `wall_lengths` plus top-level `walls` | `pipeline/models.py`, `lidar/rooms.py` | `evidence/lidar_*/property.json` | PARTIAL | Samples recovered 1-3 closed rooms from 9-26 walls; walls outside closed rooms are listed separately. |
| Dimensions | Room length x width (rectangular rooms), perimeter, wall lengths | `lidar/rooms.py` | property.json | PARTIAL | Irregular rooms get no length x width; accuracy NOT VALIDATED. |
| Ceiling height | Ceiling plane fit per level, assigned to rooms | `lidar/planes.py` | `evidence/lidar_with_ceiling` (2 ceiling levels, 2 rooms with a height) | PARTIAL | Only where a ceiling was captured and observed; none on two samples. Accuracy NOT VALIDATED. |
| Floor area | Polygon area per room, footprint of the placed rooms | `lidar/rooms.py`, `pipeline/models.py` | property.json | PARTIAL | Footprint is an envelope of recovered walls, marked `complete: false`. |
| Openings | 3D-verified doors/windows/openings with width, type, connectivity | `lidar/openings.py` | `evidence/lidar_with_ceiling` (1 verified, 1 unverified) | PARTIAL | Few verified; heights mostly unobserved; recall unmeasured. |
| Whole-property stitched plan | LiDAR: one metric frame. Photos: pairwise stitching and robust placement. Video: single trajectory | `photos/stitch.py`, `pipeline/render.py` | `evidence/*/plan.png`; photo stitching tests | PARTIAL | Photo stitching verified on synthetic data only. |
| Adjacency | Geometric adjacency plus verified-opening connectivity | `pipeline/models.py` | property.json (`adjacent_room_ids`, `connected_room_ids`) | PARTIAL | Evaluated by the harness once ground truth exists. |
| Damage regions | Crack and stain candidates mapped onto wall/floor/ceiling surfaces | `src/spatialforge/damage/` | `evidence/lidar_*` (1-3 detections each); `tests/test_damage*.py` | PARTIAL | Deterministic image processing; precision and recall NOT MEASURED. |
| Damage class | `crack`, `stain` (labels normalised, unknown refused) | `damage/classes.py` | tests | PARTIAL | Only two classes; no labelled data. |
| Metric extent | Width/length/area with interval, only when the outline sits on one surface at a usable range | `damage/geometry.py` | property.json `damage_regions[].width_m` | PARTIAL | Inherits surface uncertainty; null (with a warning) otherwise. |
| Concealed-damage flags | Rule-based risk flags that never assert damage | `damage/rules.py` | `tests/test_damage.py`; samples produced 0 flags | PARTIAL | Rules exercised on synthetic scenes only. |
| Rule IDs | `CDR-001` .. `CDR-006`, each flag names rule, condition, triggering detections | `damage/rules.py` | tests | PASS | |
| Scope line items | Keyed to surfaces; quantity from the metric extent or null | `damage/scope.py` | `evidence/lidar_*` (scope items present) | PASS | Quantities inherit the extent's uncertainty. |
| Confidence interval on every available measurement | Every measurement is `{value, unit, interval, confidence, quality, source_tier}` | `pipeline/models.py` | schema + tests | PARTIAL | Intervals exist where a stage computes one: wall lengths and opening heights are often `null`. Never invented. |
| One command per capture | `python -m spatialforge process INPUT --tier lidar\|video\|photos\|auto --output OUT` | `cli.py`, `pipeline/product.py` | `evidence/` runs; production tests | PASS | |
| JSON | `property.json`, schema v1.0, validated in tests | `schema/property.schema.json` | `tests/test_pipeline_models.py` | PASS | Schema defined by SpatialForge, not an official one. |
| Rendered plan | `plan.png` with dimensions, openings, PARTIAL badge | `pipeline/render.py` | `evidence/*/plan.png` | PASS | |

## Benchmark requirements

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| 3+ room property plus connector | Harness and ground-truth format ready | `benchmark/`, `src/spatialforge/evaluation/` | none | MISSING | Needs a real property. |
| Furnished room with damage | Damage layer and `damage[]` ground-truth fields ready | `damage/`, `benchmark/ground_truth.schema.json` | none | MISSING | Needs a real furnished damaged room. |
| Same property, all three tiers | `scripts/reproduce.ps1` runs each supplied tier and evaluates | `scripts/reproduce.ps1` | none | MISSING | Needs one property captured as photos, video and LiDAR. |
| Repeat capture | `evaluate --repeat` | `evaluation/benchmark.py` | unit tests only | MISSING | Needs a second capture. |
| Tape/laser ground truth | `ground_truth.schema.json`, `example_ground_truth.json` (placeholder) | `benchmark/` | none | MISSING | The example is labelled NOT PHYSICAL GROUND TRUTH BENCHMARK. |

## Gates

Every gate is implemented in the harness (`benchmark/gates.json`, [benchmark/README.md](../benchmark/README.md)). None has been evaluated on real data.

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Opening detection | matched / missed / phantom counts, recall | `evaluation/benchmark.py` | tests | NOT VALIDATED | No assignment threshold supplied (`null`): numbers only. |
| Opening width <= 0.02 m on >= 85% | `opening_width` gate | same | tests | NOT VALIDATED | Sample opening interval is 0.75-1.55 m: expect FAIL until measured. |
| Ceiling height <= 0.015 m | `ceiling_height` gate | same | tests | NOT VALIDATED | |
| Repeatability (ceiling spread <= 0.01 m; wall <= 0.01 m or 0.5%) | `repeat_ceiling`, `repeat_wall` | same | tests | NOT VALIDATED | |
| Drift accountability | Pose refinement with accept/fallback, before/after metrics, recorded in every report | `lidar/drift.py`, `docs/drift_ablation.md` | `evidence/drift/` (3 captures) | PASS | Accountability, not accuracy: correction accepted on 1 of 3 captures, fell back to original poses on 2. |
| Photo whole-property stitching | `photo_stitching` gate, overlap count | `photos/stitch.py`, `evaluation/benchmark.py` | synthetic tests | NOT VALIDATED | |
| Video wall length <= 3% | `video_wall_length` gate | `evaluation/benchmark.py` | tests | NOT VALIDATED | Earlier scale checks were 5-16% off: expect FAIL. |
| Photo wall length <= 8% | `photo_wall_length` gate | same | tests | NOT VALIDATED | |
| Photo footprint <= 8% | `photo_footprint` gate | same | tests | NOT VALIDATED | |
| Confidence calibration | interval coverage rate and mean width | same | tests | NOT VALIDATED | No coverage threshold supplied (`null`); intervals are engineering estimates, not calibrated. |

## Comparison with a consumer scanning app

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Consumer app, two rooms | CSV importer and template | `evaluation/incumbent.py`, `benchmark/incumbent_template.csv` | template only | MISSING | No export supplied; no app is named. |
| Dimension-by-dimension table | Win / tie / loss per dimension, beat-or-tie percentage | `evaluation/incumbent.py` | tests | PARTIAL | Generator works; no real rows. |

## Fix loop

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Infrastructure (declaration, baseline freeze, comparison) | `freeze_baseline.py`, `compare_runs.py`, `fix_declaration.md` | `scripts/`, `evaluation/fix_loop.py`, `docs/fix_loop.md` | tests | PASS | |
| Worst gate, hypothesis, predicted number | Template ready | `docs/templates/fix_declaration.md` | none | MISSING | Needs real benchmark results. |
| Shipped fix, before, after, readable diff | Not done | - | none | MISSING | **STATUS: PENDING REAL BENCHMARK.** Not fabricated. |

## Process

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Incremental Git history | One commit per ticket, on feature branches | git history | `git log --oneline --decorate --graph --all` | PASS | LiDAR validation to production hardening, plus this finalisation. |

## Deliverables

| Requirement | Implementation | Primary file(s) | Generated artifact / evidence | Status | Limitation / note |
|---|---|---|---|---|---|
| Compliance matrix | this file | `docs/compliance_matrix.md` | - | PASS | |
| Capture route | see Capture | `docs/capture_protocol.md` | - | PARTIAL | Not field-tested. |
| Device matrix | see Capture | `docs/device_matrix.md` | - | PASS | |
| Repo and README | | `README.md` | - | PASS | |
| Reproduction bundle | `scripts/setup.ps1`, `smoke_test.ps1`, `reproduce.ps1`, `build_submission.py` | `scripts/` | `dist/spatialforge_submission/` | PARTIAL | Contains code, docs and sample evidence; **no real raw benchmark data**. |
| Benchmark report | Template filled with available evidence only | `docs/benchmark_report.md` | - | PARTIAL | Real sections say NOT YET MEASURED. |
| Fix-loop bundle | Infrastructure and instructions | `docs/fix_loop.md`, `scripts/` | - | PARTIAL | No real fix. |
| Technical report | Markdown source, within the length limit | `docs/technical_report.md` | see [submission_status.md](submission_status.md) for PDF status | PARTIAL | PDF only if a converter exists locally. |
| Raw benchmark data | Directory template | `benchmark/raw/README.md` | none | MISSING | Needs external data. |
