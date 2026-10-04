# SpatialForge: technical report

*Status as of this submission: the software is complete and tested; the real tape/laser benchmark has not been run. Every accuracy statement below is either an engineering estimate or an internal sample observation, and is labelled as such.*

## 1. Executive summary

SpatialForge turns one capture of a property into a dimensioned floor plan (`plan.png`), a machine-readable description (`property.json`, schema 1.0) and visible-damage findings, with one command: `python -m spatialforge process INPUT --tier lidar|video|photos --output OUT`. Three input tiers (LiDAR, a handheld walkthrough video, room photos) feed one shared structure backend and one output contract, so a consumer sees the same schema whatever the sensor.

The design choice that matters most is **honesty about uncertainty**. Every measurement carries a value, an interval, a quality label (`strong`, `moderate`, `weak`) and its source tier. Runs end as `success`, `partial` or `failure`; a failed run writes no property. Where a quantity cannot be established (no metric scale, an opening that is not verified in 3D, a room whose walls do not close) it is reported as missing with a structured warning, not filled with a typical value.

What is demonstrated: the full pipeline runs on three real LiDAR captures (PARTIAL results, deterministic, 1-3 minutes each), the video tier runs on a real scan-pass video, all three tiers and the evaluation harness are covered by 372 automated tests, and a LiDAR drift ablation is reported honestly (accepted on one capture, rejected on two). What is **not** demonstrated: dimensional accuracy against tape/laser (**NOT VALIDATED**), real photo sets, a real walkthrough video, repeatability, or any assignment gate. An evaluation harness is ready to produce those numbers once data exists ([benchmark_report.md](benchmark_report.md)).

## 2. Architecture

    LiDAR  ──► validate, metric reconstruction, drift stage ─┐
    Video  ──► keyframes, SfM, metric scale, gravity, fusion ─┼─► MetricScene ─► shared structure backend ─► canonical Property
    Photos ──► per-room SfM + scale, cross-room stitching  ───┘      (planes, walls, rooms, openings)      │
                                                                                  damage layer (RGB evidence) ◄┤
                                                                                  plan.png, property.json, run_report.json

- **Three frontends** each end in a `MetricScene`: a gravity-aligned point cloud in metres, camera poses, and quality tags (tracking, scale, gravity) plus any tier-specific uncertainty context. Nothing downstream knows which sensor produced it, except through those tags.
- **Shared structure backend** (the same code for every tier): floor and ceiling planes from the vertical density profile; vertical walls from a deterministic Hough transform on wall-like columns with 3D verification; rooms as bounded faces of the planar wall graph; openings from gaps verified against a u-v occupancy map on the wall plane.
- **Canonical Property**: rooms, walls, openings, adjacency, damage regions, concealed-damage flags, scope line items, structured warnings and provenance. The schema is defined by SpatialForge (`schema/property.schema.json`, kept in sync with the code by a test); no official schema was supplied.
- **Product layer**: tier auto-detection that refuses to guess; a 12-stage report with timings, identical for every tier; structured warnings and errors (`{code, stage, severity, message, recoverable}`); staged output with overwrite protection; one version number everywhere; provenance with home-directory paths redacted.
- Deliberately CPU-only, no GPU, no compiled dependencies; Python 3.11.

## 3. Tier design

**LiDAR (strongest).** Input: 16-bit depth maps, poses (`odometry.csv`), intrinsics, optional RGB and confidence. Depth is 0.001 m per raw unit (an assumption supported by multi-frame overlap peaking sharply at that scale on all three samples). Intrinsics refer to the RGB image and are scaled to the depth resolution. Poses are camera-to-world with +X right, +Y down, +Z forward; world Y is vertical. The stages are deterministic.

**Video.** A deterministic keyframe selector (sharpness, parallax, gaps) feeds PyCOLMAP sparse SfM (sequential matching, one shared SIMPLE_RADIAL camera). SfM is scale-ambiguous, so metres come only from a monocular **metric** depth model (Depth Anything V2 Metric-Indoor-Small): per keyframe the metric depth is compared with SfM depth at the sparse points, per-frame scales are combined robustly, and if frames disagree too much **no scale is returned and no geometry is reported**. Each frame's depth is then rescaled to the SfM geometry at the global scale before fusion, so the only absolute quantity taken from the model is the global scale. Gravity is estimated from the geometry (slab concentration, wall-normal consistency, a floor-below-cameras sign test), never assumed.

**Photos.** 2-8 stills per room folder, no poses or depth. One COLMAP database per property with exhaustive matching; per-room SfM; each room gets its **own** metric scale (never borrowed from another room). Rooms are joined only on verified evidence: geometrically verified feature matches between rooms' photos, a joint SfM model for the relative transform, and optionally a doorway seen from both rooms. Placement is a robust (Huber) least squares over the room graph; loop-inconsistent edges are rejected, overlapping placements are refused, and a room without evidence is **listed as unplaced**, not drawn beside the property.

**Damage (all tiers).** See section 6.

Device and accuracy status per tier: [device_matrix.md](device_matrix.md).

## 4. Drift and geometry

Odometry is not trusted blindly. The drift stage registers sampled depth clouds with point-to-plane ICP (Open3D), adds conservative loop closures, optimises a pose graph with odometry anchors and gravity-constrained rotation, then **compares a corrected reconstruction against the baseline on identical frames** and uses the correction only if no consistency metric degraded by more than 3% and at least one improved by 2%. Results on the three samples ([drift_ablation.md](drift_ablation.md)): accepted on `single_room` (loop residuals 0.052 m to 0.020 m), **rejected on both larger scans** because floor flatness got worse (band thickness +8% and +25%) even though the optimiser's own constraints improved; the original poses were kept and the run report says so. We claim no universal improvement. Whether corrected poses are closer to the truth is NOT VALIDATED.

Structural extraction is a soft Manhattan prior: a wall snaps to the dominant axis only if within 5 degrees *and* snapping moves its ends by at most 10 cm. Gaps in walls are kept for the opening stage, never closed. Corners need a short extension (at most 0.5 m for strong walls), and nothing is extended further.

## 5. Error budget and uncertainty

No fixed, made-up interval is used anywhere; each interval comes from its stage. Terminology: `interval` is a measurement range, `confidence` a heuristic numeric score (**never a probability**), `quality` a label.

- **LiDAR.** Plane and wall intervals come from deterministic subsets (interleaved frame blocks, regional quadrants), fit scatter and wall position uncertainty (walls are blurred to a few centimetres by residual drift); room intervals shift every wall by its position uncertainty over fixed sign patterns. The depth-scale assumption (0.001 m/unit) is **not** included in these intervals.
- **Video.** Each length gets a relative sigma from the metric scale (spread over the square root of the frame count, plus a 10% floor because the depth model is uncalibrated), SfM quality (reprojection error, unregistered keyframes) and depth inconsistency, combined in quadrature; stage intervals are widened by +/-2 sigma (areas by twice that) and measurement quality is capped by the scale quality (never `strong`). On the samples, earlier checks against ARKit poses found depth-derived scale 5-16% off, so intervals are wide by design.
- **Photos.** Per-room sigma from scale spread (small-sample SD for 2-4 photos, 10% floor), SfM quality, single-view depth and gravity quality, then widened by cross-room scale inconsistency, loop residual and overlap conflicts. Stitch translation uncertainty widens wall positions and the footprint.
- **Openings.** The width interval combines jamb scatter across rows, the 5 cm grid, wall position and the disagreement between three independent frame subsets. Existence quality is never `strong` without independent subset confirmation. Candidates that fail 3D verification are listed as unverified and excluded from connectivity. On the `single_scan_with_ceiling` sample one verified 1.15 m door has an interval of 0.75-1.55 m, far wider than a 2 cm gate.
- **Damage extents.** The surface's own scale uncertainty (video and photos carry theirs), the plane-position uncertainty over the viewing range, and a 2 px boundary error.
- These are engineering estimates. Their calibration against ground truth (does the truth fall inside the interval?) is implemented as a metric but **NOT YET MEASURED**.

## 6. Damage layer

Deterministic OpenCV image processing on the tier's RGB evidence (LiDAR: up to 24 video frames with production poses; video: keyframes with metric poses; photos: placed rooms' photos). No learned model.

- **Candidates:** `crack` (black-hat ridges, robust MAD threshold, elongated and thin component filters, perfectly straight long lines rejected as edges) and `stain` (chroma deviation from the local wall colour; brightness-only shadows rejected; large, compact, smooth blobs). Confidence is a heuristic score.
- **Surface mapping:** rays through the outline are cast onto the property's wall, floor and ceiling planes; the nearest valid surface wins. A detection gets `surface_id`, `room_id` and a metric width/height/area/length **only if** at least 80% of its outline agrees on one surface, the angle is not grazing, the range is 0.3-10 m and the wall is not weak; otherwise it is kept with null extents and a warning. Views of the same damage are merged.
- **Concealed-damage risk rules** (never an assertion that damage exists): `CDR-001` stain within 0.30 m of the floor, `CDR-002` stain near the ceiling or on it, `CDR-003` two or more stains within 1.5 m on a wall, `CDR-004` crack near an opening's head corner, `CDR-005` crack 1.0 m or longer, `CDR-006` crack and stain within 0.30 m. Each flag names its rule, the condition that fired and the triggering detections.
- **Scope line items** are keyed to surfaces: crack -> `seal_and_repair_crack` (m), stain -> `treat_stain_and_repaint` (m2), flag -> inspection item. No metric extent means no quantity.
- A damage-stage failure is a warning; it never downgrades a run.

## 7. Calibration and benchmark

**Current validated evidence (internal, not ground truth):** the LiDAR pipeline runs on three real captures (1-3 rooms, 9-26 walls, PARTIAL status as designed), is deterministic (two full runs on one capture gave identical geometry and a byte-identical plan; drift numbers reproduced exactly), and its runtimes are 2-3 minutes per capture. One real video (a LiDAR scanning pass, not a walkthrough) produced a PARTIAL result with weak tracking (24 of 145 keyframes registered, six disconnected models). Photo behaviour is verified only on synthetic scenes. [internal_sample_report.md](internal_sample_report.md).

**Missing real evidence:** tape/laser ground truth, a multi-room property captured by all three tiers, a furnished damage room, a repeat capture, and a consumer-app export. Consequently no assignment gate (opening width <= 2 cm on 85%, ceiling <= 1.5 cm, repeat spread <= 1 cm, photo wall <= 8%, video wall <= 3%, photo footprint <= 8%) has been evaluated. We expect several to fail on the tiers with monocular scale and on opening width, given the sample intervals; that expectation is not a result.

**Harness:** `python -m spatialforge evaluate` matches ground truth to predictions conservatively (IDs first, then tolerance-limited geometry; unmatched items stay missed or phantom), computes wall, opening, ceiling, area, footprint, adjacency, repeatability, interval-coverage and runtime metrics, and reports every gate as PASS, FAIL or NOT EVALUATED. Ground truth that exists without a prediction is a FAIL; no ground truth means NOT EVALUATED, never PASS. Thresholds live in one configuration file. Interval coverage and the confidence score are reported but not gated, because no coverage target was supplied.

## 8. Fix-loop story

**STATUS: PENDING REAL BENCHMARK.** The loop is: benchmark, pick the worst gate, declare a hypothesis and a predicted number *before* coding, freeze the baseline, ship one focused fix, freeze again, compare (metric, before, after, absolute and percent change, gate before and after) and show the readable diff. The declaration template, `freeze_baseline.py` (records git SHA, capture, tier, command, ground truth, config, metrics, timestamp; never overwrites), `compare_runs.py` and their tests exist ([fix_loop.md](fix_loop.md)). No shipped fix is claimed, because without ground truth any "before/after" would be invented. The likeliest worst gates, as hypotheses only, are opening width (LiDAR) and the video and photo wall-length gates (monocular scale).

## 9. Known failure modes

- **Textureless rooms, blur, low light:** SfM fragments into disconnected models; the video tier then reports weak tracking (PARTIAL) or no coherent reconstruction (FAILURE). Only the largest connected model is used.
- **Monocular scale:** video and photo scale is typically 10-15% off and not forced to agree between rooms; intervals are wide, quality is never `strong`; with no consistent scale the run fails rather than guessing.
- **Incomplete wall loops:** only walls that close become rooms; the rest are listed as walls outside any room, the footprint is marked `complete: false`, and the property is PARTIAL.
- **Opening recall:** only openings verified in 3D are reported, heights and sills are mostly unobserved, and recall has not been measured.
- **Furniture:** tall wardrobes against walls can pass as structure; furniture and art can be detected as stains or cracks.
- **Glass and mirrors:** depth and SfM are unreliable on them.
- **Damage false positives:** classical image processing, not a trained model; precision and recall are unmeasured, so every detection is a candidate for human review. Damage on an object in front of a wall is projected onto the wall.
- **Reproducibility:** COLMAP is multi-threaded and not bit-reproducible; the video and photo tiers differ slightly between runs (documented in the provenance); LiDAR is deterministic.
- **Odometry drift:** corrected only when the metrics support it; on two of three samples the original poses were kept.
