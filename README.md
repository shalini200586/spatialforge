# SpatialForge

Turns a capture of a property into a dimensioned floor plan and a machine-readable description. The LiDAR tier
(depth + camera poses), the video tier (a normal walkthrough video) and the photo tier (room folders of still photos) are
implemented.

## Install

    pip install -e .[test]            # LiDAR tier
    pip install -e .[test,video]      # plus the video tier (see "Video tier" below)

## Usage

    python -m spatialforge process CAPTURE --tier lidar --output RESULT

One command runs every stage (capture validation, metric reconstruction with a pose-refinement check, floor and
ceiling planes, structural walls, room topology, openings) and writes:

    RESULT\property.json     canonical property description (schema v1.0, see below)
    RESULT\plan.png          rendered, dimensioned floor plan (product output, not a debug view)
    RESULT\run_report.json   operational report: status, per-stage timings and statuses, counts, warnings, errors
    RESULT\diagnostics\      small debug artifacts (wall/room/opening overlays, stage summaries); no point clouds

Exit code: 0 = success or partial result, 1 = failure (invalid capture, no metric reconstruction), 2 = bad usage.
Options: `--max-frames 400` and `--frame-step 1` (evenly spaced, deterministic frame selection).

For a walkthrough video the same command takes the video file and the video tier (needs `pip install -e .[video]`
and the one-off model download, see below):

    python -m spatialforge process "C:\capture\walkthrough.mp4" --tier video --output "C:\result"

**Results are partial by design.** Only spaces whose walls form closed outlines become rooms, and only openings
verified in 3D are reported. `property.status` says `partial` and `warnings` lists what is missing (incomplete
room topology, unverified opening recall, no ground truth); nothing missing is filled in with fake geometry.

### property.json

`schema_version "1.0"`, then `capture` (tier, source, device), `property` (status, footprint if available),
`rooms` (polygon, area, perimeter, length x width where meaningful, wall ids, ceiling height where observed,
adjacent and connected room ids, topology quality), `walls`, `openings` (type, width, height/sill where observed,
room connectivity, existence/type quality), `unverified_openings`, empty `damage`, `concealed_damage_flags` and
`scope_line_items`, `warnings`, `provenance` (stages, pose source, depth scale, assumptions) and `timing`.

Every measurement is `{value, unit, interval {low, high} | null, confidence | null, quality | null, source_tier}`.
The interval is the stage's own uncertainty range, never invented. `confidence` is a heuristic score (not a
probability) and is only present where a stage computes one (ceiling heights); other values carry a `quality` label.
Output is deterministic (sorted keys, rounded numbers); only the `timing` section differs between identical runs.

**Schema:** the case-study materials supplied no separate schema file, so SpatialForge defines schema v1.0
(`schema/property.schema.json`, kept in sync with the code by a test). It is not an official evaluator schema.

### plan.png

Rooms (id, area, length x width or "irregular", ceiling height), structural walls, edge dimensions and accepted
openings with their widths, on a metric-scaled white background. Policies: weak walls are not drawn; walls outside
any recovered room are dashed grey; unverified wall gaps are dotted; low-confidence openings are left out of the
plan and listed under `unverified_openings`; a PARTIAL badge and note are always shown for partial results.

## Video tier (tier 2)

    python -m spatialforge process "C:\capture\walkthrough.mp4" --tier video --output "C:\result"

Produces the same artifacts and the same canonical `property.json` schema as LiDAR (`source_tier` is `video`), plus
video diagnostics in `RESULT\diagnostics\`: `keyframes\`, `trajectory.png`, `trajectory.json`, `sparse_sfm.ply`,
`metric_cloud.ply`, `scale_report.json`, `video_frontend.json`.

**A monocular video has no depth, no known poses and no absolute scale.** Structure-from-Motion alone is scale
ambiguous, so the pipeline needs two things and never claims metres from SfM by itself:

    video -> validation -> deterministic keyframes -> PyCOLMAP sparse SfM (trajectory + sparse points, ARBITRARY scale)
          -> metric monocular depth on keyframes -> robust metric scale (depth vs SfM, many frames)
          -> scaled trajectory -> depth back-projected and fused -> gravity from geometry
          -> MetricScene -> the SAME floor / wall / room / opening stages as LiDAR -> canonical Property

### Prerequisites and model setup

    pip install -e .[video]              # opencv-python-headless, pycolmap, torch (CPU), transformers
    python scripts/download_models.py    # ~100 MB, cached OUTSIDE the repository

The weights (Depth Anything V2 Metric-Indoor-Small, ~25M parameters, about 99 MB) are cached in
`%SPATIALFORGE_MODELS%` or `~/.spatialforge/models` and are never committed. Behind a TLS-inspecting proxy install
`truststore` (the script uses the operating system's certificate store; verification stays on). CPU only, no CUDA, no
compilation (`pycolmap` and `torch` are prebuilt wheels). If the weights are missing the run does not fake anything: it
returns a partial result with `metric_scale_available: false` and a warning.

### How the stages work

- **Validation:** file exists, supported container, decodable, size, fps, duration (at least 5 s), seekable.
- **Keyframes (deterministic, no randomness):** candidates at a fixed rate from one decode pass; blurred candidates
  (Laplacian variance below half the local median) and whip-pans are dropped; a keyframe needs parallax (median
  optical flow) since the last one, a minimum gap, and is forced after a maximum gap. Thinned to 120 by dropping the one
  whose removal leaves the smallest gap. The container's rotation metadata is applied.
- **SfM:** PyCOLMAP SIFT features (4096 per image), sequential matching with quadratic overlap, incremental mapping,
  one shared SIMPLE_RADIAL camera with the focal length estimated (no metadata prior is used). The largest model is kept.
  Tracking quality: STRONG (>= 80% registered, >= 1500 points, <= 1 px), MODERATE (>= 50%), WEAK (>= 15%), FAILURE
  (below that, or fewer than 10 images): FAILURE stops the run (exit code 1); WEAK continues but the result is partial.
- **Metric scale:** per depth keyframe the metric depth is sampled at the SfM points' pixels (edges and extreme ratios
  rejected); `metric depth / SfM depth` gives per-frame scales (log-domain median with outlier rejection); the global scale
  is the median across frames. Reported: frames used, correspondences, spread, standard error, combined sigma.
  `scale_quality` is never `strong` for an uncalibrated monocular model; if frames disagree too much no scale is
  returned and **no geometry is reported**.
- **Per-frame depth alignment:** single-frame metric depth varies by tens of percent between frames, so each frame's depth
  is rescaled to the SfM geometry at the global metric scale before fusion. The only absolute quantity taken from the depth
  model is the global scale.
- **Gravity:** estimated from geometry, not assumed. Candidate axes are scored by height-slab concentration (floors,
  ceilings) within a cone around the camera path's least-variance axis, refined so wall normals are perpendicular to up. The
  sign comes from the floor below the cameras at a plausible distance, a strong image-up agreement, or weak cues, in that
  order, and the confidence says which.
- **Fusion:** depth is back-projected through the scaled poses (radial distortion undone), a point must be confirmed by a
  second keyframe (10 cm voxel), then voxel-downsampled (3 cm). Deterministic stride throughout.
- **Uncertainty:** every length gets a relative sigma from scale (spread/sqrt(frames) plus a 10% floor for the
  unvalidated model), SfM quality (reprojection error, unregistered keyframes) and depth inconsistency, combined in
  quadrature; stage intervals are widened by +-2 sigma in quadrature (areas by twice that), and measurement quality is
  capped by the scale quality. These are engineering estimates, not calibrated intervals.

### Known limitations

- **Scale depends on a monocular depth model.** Measured against ARKit poses on the sample captures, the depth-derived
  scale was off by +12% to +16% (oracle poses) and by -12% and +5% in two real-SfM runs. Intervals are therefore much
  wider than LiDAR's. COLMAP itself is multi-threaded and not bit-reproducible: two runs on the same video registered 25
  and 33 keyframes.
- **Performance depends on texture, blur, lighting and parallax.** The three sample videos are LiDAR scanning passes (fast
  sweeps, white walls, close range): SfM fragments into several disconnected models on them. They are not walkthroughs and
  say little about the real tier-2 benchmark video.
- Only the largest connected SfM model is used, so a fragmented trajectory covers only part of the property.
- 64 depth keyframes (default) limit coverage of long videos; denser fusion costs about 3 s per frame on CPU.
- Fused monocular clouds are far noisier than LiDAR (decimetres vs centimetres), so the LiDAR-tuned structural stages
  recover fewer walls and rooms; results are usually partial.
- Gravity sign relies on weak cues when neither floor nor image-up is decisive; the confidence reports it.
- Typical runtime on a 14-thread CPU: 6-10 minutes for a 1-3 minute video.

## Photo tier (tier 1)

    python -m spatialforge process property_photos --tier photos --output result

**Input:** a directory of room folders, 2-8 photos each (`.jpg`, `.jpeg`, `.png`; HEIC is rejected with a clear message,
convert it to JPEG first). No poses, no depth, no trajectory are needed.

    property_photos\
        room_01\   IMG_001.JPG  IMG_002.JPG  IMG_003.JPG
        room_02\   IMG_101.JPG  IMG_102.JPG  IMG_103.JPG
        hallway\   IMG_201.JPG  IMG_202.JPG

The output is ONE property (same `property.json` schema, `plan.png`, `run_report.json`, `diagnostics\`; `source_tier` is
`photos`), not separate room plans. Folder names are kept as `source_label` only; canonical ids are `room_001`, `room_002`
... in sorted folder order and no room type is ever inferred from a name.

**Capture advice** (it decides whether rooms can be stitched):
- take overlapping views and include every corner, so each wall is seen whole;
- photograph each doorway from BOTH sides: shared doorway views are the strongest stitch evidence;
- keep the original files and their EXIF (focal length is used as an SfM prior only when every photo carries a believable
  35 mm equivalent; orientation is applied in memory, originals are never modified).

**How it works** (`python -m spatialforge process ...` runs all of it):
1. *Validation:* every folder has 2-8 supported photos that decode and have sensible size; nothing is skipped silently.
2. *One COLMAP database for the whole property:* features once, **exhaustive matching** of all images (cheap: few photos).
3. *Per room:* PyCOLMAP SfM restricted to that room's photos (STRONG / MODERATE / WEAK / FAILURE; two registered photos are
   never STRONG), cached metric monocular depth (the video tier's model), the room's **own** metric scale (never borrowed from
   another room), gravity from geometry, then the shared floor / wall / room / opening stages. A failed SfM or missing scale
   gives an unreconstructed / unscaled room with **no geometry**: no typical room size, door width or ceiling height is used.
4. *Cross-room evidence:* for every room pair, geometrically VERIFIED feature matches between their photos (raw matches never
   count). A pair becomes a stitch candidate only with a verified image pair of at least 25 inliers at inlier ratio 0.25 and 50
   verified matches in total (`StitchOptions`).
5. *Relative transform:* the photos of both rooms are mapped together (a joint SfM model); aligning each room's metric local
   poses to the joint poses gives rotation and translation (yaw and x/z, both rooms being gravity-aligned). Tilt, scale
   disagreement and fit residual set the constraint's uncertainty and can reject it.
6. *Doorways:* an opening in room A and one in room B that coincide after the transform (width, parallel walls) confirm the
   stitch (`visual+doorway`). With no image evidence, a doorway-only constraint is created only if exactly one physically
   consistent placement exists (weak quality, flagged); ambiguous doorway matches are refused.
7. *Placement:* deterministic robust (Huber) least squares over the room graph, anchored at the strongest room of each
   component; loop-inconsistent edges are rejected, never averaged in; placed rooms must not overlap (alternative doorway
   hypothesis first, otherwise the weakest edge on the path is rejected). Only the largest component is placed.
8. *Property:* walls shared by two rooms and the same doorway seen from both are merged; adjacency is reported as
   `verified_connection` (same doorway), `geometric_adjacency`, or `probable_adjacency_*` (placed together by evidence only,
   lower quality); the footprint is the union of the PLACED rooms only (`complete: false`).

**Not placed:** a room that cannot be connected to the main component is not drawn beside the property. It is listed under
`provenance.photos.unplaced_rooms`, in `warnings`, and in the plan's header note; the property is then `partial`.

**Uncertainty:** per room from the scale spread (small-sample SD for 2-4 photos, a 10% floor for the unvalidated depth model),
SfM reprojection/registration, single-view depth (no multi-view cross-check with so few photos) and gravity quality, in
quadrature; then widened by cross-room scale inconsistency (`scale_inconsistent`, never forced equal), loop residual and overlap
conflicts. Stitch translation uncertainty widens the walls' position uncertainty and the footprint area. Heuristics, not
calibrated intervals.

**Diagnostics:** `diagnostics\photo_frontend.json`, `rooms\room_00N\{reconstruction.json, sparse_sfm.ply, metric_cloud.ply,
local_plan.png}`, `stitching\{room_graph.json, stitch_constraints.json, overlap_report.json, stitching_initial.png,
stitching_final.png}`.

**Known limitations**
- Monocular scale: each room's scale comes from a depth model and is typically off by 10-15%; rooms are not forced to agree.
- Sparse or textureless rooms, and rooms photographed without overlap or without corners, give weak or failed SfM.
- Poor cross-room evidence (no shared view, no doorway photographed from both sides) leaves rooms unplaced.
- A doorway-only stitch is a weak guess and is only made when unambiguous; a false stitch is possible.
- Few photos mean walls and rooms often do not close; results are usually partial.
- Reconstruction is slow-ish on CPU: a few minutes for a handful of rooms.

## Development: individual stage commands

These remain available for debugging; `process` runs them all.

    python -m spatialforge validate-lidar "C:\path\to\capture"

Exit code: 0 = VALID / VALID WITH WARNINGS, 1 = INVALID, 2 = bad usage.

## Expected capture layout

    rgb.mp4            (optional, warning if missing)
    odometry.csv       (required) header: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
    camera_matrix.csv  (optional) 3x3 intrinsics matrix
    imu.csv            (optional)
    depth/NNNNNN.png   (required) 16-bit grayscale
    confidence/NNNNNN.png (optional) 8-bit grayscale, levels 0..2

If the given folder contains a single sub-folder holding the capture, that
sub-folder is used automatically.

## Reconstruct a metric point cloud

    python -m spatialforge reconstruct-lidar "C:\spatialforge-data\single_room" --output "C:\temp\single_room.ply"

Writes an ASCII XYZ-only PLY (metres, in the capture's native world frame) and prints diagnostics:
frames used, point counts at each filtering step, camera-path and cloud extents, and a plausibility check.

Options: `--frame-step 30` (every Nth frame), `--max-frames 60` (evenly spaced cap, deterministic),
`--min-confidence 2` (levels are 0-2), `--voxel-size 0.02` (metres, keeps the first point per voxel),
`--min-range` / `--max-range` (metres), `--source-size WxH` (only if `rgb.mp4` cannot be read).

- **Depth scale:** 0.001 m per raw depth unit (`DEPTH_SCALE_M_PER_UNIT` in `lidar/geometry.py`). The PNGs do
  not state their unit, so this is an assumption supported by evidence: with poses in metres, multi-frame
  overlap peaks sharply at 0.001 on all three samples (0.67-0.74 vs <= 0.50 at 0.0005 and <= 0.31 at 0.002),
  and floors appear about 1.5 m below the handheld camera.
- **Intrinsics:** the odometry/camera-matrix intrinsics refer to the RGB image (1920x1440, read from
  `rgb.mp4`), not the 256x192 depth map. They are scaled proportionally (x 256/1920, y 192/1440; both 4:3,
  principal point near the RGB centre). Per-frame intrinsics from `odometry.csv` are preferred, with
  `camera_matrix.csv` as fallback.
- **Pose convention:** `x y z qx qy qz qw` is camera-to-world, `world = R(q) @ p_cam + t`, with camera
  axes +X right, +Y down, +Z forward. The other three interpretations (world-to-camera, and Y/Z-flipped
  camera axes) gave 0.01-0.11 neighbour-frame overlap vs 0.41-0.74 for this one. The world Y axis is vertical.
- **Limitations:** no floor/wall/ceiling extraction, no reorientation into a floor-plan frame, and no drift
  correction yet. Odometry drift is not removed, so large scans may show some misalignment.

## Drift correction ablation

    python -m spatialforge correct-lidar-drift "C:\spatialforge-data\single_scan_floor_only" --output-dir "C:\temp\drift_floor_only"

The supplied odometry is not simply trusted: the command registers the depth clouds of sampled frames,
optimises a pose graph, and compares a baseline reconstruction (OFF, supplied poses) with a corrected one
(ON) on identical frames, filtering and voxel size. Needs Open3D (CPU).

Method: odometry gives the initial guess; point-to-plane ICP refines consecutive sampled frames
(5 cm registration voxel); a few conservative loop closures are added where the camera returns to an
earlier place (far apart in the sequence, close in odometry space, similar heading) and ICP passes
stricter gates; Open3D optimises the pose graph. Every consecutive pair keeps an odometry edge as an
anchor, sequential ICP edges get a small weight and loop closures full weight. Roll/pitch stay as supplied
(only heading and position may change). All thresholds are in `DriftOptions` / `AcceptanceRules`
(`lidar/drift.py`) and are written into the report.

Safe fallback: an ICP result that fails its fitness / RMSE / correction-size gates is rejected and that
pair keeps its odometry relation. The corrected poses are only used if the before/after metrics agree
(no metric meaningfully worse, at least one meaningfully better, footprint and correction sizes plausible);
otherwise the report says `"accepted": false` and `"production_poses": "original"`.

Artifacts in `--output-dir`: `before.ply`, `after.ply`, `before_topdown.png`, `after_topdown.png`
(same bounds, scale and contrast), `drift_report.json`, `trajectory_before.csv`, `trajectory_after.csv`.
`before.ply` is identical to `reconstruct-lidar` with the same options. `after.ply` is always the corrected
candidate, even when rejected, so the ablation can be inspected.

Use enough frames for neighbouring frames to overlap (default `--max-frames 400`; 80 frames of a 9,745-frame
capture were too far apart for ICP to register most pairs). Structural extraction is not implemented yet.

## Floor and ceiling planes

    python -m spatialforge analyze-horizontal-planes "C:\spatialforge-data\single_scan_with_ceiling" --output-dir "C:\temp\planes_with_ceiling"

Detects the structural floor and ceiling(s) and measures ceiling height. It runs on the production cloud:
the drift stage (above) decides per capture whether corrected or original poses are used, and the report
states `pose_source`. World +Y is vertical; no reorientation is applied.

- **Floor:** peaks in the vertical (Y) density profile below the camera path are fitted with a robust
  (Tukey-weighted) plane `y = p*x + q*z + r`, refitted using only X-Z columns with solid coverage. The
  lowest peak with at least half the best solid area is the floor. No random sampling anywhere.
- **Ceiling:** candidates must lie 2.0-4.5 m above the floor (`PlaneOptions`) and above the camera path,
  be tilted less than 5 degrees, have a tight fit, and cover at least 3 m2 (and 10% of the floor's area) of
  *solid* X-Z area: coverage is measured on a 10 cm grid after eroding by one cell, so thin wall lines,
  lamps and shelf tops do not count. Otherwise `ceiling.observed` is false and no height is reported.
- **Several levels:** every accepted level is listed (a building can have ceilings at different heights), the
  largest is the primary, and a warning explains that assigning levels to rooms needs room segmentation.
- **Height:** perpendicular distance from the ceiling plane's inlier centroid to the floor plane.
- **Uncertainty:** interval from deterministic subsets (interleaved blocks and regional quadrants) plus fit
  scatter, with surface errors counted per 1 m patch; `confidence` falls as the interval widens or the
  supported area shrinks. It excludes the depth-scale assumption (0.001 m per unit). It is a measured
  consistency estimate, not a calibrated probability.
- **Outputs** in `--output-dir`: `horizontal_planes.json`, `vertical_profile.png`, `side_projection.png`,
  `floor_inliers.ply`, `ceiling_inliers.ply` (all accepted levels, only if observed).

Walls, rooms, openings and floor area are not implemented yet.

## Structural walls

    python -m spatialforge analyze-walls "C:\spatialforge-data\single_room" --output-dir "C:\temp\walls_room"

Extracts vertical structural walls as metric top-down (X-Z) segments, on the production cloud (corrected or
original poses as decided by the drift stage) and using the floor/ceiling planes from the horizontal-plane stage.

- **Vertical structure:** only points in a height band above the floor (0.3-2.2 m; floor and accepted ceiling
  planes excluded) are used. A 5 cm X-Z column counts as "wall-like" only if it holds many points at many
  different heights, so floors, ceilings, table tops and low furniture cannot form walls.
- **Lines:** a deterministic Hough transform on those columns (no random sampling), then 3D verification.
  Each observed segment must be long enough (0.6 m), tall enough (1.0 m span, observed at many heights),
  reach toward the floor, be near-vertical (5 degrees), fit tightly and follow the wall-like ribbon.
- **Furniture rejection:** vertical span/coverage, horizontal span, floor contact, plane fit, and structural
  context: short pieces are kept only if they join a long wall, and short planes far from both dominant wall
  axes are dropped. Tall wardrobes against a wall can still pass; each wall carries an evidence tier
  (`strong`/`moderate`/`weak`, a heuristic, not a probability).
- **Orientation:** a soft Manhattan prior. The dominant axis is the orientation cluster holding the most wall
  length. A wall snaps to it only if it is within 5 degrees AND snapping moves its ends by at most 10 cm
  (long walls therefore snap only for small deviations). Raw and snapped orientation are both reported.
- **Duplicates:** near-parallel, overlapping detections within 30 cm merge unless the points between them show
  two separate surfaces (a valley in the perpendicular profile). Walls are blurred to ~0.5 m by residual
  drift, so two real walls closer than ~0.3 m cannot be told apart.
- **Gaps are kept:** observed segments of one wall stay separate, and gaps up to 2.5 m are recorded in
  `gaps` for a later opening stage. Pieces further apart are separate walls.
- **Outputs:** `walls.json` (walls, segments, gaps, residuals, position uncertainty), `wall_candidates.json`
  (everything rejected, with reasons), `walls_topdown.png`, `wall_inliers.ply`.

Known limitations: no rooms, corners or openings yet; blurred walls make line positions uncertain by several
centimetres; some parallel duplicates, off-axis lines and furniture planes remain (see the evidence tier and
`wall_candidates.json`); thresholds were calibrated on three captures and are not production-validated.

## Room topology

    python -m spatialforge analyze-rooms "C:\spatialforge-data\single_room" --output-dir "C:\temp\rooms_room"

Turns the structural wall candidates into connected room geometry (metric X-Z, polygons counter-clockwise).
It consumes the wall stage's output (snapped orientations, evidence tiers, uncertainties, observed segments and
gaps); it does not re-detect walls.

- **Corners:** infinite wall lines are intersected for pairs more than 25 degrees apart. A corner is accepted
  only if each wall needs at most a short extension to reach it: up to 0.5 m for strong walls, 80% of that
  for moderate and 50% for weak (plus a small margin for the other wall's position uncertainty). 0.5 m comes
  from the measured near-misses on the samples (0.43-0.49 m, then 0.60 m and more). Nothing is extended further,
  and the extension of every corner is reported.
- **Gaps:** a collinear gap inside a wall is never closed or erased. The wall stays one conceptual edge across
  it, with `gap_m` and `observed_support_fraction` recorded separately, ready for opening analysis.
- **Parallel duplicates** within 35 cm are reduced to the stronger wall.
- **Rooms:** the bounded faces of the planar wall graph (half-edge face walking, no cycle enumeration), kept if
  they pass sanity filters (area, edge length, thickness, not mostly inferred, mostly observed, simple polygon,
  not nested, not an outline enclosing interior walls). Weak walls take part only in a second pass and may close
  a face only if the other edges are strong/moderate; they can never subdivide an accepted room. Open or
  unclosable regions simply produce no room.
- **Measurements:** area (shoelace), perimeter and wall lengths; length/width for rectangular rooms from the
  distance between opposite fitted wall lines (or an oriented bounding box for near-rectangles); irregular
  rooms get no length/width. Intervals come from shifting every wall by +-its position uncertainty (inflated for
  weak walls and inferred corners) over a fixed set of sign patterns. They are diagnostics, not calibrated.
- **Ceilings:** each room gets the Ticket 4 ceiling level whose solid inlier area covers most of it (at least
  25%); if two levels at different heights overlap comparably the room is reported as ambiguous, not assigned.
- **Adjacency** is geometric only (parallel polygon edges within 40 cm that overlap); it says nothing about doors.
- **Outputs:** `rooms.json`, `wall_graph.json`, `room_polygons.geojson` (local metric frame), `rooms_topdown.png`.

Known limitations: only spaces whose walls are detected AND connect become rooms, so most of a property can stay
unmodelled; rooms with undetected dividing walls come out merged; blurred walls limit corner accuracy to a few
centimetres; thresholds were calibrated on three captures; no openings, labels or room types yet.

## Structural openings

    python -m spatialforge analyze-openings "C:\spatialforge-data\single_room" --output-dir "C:\temp\openings_room"

Verifies the wall gaps preserved by the wall stage using local 3D evidence and measures the openings. It works on
structural walls, so openings are found even where no complete room polygon exists.

- **Candidates:** gaps inside a wall group, gaps between collinear walls of different groups, and unobserved
  stretches along room boundaries (away from corners). A gap alone is never an opening.
- **Verification:** for every candidate the 3D points near the wall plane (a tolerance of 2x the wall's own blur,
  0.12-0.20 m; floor and ceiling excluded) are binned into a u (along the wall) x v (height) occupancy map. An
  opening needs an empty middle over a band of rows with a solid wall run on BOTH sides in those rows. A wall that
  is complete in 3D, a wall end (only one jamb backed), poorly scanned surroundings, points just in front of the
  wall (furniture/occlusion), an unclean opening region, and widths outside 0.4-4 m are all rejected, with the
  reason and stage kept in `opening_candidates.json`.
- **Type:** reaches the floor and is not wider than 1.6 m -> door; solid wall below the gap (sill >= 0.4 m) -> window;
  anything else, including very wide openings -> generic `opening`. Height and sill are reported only when the head
  or sill is actually observed (otherwise null).
- **Jambs and width:** per row, the nearest solid run on each side gives an edge; the jamb is the median over rows.
  Width = right jamb - left jamb; the raw gap from the wall stage is not used as the width.
- **Uncertainty:** the width interval combines jamb scatter across rows, the 5 cm grid, wall position, and the
  disagreement between three deterministic frame subsets (alternating blocks of frames) re-measured independently.
- **Ratings are separate:** `observability`, `existence_quality` (never `strong` without independent subset
  confirmation), `type_quality` and `width_quality`. Openings whose existence is weak are listed as
  `low_confidence` and do not count toward connectivity.
- **Connectivity:** `connected_room_ids` only for verified openings that sit on the boundary of two rooms; an opening on one
  room's boundary (or none) connects to `unmodelled` space. This is separate from geometric adjacency.
- **Outputs:** `openings.json`, `opening_candidates.json`, `openings_topdown.png`, `opening_profiles/*.png` (the
  u-v map of every candidate).

Known limitations: most heads/sills are not observed, so heights are mostly null; walls blurred by residual drift make
jamb edges ragged (typically +-0.2 m) and widths uncertain; frame subsets often cannot all measure a candidate;
openings in walls the wall stage missed cannot be found; no ground truth was available.

## Tests

    python -m pytest
