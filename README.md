# SpatialForge

Early-stage project. Current scope: a LiDAR capture validator and metadata inspector.

## Install

    pip install -e .[test]

## Usage

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

## Tests

    python -m pytest
