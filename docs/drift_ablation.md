# Drift ablation (LiDAR tier)

**Sample/internal validation. NOT PHYSICAL GROUND TRUTH BENCHMARK.** The metrics below measure the internal consistency of the reconstruction (floor flatness, wall sharpness, frame-to-frame agreement). They do not measure dimensional accuracy against tape or laser.

The supplied odometry is not trusted blindly: `correct-lidar-drift` builds a baseline reconstruction (OFF, supplied poses) and a corrected one (ON) on identical frames, filtering and voxel size, compares them, and **only uses the corrected poses when the metrics justify it**. Every `process` run records the decision (`production_pose_source`, `fallback_decisions`) in `run_report.json`.

Regenerate (about 1-2 minutes per capture; deterministic, the numbers below were reproduced exactly on a second run):

    python -m spatialforge correct-lidar-drift C:\captures\single_room --output-dir C:\results\drift_room

## Method

1. **Odometry initial guess.** The capture's poses (camera-to-world) seed everything.
2. **Sequential point-to-plane ICP.** Depth clouds of consecutive sampled frames (400 frames, evenly spaced, 3 cm registration voxel) are registered with Open3D; each relation must pass fitness, RMSE and correction-size gates, otherwise that pair keeps its odometry relation.
3. **Conservative loop closure.** Candidate pairs that are far apart in the sequence but close in odometry space with similar heading are registered with stricter gates (fitness >= 0.6, RMSE <= 4 cm, correction <= 0.4 m and 8 degrees). At most 20 candidates are tried.
4. **Pose-graph optimisation.** Open3D optimises the graph. Every consecutive pair keeps an odometry edge as an anchor, sequential ICP edges get a small weight (0.02) and loop closures full weight. Roll and pitch stay as supplied (gravity-constrained): only heading and position may change.
5. **Safe acceptance / fallback.** The corrected poses become the production poses only if no metric got meaningfully worse (> 3% relative degradation), at least one got meaningfully better (>= 2%), the footprint changed by <= 5%, and the correction sizes are plausible (<= 0.5 m, <= 5 degrees, tilt change <= 1 degree). Otherwise the report says `"accepted": false`, `"production_poses": "original"`, and gives the reason. `after.ply` is always written so a rejected candidate can still be inspected.

## Results on the three sample captures

| Capture | Frames used / available | Sequential edges accepted | Loop closures accepted / candidates | Decision | Production poses |
|---|---|---|---|---|---|
| `single_room` | 400 / 1,715 | 379 / 399 | 6 / 20 | **accepted** | corrected |
| `single_scan_floor_only` | 400 / 5,251 | 362 / 399 | 5 / 20 | rejected: *metrics got worse: floor_band_thickness_m, floor_peak_share* | **original** |
| `single_scan_with_ceiling` | 400 / 9,745 | 325 / 399 | 5 / 20 | rejected: *metrics got worse: floor_band_thickness_m, floor_peak_share* | **original** |

Consistency metrics, OFF -> ON (supplied poses -> corrected candidate):

| Metric (lower or higher is better) | `single_room` | `single_scan_floor_only` | `single_scan_with_ceiling` |
|---|---|---|---|
| Floor band thickness m (lower) | 0.1290 -> 0.1305 | 0.1173 -> 0.1270 | 0.0996 -> 0.1244 |
| Floor peak share (higher) | 0.1674 -> 0.1740 | 0.1746 -> 0.1569 | 0.0894 -> 0.0814 |
| Neighbour residual median m (lower) | 0.01394 -> 0.01393 | 0.01881 -> 0.01877 | 0.02943 -> 0.02939 |
| Frame overlap (higher) | 0.647 -> 0.651 | 0.535 -> 0.545 | 0.418 -> 0.422 |
| Wall-slab cells per 1000 points (higher) | 14.01 -> 14.17 | 15.20 -> 15.15 | 14.37 -> 14.14 |
| Loop-constraint residual: mean translation m / rotation deg | 0.052 / 0.96 -> 0.020 / 0.45 | 0.017 / 1.04 -> 0.007 / 0.31 | 0.049 / 0.63 -> 0.037 / 0.41 |
| Sequential-constraint residual m / deg | 0.0102 / 0.260 -> 0.0101 / 0.257 | 0.0158 / 0.365 -> 0.0158 / 0.358 | 0.0231 / 0.412 -> 0.0231 / 0.409 |
| Mean / max pose correction (translation m; rotation deg) | 0.020 / 0.061; 0.32 / 0.85 | 0.048 / 0.088; 0.60 / 1.05 | 0.040 / 0.114; 0.23 / 0.55 |

## What this does and does not show

- **Correction was accepted only where the metrics justified it.** On `single_room` the loop-closure residuals more than halved and no metric degraded beyond the limit, so corrected poses are used (and a `process` run on it reports `Production poses: corrected`).
- **On the two larger scans the correction made the floor *less* flat** (floor band thickness +8% and +25%, floor peak share down), even though the loop constraints it was optimising agreed better. The acceptance rules rejected it and the **original poses were kept**. This is the safe fallback working as designed, and it is why every run records which poses it used.
- **No universal improvement is claimed.** One of three captures improved, by small margins on the consistency metrics (mostly < 5%); two fell back. The loop-residual drop on the rejected captures shows the optimiser satisfied its own constraints while worsening the structure those constraints do not see.
- Sequential constraints were already almost perfectly consistent (0.01-0.02 m) before correction: the supplied ARKit odometry is locally good, and the drift this stage addresses is the slow, loop-scale kind.
- Only 400 evenly spaced frames are used (the default); a denser sampling could change the outcome. Earlier experiments showed 80 frames of a 9,745-frame capture were too far apart for ICP to register most pairs.
- Nothing here is compared with physical measurements; whether the corrected poses are closer to the true geometry is **NOT VALIDATED**.

Artifacts for these runs (small ones only): [evidence/drift/](evidence/README.md).
