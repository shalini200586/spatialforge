# Sample evidence (internal; NOT a ground-truth benchmark)

Small artifacts from runs on the three sample LiDAR captures and one real video. They show that the pipeline runs end to end on real data, what its outputs look like, how long it takes, and what the drift stage decided. They say **nothing** about dimensional accuracy: no tape or laser measurement exists for any of these captures.

| Folder | What | Command (paths shortened) |
|---|---|---|
| `lidar_single_room/` | `property.json`, `plan.png`, `run_report.json`, `console.txt` | `python -m spatialforge process C:\captures\single_room --tier lidar --output C:\results\lidar_single_room` |
| `lidar_single_scan_floor_only/` | same | `... C:\captures\single_scan_floor_only --tier lidar ...` |
| `lidar_single_scan_with_ceiling/` | same | `... C:\captures\single_scan_with_ceiling --tier lidar ...` |
| `video_single_room_rgb/` | video-tier result on the capture's own `rgb.mp4` (a LiDAR **scanning pass**, not a walkthrough), plus `trajectory.png`, `scale_report.json` | `python -m spatialforge process C:\captures\single_room\<id>\rgb.mp4 --tier auto --diagnostics --output C:\results\video_single_room_rgb` |
| `drift/<capture>/` | `drift_report.json`, before/after top-down views | `python -m spatialforge correct-lidar-drift C:\captures\<capture> --output-dir C:\results\drift_<capture>` |

Notes:

- Paths were rewritten (`C:\captures`, `C:\results`) so no machine-specific location is exposed. Raw captures, point clouds and videos are not included.
- The video run was made before two warning-classification rules (`VIDEO_DENSER_RETRY`, `GRAVITY_AXES_DISAGREE`) were added, so four of its warnings appear as `GENERAL_WARNING` here; the readable messages are unchanged and current code classifies them.
- Every result is PARTIAL by design. See [../internal_sample_report.md](../internal_sample_report.md) for what these runs show and do not show, and [../drift_ablation.md](../drift_ablation.md) for the drift numbers.
