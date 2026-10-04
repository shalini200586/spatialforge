# Demo checklist (live defence)

Have a command window open in the repository, the virtual environment active, `SPATIALFORGE_MODELS` set, and one result folder already generated as a fallback.

1. **Check the machine:** `python -m spatialforge doctor` - show which tiers are READY and that model weights live outside the repository.
2. **Show the input capture:** open the capture folder (LiDAR: `depth\`, `odometry.csv`, `rgb.mp4`) or play the video / open a photo room folder.
3. **Run one command:** `python -m spatialforge process C:\captures\single_room --tier lidar --output C:\results\demo` - point at the `[n/N]` progress and the final status line (SUCCESS / PARTIAL / FAILURE).
4. **Open `plan.png`:** rooms with dimensions, accepted openings, dashed walls outside closed rooms, the PARTIAL badge and its note.
5. **Open `property.json`:** `rooms[].area`, `walls`, `openings`, `damage_regions`, `provenance`.
6. **Point to uncertainty:** a measurement's `interval`, `quality`, `confidence` (a heuristic score, not a probability), and `warnings` / `warning_records`.
7. **Explain partial and failure behaviour:** open `run_report.json` - the 12 standard stages with status and timing; PARTIAL means useful output with weak or missing stages; FAILURE writes no `property.json`. Show `evidence/video_single_room_rgb` (weak tracking, PARTIAL) as a real example.
8. **Show the drift ablation:** [drift_ablation.md](drift_ablation.md) - before/after metrics, accepted on one capture, fell back to the original poses on two.
9. **Show a damage region and its rule:** a `damage_regions` entry with surface, room and metric extent; a `concealed_damage_flags` entry names its rule (`CDR-00x`), condition and triggering detections; a `scope_line_items` entry keyed to the surface.
10. **Show the benchmark table:** `python -m spatialforge evaluate --property ... --ground-truth ...` - PASS / FAIL / NOT EVALUATED. Say plainly which gates have real ground truth behind them (none yet) and read the NOT YET MEASURED lines in [benchmark_report.md](benchmark_report.md).
11. **Show the fix loop:** `scripts\freeze_baseline.py`, `scripts\compare_runs.py`, the declaration template. State that **the real before/after is pending real benchmark data** and show the unit-tested mechanics.

## What to say if video or photo fails

- **Do not bluff.** A failure is a feature: the run exits non-zero, prints `ERROR [CODE] message` with a hint, writes no `property.json` or `plan.png`, and never hands over a confident-looking wrong plan.
- Open `run_report.json`: `errors[0].code` (`SFM_FAILED`, `METRIC_SCALE_UNAVAILABLE`, `NO_GEOMETRY`, `INPUT_INVALID`, ...), the failing stage, and the warnings before it (`VIDEO_TRACKING_WEAK`, `SFM_DISCONNECTED_MODELS`).
- Explain the cause in plain terms: weak tracking (blur, bare walls, fast motion fragment the SfM into disconnected models), or no consistent metric scale (the depth model and the trajectory disagree, so no metres are invented).
- Show `diagnostics\` (`trajectory.png`, `scale_report.json`, `run.log`) as the evidence, and fall back to the pre-generated result or the LiDAR tier.
- Say what capture change would fix it (slower walk, more texture, overlap, doorways from both sides: [capture_protocol.md](capture_protocol.md)).
