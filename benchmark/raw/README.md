# Raw benchmark data (not in Git)

Real captures are large (a LiDAR capture is hundreds of MB to several GB) and are **never committed** (`.gitignore` blocks the data here except this file). Keep them outside the repository and point the scripts at them. For the submission, the ZIP either contains them (`python scripts/build_submission.py --raw-benchmark-data <folder>` copies them into `raw_benchmark_data/`) or references them, depending on the size limit.

## Expected external structure

    benchmark_raw\
        property_A\                      # the multi-room property: 3+ rooms and a connector
            ground_truth.json            # tape/laser measurements (benchmark/ground_truth.schema.json)
            photos\    room_01\ ...  room_02\ ...  connector\ ...      # Tier 1
            video\     walkthrough.mp4                                   # Tier 2
            lidar\     <capture-id>\ rgb.mp4, depth\, confidence\, odometry.csv, camera_matrix.csv, imu.csv   # Tier 3
            incumbent\ export.csv  (+ the app's own export files)        # consumer-app scan of two rooms
        repeatability\
            roomX_capture1\ ...          # same room, captured again (same tier)
            roomX_capture2\ ...
            ground_truth_capture1.json   # same repeat_group, capture_id "A" / "B"
            ground_truth_capture2.json
        damage\
            furnished_room\ ...          # furnished room with real damage
            ground_truth.json            # includes damage[] measurements

The same property should be captured with **all three tiers** so the tiers can be compared on one ground truth. See [../../docs/capture_protocol.md](../../docs/capture_protocol.md).

## Running it

    powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1 `
        -LidarCapture C:\benchmark_raw\property_A\lidar\<capture-id> `
        -VideoCapture C:\benchmark_raw\property_A\video\walkthrough.mp4 `
        -PhotoCapture C:\benchmark_raw\property_A\photos `
        -GroundTruth C:\benchmark_raw\property_A\ground_truth.json -OutputDir C:\results\property_A
