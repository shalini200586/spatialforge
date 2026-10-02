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

## Tests

    python -m pytest
