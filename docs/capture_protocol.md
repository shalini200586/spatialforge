# Capture protocol (one page)

Follow this literally. Pick the tier that matches the phone you have; the folder layout at the bottom is the same for all three.

**Before you start (all tiers):** switch the lights on, open the doors between rooms and keep them open, and move loose clutter away from the walls. Do not use digital zoom, filters, portrait mode, HDR-enhancing or "beauty" modes. Keep the phone's original files; never send screenshots or messaging-app copies.

## Tier 1 - Photos (iPhone 15 or newer, 2-8 photos per room)

1. Use the **normal 1x rear camera**, phone held in portrait or landscape (stay consistent within a room).
2. Make one folder per room (`room_01`, `kitchen`, ...). Put only that room's photos in it.
3. Stand in a corner and photograph the whole room from 2-4 positions. **Every wall and every corner must appear whole** in at least two photos, with about half of each photo overlapping the previous one.
4. Photograph **each doorway from both sides** (once from each room). This is what lets the rooms be joined into one plan.
5. Hold still for each photo. Delete blurred or dark ones and retake. No flash unless the room is dark.
6. Keep the original JPEG/HEIC-converted-to-JPEG files and their metadata. HEIC is not read directly: set the camera to "Most Compatible" or convert to JPEG first.

## Tier 2 - Video (iPhone 15 or newer)

1. **Normal 1x rear camera**, 4K or 1080p, 30 fps, **one continuous walkthrough** (no pauses to start a new clip).
2. **Walk slowly** (about one step per second) at chest height, camera pointing along the walking direction, slightly tilted to see the wall/ceiling junction.
3. In each room walk the perimeter once, then pass **through every doorway** into the next room and keep going; return the same way if the route is a dead end. Do not stop between rooms.
4. Avoid fast turns (no whip pans), rotating on the spot, covering the lens, and long stretches of a bare white wall: keep some furniture, door frames or switches in view.
5. 3-8 minutes for a small property is typical.

## Tier 3 - LiDAR (iPhone Pro or iPad Pro with LiDAR)

1. **Capture app (stock, off the shelf):** the supported sample format matches the export of the free iOS app **Stray Scanner** (random hex folder names; `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`, `imu.csv`). Any app that exports this layout works; confirm the layout of your app version against `python -m spatialforge validate-lidar`.
2. Start recording, then walk slowly (about 0.5 m/s), keeping walls, corners and floor in view. Cover every room and pass through every doorway. Close the loop by returning to where you started.
3. Move the phone smoothly; avoid fast rotation. Stop recording at the starting point.
4. Export the capture folder from the app (Files / AirDrop / cable). It must contain:

| File | Required | What it is |
|---|---|---|
| `depth/000000.png ...` | yes | 16-bit depth maps (LiDAR, millimetres) |
| `odometry.csv` | yes | camera poses (`timestamp, frame, x, y, z, qx, qy, qz, qw`, per-frame intrinsics) |
| `rgb.mp4` | recommended | colour video (needed for damage analysis and plan context) |
| `camera_matrix.csv` | recommended | camera intrinsics |
| `confidence/000000.png ...` | recommended | 8-bit depth confidence (0-2) |
| `imu.csv` | optional | inertial data if the app exports it |

## Hand-off folder

    property_name\
        photos\   room_01\IMG_001.JPG ...   room_02\...        (Tier 1)
        video\    walkthrough.mp4                               (Tier 2)
        lidar\    <capture-id>\ rgb.mp4, depth\, confidence\, odometry.csv, camera_matrix.csv, imu.csv   (Tier 3)
        ground_truth.json                                       (benchmark only: tape/laser measurements)

Then run one command per capture, for example `python -m spatialforge process property_name\lidar --tier lidar --output C:\results\lidar`.

## Avoid

- **Mirrors, large glass, windows facing the sun, glossy floors:** depth and SfM fail on them; keep them out of frame or cover them where possible.
- **Low light, motion blur, exposure changes** (the phone hunting between a bright window and a dark room).
- **Extreme motion:** fast walking, whip pans, shaking.
- **Missing room transitions:** a room not connected to the others by a captured doorway cannot be placed in the plan.
- **Compressed or edited photos/video:** messaging-app copies, screenshots, trimmed or re-encoded clips.
- **Moving people or pets** in the frames.
