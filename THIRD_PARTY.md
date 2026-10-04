# Third-party software and models

SpatialForge itself has **no license file yet** (the owner has not chosen one). The table lists what it uses. "Believed" means the project's published license as I know it; **confirm against the installed package metadata (`pip show <name>`) and the upstream repository before redistributing anything.** Nothing below is bundled in the repository: packages are installed by `pip`, and model weights are downloaded by the user into a folder outside the repository.

| Component | Purpose in SpatialForge | Source / project | License |
|---|---|---|---|
| **Open3D** | point clouds, point-to-plane ICP, normals, pose-graph optimisation (LiDAR drift correction) | http://www.open3d.org / github.com/isl-org/Open3D | believed MIT; verification required |
| **COLMAP / PyCOLMAP** | structure-from-motion: features, matching, incremental mapping (video and photo tiers) | colmap.github.io / github.com/colmap/colmap | believed BSD-3-Clause; verification required |
| **OpenCV** (`opencv-python-headless`) | video decoding, optical flow, sharpness, damage detectors | opencv.org | believed Apache-2.0 for current releases; verification required |
| **NumPy** | arithmetic | numpy.org | believed BSD-3-Clause; verification required |
| **Pillow** | 16-bit PNG decoding, plan rendering | python-pillow.org | believed HPND (MIT-CMU style); verification required |
| **PyTorch** (CPU) | runtime for the metric depth model | pytorch.org | believed BSD-3-Clause style; verification required |
| **Transformers**, **huggingface_hub** | load the depth model; download script | github.com/huggingface | believed Apache-2.0; verification required |
| **Depth Anything V2 Metric-Indoor-Small** (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`) | monocular **metric** depth, used only to establish metric scale (video and photo tiers) | Hugging Face Hub model card: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf ; paper: "Depth Anything V2" (Yang et al., 2024) | **not verified here: verification required.** Read the model card's license for your use (including any non-commercial terms) before use or redistribution. This repository does not assert one. |
| **truststore** (optional) | use the OS certificate store when downloading weights behind a TLS-inspecting proxy | pypi.org/project/truststore | believed MIT; verification required |
| **pytest** (tests only) | test runner | pytest.org | believed MIT; verification required |

**Capture format reference (not a dependency, not bundled):** the LiDAR sample format matches the export of the free iOS app Stray Scanner. SpatialForge only reads files in that layout; it contains no code from the app. Check the app's own terms for your use.

**Model weights** are cached by `scripts/download_models.py` in `%SPATIALFORGE_MODELS%` or `~/.spatialforge/models`; the script prints the repository and the exact commit it downloaded. They are excluded from Git (`.gitignore`) and from the submission bundle.
