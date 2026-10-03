"""Download the metric-depth weights used by the video tier into a cache OUTSIDE the repository.

    python scripts/download_models.py [--cache-dir DIR]

Default cache: %SPATIALFORGE_MODELS% or ~/.spatialforge/models (the weights are ~100 MB and never committed).
Behind a TLS-inspecting corporate proxy, `pip install truststore` lets Python use the operating system's certificate
store (verification stays ON); this script uses it automatically when it is installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spatialforge.video.depth import MODEL_REPO, models_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    from huggingface_hub import snapshot_download

    cache = (args.cache_dir or models_dir()) / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(MODEL_REPO, cache_dir=str(cache))
    print(f"Downloaded {MODEL_REPO}\n  to {path}")
    print(f"Use it by setting SPATIALFORGE_MODELS={cache.parent} (or keep the default location).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
