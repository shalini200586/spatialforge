"""Download the metric-depth weights used by the video and photo tiers into a cache OUTSIDE the repository.

    python scripts/download_models.py [--cache-dir DIR] [--revision COMMIT]

Model:    depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (Hugging Face Hub), about 99 MB.
Default:  %SPATIALFORGE_MODELS% or ~/.spatialforge/models. The weights are never committed and no test downloads them.
Revision: by default the repository's current head; the exact commit that was downloaded is printed so a run can be reproduced.
          Pin it with --revision <commit-hash> or the SPATIALFORGE_MODEL_REVISION environment variable.
License:  see the model card at https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf. Verify the model's
          license for your use before relying on it (THIRD_PARTY.md does not assert one).
Proxy:    behind a TLS-inspecting corporate proxy, `pip install truststore` lets Python use the operating system's certificate
          store (verification stays ON); this script uses it automatically when it is installed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spatialforge.config import DEPTH_MODEL_REPO, MODELS_ENV, models_dir  # noqa: E402

REVISION_ENV = "SPATIALFORGE_MODEL_REVISION"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", type=Path, default=None, help=f"default: ${MODELS_ENV} or ~/.spatialforge/models")
    ap.add_argument("--revision", default=os.environ.get(REVISION_ENV), help="commit hash to pin (default: current head)")
    args = ap.parse_args()
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run:  pip install -e .[video]", file=sys.stderr)
        return 1

    cache = (args.cache_dir or models_dir()) / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Model:    {DEPTH_MODEL_REPO}  (Hugging Face Hub)")
    print(f"Revision: {args.revision or 'current head of the repository (the commit is printed below)'}")
    print(f"Cache:    {cache}")
    try:
        path = snapshot_download(DEPTH_MODEL_REPO, cache_dir=str(cache), revision=args.revision)
    except Exception as exc:  # network, proxy, auth and disk errors all end up here: say what to check, no traceback
        print(f"\nDownload failed: {type(exc).__name__}: {exc}\n"
              "Check the network connection and any proxy (pip install truststore helps behind a TLS-inspecting proxy), "
              "and that the cache folder is writable.", file=sys.stderr)
        return 1
    print(f"\nDownloaded {DEPTH_MODEL_REPO}\n  to {path}\n  revision (commit): {Path(path).name}")
    print(f"Use it by setting {MODELS_ENV}={cache.parent} (or keep the default location).")
    print("Check the model's license for your use: see the model card linked in this script's header.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
