"""Assemble the lightweight submission folder: dist/spatialforge_submission/ (and optionally a ZIP).

    python scripts/build_submission.py [--out dist/spatialforge_submission] [--tests-summary "352 passed"] [--zip]
                                       [--raw-benchmark-data C:\benchmark_raw] [--no-source] [--pdf]

Included: README.md, docs/, schema/, benchmark/ (templates, gate config; raw/ keeps only its README), evaluation_runs/ (if any),
scripts/, THIRD_PARTY.md, pyproject.toml, src/ and tests/ (so the bundle can be installed and reproduced; --no-source drops them),
git_commit.txt and manifest.json.
Never included: .git, virtual environments, caches, model weights, point clouds, videos, archives, large raw samples.
Real benchmark data is copied only when you pass --raw-benchmark-data, into raw_benchmark_data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spatialforge import __version__  # noqa: E402

EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build", "models", "outputs", ".mypy_cache", "node_modules"}
EXCLUDE_SUFFIXES = {".ply", ".mp4", ".mov", ".zip", ".pyc", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".heic"}
EXCLUDE_SUFFIX_DIRS = (".egg-info",)
MAX_FILE_MB = 5.0  # nothing larger than this is copied automatically
TOP_LEVEL_FILES = ["README.md", "THIRD_PARTY.md", "pyproject.toml", "LICENSE"]
TOP_LEVEL_DIRS = ["docs", "schema", "benchmark", "scripts", "evaluation_runs"]
SOURCE_DIRS = ["src", "tests"]

TIERS_IMPLEMENTED = {
    "lidar": "implemented; validated on 3 sample captures (runs, determinism, drift ablation); NOT validated against tape/laser ground truth",
    "video": "implemented; software path covered by mocked tests and one real scan-pass video; NOT validated on a real walkthrough or against ground truth",
    "photos": "implemented; software path covered by synthetic tests; NOT validated on real photos or against ground truth",
    "damage": "implemented (deterministic OpenCV crack/stain candidates, surface mapping, rule-based concealed-risk flags); NOT validated against labelled damage",
}
KNOWN_MISSING = [
    "tape/laser ground truth for any property", "real Tier-1 photo folders", "a real ordinary handheld walkthrough video",
    "a same-property all-three-tier benchmark", "a repeat capture", "an incumbent (consumer app) export",
    "a shipped before/after fix measured on real data", "confidence-interval calibration on real data",
]


def wanted(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS or any(p.endswith(EXCLUDE_SUFFIX_DIRS) for p in path.parts):
        return False
    return path.suffix.lower() not in EXCLUDE_SUFFIXES


def copy_tree(src: Path, dst: Path, skipped: list[str]) -> None:
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if not f.is_file():
            continue
        if src.name == "benchmark" and rel.parts[0] == "raw" and rel.name != "README.md":
            skipped.append(f"benchmark/{rel.as_posix()} (raw data stays outside the bundle)")
            continue
        if not wanted(rel):
            skipped.append(f"{src.name}/{rel.as_posix()}")
            continue
        if f.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            skipped.append(f"{src.name}/{rel.as_posix()} (over {MAX_FILE_MB} MB)")
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def real_benchmark_status(out: Path) -> dict:
    frozen = []
    for m in sorted(out.glob("evaluation_runs/*/*/manifest.json")):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("ground_truth_physical_measurement"):
            frozen.append(f"{d.get('slot')}/{d.get('name')}")
    raw = out / "raw_benchmark_data"
    return {"physical_ground_truth_runs_frozen": frozen, "raw_benchmark_data_included": raw.is_dir() and any(raw.iterdir()),
            "status": ("real benchmark runs are included: see evaluation_runs/" if frozen else
                       "NOT YET MEASURED: no physical (tape/laser) ground truth has been supplied, so no accuracy gate has been evaluated")}


def try_pdf(out: Path) -> str | None:
    """Render the technical report with scripts/render_report_pdf.py (Edge/Chrome + markdown). Never fatal."""
    md = out / "docs" / "technical_report.md"
    if not md.is_file():
        return None
    pdf = out / "SpatialForge_Technical_Report.pdf"
    try:
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "render_report_pdf.py"), "--input", str(md), "--output", str(pdf)],
                           capture_output=True, text=True, timeout=240)
    except (OSError, subprocess.SubprocessError):
        return None
    return pdf.name if r.returncode == 0 and pdf.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "dist" / "spatialforge_submission"))
    ap.add_argument("--tests-summary", default=None, help='e.g. "352 passed in 590 s" (recorded verbatim in manifest.json)')
    ap.add_argument("--raw-benchmark-data", help="folder with the REAL benchmark data to copy into raw_benchmark_data/")
    ap.add_argument("--no-source", action="store_true", help="leave out src/ and tests/")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    ap.add_argument("--pdf", action="store_true", help="try to render docs/technical_report.md to PDF (scripts/render_report_pdf.py; never fatal)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        if REPO in out.parents and out.parent.name == "dist":
            shutil.rmtree(out)  # only ever the dist/ build folder
        else:
            print(f"ERROR {out} exists and is not under dist/: refusing to delete it. Choose another --out.", file=sys.stderr)
            return 1
    out.mkdir(parents=True)
    skipped: list[str] = []
    for name in TOP_LEVEL_FILES:
        if (REPO / name).is_file():
            shutil.copy2(REPO / name, out / name)
    for name in TOP_LEVEL_DIRS + ([] if args.no_source else SOURCE_DIRS):
        if (REPO / name).is_dir():
            copy_tree(REPO / name, out / name, skipped)

    if args.raw_benchmark_data:
        src = Path(args.raw_benchmark_data)
        if not src.is_dir():
            print(f"ERROR raw benchmark data folder not found: {src}", file=sys.stderr)
            return 1
        shutil.copytree(src, out / "raw_benchmark_data", ignore=shutil.ignore_patterns(".git", "__pycache__"))

    sha, branch = git("rev-parse", "HEAD") or "unknown", git("branch", "--show-current") or "unknown"
    dirty = bool(git("status", "--porcelain"))
    (out / "git_commit.txt").write_text(f"{sha}\nbranch: {branch}\nworking tree had uncommitted changes when built: {dirty}\n", encoding="utf-8")

    pdf = try_pdf(out) if args.pdf else None
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "project": "SpatialForge", "version": __version__, "git_commit": sha, "git_branch": branch, "built_with_uncommitted_changes": dirty,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiers_implemented": TIERS_IMPLEMENTED,
        "test_result_summary": args.tests_summary or "not recorded: run scripts/smoke_test.ps1 -Full (python -m pytest -q)",
        "real_benchmark_status": real_benchmark_status(out),
        "known_missing_evidence": KNOWN_MISSING,
        "technical_report_pdf": pdf or "not generated: see docs/technical_report.md (conversion command inside)",
        "excluded_by_policy": sorted(EXCLUDE_DIRS) + sorted(EXCLUDE_SUFFIXES) + [f"files over {MAX_FILE_MB} MB"],
        "skipped_files": skipped[:200],
        "generated_files": [{"path": f.relative_to(out).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Built {out}\n  {len(files) + 1} files, {total / 1e6:.2f} MB; commit {sha[:12]} ({branch}){' [uncommitted changes]' if dirty else ''}")
    print(f"  real benchmark: {manifest['real_benchmark_status']['status']}")
    if args.zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, Path(out.name) / f.relative_to(out))
        print(f"  ZIP: {zpath} ({zpath.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
