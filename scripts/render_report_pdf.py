"""Render docs/technical_report.md to PDF. OPTIONAL: nothing in SpatialForge depends on it, and failure is reported, not fatal.

    python scripts/render_report_pdf.py [--input docs/technical_report.md] [--output dist/SpatialForge_Technical_Report.pdf]

Needs (both usually already present on a Windows machine, neither is a SpatialForge dependency):
  * the `markdown` Python package   (pip install markdown)
  * Microsoft Edge or Google Chrome (used headless to print the HTML to PDF)
Alternative with Pandoc installed:   pandoc docs/technical_report.md -o SpatialForge_Technical_Report.pdf
Alternative without anything:       open the generated .html in a browser and print to PDF.
Exit code 0 = PDF written, 1 = no PDF (the reason is printed).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
CSS = """
body { font-family: Segoe UI, Arial, sans-serif; font-size: 10pt; line-height: 1.35; margin: 0; color: #111; }
h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 12pt 0 4pt; border-bottom: 1px solid #ccc; } h3 { font-size: 11pt; margin: 8pt 0 3pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 4pt 0; } th, td { border: 1px solid #bbb; padding: 2pt 4pt; vertical-align: top; }
th { background: #eee; } code, pre { font-family: Consolas, monospace; font-size: 8.5pt; background: #f5f5f5; } pre { padding: 4pt; white-space: pre-wrap; }
blockquote { margin: 4pt 0; padding-left: 8pt; border-left: 3px solid #c90; color: #333; } p, li { margin: 2pt 0; }
@page { size: A4; margin: 16mm; }
"""


def find_browser() -> str | None:
    for b in BROWSERS:
        if Path(b).is_file():
            return b
    return shutil.which("msedge") or shutil.which("chrome")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=str(REPO / "docs" / "technical_report.md"))
    ap.add_argument("--output", default=str(REPO / "dist" / "SpatialForge_Technical_Report.pdf"))
    args = ap.parse_args()
    try:
        import markdown
    except ImportError:
        print("No PDF: the 'markdown' package is not installed (pip install markdown). See this script's header for alternatives.", file=sys.stderr)
        return 1
    browser = find_browser()
    if not browser:
        print("No PDF: neither Microsoft Edge nor Google Chrome was found. See this script's header for alternatives.", file=sys.stderr)
        return 1
    src, out = Path(args.input), Path(args.output)
    if not src.is_file():
        print(f"No PDF: {src} not found.", file=sys.stderr)
        return 1
    html = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>", encoding="utf-8")
        profile = Path(tmp) / "profile"
        cmd = [browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri()]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"No PDF: the browser failed ({exc}).", file=sys.stderr)
            return 1
    if not out.is_file() or out.stat().st_size == 0:
        print("No PDF: the browser produced no output. Open the report in a browser and print to PDF instead.", file=sys.stderr)
        return 1
    print(f"Wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
