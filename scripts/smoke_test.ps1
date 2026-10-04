<#
.SYNOPSIS
  Quick confidence check that an installation works.

.DESCRIPTION
  1. runs the tests: by default a FAST subset (evaluation harness, canonical models, LiDAR pipeline integration; about a minute);
     with -Full the whole suite (python -m pytest -q, roughly ten minutes on a laptop CPU);
  2. python -m spatialforge --version
  3. python -m spatialforge doctor
  4. optionally runs `process` on a capture you supply.

.PARAMETER Full
  Run the whole test suite instead of the fast subset.
.PARAMETER Sample
  A capture to process end to end (a LiDAR capture folder, a video file, or a folder of room photo folders). The tier is auto-detected.
.PARAMETER OutputDir
  Where the sample result goes (default: a new folder under the temp directory; an existing folder is never overwritten).
.PARAMETER Python
  The interpreter to use (default: .venv if present, else python on PATH).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1 -Sample C:\captures\single_room
#>
param(
    [switch]$Full,
    [string]$Sample,
    [string]$OutputDir,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $Python) {
    $venvPython = Join-Path $repo ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { $Python = $venvPython } else { $Python = "python" }
}
$failed = @()

function Step($name, [scriptblock]$action) {
    Write-Host ""
    Write-Host "== $name"
    & $action
    if ($LASTEXITCODE -ne 0) { $script:failed += $name; Write-Host "   -> FAILED (exit code $LASTEXITCODE)" } else { Write-Host "   -> ok" }
}

if ($Full) {
    Step "full test suite (python -m pytest -q)" { & $Python -m pytest -q }
} else {
    Step "fast test subset (use -Full for everything)" {
        & $Python -m pytest -q tests/test_evaluation.py tests/test_pipeline_models.py tests/test_pipeline_integration.py
    }
}
Step "spatialforge --version" { & $Python -m spatialforge --version }
Step "spatialforge doctor" { & $Python -m spatialforge doctor }

if ($Sample) {
    if (-not $OutputDir) { $OutputDir = Join-Path ([IO.Path]::GetTempPath()) ("spatialforge_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
    Step "process $Sample (tier auto-detected) -> $OutputDir" {
        & $Python -m spatialforge process $Sample --tier auto --output $OutputDir
    }
    foreach ($f in "property.json", "plan.png", "run_report.json") {
        if (-not (Test-Path (Join-Path $OutputDir $f))) { Write-Host "   missing output: $f" }
    }
}

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host "SMOKE TEST FAILED: $($failed -join '; ')"
    exit 1
}
Write-Host "SMOKE TEST PASSED"
exit 0
