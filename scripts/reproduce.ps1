<#
.SYNOPSIS
  Reproduce a benchmark: process each supplied capture and evaluate it against ground truth.

.DESCRIPTION
  Only the tiers whose capture path you supply are run. For each one:
      process  -> OutputDir\<tier>\{property.json, plan.png, run_report.json, diagnostics\}
      evaluate -> OutputDir\<tier>\evaluation\{evaluation.json, evaluation.md}      (only with -GroundTruth)
  then OutputDir\reproduction_summary.md / .json summarises every tier. A gate without ground truth is NOT EVALUATED, never PASS.
  Nothing is downloaded and no path is built in: everything comes from the parameters.

.PARAMETER LidarCapture   A LiDAR capture folder (depth/, odometry.csv, rgb.mp4, ...).
.PARAMETER VideoCapture   A walkthrough video file.
.PARAMETER PhotoCapture   A folder of room folders, each with 2-8 photos.
.PARAMETER GroundTruth    Ground-truth JSON for the property (benchmark\README.md). Used for every tier supplied.
.PARAMETER OutputDir      Result folder (required). An existing tier folder is only replaced with -Overwrite.
.PARAMETER IncumbentCsv   Optional incumbent comparison CSV (benchmark\incumbent_template.csv format).
.PARAMETER Overwrite      Replace previous results under OutputDir.
.PARAMETER Python         Interpreter (default: .venv if present, else python on PATH).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1 -LidarCapture C:\captures\prop_A\lidar -GroundTruth C:\benchmark\prop_A\ground_truth.json -OutputDir C:\results\prop_A
#>
param(
    [string]$LidarCapture,
    [string]$VideoCapture,
    [string]$PhotoCapture,
    [string]$GroundTruth,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$IncumbentCsv,
    [switch]$Overwrite,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not $Python) {
    $venvPython = Join-Path $repo ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { $Python = $venvPython } else { $Python = "python" }
}

$jobs = @()
if ($LidarCapture) { $jobs += [pscustomobject]@{ Tier = "lidar";  Capture = $LidarCapture } }
if ($VideoCapture) { $jobs += [pscustomobject]@{ Tier = "video";  Capture = $VideoCapture } }
if ($PhotoCapture) { $jobs += [pscustomobject]@{ Tier = "photos"; Capture = $PhotoCapture } }
if ($jobs.Count -eq 0) { Write-Error "Supply at least one of -LidarCapture, -VideoCapture, -PhotoCapture." }
if ($GroundTruth -and -not (Test-Path $GroundTruth)) { Write-Error "Ground truth file not found: $GroundTruth" }
foreach ($j in $jobs) { if (-not (Test-Path $j.Capture)) { Write-Error "$($j.Tier) capture not found: $($j.Capture)" } }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$summary = @()
foreach ($j in $jobs) {
    $out = Join-Path $OutputDir $j.Tier
    Write-Host ""
    Write-Host "=== $($j.Tier): process $($j.Capture)"
    $procArgs = @("-m", "spatialforge", "process", $j.Capture, "--tier", $j.Tier, "--output", $out)
    if ($Overwrite) { $procArgs += "--overwrite" }
    & $Python @procArgs
    $procExit = $LASTEXITCODE
    $row = [ordered]@{ tier = $j.Tier; process_exit_code = $procExit; status = $null; total_duration_seconds = $null
                       evaluated = $false; gates_pass = 0; gates_fail = 0; gates_not_evaluated = 0; note = "" }
    $reportPath = Join-Path $out "run_report.json"
    if (Test-Path $reportPath) {
        $rep = Get-Content $reportPath -Raw | ConvertFrom-Json
        $row.status = $rep.status
        $row.total_duration_seconds = $rep.total_duration_seconds
        if ($rep.errors -and $rep.errors.Count -gt 0) { $row.note = "$($rep.errors[0].code): $($rep.errors[0].message)" }
    } else {
        $row.note = "no run_report.json was written (see the console output above)"
    }
    if ($GroundTruth -and (Test-Path (Join-Path $out "property.json"))) {
        & $Python -m spatialforge evaluate --property $out --ground-truth $GroundTruth --output (Join-Path $out "evaluation")
        if ($LASTEXITCODE -eq 0) {
            $ev = Get-Content (Join-Path $out "evaluation\evaluation.json") -Raw | ConvertFrom-Json
            $row.evaluated = $true
            $row.gates_pass = @($ev.gates | Where-Object { $_.status -eq "PASS" }).Count
            $row.gates_fail = @($ev.gates | Where-Object { $_.status -eq "FAIL" }).Count
            $row.gates_not_evaluated = @($ev.gates | Where-Object { $_.status -eq "NOT EVALUATED" }).Count
            if ($ev.banner) { $row.note = ($row.note + " " + $ev.banner).Trim() }
        } else { $row.note = ($row.note + " evaluation failed").Trim() }
    } elseif ($GroundTruth) {
        $row.note = ($row.note + " nothing to evaluate: the run produced no property.json").Trim()
    } else {
        $row.note = ($row.note + " no ground truth supplied: processed only, no accuracy claim").Trim()
    }
    $summary += [pscustomobject]$row
}

if ($IncumbentCsv) {
    Write-Host ""
    Write-Host "=== incumbent comparison"
    & $Python -m spatialforge compare-incumbent --csv $IncumbentCsv --output (Join-Path $OutputDir "incumbent")
}

$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $OutputDir "reproduction_summary.json")
$lines = @("# Reproduction summary", "", "| Tier | Process exit | Status | Seconds | Evaluated | PASS | FAIL | NOT EVALUATED | Note |", "|---|---|---|---|---|---|---|---|---|")
foreach ($r in $summary) {
    $lines += "| $($r.tier) | $($r.process_exit_code) | $($r.status) | $($r.total_duration_seconds) | $($r.evaluated) | $($r.gates_pass) | $($r.gates_fail) | $($r.gates_not_evaluated) | $($r.note) |"
}
$lines | Set-Content -Encoding UTF8 (Join-Path $OutputDir "reproduction_summary.md")
Write-Host ""
Write-Host ($lines -join "`n")
Write-Host ""
Write-Host "Summary written to $(Join-Path $OutputDir 'reproduction_summary.md')"
if (@($summary | Where-Object { $_.process_exit_code -ne 0 }).Count -gt 0) { exit 1 }
exit 0
