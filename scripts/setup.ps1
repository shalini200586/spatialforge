<#
.SYNOPSIS
  Create a local virtual environment (.venv) and install SpatialForge into it. Never touches the global Python.

.DESCRIPTION
  - requires Python 3.11 (the tested version);
  - creates .venv next to the repository if it does not exist;
  - upgrades pip INSIDE the venv only;
  - installs the project in editable mode with the extras you ask for.

.PARAMETER Video
  Also install the video and photo tier dependencies (PyCOLMAP, PyTorch CPU, transformers, OpenCV).
.PARAMETER Damage
  Also install the damage-analysis dependency (OpenCV only). Included in -Video.
.PARAMETER Test
  Also install pytest.
.PARAMETER VenvDir
  Where to create the environment (default: .venv in the repository).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Video -Damage -Test
#>
param(
    [switch]$Video,
    [switch]$Damage,
    [switch]$Test,
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Find-Python311 {
    # the py launcher first (it can pick 3.11 even when another Python is first on PATH), then python on PATH
    $candidates = @(@("py", "-3.11"), @("python"))
    foreach ($c in $candidates) {
        $exe = $c[0]
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $args2 = @()
        if ($c.Count -gt 1) { $args2 = $c[1..($c.Count - 1)] }
        try {
            $v = & $exe @args2 -c "import sys; print('%d.%d' % sys.version_info[:2])"
        } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $v.Trim() -eq "3.11") { return ,@($exe) + $args2 }
    }
    return $null
}

$py = Find-Python311
if (-not $py) {
    Write-Error "Python 3.11 was not found. Install it from https://www.python.org/downloads/ (or 'winget install Python.Python.3.11') and run this script again. Nothing was changed."
}
Write-Host "Using Python 3.11: $($py -join ' ')"

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment in $VenvDir ..."
    $rest = @()
    if ($py.Count -gt 1) { $rest = $py[1..($py.Count - 1)] }
    & $py[0] @rest -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Could not create the virtual environment." }
} else {
    Write-Host "Reusing the existing virtual environment in $VenvDir"
}

Write-Host "Upgrading pip inside the virtual environment ..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed." }

$extras = @()
if ($Video) { $extras += "video" }
if ($Damage -and -not $Video) { $extras += "damage" }
if ($Test) { $extras += "test" }
$target = "."
if ($extras.Count -gt 0) { $target = ".[" + ($extras -join ",") + "]" }
Write-Host "Installing SpatialForge ($target) ..."
& $venvPython -m pip install -e $target
if ($LASTEXITCODE -ne 0) { Write-Error "Installation failed. Nothing outside $VenvDir was modified." }

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Activate the environment:   $VenvDir\Scripts\Activate.ps1      (or call $venvPython directly)"
if ($Video) {
    Write-Host "  2. Download the metric-depth weights ONCE (about 99 MB, stored outside the repository):"
    Write-Host "         python scripts\download_models.py"
    Write-Host "     Set SPATIALFORGE_MODELS to keep them in another folder."
} else {
    Write-Host "  2. Video and photo tiers need the extras and the model weights: rerun with -Video, then python scripts\download_models.py"
}
Write-Host "  3. Check the machine:          python -m spatialforge doctor"
Write-Host "  4. Run a tier:                 python -m spatialforge process C:\captures\room --tier lidar --output C:\results\room"
