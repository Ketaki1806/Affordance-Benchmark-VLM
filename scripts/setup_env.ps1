# Create or update the conda environment from environment.yml (Windows / local GPU).
# Usage (from project root in PowerShell):
#   .\scripts\setup_env.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvName = "affordance_benchmark"

Set-Location $ProjectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not found. Install Miniconda/Anaconda and reopen the terminal."
}

$envExists = conda env list | Select-String "^\s*$EnvName\s"

if ($envExists) {
    Write-Host "Updating existing environment: $EnvName"
    conda env update -n $EnvName -f environment.yml --prune
} else {
    Write-Host "Creating environment: $EnvName"
    conda env create -f environment.yml
}

Write-Host ""
Write-Host "Activate the environment with:"
Write-Host "  conda activate $EnvName"
Write-Host ""
Write-Host "Then run:"
Write-Host "  .\scripts\run_pipeline.ps1"
