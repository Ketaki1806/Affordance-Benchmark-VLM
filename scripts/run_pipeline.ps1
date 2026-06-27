# Run the caption generation pipeline on GPU (Windows / local).
# Usage (from project root in PowerShell, with conda env active):
#   .\scripts\run_pipeline.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvName = "affordance_benchmark"

Set-Location $ProjectRoot

if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda activate $EnvName
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:HF_HOME = Join-Path $ProjectRoot "artifacts\huggingface_cache"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "artifacts\logs") | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

Write-Host "Project root: $ProjectRoot"
Write-Host "PYTHONPATH:   $($env:PYTHONPATH)"
Write-Host "HF_HOME:      $($env:HF_HOME)"
Write-Host ""

python (Join-Path $ProjectRoot "scripts\verify_gpu.py")

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path $ProjectRoot "artifacts\logs\pipeline-$timestamp.log"

Write-Host "Running pipeline. Log: $logFile"
python (Join-Path $ProjectRoot "src\pipeline.py") 2>&1 | Tee-Object -FilePath $logFile
