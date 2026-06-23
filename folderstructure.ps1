$root = "D:\Uni\SS2026\Seminar\Project\Affordance_Benchmark"
New-Item -ItemType Directory -Force -Path $root | Out-Null

# --- Root files with content ---
Set-Content -Path "$root\.gitignore" -Value @"
# Ignore data
data/
artifacts/

# Python
__pycache__/
*.pyc
.env
.venv

# Jupyter
.ipynb_checkpoints/
"@

Set-Content -Path "$root\README.md" -Value @"
# Affordance Benchmark Project

This repository contains the codebase, configuration, and workflow for the Affordance Benchmark project.
"@

New-Item -ItemType File -Force -Path "$root\requirements.txt" | Out-Null
New-Item -ItemType File -Force -Path "$root\Dockerfile" | Out-Null
New-Item -ItemType File -Force -Path "$root\Makefile" | Out-Null

# --- Folder structure ---
$folders = @(
    "configs",
    "data\raw",
    "data\interim",
    "data\processed",
    "notebooks",
    "src",
    "artifacts\models",
    "tests"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $folder) | Out-Null
}

# --- Subfolder files ---
Set-Content "$root\configs\config.yaml" -Value "project_name: affordance_benchmark"
New-Item -ItemType File -Force -Path "$root\configs\logging_config.py" | Out-Null

New-Item -ItemType File -Force -Path "$root\notebooks\1.0-eda.ipynb" | Out-Null
New-Item -ItemType File -Force -Path "$root\notebooks\2.0-modeling.ipynb" | Out-Null

$srcFiles = @("data_loader.py","preprocessor.py","model.py","train.py","evaluate.py","pipeline.py")
New-Item -ItemType File -Force -Path "$root\src\__init__.py" | Out-Null
foreach ($file in $srcFiles) {
    New-Item -ItemType File -Force -Path "$root\src\$file" | Out-Null
}

Set-Content "$root\artifacts\metrics.json" -Value "{}"

New-Item -ItemType File -Force -Path "$root\tests\test_data.py" | Out-Null
New-Item -ItemType File -Force -Path "$root\tests\test_model.py" | Out-Null

Write-Host "Enhanced project structure created successfully!"
