# Affordance Benchmark

Hard-negative affordance caption benchmark: Qwen2.5-VL writes one valid and one
hard negative caption per PACO part image; CLIP / SigLIP / Open-VLJEPA score which
caption fits better. Paper: `papers/seminar/`.

For layout and module explanations, see [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md).
Working notes: [docs/project_notes.md](docs/project_notes.md).

## Requirements

- Python 3.10+
- Conda / micromamba (see `environment.yml`)
- GPU recommended for caption generation and Open-VLJEPA
- Pillow + Jupyter to regenerate paper figures

```bash
# Linux
bash scripts/setup.sh
source scripts/activate_env.sh
pip install -r requirements-gpu.txt

# Windows
.\scripts\setup_env.ps1
conda activate affordance_benchmark
pip install -r requirements-gpu.txt
```

## Quick start

```bash
# Sample images already under data/sample/
export PYTHONPATH=src
python src/pipeline.py --limit 2
python src/evaluate.py --backends clip
```

Paper figures:

```bash
# open notebooks/paper_figures.ipynb
# run helpers → data → "Write all paper PNGs"
```

## Configuration

Defaults live in [`configs/config.yaml`](configs/config.yaml):

| Key | Meaning |
|-----|---------|
| `data.manifest_path` | Image list for captioning |
| `output.filtered_captions` | Eval input JSON |
| `eval.backends` | `clip` / `siglip` / `open_vljepa` |
| `models.clip` | HF CLIP id |
| `models.clip_checkpoint` | Optional FT weights |

Overlays: `configs/config_train_ft.yaml`, `config_eval_*.yaml`.

## Common commands

```bash
bash scripts/run_pipeline.sh
bash scripts/run_evaluate.sh
bash scripts/run_yspace_analysis.sh
```

PACO val N=100:

```bash
bash scripts/build_paco_val_manifest.sh \
  --ann data/paco/annotations/paco_lvis_v1_val.json \
  --image-root data/paco/coco \
  --require-image --n 100 --seed 42 \
  --output data/paco/manifest_val_100.json
```

## License

Academic use. Third-party models (Qwen, CLIP, SigLIP, Open-VLJEPA,
PACO/COCO) keep their own licenses.
