# Affordance Benchmark

Qwen2.5-VL-7B writes one positive and one hard-negative affordance caption per image; CLIP / SigLIP / Open-VLJEPA score which caption fits the image better.

Notes and numbers: [docs/project_notes.md](docs/project_notes.md).

## Pipeline

1. Generate captions (Qwen)
2. Validate length / JSON
3. Save `raw.json` + `filtered.json`
4. Evaluate on `filtered.json`

## Eval

```bash
# set eval.backends in configs/config.yaml (clip | siglip | open_vljepa)
bash scripts/condor_submit_evaluate.sh

# text-only pos/neg closeness (EmbeddingGemma)
bash scripts/run_yspace_analysis.sh

# Open-VLJEPA setup (needs HF access to Llama + EmbeddingGemma)
bash scripts/setup_open_vljepa.sh
hf auth login
```

| Backend | Role |
|---------|------|
| CLIP | pos vs neg cosine |
| SigLIP | same task |
| Y-space | cos(pos, neg) text only |
| Open-VLJEPA | same task (gated deps) |

Outputs under `artifacts/eval/val_full/`.

```bash
pip install matplotlib
export PYTHONPATH=src
python src/analyze_clip_results.py
```

## Layout

```
configs/   config.yaml, train/eval overlays
data/      sample + PACO manifests
src/       pipeline, scorers, FT
scripts/   local + HTCondor
```

## Sample data

1. Images in `data/sample/` listed in `manifest.json`
2. Caption style: `[Verb] the [part] to [purpose].`

## Local (Windows)

```powershell
.\scripts\setup_env.ps1
conda activate affordance_benchmark
pip install -r requirements-gpu.txt
.\scripts\run_pipeline.ps1
```

## LST cluster

| Node | Use |
|------|-----|
| `login.lst.uni-saarland.de` | edit / setup |
| `ssh submit` | Condor |

```bash
cd ~/Affordance-Benchmark-VLM
bash scripts/install_micromamba.sh
bash scripts/setup.sh
bash scripts/condor_submit_install.sh   # on submit
```

Pipeline (GPU, ~32 GB RAM request):

```bash
ssh submit
cd ~/Affordance-Benchmark-VLM
bash scripts/condor_submit_pipeline.sh
tail -f artifacts/logs/pipeline.<ClusterId>.out
```

```bash
source scripts/activate_env.sh
bash scripts/condor_check_resources.sh
# reset: bash scripts/cluster_cleanup.sh --all
```

## PACO-LVIS

Val has ~2410 images / ~21k part segments. We use **one preferred part per image**. Seminar scale-up: **N=100** subsample (`manifest_val_100.json`). Human filter on pilot N=20 only.

Annotations:

```bash
mkdir -p data/paco/annotations data/paco/images
cd data/paco/annotations
curl -L -o paco_lvis_v1.zip https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
unzip paco_lvis_v1.zip
```

Images: COCO 2017 under `--image-root` (`train2017` / `val2017`).

Pilot:

```bash
bash scripts/build_paco_pilot_manifest.sh \
  --ann data/paco/annotations/paco_lvis_v1_val.json \
  --image-root /path/to/coco \
  --n 20 --require-image --copy-images
```

N=100:

```bash
bash scripts/build_paco_val_manifest.sh \
  --ann data/paco/annotations/paco_lvis_v1_val.json \
  --image-root data/paco/coco \
  --require-image --n 100 --seed 42 \
  --output data/paco/manifest_val_100.json
```

```bash
bash scripts/condor_submit_pipeline.sh
bash scripts/condor_submit_evaluate.sh
```

Train FT captions / ranking FT: `configs/config_train_ft.yaml`, `condor_submit_train_captions.sh`, `condor_submit_finetune_clip.sh`, `condor_submit_finetune_open_vljepa.sh`.

Human FT eval: `bash scripts/condor_submit_evaluate_human.sh`.

## Config

See [configs/config.yaml](configs/config.yaml) for caption length, backends, checkpoints.

## Cluster notes

- Prefer `~/tmp` (login `/tmp` often full)
- No `request_runtime` on LST
- `should_transfer_files = NO` (nethome shared)
