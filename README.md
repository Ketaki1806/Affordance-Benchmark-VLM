# Affordance Benchmark

Physical affordance caption pipeline: **Qwen2.5-VL-7B** generates one positive and one hard-negative action caption per image; **CLIP** evaluates frozen zero-shot discrimination on those pairs. Based on [Probing Physical Affordance Understanding](Probing%20Physical%20Affordance%20Understanding.pdf).

Draft report notes and results log: [docs/project_notes.md](docs/project_notes.md) (update as experiments complete).

## Pipeline stages

1. **Generate**: Qwen2.5-VL-7B reads each image + object label, outputs JSON captions
2. **Validate**: enforce length/word limits from config
3. **Save**: same validated pairs go to `raw.json` and `filtered.json`

Outputs:
- `artifacts/captions/raw.json`
- `artifacts/captions/filtered.json` (input to stage 4 eval)

## Stage 4 evaluation (CLIP, optional Open-VLJEPA)

After the caption pipeline completes, run evaluation on `filtered.json`:

```bash
# Optional: Open-VLJEPA only if enabled in config.yaml
bash scripts/setup_open_vljepa.sh
huggingface-cli login

bash scripts/run_evaluate.sh
# or on submit node:
bash scripts/condor_submit_evaluate.sh
```

| Backend | What it measures |
|---------|------------------|
| **CLIP** | Binary affordance choice (pos vs neg similarity) |
| **Open-VLJEPA** | Same binary task with VL-JEPA embeddings (optional) |

Outputs:
- `artifacts/eval/clip.json`
- `artifacts/eval/open_vljepa.json`
- `artifacts/eval/summary.json`

Set `models.open_vljepa.enabled: false` in `config.yaml` to skip Open-VLJEPA.

### Analyze CLIP results (local)

```bash
pip install matplotlib
export PYTHONPATH=src   # or $env:PYTHONPATH="src" on PowerShell
python src/analyze_clip_results.py
```

Prints a text summary, writes `artifacts/eval/clip_analysis.json`, and saves plots under `artifacts/eval/figures/` (PNG if matplotlib is installed, otherwise SVG).

## Project layout

```
configs/              # config.yaml, prompt templates
data/sample/          # pilot images + manifest.json
src/                  # pipeline, models, validators, CLIP filter
scripts/              # run + HTCondor cluster helpers
requirements-gpu.txt  # Python deps (PyTorch via micromamba on cluster)
```

## Sample data setup

1. Place images in `data/sample/` and list them in `data/sample/manifest.json` (`file` + `object` label per row)
2. The bundled pilot set uses 10 PNGs (`bottle.png`, `bowl.png`, …); see the manifest for the full mapping
3. Run the pipeline (GPU required for Qwen-VL)

Caption format (matches document examples):

```
[Verb] the [visible part] to [short purpose/state].
```

Example: `"Twist the cap to open the bottle."` (33 chars)

## Local setup (Windows)

```powershell
.\scripts\setup_env.ps1
conda activate affordance_benchmark
pip install -r requirements-gpu.txt
.\scripts\run_pipeline.ps1
```

## LST cluster (HTCondor)

| Node | Purpose |
|------|---------|
| `login.lst.uni-saarland.de` | Edit files, env setup |
| `ssh submit` | Submit HTCondor jobs |

### Setup

```bash
cd ~/Affordance-Benchmark-VLM
bash scripts/install_micromamba.sh
bash scripts/setup.sh
bash scripts/condor_submit_install.sh   # on submit node
```

### Run affordance caption pipeline (GPU)

Qwen2.5-VL-7B needs ~16–22 GB VRAM; submit file requests **32 GB** RAM + 1 GPU.

```bash
ssh submit
cd ~/Affordance-Benchmark-VLM
bash scripts/condor_submit_pipeline.sh
tail -f artifacts/logs/pipeline.<ClusterId>.out
```

### Check HTCondor resource fit

```bash
bash scripts/condor_check_resources.sh
```

### Each session

```bash
source scripts/activate_env.sh
```

### Reset cluster install

```bash
bash scripts/cluster_cleanup.sh --all
```

## PACO-LVIS (cluster)

### Scale choice

PACO-LVIS **val** has ~2,410 images and ~20.9k part segments. Enumerating every part instance (~21k Qwen jobs) is out of seminar scope. The **full-val** run uses **one preferred interaction part per unique image** (≤ ~2,410 pairs): `cap`/`lid`/`handle`/… ranking from `build_paco_pilot_manifest.py`. Captions are **model-generated** (Qwen → rule validation → CLIP); human filter stays on the N=20 pilot only. Open-VLJEPA is left disabled for the full-val CLIP run (same `filtered.json` can be scored later).

### Annotations + images

1. Download annotations (login node; `wget` may be missing — use `curl`):

```bash
mkdir -p data/paco/annotations data/paco/images
cd data/paco/annotations
curl -L -o paco_lvis_v1.zip https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
unzip paco_lvis_v1.zip   # or: python -c "import zipfile; zipfile.ZipFile('paco_lvis_v1.zip').extractall('.')"
```

2. Point `--image-root` at COCO 2017 images (`val2017/` / `train2017/`). PACO-LVIS reuses those files.

### Pilot (N≈20, human-check)

```bash
cd ~/Affordance-Benchmark-VLM
source scripts/activate_env.sh
bash scripts/build_paco_pilot_manifest.sh \
  --ann data/paco/annotations/paco_lvis_v1_val.json \
  --image-root /path/to/coco \
  --n 20 \
  --require-image \
  --copy-images
```

Then set `data.sample_dir` / `data.manifest_path` to the pilot paths, run `condor_submit_pipeline.sh`, human-edit captions, evaluate.

### Full val (one preferred part / image)

```bash
# Login: build manifest (prefer pointing sample_dir at COCO root; avoid copying ~2k images)
# Prefer an existing COCO tree (same root as the pilot). Do NOT use /path/to/coco.
# If images are already on disk:
bash scripts/build_paco_val_manifest.sh \
  --ann data/paco/annotations/paco_lvis_v1_val.json \
  --image-root data/paco/coco \
  --require-image

# Or build the list first, then download missing jpgs into that root (~2k files):
bash scripts/build_paco_val_manifest.sh \
  --ann data/paco/annotations/paco_lvis_v1_val.json \
  --image-root data/paco/coco \
  --download-missing
```

`configs/config.yaml` already targets:

```yaml
data:
  sample_dir: data/paco/coco          # must match --image-root above
  manifest_path: data/paco/manifest_val_full.json
output:
  raw_captions: artifacts/captions/val_full/raw.json
  filtered_captions: artifacts/captions/val_full/filtered.json
  eval_clip: artifacts/eval/val_full/clip.json
  eval_summary: artifacts/eval/val_full/summary.json
```

Sharded Qwen generation + merge + CLIP (submit node):

```bash
ssh submit
cd ~/Affordance-Benchmark-VLM
bash scripts/condor_submit_pipeline_sharded.sh 10   # 10 shards; resume-safe
# wait until all Process jobs finish
bash scripts/merge_caption_shards.sh
bash scripts/condor_submit_evaluate.sh
```

Pipeline CLI also supports `--shard-index`, `--num-shards`, `--limit`, `--no-resume`, `--save-every`.

Manifest schema fields: `paco_category`, `part`, `attributes`, `source_split` (optional extras beyond `image_id` / `file` / `object`).

## Configuration

Edit [configs/config.yaml](configs/config.yaml):

| Key | Default | Purpose |
|-----|---------|---------|
| `captions.min_chars` / `max_chars` | 25 / 55 | Caption length limits (pilot uses 25 min) |
| `captions.num_most_probable` | 1 | Positive captions per image (pilot) |
| `captions.num_negative` | 1 | Negative captions per image (pilot) |
| `models.clip_device` | `cuda` | CLIP device for stage 4 eval (full-val) |

## Notes

- Login node `/tmp` is full; scripts use `~/tmp` on nethome
- Do not use `request_runtime` in HTCondor submit files on LST
- Use `should_transfer_files = NO` (nethome is shared across nodes)
