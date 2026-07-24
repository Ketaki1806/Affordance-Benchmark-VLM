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

## PACO-LVIS pilot (cluster)

1. Download annotations (login node; `wget` may be missing — use `curl`):

```bash
mkdir -p data/paco/annotations data/paco/images
cd data/paco/annotations
curl -L -o paco_lvis_v1.zip https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
unzip paco_lvis_v1.zip   # or: python -c "import zipfile; zipfile.ZipFile('paco_lvis_v1.zip').extractall('.')"
```

2. Point `--image-root` at COCO 2017 images (`val2017/` / `train2017/`). PACO-LVIS reuses those files.

3. Build a 15–20 image pilot manifest (one image per category, prefers interaction parts like cap/handle/lid):

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

4. Point `configs/config.yaml` at the pilot:

```yaml
data:
  sample_dir: data/paco/images
  manifest_path: data/paco/manifest_pilot.json
```

5. Submit the caption pipeline on the submit node, then human-review `artifacts/captions/raw.json` before CLIP eval.

Manifest schema fields: `paco_category`, `part`, `attributes`, `source_split` (optional extras beyond `image_id` / `file` / `object`).

## Configuration

Edit [configs/config.yaml](configs/config.yaml):

| Key | Default | Purpose |
|-----|---------|---------|
| `captions.min_chars` / `max_chars` | 25 / 55 | Caption length limits (pilot uses 25 min) |
| `captions.num_most_probable` | 1 | Positive captions per image (pilot) |
| `captions.num_negative` | 1 | Negative captions per image (pilot) |
| `models.clip_device` | `cpu` | CLIP on CPU during stage 4 eval |

## Notes

- Login node `/tmp` is full; scripts use `~/tmp` on nethome
- Do not use `request_runtime` in HTCondor submit files on LST
- Use `should_transfer_files = NO` (nethome is shared across nodes)
