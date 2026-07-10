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

## PACO-LVIS scaling (future)

Use the same `manifest.json` schema with optional fields: `paco_category`, `part`, `attributes`, `source_split`.

Sampling strategy:
1. **Pilot (10–20)**: one image per diverse category (bottle, bowl, drawer, knife, …)
2. **Medium (~200)**: stratified by PACO object category
3. **Full**: LVIS images with clear part annotations and affordance-relevant attributes

Point `data.manifest_path` in `configs/config.yaml` at the new manifest; no code changes needed.

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
