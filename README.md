# Affordance Benchmark

Physical affordance caption pipeline: **Qwen2.5-VL-7B** generates most-probable and hard-negative action captions; **CLIP** adversarial filtering removes easy negatives. Based on [Probing Physical Affordance Understanding](Probing%20Physical%20Affordance%20Understanding.pdf).

Draft report notes and results log: [docs/project_notes.md](docs/project_notes.md) (update as experiments complete).

## Pipeline stages

1. **Generate**: Qwen2.5-VL-7B reads each image + object label, outputs JSON captions
2. **Validate**: enforce 30–55 chars (target 35–50), 5–10 words, affordance structure
3. **Filter**: CLIP zero-shot drops easy negatives; regenerate hard negatives with Qwen if needed

Outputs:
- `artifacts/captions/raw.json`: before filtering
- `artifacts/captions/filtered.json`: after CLIP filter + `filter_metadata`

## Stage 4 evaluation (CLIP vs Open-VLJEPA)

After the caption pipeline completes, compare frozen **CLIP** and **Open-VLJEPA** on `filtered.json`:

```bash
# One-time: clone repo + download checkpoint (needs HF access to Llama/Gemma)
bash scripts/setup_open_vljepa.sh
huggingface-cli login

# Run locally or via HTCondor GPU job
bash scripts/run_evaluate.sh
# or on submit node:
bash scripts/condor_submit_evaluate.sh
```

Outputs:
- `artifacts/eval/clip.json`
- `artifacts/eval/open_vljepa.json`
- `artifacts/eval/summary.json`

Set `models.open_vljepa.enabled: false` in `config.yaml` to run CLIP-only eval.

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
| `filter.min_similarity_gap` | 0.08 | CLIP gap threshold for hard negatives |
| `filter.max_regen_attempts` | 4 | Qwen regen rounds when negatives are too easy |
| `filter.allow_fallback` | true | Keep best rejected negative rather than empty tier |
| `filter.mode` | `gap` | Adversarial filter strategy (see below) |
| `models.clip_device` | `cpu` | CLIP on CPU while Qwen uses GPU |

### Adversarial filter modes (`filter.mode`)

| Mode | Keep negative when… | Use when… |
|------|---------------------|-----------|
| **`gap`** (default) | `sim(pos) - sim(neg) < min_similarity_gap` and CLIP does not prefer neg | Standard; matches project document |
| **`neg_sim_floor`** | `sim(neg) >= min_neg_image_sim` and CLIP does not prefer neg | Negatives were unrelated to image (too easy visually) |
| **`text_and_gap`** | gap rule **and** CLIP text similarity(pos, neg) ≥ `min_text_sim` | Negatives must use confusable affordance wording |

Other approaches (not implemented): human review, LLM critique loop, VL-JEPA distance (stage 4), pool top-k ranking among all candidates.

Change mode in `configs/config.yaml`:
```yaml
filter:
  mode: text_and_gap
  min_similarity_gap: 0.05
  min_text_sim: 0.85
```

## Notes

- Login node `/tmp` is full; scripts use `~/tmp` on nethome
- Do not use `request_runtime` in HTCondor submit files on LST
- Use `should_transfer_files = NO` (nethome is shared across nodes)
