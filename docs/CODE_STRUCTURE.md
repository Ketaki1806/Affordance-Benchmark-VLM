# Code structure

This note explains how the Affordance Benchmark repo is organized and what each
major piece is for.

## What the project does (in one paragraph)

We take PACO-LVIS part images, ask Qwen2.5-VL to write one **valid** affordance
caption and one **hard negative** (same part, wrong action / attribute / place),
then ask vision–language models (CLIP, SigLIP, Open-VLJEPA) which caption fits
the image better. Accuracy is simply “did the positive score higher?”. We also
run ranking fine-tuning, tag frozen-model failures, and compute light XAI
metrics (occlusion vision share, embedding modality gap, word heatmaps). The
paper lives under `papers/seminar/`; figures are regenerated from
`notebooks/paper_figures.ipynb`.

## Top-level layout

```
Affordance_Benchmark/
├── configs/          YAML configs + caption prompts
├── data/sample/      Tiny demo images + manifest (safe to ship)
├── src/              Python pipeline, scorers, FT, analysis
├── scripts/          Shell wrappers (local + HTCondor) and small plot helpers
├── notebooks/        Paper figure notebook
├── papers/seminar/   LaTeX paper, bib, final fig_*.png
├── humaneval/        Frozen / FT eval JSON dumps used in the paper
├── tests/            Unit tests for filtering, grounding, XAI helpers
├── docs/             Project notes + this file
├── artifacts/        LOCAL ONLY — captions, eval, logs, checkpoints (gitignored)
└── vendor/           LOCAL ONLY — large third-party checkouts (gitignored)
```

Anything under `artifacts/` is produced by runs on the LST cluster or a local
GPU machine. Do not submit those folders; they are huge and regenerable.

## End-to-end data flow

1. **Manifest** — JSON list of `{image_id, object, part, image_path, ...}`.
   Built from PACO with `src/build_paco_val_manifest.py` (and train variants).
2. **Caption generation** — `src/pipeline.py` loads Qwen, builds a prompt from
   `caption_generation_prompt.py` / `configs/prompt_template.txt`, validates
   JSON with `caption_validator.py`, writes `raw.json` and `filtered.json`
   (same content in the current setup; no adversarial CLIP filter).
3. **Scoring** — `src/evaluate.py` loads each backend scorer, scores pos and
   neg, marks `correct` if `pos_score >= neg_score`, writes per-backend JSON
   plus a summary.
4. **Optional FT** — `finetune_clip.py` / `finetune_open_vljepa.py` train a
   ranking objective on train captions, then eval configs point scorers at the
   new checkpoint.
5. **Error tags / XAI** — `scripts/plot_error_tags.py` tags failures;
   `attribution_occlusion.py`, `compute_modality_gap.py`, `word_grounding.py`
   produce the JSON/PNG used in the paper notebook.
6. **Paper figures** — `notebooks/paper_figures.ipynb` turns table numbers +
   artifact PNGs into `papers/seminar/fig_*.png` for `main.tex`.

## `src/` module map

| File | Role |
|------|------|
| `pipeline.py` | Caption generation loop (resume, sharding, checkpoints) |
| `model.py` | Qwen2.5-VL wrapper |
| `caption_generation_prompt.py` | Builds the affordance prompt |
| `caption_validator.py` | Length / JSON / tier checks |
| `dataset_io.py` | Manifest + caption JSON I/O, shard helpers |
| `config_loader.py` | Load YAML, resolve paths relative to repo root |
| `evaluate.py` | Binary pos-vs-neg evaluation driver |
| `clip_scorer.py` / `siglip_scorer.py` / `open_vljepa_scorer.py` | Image–text cosine scorers |
| `finetune_clip.py` / `finetune_open_vljepa.py` | Ranking fine-tuning |
| `build_paco_*.py` | PACO → manifest builders |
| `adversarial_filter.py` / `pairs_to_filtered.py` | Older / optional filter utilities |
| `analyze_clip_results.py` / `analyze_caption_yspace.py` | Post-eval analysis |
| `attribution_occlusion.py` | Patch occlusion → vision share |
| `compute_modality_gap.py` | Image/text centroid gap + matched cosines |
| `word_grounding.py` / `grounding_text.py` | Word / phrase heatmaps |
| `compute_cider.py` | Caption quality side metric |
| `logger.py` | Shared logging |

### Scoring rule (important)

For every `(image, positive, negative)`:

```
correct = score(image, positive) >= score(image, negative)
```

`confidence_gap` is `|pos_score - neg_score|`. That is the whole metric used for
frozen and fine-tuned tables in the paper.

## `scripts/` conventions

- `run_*.sh` / `run_*.ps1` — local entry points (activate env, call `python -m` or `src/...`).
- `condor_submit_*.sh` — write/submit HTCondor jobs on the LST cluster.
- `condor_run_*.sh` — what the Condor job actually executes on the worker.
- `setup*.sh` / `install_*.sh` — micromamba env + deps.
- `plot_error_tags.py` — failure-tag SVG + shared `tag()` used by the notebook.
- `build_seminar_report_docx.py` / `recheck_report_analysis.py` — local-only report tooling (gitignored).

## Configs

- `configs/config.yaml` — default paths, caption settings, eval backends.
- Overlays such as `config_train_ft.yaml`, `config_eval_clip_ft_train1000.yaml`
  swap manifests, output dirs, and checkpoints for FT experiments.
- `prompt_template.txt` / `prompt_regeneration.txt` — caption prompts.

## Paper + figures

- `papers/seminar/main.tex` + `custom.bib` — LaTeX write-up.
- `papers/seminar/fig_*.png` — figures included by LaTeX (keep these).
- `papers/seminar/assets/proposal_pipeline.png` — source art for `fig_pipeline.png`.
- `notebooks/paper_figures.ipynb` — regenerate all paper PNGs (Pillow only).

Hard-coded table numbers live in the notebook data cell so figures can be
redrawn without re-running cluster jobs. If local eval dumps exist under
`artifacts/eval/...`, the fix-rate cell can recompute CLIP FT1000 rates live.

## `humaneval/`

Dated folders (`30jul`, `31jul`, `1aug`, …) hold the JSON eval dumps that
backed the paper numbers (frozen CLIP / SigLIP / Open-VLJEPA and FT variants).
`humaneval/images` is local preview media and is gitignored.

## What not to ship

Keep these **local only** (already gitignored or deleted from the tree):

- `artifacts/` — captions, checkpoints, logs, attribution dumps
- `vendor/` — large model / third-party checkouts
- `data/paco/` COCO images and PACO zips (download separately)
- `papers/seminar/_proposal_imgs/` — temporary PDF extract junk
- `.pytest_cache/`, `.superpowers/`, `.ipynb_checkpoints/`, `__pycache__/`
- Scratch `_tmp_*` scripts

## Typical reproduction path

```bash
# 1) env
bash scripts/setup.sh          # or scripts/setup_env.ps1 on Windows

# 2) captions (needs GPU + Qwen)
bash scripts/run_pipeline.sh

# 3) eval
# set eval.backends in configs/config.yaml
bash scripts/run_evaluate.sh

# 4) paper figures
# open notebooks/paper_figures.ipynb → run helpers, data, then "Write all"
```

For PACO download steps, see `README.md`.
For experimental notes and number provenance, see `docs/project_notes.md`.
