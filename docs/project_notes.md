# Project notes (living document)

Draft material for the AVPR seminar report. Update sections as pipeline runs and evaluation results arrive.

Last updated: 2026-06-30

---

## 1. Positioning relative to AffordanceCLIP

**Target report section:** Chapter 2 Literature Review, Section 2.3

Cuttano et al. (2024) introduce AffordanceCLIP, which unlocks affordance grounding from frozen CLIP via a Feature Pyramid Network trained on referring segmentation. Their method avoids pixel-level affordance annotations and supports open-vocabulary action prompts, but global CLIP features still lack spatial precision without this additional module. Evaluation on AGD20K measures localization quality (mask overlap), not whether the model selects the correct affordance among semantically similar alternatives. Reported failures include associating an action with the effect region rather than the grasped part (pencil tip for writing) and missing occluded interaction parts (bicycle seat for riding).

This project targets those gaps from a complementary angle. Rather than predicting where an action applies, the benchmark tests whether frozen CLIP and VL-JEPA representations can discriminate valid affordance captions from hard negatives that reference the same visible part with a wrong action (e.g., twist the cap to open versus twist the cap to refill). Qwen2.5-VL-7B generates open-vocabulary imperative captions; a CLIP adversarial filter retains visually and lexically confusable negatives; metrics include binary accuracy, confidence gap, and failure breakdown into spatial, attribute, and fluency types. PACO-LVIS images provide diverse everyday objects and states without manual affordance masks. VL-JEPA comparison extends beyond the CLIP-only analysis suggested in AffordanceCLIP's future work. Pixel grounding is not attempted; the focus is semantic affordance understanding at the representation level. Targeted contrastive fine-tuning is planned when failure analysis indicates representational limits rather than caption noise.

**Reference:** Cuttano, F., et al. (2024). What Does CLIP Know About Peeling a Banana? CVPR Workshop on Multimodal Algorithmic Reasoning (MAR). arXiv:2404.12015.

### What we address vs. what we do not

| AffordanceCLIP limitation | Addressed here? | Notes |
|---------------------------|-----------------|-------|
| Global CLIP, no spatial detail | Reframed | Probe global embeddings before adding FPN-style heads |
| Action vs. part confusion | Yes | Part-tied captions, hard negatives, attribute failure tags |
| Occlusion / part visibility | Partial | PACO states + spatial failure tags; no segmentation masks |
| Closed action vocabulary | Yes | Open Qwen captions |
| Costly affordance labels | Yes | VLM-generated captions + CLIP filter |
| Segmentation-only evaluation | Yes | Binary affordance choice + confidence gap |
| Pixel-level grounding | No | Out of scope |

---

## 2. Model and pipeline decisions

### Caption pipeline (stages 1 to 2, implemented)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Caption model | Qwen2.5-VL-7B-Instruct | Image-conditioned JSON captions with part references |
| Filter | CLIP ViT-L/14 adversarial | Drop easy negatives; keep confusable distractors |
| Filter mode | `gap` (default) | `sim(pos) - sim(neg) < min_similarity_gap` |
| Dataset (pilot) | 10 PNGs, PACO-style manifest | `data/sample/manifest.json` |

### Stage 4 inference (implemented)

| Backend | Module | Status |
|---------|--------|--------|
| CLIP (baseline) | `src/clip_scorer.py`, `src/evaluate.py` | Ready |
| Open-VLJEPA | `src/open_vljepa_scorer.py`, `src/evaluate.py` | Ready (needs `setup_open_vljepa.sh`) |

### Fine-tuning plan (after frozen baseline)

1. Frozen CLIP + frozen Open-VLJEPA evaluation
2. Failure analysis (spatial / attribute / fluency)
3. Targeted contrastive fine-tune on hard triplets from `filtered.json`
4. If no gain: LoRA on predictor, then Y-encoder ablation

**Success criteria (held-out split):** binary accuracy up, confidence gap up on previously wrong pairs, attribute failures down.

---

## 3. Failure attribution (for report Section 4 / 5)

Tag wrong predictions after stage 4 argmax:

| Type | Definition | Example |
|------|------------|---------|
| Spatial | Different visible parts in captions | cap vs spout |
| Attribute | Same part, wrong affordance | open vs refill |
| Fluency | Caption or Qwen artifact | awkward wording, not a model limit |

Optional later: occlusion / region masking pilot on CLIP (5 to 10 images).

---

## 4. Results log

Fill in after each experiment run. Copy numbers from `artifacts/eval/` when available.

### 4.1 Caption pipeline (pilot, N = 10)

Pilot settings (see `configs/config.yaml`):

- **1 positive + 1 negative** per image (`captions.num_most_probable` / `num_negative`)
- CLIP **gap** filter with up to **4** regen attempts; **fallback** to best rejected negative if none pass
- **25–55** character captions; Qwen retries when tiers come back empty
- Stage 4 eval: **CLIP only** for pilot (`eval.backends: [clip]`)

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Images processed | TBD | | |
| Images with pos+neg pair | TBD | | target 10/10 via fallback |
| Avg regen attempts | TBD | | from `pair_metadata.regen_attempts` |
| Fallback selections (%) | TBD | | `selection: fallback_best_rejected` |
| Filter mode | `gap` | | `min_similarity_gap: 0.08` |

### 4.2 Frozen CLIP inference

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Binary accuracy | TBD | | pos vs neg argmax |
| Mean confidence gap | TBD | | |
| Spatial failures (%) | TBD | | |
| Attribute failures (%) | TBD | | |
| Fluency failures (%) | TBD | | |

### 4.3 Frozen Open-VLJEPA inference

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Binary accuracy | TBD | | |
| Mean confidence gap | TBD | | |
| vs CLIP delta (accuracy) | TBD | | |

### 4.4 After targeted contrastive fine-tune (if run)

| Metric | Frozen | Fine-tuned | Delta | Date |
|--------|--------|------------|-------|------|
| Binary accuracy | TBD | TBD | TBD | |
| Mean confidence gap | TBD | TBD | TBD | |
| Attribute failure rate | TBD | TBD | TBD | |

### 4.5 Qualitative observations

- TBD: PCA clustering of image-caption pairs
- TBD: Example wins and failures (image id, captions, model choice)
- TBD: Cases analogous to AffordanceCLIP part confusion

---

## 5. Report snippets (draft)

Short paragraphs to paste or adapt into the final AVPR report. Expand when results exist.

### 5.1 Motivation (draft)

Physical affordance understanding asks what actions an object supports, not only what category it belongs to. Prior work such as AffordanceCLIP localizes affordance regions in images using adapted CLIP features. This project instead probes whether frozen vision-language embeddings already encode enough affordance knowledge to choose between valid and confusable action descriptions, and whether VL-JEPA representations differ from CLIP on that task.

### 5.2 Methods summary (draft)

Captions follow `[Verb] the [visible part] to [purpose]` (30 to 55 characters). Hard negatives share object and part wording but swap affordance. CLIP adversarial filtering removes trivial negatives. Inference compares cosine similarity (CLIP) or embedding distance (VL-JEPA) between the image and each caption tier.

### 5.3 Discussion hooks (fill after results)

- TBD: Did CLIP fail mainly on attribute or spatial pairs?
- TBD: Did hard negatives from the filter correlate with low confidence gaps?
- TBD: Does VL-JEPA improve attribute confusions where AffordanceCLIP reported semantic part errors?
- TBD: Fine-tune outcome vs failure type (contrastive fix vs frozen vision limit)

---

## 6. Changelog

| Date | Update |
|------|--------|
| 2026-06-30 | Initial notes: AffordanceCLIP positioning, pipeline decisions, empty results tables |
