# Project notes (living document)

Draft material for the AVPR seminar report (`D:\Uni\SS2026\Seminar\Project\Report\main.tex`).  
Update sections as pipeline runs and evaluation results arrive.

Last updated: 2026-07-25

### Report structure map (ACL template)

| Report section (`main.tex`) | Content in these notes |
|-----------------------------|------------------------|
| Abstract + Introduction | §5.1 motivation; research question: do frozen VL embeddings discriminate valid vs hard-negative affordance captions? |
| Related Work | §1 AffordanceCLIP positioning |
| Experimental Setup | §2 pipeline; §4.1 PACO pilot; **§7 human-check** |
| Results | §4 metrics tables; **§7.2 qualitative caption examples** |
| Conclusion | §5.3 discussion hooks |
| Appendix | prompts, AI use disclaimer |

---

## 1. Positioning relative to AffordanceCLIP

**Target report section:** Related Work

Cuttano et al. (2024) introduce AffordanceCLIP, which unlocks affordance grounding from frozen CLIP via a Feature Pyramid Network trained on referring segmentation. Their method avoids pixel-level affordance annotations and supports open-vocabulary action prompts, but global CLIP features still lack spatial precision without this additional module. Evaluation on AGD20K measures localization quality (mask overlap), not whether the model selects the correct affordance among semantically similar alternatives. Reported failures include associating an action with the effect region rather than the grasped part (pencil tip for writing) and missing occluded interaction parts (bicycle seat for riding).

This project targets those gaps from a complementary angle. Rather than predicting where an action applies, the benchmark tests whether frozen CLIP and VL-JEPA representations can discriminate valid affordance captions from hard negatives that reference the same visible part with a wrong action (e.g., twist the cap to open versus twist the cap to refill). Qwen2.5-VL-7B generates open-vocabulary imperative captions; captions are treated as **candidates** and human-validated on the pilot; metrics include binary accuracy, confidence gap, and failure breakdown into spatial, attribute, and fluency types. PACO-LVIS images provide diverse everyday objects and parts. VL-JEPA comparison extends beyond the CLIP-only analysis suggested in AffordanceCLIP's future work. Pixel grounding is not attempted; the focus is semantic affordance understanding at the representation level.

**Reference:** Cuttano, F., et al. (2024). What Does CLIP Know About Peeling a Banana? CVPR Workshop on Multimodal Algorithmic Reasoning (MAR). arXiv:2404.12015.

### What we address vs. what we do not

| AffordanceCLIP limitation | Addressed here? | Notes |
|---------------------------|-----------------|-------|
| Global CLIP, no spatial detail | Reframed | Probe global embeddings before adding FPN-style heads |
| Action vs. part confusion | Yes | Part-tied captions, hard negatives, attribute failure tags |
| Occlusion / part visibility | Partial | PACO parts + spatial failure tags; no segmentation masks |
| Closed action vocabulary | Yes | Open Qwen captions |
| Costly affordance labels | Yes | VLM candidates + **human review** on pilot |
| Segmentation-only evaluation | Yes | Binary affordance choice + confidence gap |
| Pixel-level grounding | No | Out of scope |

---

## 2. Model and pipeline decisions

### Caption pipeline (implemented)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Caption model | Qwen2.5-VL-7B-Instruct | Image-conditioned JSON: 1 pos + 1 hard neg |
| Adversarial CLIP filter | **Removed** | Circular if CLIP both filters and evaluates |
| Human validation | Pilot (N=20) | Ground captions; fix Qwen mistakes |
| Dataset (current) | PACO-LVIS val, 1 image/category | `data/paco/manifest_pilot.json` + COCO jpgs |

### Stage 4 inference

| Backend | Module | Status |
|---------|--------|--------|
| CLIP (baseline) | `src/clip_scorer.py`, `src/evaluate.py` | Ready |
| Open-VLJEPA | `src/open_vljepa_scorer.py` | Optional (`enabled: false`) |

### Scope of “negative”

Not task-/goal-conditioned (screwdriver for turning vs tapping). Negatives = physically implausible or wrong affordance for the **visible object/part/state** in the image. PACO supplies object, part, and appearance attributes; Qwen proposes the action; human review confirms grounding.

---

## 3. Failure attribution (for report Results)

Tag wrong predictions after stage 4 argmax:

| Type | Definition | Example |
|------|------------|---------|
| Spatial | Different visible parts in captions | cap vs spout; handle vs rim |
| Attribute | Same part, wrong affordance / state | open vs refill; pour onto floor |
| Fluency | Caption or Qwen artifact | wrong object, unrelated action |

---

## 4. Results log

Fill in after each experiment run. Copy numbers from `artifacts/eval/` when available.

### 4.1 Caption pipeline (PACO-LVIS pilot, N = 20)

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Images processed | 20 | 2026-07-25 | Qwen job completed |
| Images with pos+neg pair | 20/20 | 2026-07-25 | format-validated only |
| Human-validated pairs | TBD | | see §7 |
| Dataset | PACO-LVIS val | | 1 category / image |
| Filter | none | | human review instead |

### 4.2 Frozen CLIP inference

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Binary accuracy | **0.70** (14/20) | 2026-07-25 | raw Qwen pairs (not yet human-validated) |
| Mean confidence gap | **0.030** | 2026-07-25 | small margin → many near-ties |
| Spatial / attribute / fluency failures | TBD | | run `analyze_clip_results.py` + human tags |

**Caveat for the report:** these numbers use format-valid Qwen captions only. Several pairs fail human-check (wrong object, unrelated negative; see §7). Re-run CLIP after editing `filtered.json` before claiming PACO-LVIS pilot results.

### 4.3 Frozen Open-VLJEPA inference

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Binary accuracy | TBD | | optional |
| vs CLIP delta | TBD | | |

---

## 5. Report snippets (draft)

### 5.1 Motivation (draft)

Physical affordance understanding asks what actions an object supports, not only what category it belongs to. Prior work such as AffordanceCLIP localizes affordance regions. This project probes whether frozen vision-language embeddings already encode enough affordance knowledge to choose between valid and confusable action descriptions.

### 5.2 Methods summary (draft)

Captions follow `[Verb] the [visible part] to [purpose]`. Hard negatives should share object/part wording but swap affordance. On the pilot, Qwen proposes pairs; humans accept, edit, or drop them before CLIP/VL-JEPA eval. Inference compares cosine similarity between the image and each caption.

### 5.3 Discussion hooks

- TBD: Did CLIP fail mainly on attribute or spatial pairs?
- TBD: How often did human review reject Qwen negatives as ungrounded?
- TBD: Does VL-JEPA improve attribute confusions?

---

## 6. Changelog

| Date | Update |
|------|--------|
| 2026-07-26 | CLIP PACO pilot: 70% (14/20), mean gap 0.030 on raw Qwen captions |
| 2026-07-25 | PACO N=20 pipeline; human-check protocol + qualitative caption examples (§7); aligned to Report `main.tex` |
| 2026-07-08 | Ablation study table |
| 2026-06-30 | Initial notes |

---

## 7. Human-check captions (Experimental Setup → Results qualitative)

**Target report sections:** Experimental Setup (procedure); Results (qualitative examples).

### 7.1 Validation protocol

Qwen outputs are **candidates**, not ground truth. For each of the 20 PACO pilot images, open the image next to `artifacts/captions/raw.json` and mark:

| Verdict | Meaning |
|---------|---------|
| **keep** | Positive is valid for visible state/part; negative is same object/part, wrong affordance |
| **edit** | Fix wording by hand (preferred when close) |
| **regen / drop** | Wrong object, unrelated action, or unusable negative |

Checklist:

1. **Part match** — caption uses the annotated / clearly visible interaction part  
2. **State match** — positive is plausible for what is visible  
3. **Hard negative** — same object/part, wrong affordance (not a different valid goal)  
4. **Not unrelated** — negative must not jump to another object  
5. **Format** — imperative action phrase  

Only human-validated (or edited) pairs go into CLIP eval numbers claimed as PACO-LVIS pilot results.

Source files (local copies for writing):

- Captions: `raw.json` (from cluster `artifacts/captions/raw.json`)
- Images: `images/XXXXXXXXXXXX.jpg`

### 7.2 Qualitative examples (for report Results figure)

Four PACO-LVIS pilot examples with Qwen positive / negative captions and a provisional human verdict. Use these as figure panels in the report (copy images into `Report/figures/` when drafting LaTeX).

---

#### Example A — Bottle (`lvis_364437`) — **keep (with note)**

![Bottle / mouthwash pump](../images/000000364437.jpg)

| Field | Value |
|-------|-------|
| PACO object / part (manifest) | bottle / cap |
| Image file | `images/000000364437.jpg` |
| **Positive (Qwen)** | Press the pump to dispense mouthwash. |
| **Negative (Qwen)** | Open the cap to drink from the bottle. |
| Human verdict | **keep** |
| Notes | Positive matches visible pump dispenser and product. Negative is wrong (mouthwash is not for drinking) and still references the same bottle; good attribute-style hard negative. Manifest part was `cap`; visible interaction is really the **pump**. |

---

#### Example B — Mug (`lvis_93765`) — **edit**

![Mug on table](../images/000000093765.jpg)

| Field | Value |
|-------|-------|
| PACO object / part (manifest) | mug / handle |
| Image file | `images/000000093765.jpg` |
| **Positive (Qwen)** | Pick up the mug to drink from it. |
| **Negative (Qwen)** | Open the mug to drink from it. |
| Human verdict | **edit** |
| Notes | Positive is fine but does not name the **handle** (PACO part). Negative “open the mug” is weakly grounded (no lid). Prefer e.g. pos: “Grip the handle to lift the mug.” / neg: “Grip the handle to pour out the mug.” (same part, wrong purpose). |

---

#### Example C — Hammer (`lvis_410924`) — **drop / regen**

![Medical hammers and scissors in cup](../images/000000410924.jpg)

| Field | Value |
|-------|-------|
| PACO object / part (manifest) | hammer / handle |
| Image file | `images/000000410924.jpg` |
| **Positive (Qwen)** | Hold the scissors to cut paper. |
| **Negative (Qwen)** | Squeeze the scissors to sharpen pencils. |
| Human verdict | **drop** |
| Notes | Image contains reflex hammers **and** scissors; Qwen captioned the scissors and ignored the hammer category. Both tiers are about the wrong object → unusable for a hammer affordance pair. |

---

#### Example D — Blender (`lvis_258649`) — **keep**

![Blender with lid being pressed](../images/000000258649.jpg)

| Field | Value |
|-------|-------|
| PACO object / part (manifest) | blender / handle |
| Image file | `images/000000258649.jpg` |
| **Positive (Qwen)** | Push down the lid to blend the mixture. |
| **Negative (Qwen)** | Pull up the lid to open the blender. |
| Human verdict | **keep** |
| Notes | Strong visible-state grounding (hand on lid while blending). Same part (`lid`), opposite actions — good hard negative. Manifest preferred part was `handle`; Qwen correctly focused on the **lid** interaction in the image. |

---

#### Extra contrast — Scissors (`lvis_517182`) — **edit negative**

![Scissors on lap](../images/000000517182.jpg)

| Field | Value |
|-------|-------|
| PACO object / part (manifest) | scissors / handle |
| Image file | `images/000000517182.jpg` |
| **Positive (Qwen)** | Cut the paper with the blade to separate sheets. |
| **Negative (Qwen)** | Squeeze the lid to open the bottle. |
| Human verdict | **edit** (negative) |
| Notes | Positive is OK (blade). Negative jumps to an unrelated bottle → **fluency / object error**. Replace with same-part wrong affordance, e.g. “Cut the paper with the blade to sharpen pencils.” |

### 7.3 Takeaway for the report

Human review is necessary: format-valid Qwen JSON still includes **wrong-object** pairs (Example C) and **ungrounded negatives** (scissors example). Report PACO-LVIS numbers only on **human-checked** captions; use Examples A–D as qualitative evidence that the benchmark inherits Qwen mistakes unless validated.
