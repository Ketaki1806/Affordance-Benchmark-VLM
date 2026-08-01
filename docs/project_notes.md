# Project notes

Working notes for the AVPR seminar project.  
Last updated: 2026-08-01

### Report map

| Report section | Notes section |
|----------------|---------------|
| Abstract / Intro | §5 |
| Related Work | §1 |
| Experimental Setup | §2, §4.1, §7 |
| Results | §4, §7.2 |
| Conclusion | §5.3 |
| Appendix | prompts, setup |

---

## 1. vs AffordanceCLIP

**Ref:** Cuttano et al. (2024). What Does CLIP Know About Peeling a Banana? arXiv:2404.12015.

AffordanceCLIP: frozen CLIP + FPN for affordance localization (AGD20K). Issues they note: weak spatial detail in global CLIP, action/part mixups, closed vocab, costly labels.

This project:
- binary choice: valid affordance caption vs hard negative (same part, wrong action)
- Qwen2.5-VL captions (candidates); human check on pilot
- CLIP / SigLIP / Open-VLJEPA scoring (cosine)
- PACO-LVIS parts; no pixel masks

| AffordanceCLIP issue | Here |
|----------------------|------|
| Global CLIP only | still global embeddings; no FPN |
| Action vs part | part-tied captions + hard negs |
| Occlusion | PACO parts; no masks |
| Closed vocab | open Qwen captions |
| Costly labels | VLM + human on pilot |
| Seg-only eval | binary acc + confidence gap |
| Pixel grounding | out of scope |

---

## 2. Pipeline choices

| Component | Choice |
|-----------|--------|
| Captions | Qwen2.5-VL-7B, 1 pos + 1 hard neg |
| CLIP filter before eval | removed (circular) |
| Human check | pilot N=20 |
| Pilot data | `manifest_pilot.json` |
| Full preferred-part pool | ~1129 (`manifest_val_full.json`) |
| Scale-up eval | N=100 (`manifest_val_100.json`) |

| Backend | Code | Notes |
|---------|------|-------|
| CLIP | `clip_scorer.py` | frozen + optional FT |
| SigLIP | `siglip_scorer.py` | frozen |
| Y-space | `analyze_caption_yspace.py` | text-only EmbeddingGemma |
| Open-VLJEPA | `open_vljepa_scorer.py` | ZS + FT |

Negatives = wrong affordance for the visible part/state (not alternate valid goals).

---

## 3. Failure tags

| Type | Meaning |
|------|---------|
| Spatial | wrong part named |
| Attribute | same part, wrong action/state |
| Fluency | Qwen junk / wrong object |

---

## 4. Results log

### 4.1 Pilot captions (N=20)

| Metric | Value | Date |
|--------|-------|------|
| Images | 20/20 with pos+neg | 2026-07-25 |
| Human-checked | see §7 | |

### 4.2 CLIP (pilot)

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Acc (raw Qwen) | 0.70 (14/20) | 2026-07-25 | format-valid only |
| Mean gap | 0.030 | 2026-07-25 | |
| Acc (human-filtered) | 0.50 (10/20) | 2026-07-27 | `clip_human.json` |

Raw numbers are not human-validated; several pairs fail §7 checks.

### 4.3 Scale-up N=100 (Qwen captions, no human filter)

Pool ~1129 preferred-part images; subsample N=100 (`--seed 42`) for one-GPU runs.

| Metric | Value | Date |
|--------|-------|------|
| CLIP frozen | 0.63 | 2026-07-30 |
| Mean gap | 0.021 | 2026-07-30 |
| Open-VLJEPA ZS | 0.54 | 2026-07-31 |
| Open-VLJEPA FT | 0.64 | 2026-07-31 |
| CLIP FT | 0.87 | 2026-08-01 |

FT on ~500 held-out Qwen pairs (ranking loss). High CLIP-FT may partly match caption style; see pilot FT below.

### 4.4 SigLIP / Y-space

Optional; same N=100 captions. Y-space = EmbeddingGemma cos(pos, neg) only.

### 4.5 Open-VLJEPA

Llama-3.2 access accepted 2026-07-30. Checkpoint: `best.pt` (ZS), `finetuned_affordance_ep5.pt` (FT). Use bf16 (fp16 → NaNs with EmbeddingGemma).

### 4.6 Human pilot + FT models (N=20)

| Model | Acc | Notes |
|-------|-----|-------|
| CLIP FT | 0.90 (18/20) | `pilot_human/clip_ft.json` |
| VLJEPA FT | 0.55 (11/20) | `pilot_human/open_vljepa_ft.json` |

---

## 5. Short drafts

**RQ:** Can VL embeddings pick the valid affordance caption over a hard negative?

**Method sketch:** `[Verb] the [part] to [purpose]`; score image–caption cosine; human review on pilot.

**Open questions:**
- failure mix (spatial / attribute / fluency)?
- how often humans drop Qwen pairs?
- does VLJEPA help on attribute errors?

---

## 6. Changelog

| Date | Update |
|------|--------|
| 2026-08-01 | CLIP FT 0.87; human FT CLIP 0.90 / VLJEPA 0.55 |
| 2026-07-31 | VLJEPA ZS 0.54, FT 0.64 |
| 2026-07-30 | N=100 CLIP 0.63 |
| 2026-07-26 | pilot CLIP raw 0.70 |
| 2026-07-25 | PACO N=20 pipeline + human-check |
| 2026-06-30 | initial notes |

---

## 7. Human-check (pilot)

Qwen captions are candidates. Per image: **keep** / **edit** / **drop**.

Checks: part match, state match, hard neg (same part), not unrelated object, format.

Sources: `humaneval/26jul/` captions + `images/`.

### Examples

#### A — Bottle (`lvis_364437`) — keep

| | |
|--|--|
| Pos | Press the pump to dispense mouthwash. |
| Neg | Open the cap to drink from the bottle. |
| Notes | Good attribute neg; visible interaction is pump more than cap. |

#### B — Mug (`lvis_93765`) — edit

| | |
|--|--|
| Pos | Pick up the mug to drink from it. |
| Neg | Open the mug to drink from it. |
| Notes | Prefer naming handle; “open mug” weak. |

#### C — Hammer (`lvis_410924`) — drop

| | |
|--|--|
| Pos/Neg | about scissors, not hammer |
| Notes | Wrong object in scene → drop. |

#### D — Blender (`lvis_258649`) — keep

| | |
|--|--|
| Pos | Push down the lid to blend the mixture. |
| Neg | Pull up the lid to open the blender. |
| Notes | Same part, opposite actions. |

#### Scissors (`lvis_517182`) — edit neg

Neg jumps to a bottle → replace with same-part wrong affordance.

### Note

Format-valid JSON can still be wrong-object or ungrounded. Prefer human-checked numbers for pilot claims.
