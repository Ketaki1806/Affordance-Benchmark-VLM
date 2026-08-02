# Project notes

Working notes for the AVPR seminar project.  
Last updated: 2026-08-01

### Report map

| Report section | Notes section |
|----------------|---------------|
| Abstract / Intro | §5 |
| Related Work | §1, §8 |
| Experimental Setup | §2, §4.1, §7 |
| Results | §4, §7.2 |
| Error analysis / figures | §3 |
| Open-VLJEPA vs Meta VL-JEPA | §8 |
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

Negatives = wrong affordance for the visible part/state (not alternate valid goals). Hard negatives follow SugarCrepe’s principle of fluent, human-validated distractors, specialized to affordance/purpose swaps on PACO parts rather than generic attribute/object edits.

---

## 3. Error analysis (failure tags)

Tag each **model-wrong** pair by the dominant pos/neg contrast (heuristic + spot check). Same tags also cover caption-quality issues when the “gold” pos is itself bad.

| Type | Meaning | Typical contrast |
|------|---------|------------------|
| Attribute | same part/object, wrong action or purpose | open↔close, carry↔empty, answer↔power-off |
| Spatial | location or part mismatch | wall↔ceiling, teeth↔face, handle vs whole object |
| Fluency | junk / wrong-object / ungrounded Qwen text | lamp captions on a pillow; tautology; “spoon spoon” |

Sources (N=100, Qwen captions, frozen models): `humaneval/30jul/clip.json`, `1aug/siglip.json`, `31jul/open_vljepa.json`.  
N=100 CLIP-FT / VLJEPA-FT pair dumps still on cluster if needed for a second pass.

### 3.1 Counts (N=100 wrongs)

| Model | Wrong | Attribute | Spatial | Fluency |
|-------|------:|----------:|--------:|--------:|
| CLIP frozen | 37 | 24 (65%) | 8 (22%) | 5 (14%) |
| SigLIP | 40 | 30 (75%) | 6 (15%) | 4 (10%) |
| Open-VLJEPA ZS | 46 | 38 (83%) | 5 (11%) | 3 (7%) |

**Plot:** `artifacts/figures/error_tags_n100.svg` (regen: `py -3 scripts/plot_error_tags.py`).

Overlap of wrong `image_id`s: CLIP∩SigLIP 22 · CLIP∩VLJEPA 21 · SigLIP∩VLJEPA 24 · **all three 14**.

**Takeaway:** most frozen errors are **attribute** (purpose/action swaps) — the intended hard-neg type. Spatial share is smaller but shared across encoders (e.g. wall/ceiling). Fluency share is the caption-noise floor (~7–14%); those pairs should not carry much weight in model comparisons.

Shared fails (all three wrong) — good qualitative set:  
`lvis_52835` box, `81841` chair, `88609` tray, `135976` telephone, `237944` dog+toothbrush, `395975` fan, `438922` spoon, `446014` soap, `465786` mirror/frame, `479944` bag, `486018` microwave, `501118` dog, `512070` / `533706` phone.

### 3.2 Figure examples (use with PACO image)

Paths on cluster: `data/paco/images/<COCO id>.jpg` (id from `lvis_<id>`).

#### Fig A — Attribute (all three fail)

| | |
|--|--|
| ID | `lvis_486018` microwave |
| Pos | Open the microwave by pulling the door handle. |
| Neg | Close the microwave by pulling the door handle. |
| Tag | Attribute — same part/verb, purpose polarity flipped |
| Note | Classic affordance hard neg; dual-encoders and VLJEPA all pick close. |

#### Fig B — Spatial

| | |
|--|--|
| ID | `lvis_75183` mirror |
| Pos | Hang the mirror on the wall. |
| Neg | Hang the mirror on the ceiling. |
| Tag | Spatial — same action, wrong place |
| Note | CLIP + SigLIP wrong (VLJEPA correct on this one). Good “global embedding misses layout” example. |

#### Fig C — Fluency / caption noise

| | |
|--|--|
| ID | `lvis_422959` pillow (CLIP wrong) |
| Pos | Press the button to activate the lamp. |
| Neg | Turn the switch to turn off the lamp. |
| Tag | Fluency — captions describe a lamp, label is pillow |
| Note | Model “error” is not a fair affordance fail; argue for human filter / drop. |

#### Fig D — Human pilot, FT models (attribute)

| | |
|--|--|
| ID | `lvis_258649` blender |
| Pos | Push down the lid to blend the mixture. |
| Neg | Push the blender lid to serve the mixture. |
| Tag | Attribute — same lid, wrong purpose |
| Note | Both CLIP-FT and VLJEPA-FT wrong on human pilot (pre-edit dump). Shows residual purpose confusion even after FT. |

### 3.3 Human pilot FT wrongs (brief)

Local `humaneval/1aug/pilot_human/` is still **pre-edit** (VLJEPA 0.55). Post-edit cluster: CLIP-FT 0.90 (2 wrong) / VLJEPA-FT 0.45 (11 wrong).

| Model | Wrong (pre-edit local) | Dominant tag |
|-------|------------------------|--------------|
| CLIP-FT | blender; can (tab vs cap) | attribute |
| VLJEPA-FT | 11/20 — mug open, hammer/pencils, knife peel, kettle→floor, scissors→pencils, blender, pan wash, microwave wash, mirror hang, ladder slide, bucket food | mostly **attribute** purpose swaps |

VLJEPA-FT’s pilot misses stay attribute-heavy; SugarCrepe-style edits made that worse (0.55→0.45), consistent with less leaky negs.

### 3.4 Report bullets

- Lead with attribute dominance (~65–83% of frozen wrongs).
- One spatial figure (mirror wall/ceiling) + one attribute figure (microwave open/close).
- One fluency figure to justify human check / why raw Qwen N=100 is an upper-noise bound.
- Caveat: tags are contrast-based heuristics; re-spot-check figure captions against the image before camera-ready.

### 3.5 Occlusion attribution (planned / running)

Protocol: text leave-one-out + 3×3 image blackout on Δ = s_pos − s_neg.  
Pairs: 8 (§3 figures + shared wrongs). Backends: CLIP, SigLIP, Open-VLJEPA ZS.  
Outputs: `artifacts/attribution/`. Spec: `docs/superpowers/specs/2026-08-02-occlusion-attribution-design.md`.  
Cluster: `bash scripts/condor_submit_attribution.sh`

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
| Mean gap (CLIP) | 0.021 | 2026-07-30 |
| SigLIP | 0.60 | 2026-08-01 |
| Mean gap (SigLIP) | 0.029 | 2026-08-01 |
| Open-VLJEPA ZS | 0.54 | 2026-07-31 |
| Open-VLJEPA FT | 0.64 | 2026-07-31 |
| CLIP FT | 0.87 | 2026-08-01 |

FT on ~500 held-out Qwen pairs (ranking loss). High CLIP-FT may partly match caption style; see pilot FT below. SigLIP ≈ CLIP → dual-encoder family, not CLIP-only.

### 4.4 Y-space (EmbeddingGemma, N=100)

Text-only cos(pos, neg) on same captions; join with frozen CLIP correct/wrong.

| Metric | Value |
|--------|-------|
| Mean cos | 0.815 |
| Median cos | 0.809 |
| Min / max | 0.654 / 0.977 |
| Frac cos > 0.8 | 0.55 |
| Frac cos > 0.9 | 0.15 |
| Mean cos \| CLIP correct | 0.807 |
| Mean cos \| CLIP wrong | 0.827 |

Hard negs are close in Y-space; CLIP errors slightly higher cos (weak signal). Files: `siglip.json`, `yspace_caption_analysis.json`.

### 4.5 Open-VLJEPA

Llama-3.2 access accepted 2026-07-30. Checkpoint: `best.pt` (ZS), `finetuned_affordance_ep5.pt` (FT). Use bf16 (fp16 → NaNs with EmbeddingGemma).  
Not Meta’s released VL-JEPA weights — see §8 (architecture-close, scale-far).

### 4.6 Human pilot + FT models (N=20)

After SugarCrepe-style caption edits (2026-08-01):

| Model | Acc | Notes |
|-------|-----|-------|
| CLIP FT | 0.90 (18/20) | gap ~0.070 |
| VLJEPA FT | 0.45 (9/20) | gap ~0.077; below chance |

Pre-edit FT: CLIP 0.90 / VLJEPA 0.55. Cleaner hard negs hurt VLJEPA more.

---

## 5. Short drafts

**RQ:** Can VL embeddings pick the valid affordance caption over a hard negative?

**Method sketch:** `[Verb] the [part] to [purpose]`; score image–caption cosine; human review on pilot (SugarCrepe-style hard negatives; see §2).

**Open questions:**
- failure mix — see §3 (attribute-dominant; VLJEPA ZS still attribute-heavy)
- how often humans drop Qwen pairs?
- does VLJEPA help on attribute errors? — not on pilot FT (worse than CLIP-FT)

---

## 6. Changelog

| Date | Update |
|------|--------|
| 2026-08-01 | §8 Open-VLJEPA vs Meta VL-JEPA (arch close, scale far) |
| 2026-08-01 | Error analysis §3: tag CLIP/SigLIP/VLJEPA wrongs + figure examples |
| 2026-08-01 | SigLIP 0.60; Y-space mean cos 0.815; human FT after edits CLIP 0.90 / VLJEPA 0.45 |
| 2026-08-01 | CLIP FT 0.87; human FT (pre-edit) CLIP 0.90 / VLJEPA 0.55 |
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

---

## 8. Open-VLJEPA vs Meta VL-JEPA (how close?)

**Refs:** Chen et al., *VL-JEPA* (arXiv:2512.10942, Meta). Open re-impl: [dion-jy/open-vljepa](https://github.com/dion-jy/open-vljepa) / checkpoint `cun-bjy/open-vljepa` (Baek, 2026).

We use **Open-VLJEPA**, an unofficial small-scale re-implementation. Meta did **not** release full VL-JEPA weights for drop-in use; Action100M and full pretraining mix are partly internal. So “how close?” splits into **architecture** vs **trained system**.

### 8.1 Architecture — close (same recipe)

Paper §3.1 and Open-VLJEPA match on the pieces that matter for our retrieval-style scoring:

| Piece | Meta VL-JEPA (paper) | Open-VLJEPA |
|-------|----------------------|------------|
| X-Encoder | frozen V-JEPA 2 ViT-L (304M) | same (`facebook/vjepa2-vitl-fpc64-256`) |
| Predictor | last 8 layers Llama-3.2-1B, bi-dir attn (~490M) | same |
| Y-Encoder | EmbeddingGemma-300M + proj | same |
| Loss | bi-directional InfoNCE | same (τ=0.07) |
| Shared space | continuous embedding (CLIP-style nearest) | 1536-D |
| Inference we use | encode image (+ prompt) → ˆy; encode captions → pick max cos | same in `open_vljepa_scorer.py` |

So for the report: Open-VLJEPA is a **faithful JEPA VL stack**, not a CLIP clone with a trendy name. Same X / Predictor / Y roles as Fig. 1–2 in the paper.

### 8.2 Training & weights — far (scale gap)

| | Meta VL-JEPA | Open-VLJEPA |
|--|--------------|------------|
| Samples seen | ~2B+ (image then image+video); paper cites large Datacomp/YFCC/Action100M mix | ~0.14–1.8M (WebVid-scale / small public mix) |
| Compute | ~192× H200, ~2 weeks (paper) | ~8× RTX 4090, days |
| Batch | up to ~24k (image stage) | much smaller |
| Model scale (paper) | ~1.6B total reported | ~800M trainable in ckpt; frozen V-JEPA 2 loaded from HF |
| Weights | Meta’s trained predictor/Y (not what we load) | community `best.pt` — **re-trained**, not distilled from Meta |

Open authors’ own retrieval sanity check (their README): domain MSRVTT finetune **R@1 ≈ 23.9** vs paper **≈ 51.6** on the comparable setting → roughly **~45–50% of Meta’s reported retrieval score**. They attribute the gap to **data/compute scale**, not a different architecture. Zero-shot retrieval for Open is much weaker still.

Rough “closeness” summary:

| Axis | Closeness | One-liner |
|------|-----------|-----------|
| Method / modules | **High** | Same JEPA VL blueprint |
| Loss & embedding use | **High** | InfoNCE + cosine retrieval |
| Pretraining data | **Low** | Orders of magnitude fewer samples; no Action100M |
| Published retrieval strength | **Medium-low** | ~½ paper MSRVTT after Open’s domain FT |
| Our affordance numbers | **Not transferable** | 0.54 ZS / 0.64 FT are Open’s, not Meta’s |

### 8.3 What this means for the seminar report

1. **Cite both:** Meta VL-JEPA for the idea; Open-VLJEPA for the runnable model we evaluate.
2. **Do not claim** we measured Meta VL-JEPA on PACO affordance. Phrase as: *“Open-VLJEPA (architecture-matched open re-impl of Chen et al.; much smaller pretraining).”*
3. **Interpretation of weak ZS (0.54):** partly expected under-scale — Meta’s model would likely be stronger on the same binary task, but that is a hypothesis, not a result we have.
4. **FT (0.64) still modest vs CLIP-FT (0.87):** even adapting Open on ~500 affordance pairs does not close the dual-encoder gap on this task; that comparison is fair **among models we ran**, not vs Meta.
5. **Y-space section** already uses EmbeddingGemma — same Y-init family as both VL-JEPA variants; useful bridge in related work.

### 8.4 Report sentence (copy-ready)

> We evaluate Open-VLJEPA (Baek, 2026), an open re-implementation that mirrors the VL-JEPA architecture of Chen et al. (frozen V-JEPA 2, Llama-3.2 predictor, EmbeddingGemma Y-encoder, InfoNCE). It is **not** Meta’s released VL-JEPA checkpoint: pretraining is two-plus orders of magnitude smaller, and published MSRVTT retrieval is roughly half the paper figure after domain finetuning. Our affordance accuracies therefore bound the open model, not the closed-scale Meta system.
