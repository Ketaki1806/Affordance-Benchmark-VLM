"""
Fine-tune CLIP with pairwise ranking on affordance caption pairs.

Same loss as Open-VLJEPA FT: relu(margin - cos(img, pos) + cos(img, neg)).
Trains the full CLIP model (ViT-L/14) lightly on the train_500 captions.

Usage:
  PYTHONPATH=src python src/finetune_clip.py --config configs/config_train_ft.yaml

Then set in configs/config.yaml:
  models.clip_checkpoint: artifacts/checkpoints/clip/finetuned_affordance_ep5.pt
  output.eval_clip: artifacts/eval/val_full/clip_ft.json
  eval.backends: [clip]
and run: bash scripts/condor_submit_evaluate.sh
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from config_loader import PROJECT_ROOT, load_config, resolve_path
from logger import get_logger

logger = get_logger(__name__)


def _load_pairs(filtered_path: Path) -> list[dict[str, Any]]:
    with open(filtered_path, encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", data)
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {filtered_path}")

    pairs: list[dict[str, Any]] = []
    for record in objects:
        image_path = record.get("image_path") or record.get("file")
        if not image_path:
            continue
        tiers = record.get("affordance_tiers") or {}
        positives = [
            str(p).strip() for p in (tiers.get("most_probable") or []) if str(p).strip()
        ]
        negatives = [
            str(n).strip() for n in (tiers.get("negative") or []) if str(n).strip()
        ]
        if not positives or not negatives:
            continue
        for pos in positives:
            for neg in negatives:
                pairs.append(
                    {
                        "image_id": record.get("image_id"),
                        "image_path": image_path,
                        "positive": pos,
                        "negative": neg,
                    }
                )
    return pairs


class AffordancePairDataset(Dataset):
    def __init__(self, pairs: list[dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.pairs[idx]


def _clip_image_embeds(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Image tower → pooled → visual projection (CLIP shared space)."""
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.pooler_output
    if pooled is None:
        pooled = vision_outputs.last_hidden_state[:, 0, :]
    return model.visual_projection(pooled)


def _clip_text_embeds(
    model: CLIPModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **_: Any,
) -> torch.Tensor:
    """Text tower → pooled → text projection (CLIP shared space)."""
    text_outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    pooled = text_outputs.pooler_output
    if pooled is None:
        pooled = text_outputs.last_hidden_state[:, 0, :]
    return model.text_projection(pooled)


def train(config: dict[str, Any], *, seed: int = 42) -> Path:
    ft_cfg = config.get("finetune_clip", config.get("finetune", {}))
    filtered_path = resolve_path(config["output"]["filtered_captions"], PROJECT_ROOT)
    out_path = resolve_path(
        config["output"].get(
            "finetuned_clip",
            "artifacts/checkpoints/clip/finetuned_affordance.pt",
        ),
        PROJECT_ROOT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(filtered_path)
    if not pairs:
        raise SystemExit(f"No train pairs in {filtered_path}")

    random.seed(seed)
    random.shuffle(pairs)
    logger.info("Loaded %d train pairs from %s", len(pairs), filtered_path)

    model_name = config["models"]["clip"]
    device_pref = config["models"].get("clip_device", "cuda")
    device = torch.device(
        "cuda" if device_pref == "cuda" and torch.cuda.is_available() else "cpu"
    )

    logger.info("Loading CLIP %s on %s", model_name, device)
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.train()

    lr = float(ft_cfg.get("lr", 1e-5))
    epochs = int(ft_cfg.get("epochs", 5))
    batch_size = int(ft_cfg.get("batch_size", 4))
    grad_accum = max(1, int(ft_cfg.get("grad_accum", 4)))
    margin = float(ft_cfg.get("margin", 0.05))
    weight_decay = float(ft_cfg.get("weight_decay", 0.01))
    max_grad_norm = float(ft_cfg.get("max_grad_norm", 1.0))
    log_every = int(ft_cfg.get("log_every", 20))
    save_every = int(ft_cfg.get("save_every_epochs", 1))
    max_text_length = int(ft_cfg.get("max_text_length", 77))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        AffordancePairDataset(pairs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda xs: xs,
    )

    use_amp = device.type == "cuda"
    # CLIP is stable in fp16; prefer bf16 when available
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader, start=1):
            images = [Image.open(b["image_path"]).convert("RGB") for b in batch]
            positives = [b["positive"] for b in batch]
            negatives = [b["negative"] for b in batch]

            # One forward for image+pos, one for image+neg would double image encode;
            # encode image once, texts separately.
            img_inputs = processor(images=images, return_tensors="pt", padding=True)
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

            pos_inputs = processor(
                text=positives,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_text_length,
            )
            neg_inputs = processor(
                text=negatives,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_text_length,
            )
            pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
            neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                image_embeds = _clip_image_embeds(model, img_inputs["pixel_values"])
                pos_embeds = _clip_text_embeds(
                    model,
                    input_ids=pos_inputs["input_ids"],
                    attention_mask=pos_inputs.get("attention_mask"),
                )
                neg_embeds = _clip_text_embeds(
                    model,
                    input_ids=neg_inputs["input_ids"],
                    attention_mask=neg_inputs.get("attention_mask"),
                )

                image_embeds = F.normalize(image_embeds.float(), dim=-1)
                pos_embeds = F.normalize(pos_embeds.float(), dim=-1)
                neg_embeds = F.normalize(neg_embeds.float(), dim=-1)

                sim_pos = (image_embeds * pos_embeds).sum(dim=-1)
                sim_neg = (image_embeds * neg_embeds).sum(dim=-1)
                loss = F.relu(margin - sim_pos + sim_neg).mean()
                loss = loss / grad_accum

            loss.backward()
            running += float(loss.item()) * grad_accum
            n_steps += 1
            global_step += 1

            if batch_idx % grad_accum == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % log_every == 0:
                logger.info(
                    "epoch %d step %d loss=%.4f sim_pos=%.3f sim_neg=%.3f",
                    epoch,
                    global_step,
                    float(loss.item()) * grad_accum,
                    float(sim_pos.mean().item()),
                    float(sim_neg.mean().item()),
                )

        avg = running / max(1, n_steps)
        logger.info("epoch %d done avg_loss=%.4f", epoch, avg)

        if save_every > 0 and (epoch % save_every == 0 or epoch == epochs):
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "finetune": {
                    "epoch": epoch,
                    "margin": margin,
                    "lr": lr,
                    "source_captions": str(filtered_path),
                    "avg_loss": avg,
                },
            }
            epoch_path = out_path.with_name(f"{out_path.stem}_ep{epoch}{out_path.suffix}")
            torch.save(ckpt, epoch_path)
            torch.save(ckpt, out_path)
            logger.info("Saved checkpoint -> %s (and %s)", out_path, epoch_path)

    logger.info("CLIP fine-tune complete: %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config_train_ft.yaml"),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    path = train(cfg, seed=args.seed)
    print(f"Finetuned CLIP checkpoint: {path}")
