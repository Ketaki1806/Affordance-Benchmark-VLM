"""Open-VLJEPA ranking fine-tune (freeze X-encoder).

Loss: relu(margin - cos(pred, y_pos) + cos(pred, y_neg)).
  PYTHONPATH=src python src/finetune_open_vljepa.py --config configs/config_train_ft.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config_loader import PROJECT_ROOT, load_config, resolve_path
from logger import get_logger
from open_vljepa_scorer import OpenVLJEPAScorer

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
        positives = [str(p).strip() for p in (tiers.get("most_probable") or []) if str(p).strip()]
        negatives = [str(n).strip() for n in (tiers.get("negative") or []) if str(n).strip()]
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


def _freeze_x_encoder(model) -> None:
    if hasattr(model, "x_encoder"):
        for p in model.x_encoder.parameters():
            p.requires_grad = False


def _build_optimizer(model, ft_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    pred_params = [p for p in model.predictor.parameters() if p.requires_grad]
    param_groups = [
        {
            "params": pred_params,
            "lr": float(ft_cfg.get("lr_predictor", 1e-5)),
        }
    ]
    if ft_cfg.get("train_y_encoder", True):
        y_params = [p for p in model.y_encoder.parameters() if p.requires_grad]
        param_groups.append(
            {
                "params": y_params,
                "lr": float(ft_cfg.get("lr_y_encoder", 5e-7)),
            }
        )
    else:
        for p in model.y_encoder.parameters():
            p.requires_grad = False

    return torch.optim.AdamW(
        param_groups,
        weight_decay=float(ft_cfg.get("weight_decay", 0.01)),
    )


def _tokenize_batch(tokenizer, texts: list[str], max_len: int, device: torch.device):
    enc = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def train(config: dict[str, Any], *, seed: int = 42) -> Path:
    ft_cfg = config.get("finetune", {})
    ocfg = config["models"]["open_vljepa"]

    filtered_path = resolve_path(config["output"]["filtered_captions"], PROJECT_ROOT)
    out_path = resolve_path(
        config["output"].get(
            "finetuned_vljepa",
            "artifacts/checkpoints/open-vljepa/finetuned_affordance.pt",
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

    scorer = OpenVLJEPAScorer(config)
    scorer.load()
    model = scorer.model
    assert model is not None
    _freeze_x_encoder(model)
    model.train()

    # X-encoder stays eval/frozen for BN/dropout stability if any
    if hasattr(model, "x_encoder"):
        model.x_encoder.eval()

    optimizer = _build_optimizer(model, ft_cfg)
    margin = float(ft_cfg.get("margin", 0.05))
    epochs = int(ft_cfg.get("epochs", 5))
    batch_size = int(ft_cfg.get("batch_size", 1))
    grad_accum = max(1, int(ft_cfg.get("grad_accum", 8)))
    max_grad_norm = float(ft_cfg.get("max_grad_norm", 1.0))
    log_every = int(ft_cfg.get("log_every", 20))
    save_every = int(ft_cfg.get("save_every_epochs", 1))

    dataset = AffordancePairDataset(pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda xs: xs,
    )

    device = scorer.device
    autocast_dtype = scorer.dtype if device.type == "cuda" else torch.float32
    use_amp = device.type == "cuda"

    global_step = 0
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader, start=1):
            image_paths = [b["image_path"] for b in batch]
            positives = [b["positive"] for b in batch]
            negatives = [b["negative"] for b in batch]

            frames = torch.stack(
                [scorer._load_image_video(p) for p in image_paths]
            ).to(device)
            q_ids = scorer._query_ids.expand(len(batch), -1)
            q_mask = scorer._query_mask.expand(len(batch), -1)
            pos_ids, pos_mask = _tokenize_batch(
                scorer.target_tokenizer,
                positives,
                scorer._max_caption_len,
                device,
            )
            neg_ids, neg_mask = _tokenize_batch(
                scorer.target_tokenizer,
                negatives,
                scorer._max_caption_len,
                device,
            )

            with torch.amp.autocast(device.type, dtype=autocast_dtype, enabled=use_amp):
                pred = model(frames, q_ids, q_mask)
                y_pos = model.y_encoder(pos_ids, pos_mask)
                y_neg = model.y_encoder(neg_ids, neg_mask)
                pred = F.normalize(pred.float(), dim=-1)
                y_pos = F.normalize(y_pos.float(), dim=-1)
                y_neg = F.normalize(y_neg.float(), dim=-1)
                sim_pos = (pred * y_pos).sum(dim=-1)
                sim_neg = (pred * y_neg).sum(dim=-1)
                loss = F.relu(margin - sim_pos + sim_neg).mean()
                loss = loss / grad_accum

            loss.backward()
            running += float(loss.item()) * grad_accum
            n_steps += 1
            global_step += 1

            if batch_idx % grad_accum == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
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
                "config": scorer.model_cfg,
                "finetune": {
                    "epoch": epoch,
                    "margin": margin,
                    "source_captions": str(filtered_path),
                    "init_checkpoint": str(resolve_path(ocfg["checkpoint"], PROJECT_ROOT)),
                    "avg_loss": avg,
                },
            }
            epoch_path = out_path.with_name(f"{out_path.stem}_ep{epoch}{out_path.suffix}")
            torch.save(ckpt, epoch_path)
            torch.save(ckpt, out_path)
            logger.info("Saved checkpoint -> %s (and %s)", out_path, epoch_path)

    scorer.unload()
    logger.info("Fine-tune complete: %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config_train_ft.yaml"),
        help="Train/FT YAML config",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    path = train(cfg, seed=args.seed)
    print(f"Finetuned checkpoint: {path}")
