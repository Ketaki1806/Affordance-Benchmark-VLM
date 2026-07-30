"""
Y-space caption confusability analysis (EmbeddingGemma).

Embeds positive/negative affordance captions with the same family of text
encoder Open-VLJEPA uses as Y (default: google/embeddinggemma-300m).
No images — answers how close hard-negative pairs are in text space.

Usage:
  export PYTHONPATH=src
  python src/analyze_caption_yspace.py
  python src/analyze_caption_yspace.py --captions artifacts/captions/val_full/filtered.json \\
      --clip-json artifacts/eval/val_full/clip.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config_loader import PROJECT_ROOT, load_config, resolve_path
from logger import get_logger

logger = get_logger(__name__)


def _load_objects(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", data)
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {path}")
    return objects


def _first_caption(tiers: dict[str, Any], key: str) -> str | None:
    vals = tiers.get(key) or []
    if not vals:
        return None
    text = str(vals[0]).strip()
    return text or None


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class YEncoderEmbedder:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        logger.info("Loading Y-encoder: %s on %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        assert self.tokenizer is not None and self.model is not None
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled.float(), dim=-1).squeeze(0).cpu()


def _clip_correct_by_id(clip_path: Path | None) -> dict[str, bool]:
    if clip_path is None or not clip_path.is_file():
        return {}
    with open(clip_path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, bool] = {}
    for pair in data.get("pairs", []):
        image_id = pair.get("image_id")
        if image_id is not None and "correct" in pair:
            out[str(image_id)] = bool(pair["correct"])
    return out


def analyze(
    captions_path: Path,
    out_path: Path,
    model_name: str,
    device: torch.device,
    clip_path: Path | None = None,
) -> dict[str, Any]:
    objects = _load_objects(captions_path)
    clip_map = _clip_correct_by_id(clip_path)

    embedder = YEncoderEmbedder(model_name, device)
    embedder.load()
    pairs_out: list[dict[str, Any]] = []
    try:
        for obj in objects:
            tiers = obj.get("affordance_tiers") or {}
            pos = _first_caption(tiers, "most_probable")
            neg = _first_caption(tiers, "negative")
            if not pos or not neg:
                continue
            pos_emb = embedder.embed(pos)
            neg_emb = embedder.embed(neg)
            cos = float((pos_emb @ neg_emb).item())
            image_id = str(obj.get("image_id", ""))
            row: dict[str, Any] = {
                "image_id": image_id,
                "object": obj.get("object"),
                "positive": pos,
                "negative": neg,
                "cosine_pos_neg": cos,
            }
            if image_id in clip_map:
                row["clip_correct"] = clip_map[image_id]
            pairs_out.append(row)
            logger.info("%s cos(pos,neg)=%.4f", image_id, cos)
    finally:
        embedder.unload()

    cosines = [p["cosine_pos_neg"] for p in pairs_out]
    summary: dict[str, Any] = {
        "num_pairs": len(cosines),
        "mean_cosine": statistics.fmean(cosines) if cosines else 0.0,
        "median_cosine": statistics.median(cosines) if cosines else 0.0,
        "min_cosine": min(cosines) if cosines else 0.0,
        "max_cosine": max(cosines) if cosines else 0.0,
        "frac_cosine_gt_0_8": (
            sum(1 for c in cosines if c > 0.8) / len(cosines) if cosines else 0.0
        ),
        "frac_cosine_gt_0_9": (
            sum(1 for c in cosines if c > 0.9) / len(cosines) if cosines else 0.0
        ),
        "model": model_name,
        "source": str(captions_path),
    }

    with_clip = [p for p in pairs_out if "clip_correct" in p]
    if with_clip:
        correct = [p["cosine_pos_neg"] for p in with_clip if p["clip_correct"]]
        wrong = [p["cosine_pos_neg"] for p in with_clip if not p["clip_correct"]]
        summary["clip_join"] = {
            "num_joined": len(with_clip),
            "mean_cosine_when_clip_correct": (
                statistics.fmean(correct) if correct else None
            ),
            "mean_cosine_when_clip_wrong": statistics.fmean(wrong) if wrong else None,
        }

    result = {"summary": summary, "pairs": pairs_out}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    models = cfg.get("models", {})
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--captions",
        type=Path,
        default=None,
        help="filtered.json (default: config output.filtered_captions)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: config output.eval_yspace)",
    )
    p.add_argument(
        "--clip-json",
        type=Path,
        default=None,
        help="Optional clip.json to join correct/wrong with text cosine",
    )
    p.add_argument(
        "--model",
        default=models.get("y_encoder", "google/embeddinggemma-300m"),
    )
    p.add_argument(
        "--device",
        default=models.get("y_encoder_device", "cuda"),
        choices=("cuda", "cpu"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    captions = resolve_path(
        args.captions or cfg["output"]["filtered_captions"],
        PROJECT_ROOT,
    )
    out = resolve_path(
        args.out
        or cfg["output"].get(
            "eval_yspace",
            "artifacts/eval/val_full/yspace_caption_analysis.json",
        ),
        PROJECT_ROOT,
    )
    clip_path = None
    if args.clip_json is not None:
        clip_path = resolve_path(args.clip_json, PROJECT_ROOT)
    else:
        default_clip = resolve_path(cfg["output"]["eval_clip"], PROJECT_ROOT)
        if default_clip.is_file():
            clip_path = default_clip

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    result = analyze(captions, out, args.model, device, clip_path=clip_path)
    summary = result["summary"]
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
