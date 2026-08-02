"""Word–patch / word–window similarity heatmaps for CLIP / SigLIP (dual-encoder XAI).

For each content word in pos/neg captions, build a spatial map of similarity to
that word and save an overlay PNG. Complements occlusion attribution with
word-level visual grounding (not B-cos / generative token grounding).

CLIP: ViT patch tokens → visual_projection → cos with text embed.
SigLIP: 7×7 window crops → full get_image_features (dim-safe).

Usage:
  PYTHONPATH=src python src/word_grounding.py \\
    --pairs-json artifacts/eval/val_full/clip.json \\
    --extra-pairs-json humaneval/1aug/pilot_human/clip_ft.json \\
    --backends clip siglip \\
    --out-dir artifacts/grounding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from attribution_occlusion import DEFAULT_IMAGE_IDS, load_pairs_by_id
from config_loader import PROJECT_ROOT, load_config, resolve_path
from grounding_text import content_words, safe_name
from logger import get_logger

logger = get_logger(__name__)


def heatmap_overlay(image: Image.Image, heat, *, alpha: float = 0.45) -> Image.Image:
    import numpy as np

    img = image.convert("RGB")
    w, h = img.size
    heat = np.asarray(heat, dtype=np.float32)
    hmin, hmax = float(heat.min()), float(heat.max())
    norm = np.zeros_like(heat) if hmax - hmin < 1e-8 else (heat - hmin) / (hmax - hmin)
    t = torch.from_numpy(norm)[None, None]
    up = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    norm_up = up.squeeze().numpy()
    r = norm_up
    g = 0.6 * norm_up + 0.2
    b = 1.0 - 0.7 * norm_up
    heat_rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    base = np.asarray(img).astype(np.float32) / 255.0
    a = (alpha * norm_up)[..., None]
    blend = (1.0 - a) * base + a * heat_rgb
    return Image.fromarray((np.clip(blend, 0, 1) * 255).astype(np.uint8))


class PatchGrounder:
    def __init__(self, backend: str, config: dict):
        self.backend = backend
        self.config = config
        device_pref = config.get("models", {}).get(
            f"{backend}_device", config.get("models", {}).get("clip_device", "cuda")
        )
        if device_pref == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = None
        self.processor = None

    def load(self) -> None:
        models = self.config["models"]
        if self.backend == "clip":
            from transformers import CLIPModel, CLIPProcessor

            name = models["clip"]
            logger.info("Loading CLIP for grounding: %s on %s", name, self.device)
            self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(name)
        elif self.backend == "siglip":
            from transformers import AutoModel, AutoProcessor

            name = models.get("siglip", "google/siglip-so400m-patch14-384")
            logger.info("Loading SigLIP for grounding: %s on %s", name, self.device)
            self.model = AutoModel.from_pretrained(name).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(name)
        else:
            raise ValueError(f"Grounding supports clip/siglip only, got {self.backend}")

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def word_embed(self, word: str) -> torch.Tensor:
        assert self.model is not None and self.processor is not None
        prompt = f"a photo of {word}"
        if self.backend == "clip":
            inputs = self.processor(
                text=[prompt], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            emb = self.model.get_text_features(**inputs)
        else:
            inputs = self.processor(
                text=[prompt],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if k in ("input_ids", "attention_mask")
            }
            emb = self.model.get_text_features(**inputs)
        return F.normalize(emb.float(), dim=-1).squeeze(0)

    @torch.no_grad()
    def word_heatmap(self, image: Image.Image, word: str):
        if self.backend == "clip":
            return self._clip_patch_heatmap(image, word)
        return self._window_heatmap(image, word, grid=7)

    @torch.no_grad()
    def _clip_patch_heatmap(self, image: Image.Image, word: str):
        import numpy as np

        assert self.model is not None and self.processor is not None
        inputs = self.processor(images=image, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device)
        vision = self.model.vision_model(pixel_values=pixel)
        hidden = vision.last_hidden_state[:, 1:, :]
        patches = F.normalize(self.model.visual_projection(hidden).float(), dim=-1)
        text = self.word_embed(word)
        sims = (patches.squeeze(0) @ text).cpu().numpy()
        n = sims.shape[0]
        side = int(n**0.5)
        return sims[: side * side].reshape(side, side)

    @torch.no_grad()
    def _window_heatmap(self, image: Image.Image, word: str, grid: int = 7):
        import numpy as np

        assert self.model is not None and self.processor is not None
        text = self.word_embed(word)
        w, h = image.size
        heat = np.zeros((grid, grid), dtype=np.float32)
        for r in range(grid):
            for c in range(grid):
                x0, x1 = c * w // grid, (c + 1) * w // grid
                y0, y1 = r * h // grid, (r + 1) * h // grid
                crop = image.crop((x0, y0, x1, y1))
                inputs = self.processor(images=crop, return_tensors="pt")
                pixel = inputs["pixel_values"].to(self.device)
                img_emb = self.model.get_image_features(pixel_values=pixel)
                img_emb = F.normalize(img_emb.float(), dim=-1).squeeze(0)
                heat[r, c] = float((img_emb @ text).item())
        return heat


def ground_pair(
    grounder: PatchGrounder,
    pair: dict[str, Any],
    out_dir: Path,
    max_words: int = 6,
) -> dict[str, Any]:
    image = Image.open(pair["image_path"]).convert("RGB")
    image_id = pair["image_id"]
    backend = grounder.backend
    dest = out_dir / backend / image_id
    dest.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "image_id": image_id,
        "backend": backend,
        "image_path": pair["image_path"],
        "positive": pair["positive"],
        "negative": pair["negative"],
        "words": {"positive": [], "negative": []},
    }

    for side in ("positive", "negative"):
        words = content_words(pair[side])[:max_words]
        for word in words:
            heat = grounder.word_heatmap(image, word)
            overlay = heatmap_overlay(image, heat)
            fname = f"{side}_{safe_name(word)}.png"
            overlay.save(dest / fname)
            record["words"][side].append(
                {
                    "word": word,
                    "file": str((dest / fname).as_posix()),
                    "heat_min": float(heat.min()),
                    "heat_max": float(heat.max()),
                    "heat_mean": float(heat.mean()),
                }
            )
            logger.info("%s %s %s/%s -> %s", backend, image_id, side, word, fname)

    (dest / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def run_grounding(
    pairs: list[dict],
    backends: list[str],
    out_dir: Path,
    config: dict,
    max_words: int = 6,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"backends": {}, "num_pairs": len(pairs)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        grounder = PatchGrounder(backend, config)
        grounder.load()
        try:
            results = [ground_pair(grounder, pair, out_dir, max_words=max_words) for pair in pairs]
            summary["backends"][backend] = {
                "num_pairs": len(results),
                "pairs": [
                    {
                        "image_id": r["image_id"],
                        "n_pos_words": len(r["words"]["positive"]),
                        "n_neg_words": len(r["words"]["negative"]),
                    }
                    for r in results
                ],
            }
        finally:
            grounder.unload()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Word–region grounding heatmaps")
    parser.add_argument("--pairs-json", required=True)
    parser.add_argument("--extra-pairs-json", nargs="+", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/grounding")
    parser.add_argument("--backends", nargs="+", default=["clip", "siglip"])
    parser.add_argument("--image-ids", nargs="+", default=None)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    for b in args.backends:
        if b not in ("clip", "siglip"):
            raise SystemExit(f"Unsupported backend for grounding: {b}")

    config = load_config(args.config) if args.config else load_config()
    config = dict(config)
    config["models"] = dict(config.get("models", {}))
    config["models"].pop("clip_checkpoint", None)

    pairs = load_pairs_by_id(
        resolve_path(args.pairs_json, PROJECT_ROOT),
        args.image_ids or DEFAULT_IMAGE_IDS,
        extra_json=(
            [resolve_path(p, PROJECT_ROOT) for p in args.extra_pairs_json]
            if args.extra_pairs_json
            else None
        ),
        skip_missing=args.skip_missing,
    )
    summary = run_grounding(
        pairs,
        args.backends,
        resolve_path(args.out_dir, PROJECT_ROOT),
        config,
        max_words=args.max_words,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
