"""SigLIP image–caption cosine scorer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)


class SigLIPScorer:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        models = self.config["models"]
        self.model_name = models.get("siglip", "google/siglip-so400m-patch14-384")
        device_pref = models.get("siglip_device", models.get("clip_device", "cuda"))
        if device_pref == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = None
        self.processor = None

    def load(self) -> None:
        logger.info("Loading SigLIP model: %s on %s", self.model_name, self.device)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _as_embed_tensor(emb):
        """Normalize HF quirks: tensor vs BaseModelOutputWithPooling."""
        if torch.is_tensor(emb):
            return emb
        pooled = getattr(emb, "pooler_output", None)
        if pooled is None and isinstance(emb, (tuple, list)):
            pooled = emb[1] if len(emb) > 1 else emb[0]
        if pooled is None:
            raise TypeError(f"Cannot extract embedding from {type(emb)}")
        return pooled

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        """L2-normalized SigLIP image embedding (CPU float tensor, shape [D])."""
        if not self.is_loaded():
            raise RuntimeError("SigLIP not loaded. Call load() first.")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        raw = self.model.get_image_features(pixel_values=pixel_values)
        if torch.is_tensor(raw):
            feats = raw
        else:
            pooled = self._as_embed_tensor(raw)
            proj = getattr(self.model, "visual_projection", None)
            feats = proj(pooled) if proj is not None else pooled
        feats = F.normalize(feats.float(), dim=-1)
        return feats.squeeze(0).cpu()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """L2-normalized SigLIP text embedding (CPU float tensor, shape [D])."""
        if not self.is_loaded():
            raise RuntimeError("SigLIP not loaded. Call load() first.")
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        text_kwargs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }
        raw = self.model.get_text_features(**text_kwargs)
        if torch.is_tensor(raw):
            feats = raw
        else:
            pooled = self._as_embed_tensor(raw)
            proj = getattr(self.model, "text_projection", None)
            feats = proj(pooled) if proj is not None else pooled
        feats = F.normalize(feats.float(), dim=-1)
        return feats.squeeze(0).cpu()

    @torch.no_grad()
    def score(self, image_path: str, text: str) -> float:
        """Image–text cosine similarity in SigLIP embedding space."""
        if not self.is_loaded():
            raise RuntimeError("SigLIP not loaded. Call load() first.")

        # Joint forward is the historically stable path on cluster transformers.
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)
        if image_embeds is None or text_embeds is None:
            return float(
                (self.encode_image(image_path) @ self.encode_text(text)).item()
            )
        image_embeds = F.normalize(image_embeds.float(), dim=-1)
        text_embeds = F.normalize(text_embeds.float(), dim=-1)
        return float((image_embeds @ text_embeds.T).squeeze().item())
