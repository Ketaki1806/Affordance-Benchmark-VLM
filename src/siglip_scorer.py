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

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        """L2-normalized SigLIP image embedding (CPU float tensor, shape [D])."""
        if not self.is_loaded():
            raise RuntimeError("SigLIP not loaded. Call load() first.")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).float().cpu()

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
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        feats = self.model.get_text_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).float().cpu()

    @torch.no_grad()
    def score(self, image_path: str, text: str) -> float:
        """Image–text cosine similarity in SigLIP embedding space."""
        image_embeds = self.encode_image(image_path)
        text_embeds = self.encode_text(text)
        return float((image_embeds @ text_embeds).item())
