"""
CLIP zero-shot scorer for adversarial filtering.

Encodes image and text into a shared embedding space; higher cosine similarity
means the caption is more compatible with the image (used in stage 2 filtering).
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)


class CLIPScorer:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model_name = self.config["models"]["clip"]
        # CPU by default so Qwen-VL can keep the GPU; CLIP on 10 images is fast on CPU.
        device_pref = self.config["models"].get("clip_device", "cpu")
        if device_pref == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = None
        self.processor = None

    def load(self) -> None:
        logger.info("Loading CLIP model: %s on %s", self.model_name, self.device)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
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
    def score(self, image_path: str, text: str) -> float:
        """Image–text cosine similarity in CLIP embedding space."""
        if not self.is_loaded():
            raise RuntimeError("CLIP not loaded. Call load() first.")

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarity = (image_embeds @ text_embeds.T).squeeze().item()
        return float(similarity)

    @torch.no_grad()
    def score_text_pair(self, text_a: str, text_b: str) -> float:
        """
        Text–text cosine similarity (text_and_gap filter mode).
        High score → captions use similar affordance wording (confusable hard negatives).
        """
        if not self.is_loaded():
            raise RuntimeError("CLIP not loaded. Call load() first.")

        inputs = self.processor(
            text=[text_a, text_b],
            images=None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        text_embeds = self.model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        similarity = (text_embeds[0] @ text_embeds[1]).item()
        return float(similarity)
