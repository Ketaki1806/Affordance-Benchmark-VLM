"""CLIP image–caption cosine scorer."""

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
        self.checkpoint = self.config["models"].get("clip_checkpoint")
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
        if self.checkpoint:
            from pathlib import Path

            from config_loader import PROJECT_ROOT, resolve_path

            ckpt_path = resolve_path(str(self.checkpoint), PROJECT_ROOT)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"CLIP checkpoint not found: {ckpt_path}")
            logger.info("Loading CLIP weights: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state, strict=True)
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
        """L2-normalized CLIP image embedding (CPU float tensor, shape [D])."""
        if not self.is_loaded():
            raise RuntimeError("CLIP not loaded. Call load() first.")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        feats = self.model.get_image_features(pixel_values=pixel_values)
        if not torch.is_tensor(feats):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).float().cpu()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """L2-normalized CLIP text embedding (CPU float tensor, shape [D])."""
        if not self.is_loaded():
            raise RuntimeError("CLIP not loaded. Call load() first.")
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        # Explicit path: some HF builds return BaseModelOutput from get_text_features.
        text_out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feats = self.model.text_projection(text_out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).float().cpu()

    @torch.no_grad()
    def score(self, image_path: str, text: str) -> float:
        """Image–text cosine similarity in CLIP embedding space."""
        image_embeds = self.encode_image(image_path)
        text_embeds = self.encode_text(text)
        return float((image_embeds @ text_embeds).item())

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
