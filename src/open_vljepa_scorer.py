"""Open-VLJEPA image–caption scorer (vendor/open-vljepa + checkpoint)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer

from config_loader import PROJECT_ROOT, load_config, resolve_path
from logger import get_logger

logger = get_logger(__name__)


def _resolve_repo_root(config: dict) -> Path:
    ocfg = config.get("models", {}).get("open_vljepa", {})
    repo = ocfg.get("repo_root") or os.environ.get("OPENVLJEPA_ROOT")
    if not repo:
        raise RuntimeError(
            "Open-VLJEPA repo not configured. Run: bash scripts/setup_open_vljepa.sh "
            "or set models.open_vljepa.repo_root in config.yaml"
        )
    root = resolve_path(str(repo), PROJECT_ROOT)
    if not (root / "openvljepa").is_dir():
        raise RuntimeError(f"Open-VLJEPA package not found under: {root}")
    return root


def _resolve_torch_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


class OpenVLJEPAScorer:
    """Score image-caption pairs in Open-VLJEPA shared embedding space."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        ocfg = self.config["models"]["open_vljepa"]
        device_pref = ocfg.get("device", "cuda")
        if device_pref == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # EmbeddingGemma: avoid float16 (NaNs).
        raw_dtype = ocfg.get("dtype", "bfloat16")
        self.dtype = _resolve_torch_dtype(raw_dtype)
        if self.dtype == torch.float16:
            fallback = (
                torch.bfloat16
                if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float32
            )
            logger.warning("dtype=float16 unsupported here; using %s", fallback)
            self.dtype = fallback
        self.checkpoint_path = resolve_path(ocfg["checkpoint"], PROJECT_ROOT)
        self.retrieval_prompt = ocfg.get("retrieval_prompt", "Describe the image.")
        self.model = None
        self.model_cfg: dict | None = None
        self.query_tokenizer = None
        self.target_tokenizer = None
        self._query_ids: torch.Tensor | None = None
        self._query_mask: torch.Tensor | None = None
        self._image_transform = None
        self._num_frames = 1
        self._image_size = 256
        self._max_query_len = 64
        self._max_caption_len = 64
        self._image_embed_cache: dict[str, torch.Tensor] = {}

    def _import_openvljepa(self):
        repo_root = _resolve_repo_root(self.config)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from openvljepa.data.msrvtt import _ensure_pad_token  # noqa: WPS433
        from openvljepa.models.vljepa import OpenVLJEPA  # noqa: WPS433

        return OpenVLJEPA, _ensure_pad_token

    def load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Open-VLJEPA checkpoint missing: {self.checkpoint_path}. "
                "Run: bash scripts/setup_open_vljepa.sh"
            )

        OpenVLJEPA, ensure_pad = self._import_openvljepa()
        logger.info("Loading Open-VLJEPA checkpoint: %s", self.checkpoint_path)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model_cfg = ckpt["config"]

        data_cfg = self.model_cfg.get("data", {})
        self._num_frames = int(data_cfg.get("num_frames", 1))
        self._image_size = int(data_cfg.get("image_size", 256))
        self._max_query_len = int(data_cfg.get("max_query_len", 64))
        self._max_caption_len = int(data_cfg.get("max_caption_len", 64))

        self.model = OpenVLJEPA(
            self.model_cfg["encoder"],
            self.model_cfg["y_encoder"],
            self.model_cfg["predictor"],
            torch_dtype=self.dtype,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval().to(self.device)

        pred_cfg = self.model_cfg["predictor"]
        y_cfg = self.model_cfg["y_encoder"]
        self.query_tokenizer = ensure_pad(
            AutoTokenizer.from_pretrained(pred_cfg["llama_name"])
        )
        self.target_tokenizer = ensure_pad(
            AutoTokenizer.from_pretrained(y_cfg["model_name"])
        )

        qenc = self.query_tokenizer(
            self.retrieval_prompt,
            max_length=self._max_query_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self._query_ids = qenc["input_ids"].to(self.device)
        self._query_mask = qenc["attention_mask"].to(self.device)

        self._image_transform = T.Compose(
            [
                T.Resize(self._image_size, antialias=True),
                T.CenterCrop(self._image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        logger.info(
            "Open-VLJEPA loaded on %s (frames=%d, size=%d, dtype=%s)",
            self.device,
            self._num_frames,
            self._image_size,
            self.dtype,
        )

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        self.model = None
        self.query_tokenizer = None
        self.target_tokenizer = None
        self._query_ids = None
        self._query_mask = None
        self._image_embed_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_image_video(self, image_path: str) -> torch.Tensor:
        if self._image_transform is None:
            raise RuntimeError("Open-VLJEPA not loaded. Call load() first.")
        img = Image.open(image_path).convert("RGB")
        frame = self._image_transform(img)
        return frame.unsqueeze(0).repeat(self._num_frames, 1, 1, 1)

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        """Predicted embedding for image + default retrieval prompt."""
        if not self.is_loaded():
            raise RuntimeError("Open-VLJEPA not loaded. Call load() first.")

        cache_key = str(Path(image_path).resolve())
        if cache_key in self._image_embed_cache:
            return self._image_embed_cache[cache_key]

        frames = self._load_image_video(image_path).unsqueeze(0).to(self.device)
        q_ids = self._query_ids.expand(1, -1)
        q_mask = self._query_mask.expand(1, -1)

        autocast_dtype = self.dtype if self.device.type == "cuda" else torch.float32
        with torch.amp.autocast(self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
            pred = self.model(frames, q_ids, q_mask)
            pred = F.normalize(pred.float(), dim=-1)

        embed = pred.squeeze(0).cpu()
        self._image_embed_cache[cache_key] = embed
        return embed

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Target embedding from Y-encoder."""
        if not self.is_loaded():
            raise RuntimeError("Open-VLJEPA not loaded. Call load() first.")

        tenc = self.target_tokenizer(
            text,
            max_length=self._max_caption_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        t_ids = tenc["input_ids"].to(self.device)
        t_mask = tenc["attention_mask"].to(self.device)

        autocast_dtype = self.dtype if self.device.type == "cuda" else torch.float32
        with torch.amp.autocast(self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
            target = self.model.y_encoder(t_ids, t_mask)
            target = F.normalize(target.float(), dim=-1)

        return target.squeeze(0).cpu()

    @torch.no_grad()
    def score(self, image_path: str, text: str) -> float:
        """Cosine similarity between image prediction and caption target."""
        image_embed = self.encode_image(image_path)
        text_embed = self.encode_text(text)
        return float((image_embed @ text_embed).item())

    @torch.no_grad()
    def distance(self, image_path: str, text: str) -> float:
        """L2 distance in shared space (lower = closer caption)."""
        image_embed = self.encode_image(image_path)
        text_embed = self.encode_text(text)
        return float(torch.norm(image_embed - text_embed, p=2).item())
