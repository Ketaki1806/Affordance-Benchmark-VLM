"""
Qwen2.5-VL-7B caption generator (stage 1).

Vision-language model: takes an image path + text prompt, returns JSON with
most_probable and negative affordance captions.
"""

import json

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from caption_validator import extract_json_from_text
from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)


class QwenVLCaptionModel:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model_name = self.config["models"]["qwen_vl"]
        self.model = None
        self.processor = None

    # Vision pixel limits for Qwen-VL model, because the GPU cannot process such high vision tokens
    def _vision_pixel_limits(self) -> tuple[int, int]:
        models_cfg = self.config["models"]
        min_pixels = int(models_cfg.get("min_pixels", 256 * 28 * 28))
        max_pixels = int(models_cfg.get("max_pixels", 1280 * 28 * 28))
        return min_pixels, max_pixels

    def _resolve_torch_dtype(self):
        dtype_name = self.config["models"].get("qwen_dtype", "float16")
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def load_model(self) -> None:
        logger.info("Loading Qwen-VL model: %s", self.model_name)
        min_pixels, max_pixels = self._vision_pixel_limits()
        dtype = self._resolve_torch_dtype()
        logger.info(
            "Qwen-VL settings: dtype=%s min_pixels=%d max_pixels=%d",
            dtype,
            min_pixels,
            max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        logger.info("Qwen-VL model loaded")

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
        logger.info("Qwen-VL model unloaded")

    def _build_messages(self, image_path: str, prompt: str) -> list[dict]:
        """Chat format expected by Qwen2.5-VL: one user turn with image + text."""
        min_pixels, max_pixels = self._vision_pixel_limits()
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _generate_raw(self, image_path: str, prompt: str) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        path = str(image_path)
        # Fail fast if the image path from manifest is wrong.
        Image.open(path).convert("RGB").close()

        messages = self._build_messages(path, prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        gen_cfg = self.config["generation"]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                do_sample=True,
            )

        # Decode only new tokens (strip the prompt echo).
        trimmed = [
            out[len(inp) :]
            for inp, out in zip(inputs.input_ids, output_ids, strict=True)
        ]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

    def generate_captions(self, image_path: str, prompt: str) -> dict:
        """Call Qwen-VL and parse JSON; retry with stricter instruction if parse fails."""
        max_retries = self.config["captions"]["max_json_retries"]
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                raw = self._generate_raw(image_path, prompt)
                logger.debug("Raw model output: %s", raw[:300])
                data = extract_json_from_text(raw)
                # Normalize string tiers early so structure checks pass
                for key in ("most_probable", "negative", "positive", "positives", "negatives"):
                    if key in data and isinstance(data[key], str):
                        data[key] = [data[key]]
                if "most_probable" in data and "negative" in data:
                    return data
                # Regeneration prompts return negatives only.
                if "negative" in data and "most_probable" not in data:
                    return data
                # Accept positive/positives as most_probable
                if "negative" in data and (
                    "positive" in data or "positives" in data
                ):
                    data["most_probable"] = data.get("positive") or data.get("positives")
                    if isinstance(data["most_probable"], str):
                        data["most_probable"] = [data["most_probable"]]
                    return data
                raise ValueError(f"Unexpected JSON keys: {list(data.keys())}")
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning("JSON parse attempt %d failed: %s", attempt + 1, exc)
                prompt = prompt + "\n\nReturn ONLY the JSON object. No extra text."
        raise RuntimeError(f"Failed to parse model JSON after retries: {last_error}")

    def regenerate_negatives(
        self,
        image_path: str,
        prompt: str,
    ) -> list[str]:
        """Used by adversarial filter when CLIP rejects too many negatives as too easy."""
        data = self.generate_captions(image_path, prompt)
        negatives = data.get("negative", [])
        if not isinstance(negatives, list):
            raise ValueError("Regeneration response missing 'negative' list")
        return [str(c).strip() for c in negatives]
