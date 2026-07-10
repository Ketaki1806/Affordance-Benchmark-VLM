"""
Caption validation (stage 1c).

Enforces document-style constraints: min/max chars and words from config,
valid JSON structure. Does not check affordance correctness (that's Qwen's job).
"""

import json
import re

from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)


class CaptionValidator:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.caps = self.config["captions"]

    def char_count(self, text: str) -> int:
        return len(text.strip())

    def word_count(self, text: str) -> int:
        return len(text.strip().split())

    def validate_length(self, text: str) -> bool:
        n = self.char_count(text)
        return self.caps["min_chars"] <= n <= self.caps["max_chars"]

    def validate_word_count(self, text: str) -> bool:
        w = self.word_count(text)
        return self.caps["min_words"] <= w <= self.caps["max_words"]

    def validate_caption(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if not self.validate_length(text):
            return False
        if not self.validate_word_count(text):
            return False
        return True

    def validate_tier(self, captions: list[str]) -> list[str]:
        """Keep only captions that pass length/word rules; deduplicate."""
        valid: list[str] = []
        seen: set[str] = set()
        for caption in captions:
            cleaned = caption.strip()
            if cleaned in seen:
                continue
            if self.validate_caption(cleaned):
                valid.append(cleaned)
                seen.add(cleaned)
            else:
                logger.warning(
                    "Dropped caption (len=%d, words=%d): %s",
                    self.char_count(cleaned),
                    self.word_count(cleaned),
                    cleaned[:60],
                )
        return valid

    def validate_json_structure(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "most_probable" not in data or "negative" not in data:
            return False
        if not isinstance(data["most_probable"], list) or not isinstance(data["negative"], list):
            return False
        return True

    def validate_record(
        self,
        most_probable: list[str],
        negative: list[str],
    ) -> tuple[list[str], list[str]]:
        pos = self.validate_tier(most_probable)[: self.caps["num_most_probable"]]
        neg = self.validate_tier(negative)[: self.caps["num_negative"]]
        min_pos = self.caps["num_most_probable"]
        min_neg = self.caps["num_negative"]
        if len(pos) < min_pos:
            logger.warning("Only %d valid most_probable captions (want %d)", len(pos), min_pos)
        if len(neg) < min_neg:
            logger.warning("Only %d valid negative captions (want >= %d)", len(neg), min_neg)
        return pos, neg

    def length_delta_ok(self, positive: str, negative: str) -> bool:
        """Pos/neg pairs should be similar length for fair CLIP comparison."""
        delta = abs(self.char_count(positive) - self.char_count(negative))
        return delta <= self.caps["max_length_delta"]


def extract_json_from_text(text: str) -> dict:
    """Parse JSON from model output; handles markdown fences and trailing prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in model output: {text[:200]}...")
