"""
Adversarial filtering with CLIP zero-shot (stage 2 of the affordance pipeline).

Goal: drop *easy* hard negatives that CLIP can trivially reject, and keep negatives
that are semantically close to the image but affordance-wrong.

Filter modes (set filter.mode in configs/config.yaml):
  gap (default)
      Keep negative if: sim(image, pos) - sim(image, neg) < min_similarity_gap
      AND sim(image, neg) <= sim(image, pos).
      Intuition: CLIP barely prefers the positive → negative is a hard distractor.

  neg_sim_floor
      Keep negative if: sim(image, neg) >= min_neg_image_sim
      AND sim(image, neg) <= sim(image, pos).
      Intuition: negative must be visually grounded in the image (not unrelated),
      but still lose to the positive. Good when gap alone drops grounded captions.

  text_and_gap
      Same as gap, plus require high CLIP *text* similarity between pos and neg.
      Intuition: hard negatives should use confusable affordance wording (same parts,
      similar verbs) even before comparing to the image.

Other methods used in literature (not implemented here):
  - Human review loop after automatic pre-filter
  - Round-trip LLM critique ("would a human pick the negative for this image?")
  - VL-JEPA distance instead of CLIP (stage 4 in the project document)
  - Pool ranking: negative must be in top-k CLIP scores among all candidate captions
"""

from typing import Any

from caption_generation_prompt import CaptionGenerationPrompt
from caption_validator import CaptionValidator
from clip_scorer import CLIPScorer
from config_loader import load_config
from dataset_io import build_image_record
from logger import get_logger
from model import QwenVLCaptionModel

logger = get_logger(__name__)


class AdversarialFilter:
    def __init__(
        self,
        clip: CLIPScorer,
        validator: CaptionValidator,
        config: dict | None = None,
        qwen: QwenVLCaptionModel | None = None,
        prompt_builder: CaptionGenerationPrompt | None = None,
    ):
        self.config = config or load_config()
        self.clip = clip
        self.validator = validator
        self.qwen = qwen
        self.prompt_builder = prompt_builder or CaptionGenerationPrompt()
        self.filter_cfg = self.config["filter"]
        # gap | neg_sim_floor | text_and_gap
        self.mode = self.filter_cfg.get("mode", "gap")

    def _should_keep_negative(
        self,
        sim_pos: float,
        sim_neg: float,
        text_sim: float | None,
    ) -> bool:
        """Apply the configured filter mode to decide if a negative is hard enough."""
        gap = sim_pos - sim_neg

        # Always reject if CLIP prefers the wrong (negative) caption for this image.
        if sim_neg > sim_pos:
            return False

        if self.mode == "neg_sim_floor":
            floor = self.filter_cfg.get("min_neg_image_sim", 0.20)
            return sim_neg >= floor

        if self.mode == "text_and_gap":
            min_text = self.filter_cfg.get("min_text_sim", 0.85)
            max_gap = self.filter_cfg["min_similarity_gap"]
            if text_sim is None:
                return gap < max_gap
            return gap < max_gap and text_sim >= min_text

        # Default: gap mode (same idea as rank margin: sim_neg > sim_pos - margin).
        return gap < self.filter_cfg["min_similarity_gap"]

    def _score_pair(
        self,
        image_path: str,
        positive: str,
        negative: str,
    ) -> dict[str, Any]:
        # Step 1: CLIP image–text cosine similarity for positive and negative.
        sim_pos = self.clip.score(image_path, positive)
        sim_neg = self.clip.score(image_path, negative)
        gap = sim_pos - sim_neg

        # Step 2 (text_and_gap only): confusable wording in embedding space.
        text_sim: float | None = None
        if self.mode == "text_and_gap":
            text_sim = round(self.clip.score_text_pair(positive, negative), 4)

        kept = self._should_keep_negative(sim_pos, sim_neg, text_sim)
        return {
            "positive": positive,
            "negative": negative,
            "sim_pos": round(sim_pos, 4),
            "sim_neg": round(sim_neg, 4),
            "gap": round(gap, 4),
            "text_sim": text_sim,
            "filter_mode": self.mode,
            "kept": kept,
            "reason": self._reject_reason(sim_pos, sim_neg, gap, kept),
        }

    def _reject_reason(
        self,
        sim_pos: float,
        sim_neg: float,
        gap: float,
        kept: bool,
    ) -> str:
        if sim_neg > sim_pos:
            return "clip_prefers_negative"
        if kept:
            return "hard_negative_kept"
        if self.mode == "neg_sim_floor":
            return "neg_not_grounded_in_image"
        if gap >= self.filter_cfg["min_similarity_gap"]:
            return "too_easy_large_gap"
        return "rejected"

    def filter_negatives(
        self,
        image_path: str,
        most_probable: list[str],
        negatives: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if not most_probable:
            return [], []

        # Compare each negative against the first (anchor) positive caption.
        anchor_positive = most_probable[0]
        metadata: list[dict[str, Any]] = []
        kept_negatives: list[str] = []

        for negative in negatives:
            result = self._score_pair(image_path, anchor_positive, negative)
            metadata.append(result)
            if result["kept"]:
                kept_negatives.append(negative)
            else:
                logger.info(
                    "Dropped negative (%s): %s",
                    result["reason"],
                    negative,
                )

        return kept_negatives, metadata

    def filter_with_regeneration(
        self,
        entry: dict[str, Any],
        most_probable: list[str],
        negatives: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Filter negatives; if too few survive, ask Qwen-VL to regenerate harder ones.
        Loop at most filter.max_regen_attempts times.
        """
        image_path = entry["image_path"]
        object_label = entry["object"]
        all_metadata: list[dict[str, Any]] = []
        current_negatives = list(negatives)

        for attempt in range(self.filter_cfg["max_regen_attempts"] + 1):
            kept, metadata = self.filter_negatives(
                image_path,
                most_probable,
                current_negatives,
            )
            all_metadata.extend(metadata)

            min_kept = self.filter_cfg["min_negatives_kept"]
            if len(kept) >= min_kept:
                return build_image_record(entry, most_probable, kept), all_metadata

            if attempt >= self.filter_cfg["max_regen_attempts"]:
                logger.warning(
                    "Keeping %d negatives after max regen attempts for %s",
                    len(kept),
                    entry["image_id"],
                )
                return build_image_record(entry, most_probable, kept), all_metadata

            if self.qwen is None or not self.qwen.is_loaded():
                logger.warning("Cannot regenerate negatives without Qwen model loaded")
                return build_image_record(entry, most_probable, kept), all_metadata

            # Tell Qwen which negatives were too easy so it can propose harder ones.
            rejected = [m["negative"] for m in metadata if not m["kept"]]
            if not rejected:
                rejected = current_negatives

            regen_prompt = self.prompt_builder.build_regeneration_prompt(
                object_label=object_label,
                rejected_negatives=rejected,
                positive_captions=most_probable,
            )
            logger.info(
                "Regenerating negatives for %s (attempt %d)",
                entry["image_id"],
                attempt + 1,
            )
            new_negatives = self.qwen.regenerate_negatives(image_path, regen_prompt)
            _, validated_neg = self.validator.validate_record(most_probable, new_negatives)
            current_negatives = validated_neg if validated_neg else new_negatives

        return build_image_record(entry, most_probable, kept), all_metadata
