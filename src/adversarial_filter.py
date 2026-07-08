"""
Adversarial filtering with CLIP zero-shot (stage 2 of the affordance pipeline).

Pilot mode (filter.export_single_pair: true):
  - One anchor positive (highest CLIP image similarity if several candidates).
  - Regenerate until one hard negative passes the filter, or use the best rejected
    fallback so filtered.json never ships an empty negative tier when avoidable.
"""

from typing import Any

from caption_generation_prompt import CaptionGenerationPrompt
from caption_validator import CaptionValidator
from clip_scorer import CLIPScorer
from config_loader import load_config
from dataset_io import build_image_record, collapse_to_single_pair
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
        self.mode = self.filter_cfg.get("mode", "gap")

    def _should_keep_negative(
        self,
        sim_pos: float,
        sim_neg: float,
        text_sim: float | None,
    ) -> bool:
        gap = sim_pos - sim_neg
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

        return gap < self.filter_cfg["min_similarity_gap"]

    def _score_pair(
        self,
        image_path: str,
        positive: str,
        negative: str,
    ) -> dict[str, Any]:
        sim_pos = self.clip.score(image_path, positive)
        sim_neg = self.clip.score(image_path, negative)
        gap = sim_pos - sim_neg

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

    def _select_anchor_positive(self, image_path: str, most_probable: list[str]) -> str:
        if len(most_probable) == 1:
            return most_probable[0]
        best = most_probable[0]
        best_sim = float("-inf")
        for positive in most_probable:
            sim = self.clip.score(image_path, positive)
            if sim > best_sim:
                best_sim = sim
                best = positive
        return best

    def _pick_hardest_kept(self, metadata: list[dict[str, Any]]) -> dict[str, Any]:
        kept = [m for m in metadata if m["kept"]]
        if not kept:
            raise ValueError("No kept negatives to pick from")
        return min(kept, key=lambda m: m["gap"])

    def _pick_fallback_negative(self, all_metadata: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not all_metadata:
            return None
        grounded = [m for m in all_metadata if m["sim_neg"] <= m["sim_pos"]]
        pool = grounded if grounded else all_metadata
        return min(pool, key=lambda m: (m["gap"], m["sim_neg"] - m["sim_pos"]))

    def _export_record(
        self,
        entry: dict[str, Any],
        positives: list[str],
        negatives: list[str],
        pair_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.filter_cfg.get("export_single_pair", True):
            positives, negatives = collapse_to_single_pair(positives, negatives)
        return build_image_record(entry, positives, negatives, pair_meta)

    def filter_negatives(
        self,
        image_path: str,
        most_probable: list[str],
        negatives: list[str],
    ) -> tuple[list[str], list[dict[str, Any]], str]:
        if not most_probable:
            return [], [], ""

        anchor = self._select_anchor_positive(image_path, most_probable)
        metadata: list[dict[str, Any]] = []
        kept_negatives: list[str] = []

        for negative in negatives:
            result = self._score_pair(image_path, anchor, negative)
            metadata.append(result)
            if result["kept"]:
                kept_negatives.append(negative)
            else:
                logger.info(
                    "Dropped negative (%s): %s",
                    result["reason"],
                    negative,
                )

        return kept_negatives, metadata, anchor

    def filter_with_regeneration(
        self,
        entry: dict[str, Any],
        most_probable: list[str],
        negatives: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Filter negatives; regenerate harder ones until the pilot pair is filled.
        Never returns an empty negative list when allow_fallback is true and any
        candidate was scored.
        """
        image_path = entry["image_path"]
        object_label = entry["object"]
        all_metadata: list[dict[str, Any]] = []
        current_negatives = list(negatives)
        anchor = ""
        export_positives = most_probable

        if not most_probable:
            logger.warning("No positives for %s; skipping filter", entry["image_id"])
            return self._export_record(entry, [], [], {"selection": "no_positive"}), all_metadata

        max_regen = self.filter_cfg["max_regen_attempts"]
        allow_fallback = self.filter_cfg.get("allow_fallback", True)

        for attempt in range(max_regen + 1):
            kept, metadata, anchor = self.filter_negatives(
                image_path,
                most_probable,
                current_negatives,
            )
            all_metadata.extend(metadata)
            export_positives = [anchor] if self.filter_cfg.get("export_single_pair", True) else most_probable

            if kept:
                chosen_meta = self._pick_hardest_kept(metadata)
                pair_meta = {
                    "selection": "filter_kept",
                    "anchor_positive": anchor,
                    "regen_attempts": attempt,
                    **chosen_meta,
                }
                return (
                    self._export_record(entry, export_positives, [chosen_meta["negative"]], pair_meta),
                    all_metadata,
                )

            if attempt >= max_regen:
                break

            if self.qwen is None or not self.qwen.is_loaded():
                logger.warning("Cannot regenerate negatives without Qwen model loaded")
                break

            rejected_meta = metadata if metadata else [
                {"negative": n, "reason": "not_scored", "sim_pos": 0, "sim_neg": 0, "gap": 0}
                for n in current_negatives
            ]
            regen_prompt = self.prompt_builder.build_regeneration_prompt(
                object_label=object_label,
                positive_captions=[anchor],
                rejection_details=rejected_meta,
            )
            logger.info(
                "Regenerating negatives for %s (attempt %d)",
                entry["image_id"],
                attempt + 1,
            )
            new_negatives = self.qwen.regenerate_negatives(image_path, regen_prompt)
            _, validated_neg = self.validator.validate_record(most_probable, new_negatives)
            current_negatives = validated_neg if validated_neg else new_negatives
            if not current_negatives:
                logger.warning("Regeneration returned no valid negatives for %s", entry["image_id"])

        if allow_fallback:
            fallback = self._pick_fallback_negative(all_metadata)
            if fallback:
                pair_meta = {
                    "selection": "fallback_best_rejected",
                    "anchor_positive": anchor or most_probable[0],
                    "regen_attempts": max_regen,
                    **fallback,
                }
                logger.warning(
                    "Using fallback negative for %s (reason=%s, gap=%s)",
                    entry["image_id"],
                    fallback["reason"],
                    fallback["gap"],
                )
                return (
                    self._export_record(
                        entry,
                        export_positives or [most_probable[0]],
                        [fallback["negative"]],
                        pair_meta,
                    ),
                    all_metadata,
                )

        logger.warning("No negative kept for %s after filter + regen", entry["image_id"])
        pair_meta = {
            "selection": "empty",
            "anchor_positive": anchor or most_probable[0],
            "regen_attempts": max_regen,
        }
        return self._export_record(entry, export_positives or most_probable[:1], [], pair_meta), all_metadata
