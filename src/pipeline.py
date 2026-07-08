"""
Affordance caption pipeline (stages 1–2 from the project document).

Pilot: one positive + one negative per image; CLIP filter with regen + fallback.

Outputs:
  artifacts/captions/raw.json       validated captions before filtering
  artifacts/captions/filtered.json  after CLIP filter + filter_metadata
"""

from pathlib import Path

from adversarial_filter import AdversarialFilter
from caption_generation_prompt import CaptionGenerationPrompt
from caption_validator import CaptionValidator
from clip_scorer import CLIPScorer
from config_loader import PROJECT_ROOT, load_config, resolve_path
from dataset_io import build_image_record, load_manifest, save_captions
from logger import get_logger
from model import QwenVLCaptionModel

logger = get_logger(__name__)


class AffordanceCaptionPipeline:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.prompt_builder = CaptionGenerationPrompt()
        self.validator = CaptionValidator(self.config)
        self.qwen = QwenVLCaptionModel(self.config)
        self.clip = CLIPScorer(self.config)

    def _output_paths(self) -> tuple[Path, Path]:
        raw = resolve_path(self.config["output"]["raw_captions"], PROJECT_ROOT)
        filtered = resolve_path(self.config["output"]["filtered_captions"], PROJECT_ROOT)
        return raw, filtered

    def _generate_validated_captions(
        self,
        entry: dict,
        prompt: str,
    ) -> tuple[list[str], list[str], dict]:
        """Stage 1: Qwen generation with retries when tiers come back empty."""
        image_path = entry["image_path"]
        max_retries = self.config["captions"].get("max_generation_retries", 0)
        last_raw: dict = {}

        for attempt in range(max_retries + 1):
            raw_data = self.qwen.generate_captions(image_path, prompt)
            if not self.validator.validate_json_structure(raw_data):
                raise ValueError(f"Invalid JSON structure for {entry['image_id']}")

            most_probable, negatives = self.validator.validate_record(
                raw_data.get("most_probable", []),
                raw_data.get("negative", []),
            )
            last_raw = raw_data

            if most_probable and negatives:
                return most_probable, negatives, last_raw

            if attempt < max_retries:
                logger.warning(
                    "Retrying caption generation for %s (empty tiers, attempt %d)",
                    entry["image_id"],
                    attempt + 1,
                )
                prompt = (
                    prompt
                    + "\n\nYou must return exactly one most_probable and one negative caption."
                )
            else:
                logger.warning(
                    "Generation finished with empty tiers for %s after %d attempts",
                    entry["image_id"],
                    max_retries + 1,
                )

        return (
            self.validator.validate_tier(last_raw.get("most_probable", []))[:1],
            self.validator.validate_tier(last_raw.get("negative", []))[:1],
            last_raw,
        )

    def _process_image(
        self,
        entry: dict,
        adv_filter: AdversarialFilter,
    ) -> tuple[dict, dict, list[dict]]:
        """Run stages 1 and 2 for a single manifest entry."""
        image_path = entry["image_path"]
        if not Path(image_path).is_file():
            raise FileNotFoundError(
                f"Image not found for {entry['image_id']}: {image_path}. "
                "Add images under data/sample/ matching manifest.json."
            )

        prompt = self.prompt_builder.build_prompt(entry["object"])
        logger.info("Generating captions for %s (%s)", entry["image_id"], entry["object"])

        most_probable, negatives, _raw_data = self._generate_validated_captions(entry, prompt)
        raw_record = build_image_record(entry, most_probable, negatives)

        filtered_record, filter_metadata = adv_filter.filter_with_regeneration(
            entry,
            most_probable,
            negatives,
        )
        return raw_record, filtered_record, filter_metadata

    def run(self, manifest_path: str | None = None) -> tuple[Path, Path]:
        entries = load_manifest(manifest_path)
        if not entries:
            raise ValueError("Manifest contains no images")

        raw_path, filtered_path = self._output_paths()

        logger.info("Loading Qwen-VL for caption generation")
        self.qwen.load_model()

        logger.info("Loading CLIP for adversarial filtering (mode=%s)", self.config["filter"].get("mode", "gap"))
        self.clip.load()
        adv_filter = AdversarialFilter(
            clip=self.clip,
            validator=self.validator,
            config=self.config,
            qwen=self.qwen,
            prompt_builder=self.prompt_builder,
        )

        raw_objects: list[dict] = []
        filtered_objects: list[dict] = []
        all_filter_metadata: dict[str, list] = {}

        try:
            for entry in entries:
                raw_record, filtered_record, meta = self._process_image(entry, adv_filter)
                raw_objects.append(raw_record)
                filtered_objects.append(filtered_record)
                all_filter_metadata[entry["image_id"]] = meta
                pair_meta = filtered_record.get("pair_metadata", {})
                logger.info(
                    "Done %s: %d pos, %d neg -> %d neg after filter (%s)",
                    entry["image_id"],
                    len(raw_record["affordance_tiers"]["most_probable"]),
                    len(raw_record["affordance_tiers"]["negative"]),
                    len(filtered_record["affordance_tiers"]["negative"]),
                    pair_meta.get("selection", "unknown"),
                )
        finally:
            self.qwen.unload()
            self.clip.unload()

        raw_output = {"objects": raw_objects}
        filtered_output = {
            "objects": filtered_objects,
            "filter_metadata": all_filter_metadata,
        }

        save_captions(raw_output, raw_path)
        save_captions(filtered_output, filtered_path)
        logger.info("Saved raw captions: %s", raw_path)
        logger.info("Saved filtered captions: %s", filtered_path)
        return raw_path, filtered_path


if __name__ == "__main__":
    pipeline = AffordanceCaptionPipeline()
    raw, filtered = pipeline.run()
    print(f"Raw:      {raw}")
    print(f"Filtered: {filtered}")
