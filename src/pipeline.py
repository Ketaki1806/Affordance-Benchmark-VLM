"""
Affordance caption pipeline.

Generate and validate captions with Qwen2.5-VL; write the same pairs to raw.json
and filtered.json (no CLIP adversarial filtering).

Outputs:
  artifacts/captions/raw.json       validated captions
  artifacts/captions/filtered.json  same pairs, used by stage 4 eval
"""

from pathlib import Path

from caption_generation_prompt import CaptionGenerationPrompt
from caption_validator import CaptionValidator
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

    def _output_paths(self) -> tuple[Path, Path]:
        raw = resolve_path(self.config["output"]["raw_captions"], PROJECT_ROOT)
        filtered = resolve_path(self.config["output"]["filtered_captions"], PROJECT_ROOT)
        return raw, filtered

    def _generate_validated_captions(
        self,
        entry: dict,
        prompt: str,
    ) -> tuple[list[str], list[str]]:
        """Qwen generation with retries when JSON/tiers are invalid or empty."""
        image_path = entry["image_path"]
        max_retries = self.config["captions"].get("max_generation_retries", 0)
        last_raw: dict = {}

        for attempt in range(max_retries + 1):
            raw_data = self.qwen.generate_captions(image_path, prompt)
            normalized = self.validator.normalize_caption_payload(raw_data)
            if not self.validator.validate_json_structure(normalized):
                logger.warning(
                    "Invalid JSON structure for %s (attempt %d): %s",
                    entry["image_id"],
                    attempt + 1,
                    raw_data,
                )
                if attempt < max_retries:
                    prompt = (
                        prompt
                        + "\n\nReturn ONLY valid JSON with keys "
                        '"most_probable" and "negative", each a list of one string.'
                    )
                    continue
                raise ValueError(
                    f"Invalid JSON structure for {entry['image_id']}: {raw_data}"
                )

            most_probable, negatives = self.validator.validate_record(
                normalized.get("most_probable", []),
                normalized.get("negative", []),
            )
            last_raw = normalized

            if most_probable and negatives:
                return most_probable, negatives

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
        )

    def _process_image(self, entry: dict) -> dict:
        image_path = entry["image_path"]
        if not Path(image_path).is_file():
            raise FileNotFoundError(
                f"Image not found for {entry['image_id']}: {image_path}. "
                "Add images under data/sample/ matching manifest.json."
            )

        prompt = self.prompt_builder.build_prompt(entry["object"])
        logger.info("Generating captions for %s (%s)", entry["image_id"], entry["object"])

        most_probable, negatives = self._generate_validated_captions(entry, prompt)
        return build_image_record(entry, most_probable, negatives)

    def run(self, manifest_path: str | None = None) -> tuple[Path, Path]:
        entries = load_manifest(manifest_path)
        if not entries:
            raise ValueError("Manifest contains no images")

        raw_path, filtered_path = self._output_paths()

        logger.info("Loading Qwen-VL for caption generation")
        self.qwen.load_model()

        records: list[dict] = []
        try:
            for entry in entries:
                record = self._process_image(entry)
                records.append(record)
                tiers = record["affordance_tiers"]
                logger.info(
                    "Done %s: %d pos, %d neg",
                    entry["image_id"],
                    len(tiers["most_probable"]),
                    len(tiers["negative"]),
                )
        finally:
            self.qwen.unload()

        output = {"objects": records}
        save_captions(output, raw_path)
        save_captions(output, filtered_path)
        logger.info("Saved raw captions: %s", raw_path)
        logger.info("Saved filtered captions: %s", filtered_path)
        return raw_path, filtered_path


if __name__ == "__main__":
    pipeline = AffordanceCaptionPipeline()
    raw, filtered = pipeline.run()
    print(f"Raw:      {raw}")
    print(f"Filtered: {filtered}")
