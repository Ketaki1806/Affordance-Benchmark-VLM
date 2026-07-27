"""
Affordance caption pipeline.

Generate and validate captions with Qwen2.5-VL; write the same pairs to raw.json
and filtered.json (no CLIP adversarial filtering).

Supports resume (skip completed image_ids), incremental saves, and Condor sharding.

Outputs (from config; shard runs write *.shard{K}.json):
  artifacts/captions/.../raw.json
  artifacts/captions/.../filtered.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from caption_generation_prompt import CaptionGenerationPrompt
from caption_validator import CaptionValidator
from config_loader import PROJECT_ROOT, load_config, resolve_path
from dataset_io import (
    build_image_record,
    completed_image_ids,
    load_captions,
    load_manifest,
    save_captions,
    shard_entries,
    shard_output_path,
)
from logger import get_logger
from model import QwenVLCaptionModel

logger = get_logger(__name__)


class AffordanceCaptionPipeline:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.prompt_builder = CaptionGenerationPrompt()
        self.validator = CaptionValidator(self.config)
        self.qwen = QwenVLCaptionModel(self.config)

    def _base_output_paths(self) -> tuple[Path, Path]:
        raw = resolve_path(self.config["output"]["raw_captions"], PROJECT_ROOT)
        filtered = resolve_path(self.config["output"]["filtered_captions"], PROJECT_ROOT)
        return raw, filtered

    def _output_paths(self, shard_index: int | None) -> tuple[Path, Path]:
        raw, filtered = self._base_output_paths()
        return shard_output_path(raw, shard_index), shard_output_path(filtered, shard_index)

    def _generate_validated_captions(
        self,
        entry: dict,
        prompt: str,
    ) -> tuple[list[str], list[str]]:
        """Qwen generation with retries when JSON/tiers are invalid or empty."""
        max_retries = self.config["captions"].get("max_generation_retries", 0)
        last_raw: dict = {}

        for attempt in range(max_retries + 1):
            raw_data = self.qwen.generate_captions(entry["image_path"], prompt)
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
                "Check data.sample_dir / image_root and the manifest file paths."
            )

        prompt = self.prompt_builder.build_prompt(
            entry["object"],
            part=entry.get("part"),
            attributes=entry.get("attributes"),
        )
        logger.info("Generating captions for %s (%s)", entry["image_id"], entry["object"])

        most_probable, negatives = self._generate_validated_captions(entry, prompt)
        return build_image_record(entry, most_probable, negatives)

    def run(
        self,
        manifest_path: str | None = None,
        *,
        resume: bool = True,
        limit: int | None = None,
        shard_index: int | None = None,
        num_shards: int = 1,
        save_every: int = 1,
    ) -> tuple[Path, Path]:
        entries = load_manifest(manifest_path)
        if not entries:
            raise ValueError("Manifest contains no images")

        if num_shards > 1:
            if shard_index is None:
                raise ValueError("num_shards > 1 requires shard_index")
            entries = shard_entries(entries, shard_index, num_shards)
            logger.info(
                "Shard %d/%d: %d manifest entries",
                shard_index,
                num_shards,
                len(entries),
            )

        raw_path, filtered_path = self._output_paths(shard_index)

        existing = load_captions(raw_path) if resume else {"objects": []}
        done_ids = completed_image_ids(existing) if resume else set()
        # Keep only complete pairs so retries do not duplicate image_ids
        records: list[dict] = [
            obj
            for obj in existing.get("objects", [])
            if str(obj.get("image_id", "")) in done_ids
        ]
        if done_ids:
            logger.info("Resume: skipping %d completed image_ids in %s", len(done_ids), raw_path)

        pending = [e for e in entries if e["image_id"] not in done_ids]
        if limit is not None:
            pending = pending[: max(0, limit)]

        if not pending:
            logger.info("Nothing to process; writing existing records to filtered path")
            output = {"objects": records}
            save_captions(output, raw_path)
            save_captions(output, filtered_path)
            return raw_path, filtered_path

        logger.info("Loading Qwen-VL for caption generation (%d pending)", len(pending))
        self.qwen.load_model()

        try:
            for i, entry in enumerate(pending, start=1):
                record = self._process_image(entry)
                records.append(record)
                tiers = record["affordance_tiers"]
                logger.info(
                    "Done %s (%d/%d): %d pos, %d neg",
                    entry["image_id"],
                    i,
                    len(pending),
                    len(tiers["most_probable"]),
                    len(tiers["negative"]),
                )
                if save_every > 0 and (i % save_every == 0 or i == len(pending)):
                    output = {"objects": records}
                    save_captions(output, raw_path)
                    save_captions(output, filtered_path)
                    logger.info("Checkpoint saved (%d records) -> %s", len(records), raw_path)
        finally:
            self.qwen.unload()

        output = {"objects": records}
        save_captions(output, raw_path)
        save_captions(output, filtered_path)
        logger.info("Saved raw captions: %s", raw_path)
        logger.info("Saved filtered captions: %s", filtered_path)
        return raw_path, filtered_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        default=None,
        help="Override config data.manifest_path",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending images (after resume/shard filters)",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="0-based shard index (with --num-shards)",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards (default 1 = no sharding)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing shard/output JSON and regenerate",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write checkpoint every N images (default 1)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = AffordanceCaptionPipeline()
    raw, filtered = pipeline.run(
        args.manifest,
        resume=not args.no_resume,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        save_every=max(1, args.save_every),
    )
    print(f"Raw:      {raw}")
    print(f"Filtered: {filtered}")
