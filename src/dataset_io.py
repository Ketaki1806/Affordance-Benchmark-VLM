"""Load/save manifest and caption JSON (PACO-LVIS-ready schema)."""

import json
from pathlib import Path
from typing import Any

from config_loader import PROJECT_ROOT, load_config, resolve_path


def load_manifest(manifest_path: Path | str | None = None) -> list[dict[str, Any]]:
    """
    Read data/sample/manifest.json (or PACO manifest with same schema).
    Resolves image paths relative to data.sample_dir in config.
    """
    config = load_config()
    path = resolve_path(
        manifest_path or config["data"]["manifest_path"],
        PROJECT_ROOT,
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", data)
    if not isinstance(images, list):
        raise ValueError(f"Manifest must contain an 'images' list: {path}")

    entries: list[dict[str, Any]] = []
    sample_dir = resolve_path(config["data"]["sample_dir"], PROJECT_ROOT)
    for item in images:
        image_id = item["image_id"]
        file_name = item["file"]
        image_path = sample_dir / file_name
        entries.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "file": file_name,
                "object": item["object"],
                "paco_category": item.get("paco_category"),
                "part": item.get("part"),
                "attributes": item.get("attributes"),
                "source_split": item.get("source_split"),
            }
        )
    return entries


def save_captions(data: dict[str, Any], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_image_record(
    entry: dict[str, Any],
    most_probable: list[str],
    negative: list[str],
    pair_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One record in raw.json / filtered.json matching the project document shape."""
    record: dict[str, Any] = {
        "image_id": entry["image_id"],
        "image_path": entry["image_path"],
        "object": entry["object"],
        "affordance_tiers": {
            "most_probable": most_probable,
            "negative": negative,
        },
    }
    if pair_meta:
        record["pair_metadata"] = pair_meta
    return record


def collapse_to_single_pair(
    most_probable: list[str],
    negative: list[str],
) -> tuple[list[str], list[str]]:
    """Pilot export: at most one positive and one negative caption."""
    pos = [most_probable[0]] if most_probable else []
    neg = [negative[0]] if negative else []
    return pos, neg
