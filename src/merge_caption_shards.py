"""
Merge sharded caption JSON files into raw.json + filtered.json.

Looks for raw.shard*.json next to the config raw_captions path (or --raw-glob).

Usage:
  export PYTHONPATH=src
  python src/merge_caption_shards.py
  python src/merge_caption_shards.py --raw-dir artifacts/captions/val_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config_loader import PROJECT_ROOT, load_config, resolve_path
from dataset_io import save_captions


def _find_shard_files(raw_base: Path) -> list[Path]:
    parent = raw_base.parent
    stem = raw_base.stem  # e.g. "raw"
    suffix = raw_base.suffix  # .json
    pattern = f"{stem}.shard*{suffix}"
    files = sorted(parent.glob(pattern), key=lambda p: p.name)
    return files


def _load_objects(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        objects = data.get("objects", [])
    elif isinstance(data, list):
        objects = data
    else:
        objects = []
    if not isinstance(objects, list):
        raise ValueError(f"Invalid caption file: {path}")
    return objects


def merge_objects(shard_files: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in shard_files:
        for obj in _load_objects(path):
            image_id = str(obj.get("image_id", ""))
            if not image_id:
                continue
            tiers = obj.get("affordance_tiers") or {}
            if not (tiers.get("most_probable") and tiers.get("negative")):
                continue
            by_id[image_id] = obj
    return [by_id[k] for k in sorted(by_id.keys())]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/config.yaml); used for default output paths",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing raw.shard*.json (default: parent of config raw_captions)",
    )
    p.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Merged raw output (default: config output.raw_captions)",
    )
    p.add_argument(
        "--filtered-out",
        type=Path,
        default=None,
        help="Merged filtered output (default: config output.filtered_captions)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    raw_out = resolve_path(args.raw_out or cfg["output"]["raw_captions"], PROJECT_ROOT)
    filtered_out = resolve_path(
        args.filtered_out or cfg["output"]["filtered_captions"],
        PROJECT_ROOT,
    )
    if args.raw_dir is not None:
        raw_base = Path(args.raw_dir) / "raw.json"
        if not raw_base.is_absolute():
            raw_base = PROJECT_ROOT / raw_base
    else:
        raw_base = raw_out

    shard_files = _find_shard_files(raw_base)
    if not shard_files:
        raise SystemExit(f"No shard files matching {raw_base.stem}.shard*{raw_base.suffix} in {raw_base.parent}")

    print(f"Merging {len(shard_files)} shards from {raw_base.parent}:")
    for path in shard_files:
        print(f"  {path.name}")

    objects = merge_objects(shard_files)
    output = {"objects": objects}
    save_captions(output, raw_out)
    save_captions(output, filtered_out)
    print(f"Wrote {len(objects)} unique pairs -> {raw_out}")
    print(f"Wrote {len(objects)} unique pairs -> {filtered_out}")


if __name__ == "__main__":
    main()
