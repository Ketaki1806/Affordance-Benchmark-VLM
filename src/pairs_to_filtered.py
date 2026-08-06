"""Convert humaneval / attribution pair dumps into filtered caption JSON for evaluate.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config_loader import PROJECT_ROOT, resolve_path
from dataset_io import save_captions


def pairs_to_filtered(pairs: list[dict]) -> dict:
    objects: list[dict] = []
    for p in pairs:
        pos = str(p.get("positive", "")).strip()
        neg = str(p.get("negative", "")).strip()
        image_path = p.get("image_path") or p.get("file")
        image_id = p.get("image_id")
        if not (pos and neg and image_path and image_id):
            continue
        objects.append(
            {
                "image_id": str(image_id),
                "object": p.get("object"),
                "image_path": str(image_path),
                "affordance_tiers": {
                    "most_probable": [pos],
                    "negative": [neg],
                },
            }
        )
    return {"objects": objects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-json",
        required=True,
        help="Eval dump with top-level 'pairs' (e.g. humaneval/30jul/clip.json)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/captions/val_100_pairs/filtered.json",
        help="Filtered caption JSON for evaluate.py",
    )
    args = parser.parse_args()

    src = resolve_path(args.pairs_json, PROJECT_ROOT)
    out = resolve_path(args.out, PROJECT_ROOT)
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", data)
    if not isinstance(pairs, list):
        raise SystemExit(f"No pairs list in {src}")

    payload = pairs_to_filtered(pairs)
    save_captions(payload, out)
    print(f"Wrote {len(payload['objects'])} pairs -> {out}")


if __name__ == "__main__":
    main()
