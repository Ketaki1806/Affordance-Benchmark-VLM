"""
Build a full PACO-LVIS val manifest: one preferred interaction part per unique image.

Reuses candidate construction from build_paco_pilot_manifest (PREFERRED_PARTS ranking).
Per image, keeps the best (part_rank ascending, object area descending) candidate
that has a non-null preferred/any part annotation.

Usage:
  python src/build_paco_val_manifest.py \\
    --ann data/paco/annotations/paco_lvis_v1_val.json \\
    --image-root /path/to/coco \\
    --require-image \\
    --download-missing
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_paco_pilot_manifest import (
    _build_candidates,
    _diagnose_image_root,
    _load_json,
    _manifest_entry,
    copy_images,
    ensure_images,
)
from config_loader import PROJECT_ROOT


def select_one_per_image(
    by_object: dict[str, list[dict[str, Any]]],
    *,
    require_part: bool = True,
) -> list[dict[str, Any]]:
    """
    One candidate per unique lvis_image_id.

    Sort key: (part_rank ascending, area descending).
    """
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for items in by_object.values():
        for item in items:
            if require_part and not item.get("part"):
                continue
            by_image[item["lvis_image_id"]].append(item)

    selected: list[dict[str, Any]] = []
    for image_id in sorted(by_image.keys()):
        candidates = by_image[image_id]
        candidates.sort(key=lambda x: (x["part_rank"], -x["area"]))
        selected.append(candidates[0])

    selected.sort(key=lambda x: x["lvis_image_id"])
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ann",
        type=Path,
        default=PROJECT_ROOT / "data/paco/annotations/paco_lvis_v1_val.json",
        help="PACO-LVIS annotation JSON (val)",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="COCO/LVIS image root (contains val2017/train2017 or flat jpgs)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/paco/manifest_val_full.json",
        help="Output manifest path",
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=5000.0,
        help="Minimum object box area (pixels^2)",
    )
    p.add_argument(
        "--require-image",
        action="store_true",
        help="Only keep candidates whose image file exists under --image-root",
    )
    p.add_argument(
        "--allow-no-part",
        action="store_true",
        help="Keep images even when no part annotation exists (default: skip)",
    )
    p.add_argument(
        "--download-missing",
        action="store_true",
        help=(
            "Download missing images from the COCO CDN "
            "(tries val2017 then train2017). Needs --image-root."
        ),
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy selected images into --sample-dir and rewrite file paths",
    )
    p.add_argument(
        "--sample-dir",
        type=Path,
        default=PROJECT_ROOT / "data/paco/images_val_full",
        help="Destination for --copy-images",
    )
    p.add_argument(
        "--source-split",
        default="val",
        help="Stored in manifest source_split field",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ann.is_file():
        raise SystemExit(f"Annotation file not found: {args.ann}")

    if args.require_image and args.image_root is None:
        raise SystemExit("--require-image needs --image-root")

    print(f"Loading {args.ann} ...")
    data = _load_json(args.ann)
    print(
        f"  images={len(data['images'])}  "
        f"annotations={len(data['annotations'])}  "
        f"categories={len(data['categories'])}"
    )
    print(
        "  Scale note: PACO-LVIS val has ~20.9k part segments; "
        "this script emits one preferred part per unique image (~<=2410), "
        "not every part instance."
    )

    if args.image_root is not None:
        _diagnose_image_root(data, args.image_root)

    by_object = _build_candidates(
        data,
        image_root=args.image_root,
        require_image=args.require_image,
        min_area=args.min_area,
    )
    n_object_candidates = sum(len(v) for v in by_object.values())
    print(f"  object-instance candidates: {n_object_candidates}")

    selected = select_one_per_image(
        by_object,
        require_part=not args.allow_no_part,
    )
    if not selected:
        _diagnose_image_root(data, args.image_root)
        raise SystemExit(
            "No candidates found. Check --image-root / --require-image / --min-area."
        )

    categories = sorted({item["paco_category"] for item in selected})
    with_preferred = sum(1 for item in selected if item["part_rank"] < 1000)
    print(
        f"  selected images: {len(selected)}  "
        f"categories: {len(categories)}  "
        f"preferred-part hits: {with_preferred}"
    )

    if args.download_missing or args.copy_images:
        if args.image_root is None:
            raise SystemExit("--download-missing / --copy-images need --image-root")
        ensure_images(
            selected,
            image_root=args.image_root,
            download_missing=args.download_missing,
        )

    if args.copy_images:
        copy_images(selected, args.sample_dir)
        print(f"Copied images to {args.sample_dir}")

    # Drop unresolved rows only after an image-ensure pass
    if args.download_missing or args.copy_images:
        selected = [item for item in selected if item.get("resolved_path")]

    if not selected:
        raise SystemExit("No images left after resolve/download/copy.")

    manifest = {
        "images": [_manifest_entry(item, args.source_split) for item in selected]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(manifest['images'])} entries -> {args.output}")
    print("\nNext: point configs/config.yaml at the full-val set:")
    sample_dir = (
        args.sample_dir if args.copy_images else (args.image_root or "data/paco/images")
    )
    print(f"  data.sample_dir: {sample_dir}")
    print(f"  data.manifest_path: {args.output}")
    print("  output.*: artifacts/captions/val_full/ and artifacts/eval/val_full/")


if __name__ == "__main__":
    main()
