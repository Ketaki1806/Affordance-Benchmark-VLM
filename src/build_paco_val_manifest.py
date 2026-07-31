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
import random
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


def load_exclude_image_ids(path: Path) -> set[str]:
    """Image IDs from a prior manifest (e.g. eval N=100) to hold out of train."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", data)
    if not isinstance(images, list):
        raise ValueError(f"Exclude manifest must contain an 'images' list: {path}")
    return {str(item["image_id"]) for item in images if "image_id" in item}


def subsample_selected(
    selected: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Cap to n images: prefer category coverage, then random fill.

    First take one image per paco_category (shuffle within cat),
    then fill remaining slots from leftover images.
    """
    if n <= 0 or n >= len(selected):
        return selected

    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_cat[item["paco_category"]].append(item)

    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    cats = sorted(by_cat.keys())
    rng.shuffle(cats)
    for cat in cats:
        if len(picked) >= n:
            break
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        choice = pool[0]
        picked.append(choice)
        used.add(choice["lvis_image_id"])

    if len(picked) < n:
        rest = [item for item in selected if item["lvis_image_id"] not in used]
        rng.shuffle(rest)
        for item in rest:
            if len(picked) >= n:
                break
            picked.append(item)

    picked.sort(key=lambda x: x["lvis_image_id"])
    return picked[:n]


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
        "--n",
        type=int,
        default=None,
        help="Optional cap on number of images (category-diverse subsample)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed used when --n is set",
    )
    p.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Drop image_ids present in this manifest (hold out eval set from train)",
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
    if args.download_missing and args.image_root is None:
        raise SystemExit("--download-missing needs --image-root")

    if args.image_root is not None:
        root_str = str(args.image_root).replace("\\", "/").lower()
        if "path/to" in root_str or root_str in {"/path/to/coco", "path/to/coco"}:
            raise SystemExit(
                f"--image-root looks like a placeholder: {args.image_root}\n"
                "Use your real COCO root, e.g. data/paco/coco "
                "(must contain train2017/ and/or val2017/), "
                "or the path you used for the pilot."
            )

    # With --download-missing, do not filter on disk yet; ensure_images fetches later.
    require_image = args.require_image and not args.download_missing
    if args.require_image and args.download_missing:
        print(
            "Note: --download-missing disables upfront --require-image filtering; "
            "missing files are fetched after selection."
        )

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
        require_image=require_image,
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

    if args.exclude_manifest is not None:
        if not args.exclude_manifest.is_file():
            raise SystemExit(f"Exclude manifest not found: {args.exclude_manifest}")
        exclude_ids = load_exclude_image_ids(args.exclude_manifest)
        before = len(selected)
        selected = [item for item in selected if item["image_id"] not in exclude_ids]
        print(
            f"  excluded {before - len(selected)} images from "
            f"{args.exclude_manifest.name} ({len(exclude_ids)} ids); "
            f"{len(selected)} remain"
        )

    pool_size = len(selected)
    if args.n is not None:
        selected = subsample_selected(selected, args.n, args.seed)
        print(
            f"  subsampled with --n {args.n} seed={args.seed}: "
            f"{len(selected)} / {pool_size} pool images"
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
