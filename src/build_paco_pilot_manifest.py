"""
Build a PACO-LVIS pilot manifest (one image per object category).

Run on the cluster login node after downloading PACO annotations:

  cd ~/Affordance-Benchmark-VLM
  python src/build_paco_pilot_manifest.py \\
    --ann data/paco/annotations/paco_lvis_v1_val.json \\
    --image-root /path/to/coco/images \\
    --n 20 \\
    --require-image \\
    --copy-images

Writes data/paco/manifest_pilot.json compatible with dataset_io.load_manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from config_loader import PROJECT_ROOT

# Prefer interaction parts over generic body/base for affordance captions.
PREFERRED_PARTS = (
    "cap",
    "lid",
    "handle",
    "blade",
    "rim",
    "spout",
    "tip",
    "shank",
    "grip",
    "face",
    "head",
    "drawer",
    "switch",
    "button",
    "knob",
    "nozzle",
    "pull_tab",
    "door_handle",
)

# Categories aligned with the existing sample set + related PACO objects.
DEFAULT_CATEGORIES = (
    "bottle",
    "bowl",
    "mug",
    "hammer",
    "screwdriver",
    "knife",
    "jar",
    "kettle",
    "spoon",
    "scissors",
    "blender",
    "pan_(for_cooking)",
    "cup",
    "can",
    "box",
    "chair",
    "microwave_oven",
    "broom",
    "pliers",
    "wrench",
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_part_category(cat: dict[str, Any]) -> bool:
    if cat.get("supercategory") == "PART":
        return True
    name = cat.get("name", "")
    return ":" in name


def _object_and_part(cat_name: str) -> tuple[str, str | None]:
    if ":" in cat_name:
        obj, part = cat_name.split(":", 1)
        return obj, part
    return cat_name, None


def _part_rank(part: str | None) -> int:
    if part is None:
        return 10_000
    try:
        return PREFERRED_PARTS.index(part)
    except ValueError:
        return 1_000 + len(part)


def _attr_names(attr_ids: list[int] | None, id_to_attr: dict[int, str]) -> list[str]:
    if not attr_ids:
        return []
    names: list[str] = []
    for aid in attr_ids:
        name = id_to_attr.get(aid)
        if name and not name.startswith("other("):
            names.append(name)
    return names


def _resolve_image_path(image_root: Path | None, file_name: str) -> Path | None:
    if image_root is None:
        return None
    direct = image_root / file_name
    if direct.is_file():
        return direct
    # COCO/LVIS layout: train2017/ or val2017/
    # PACO-LVIS val images often live under train2017 (LVIS ≠ COCO split).
    for split in ("val2017", "train2017", "test2017"):
        candidate = image_root / split / Path(file_name).name
        if candidate.is_file():
            return candidate
    # Bare filename under root
    bare = image_root / Path(file_name).name
    if bare.is_file():
        return bare
    return None


def _download_coco_image(file_name: str, image_root: Path) -> Path | None:
    """Fetch one COCO image; try val2017 then train2017 CDN paths."""
    import urllib.error
    import urllib.request

    name = Path(file_name).name
    existing = _resolve_image_path(image_root, name)
    if existing is not None:
        return existing

    for split in ("val2017", "train2017"):
        url = f"http://images.cocodataset.org/{split}/{name}"
        dest_dir = image_root / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        try:
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, dest)
            if dest.is_file() and dest.stat().st_size > 1000:
                return dest
            dest.unlink(missing_ok=True)
        except urllib.error.HTTPError as exc:
            print(f"  skip {split}: HTTP {exc.code}")
            dest.unlink(missing_ok=True)
        except OSError as exc:
            print(f"  skip {split}: {exc}")
            dest.unlink(missing_ok=True)
    return None


def _diagnose_image_root(data: dict[str, Any], image_root: Path | None) -> None:
    if image_root is None:
        print("  (no --image-root; cannot check which files exist on disk)")
        return
    names = [Path(img["file_name"]).name for img in data["images"][:200]]
    in_val = sum(1 for n in names if (image_root / "val2017" / n).is_file())
    in_train = sum(1 for n in names if (image_root / "train2017" / n).is_file())
    print(
        f"  image check (first {len(names)} PACO files): "
        f"{in_val} in val2017/, {in_train} in train2017/"
    )
    if in_val == 0 and in_train == 0:
        sample = names[:3]
        print(f"  example PACO file_name values: {sample}")
        print(
            "  Hint: PACO-LVIS val images are often under COCO train2017. "
            "Use --download-missing to fetch only the pilot images."
        )


def _build_candidates(
    data: dict[str, Any],
    image_root: Path | None,
    require_image: bool,
    min_area: float,
) -> dict[str, list[dict[str, Any]]]:
    id_to_cat = {c["id"]: c for c in data["categories"]}
    id_to_attr = {a["id"]: a["name"] for a in data.get("attributes", [])}
    id_to_image = {img["id"]: img for img in data["images"]}

    # Object annotations keyed by ann id
    obj_anns: dict[int, dict[str, Any]] = {}
    part_by_obj: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ann in data["annotations"]:
        cat = id_to_cat.get(ann["category_id"])
        if cat is None:
            continue
        if _is_part_category(cat):
            parent = ann.get("obj_ann_id")
            if parent is not None and parent >= 0:
                part_by_obj[parent].append(ann)
        else:
            obj_anns[ann["id"]] = ann

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for obj_ann_id, obj_ann in obj_anns.items():
        cat = id_to_cat[obj_ann["category_id"]]
        obj_name, _ = _object_and_part(cat["name"])
        area = float(obj_ann.get("area") or 0.0)
        if area < min_area:
            continue

        img = id_to_image.get(obj_ann["image_id"])
        if img is None:
            continue
        file_name = img["file_name"]
        resolved = _resolve_image_path(image_root, file_name)
        if require_image and resolved is None:
            continue

        parts = part_by_obj.get(obj_ann_id, [])
        best_part_name: str | None = None
        best_attrs: list[str] = []
        best_rank = 10_000

        for part_ann in parts:
            part_cat = id_to_cat.get(part_ann["category_id"])
            if part_cat is None:
                continue
            _, part_name = _object_and_part(part_cat["name"])
            if part_name is None:
                part_name = part_cat.get("synset") or part_cat["name"]
            rank = _part_rank(part_name)
            if rank < best_rank:
                best_rank = rank
                best_part_name = part_name
                best_attrs = _attr_names(part_ann.get("attribute_ids"), id_to_attr)

        if best_part_name is None:
            # Fall back to object-level attributes if no part exists
            best_attrs = _attr_names(obj_ann.get("attribute_ids"), id_to_attr)

        if resolved is not None and image_root is not None:
            try:
                rel_file = str(Path(resolved).relative_to(image_root))
            except ValueError:
                rel_file = Path(resolved).name
        else:
            rel_file = file_name

        by_object[obj_name].append(
            {
                "image_id": f"lvis_{img['id']}",
                "lvis_image_id": img["id"],
                "file": rel_file,
                "resolved_path": str(resolved) if resolved else None,
                "object": obj_name.replace("_", " "),
                "paco_category": obj_name,
                "part": best_part_name,
                "attributes": best_attrs,
                "area": area,
                "part_rank": best_rank,
            }
        )

    # Prefer preferred parts, then larger boxes
    for obj_name, items in by_object.items():
        items.sort(key=lambda x: (x["part_rank"], -x["area"]))
    return by_object


def sample_pilot(
    by_object: dict[str, list[dict[str, Any]]],
    categories: list[str],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_images: set[int] = set()

    # First pass: requested categories in order
    for cat in categories:
        if len(selected) >= n:
            break
        candidates = [
            c for c in by_object.get(cat, []) if c["lvis_image_id"] not in used_images
        ]
        if not candidates:
            continue
        # Keep top-ranked (preferred part / large), shuffle among top few
        top = candidates[: min(8, len(candidates))]
        pick = rng.choice(top)
        used_images.add(pick["lvis_image_id"])
        selected.append(pick)

    # Fill remaining slots from any remaining categories
    if len(selected) < n:
        remaining_cats = [c for c in sorted(by_object.keys()) if c not in categories]
        rng.shuffle(remaining_cats)
        for cat in remaining_cats:
            if len(selected) >= n:
                break
            candidates = [
                c for c in by_object.get(cat, []) if c["lvis_image_id"] not in used_images
            ]
            if not candidates:
                continue
            pick = candidates[0]
            used_images.add(pick["lvis_image_id"])
            selected.append(pick)

    return selected[:n]


def _manifest_entry(item: dict[str, Any], source_split: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "image_id": item["image_id"],
        "file": item["file"],
        "object": item["object"],
        "paco_category": item["paco_category"],
        "source_split": source_split,
    }
    if item.get("part"):
        entry["part"] = item["part"]
    if item.get("attributes"):
        entry["attributes"] = item["attributes"]
    return entry


def ensure_images(
    selected: list[dict[str, Any]],
    image_root: Path,
    download_missing: bool,
) -> None:
    """Resolve each selected image; optionally download missing COCO files."""
    missing = 0
    for item in selected:
        resolved = item.get("resolved_path")
        path = Path(resolved) if resolved else None
        if path is None or not path.is_file():
            path = _resolve_image_path(image_root, item["file"])
        if (path is None or not path.is_file()) and download_missing:
            path = _download_coco_image(item["file"], image_root)
        if path is None or not path.is_file():
            missing += 1
            print(f"  MISSING image for {item['image_id']}: {item['file']}")
            item["resolved_path"] = None
            continue
        item["resolved_path"] = str(path)
        # Keep relative path under image_root when possible
        try:
            item["file"] = str(Path(path).relative_to(image_root))
        except ValueError:
            item["file"] = Path(path).name
    if missing:
        raise SystemExit(
            f"{missing}/{len(selected)} images missing. "
            "Re-run with --download-missing or add train2017 images."
        )


def copy_images(
    selected: list[dict[str, Any]],
    dest_dir: Path,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        src = item.get("resolved_path")
        if not src:
            continue
        src_path = Path(src)
        # Flat copy using LVIS image id for stable names
        dest_name = f"{item['lvis_image_id']:012d}{src_path.suffix}"
        dest = dest_dir / dest_name
        if not dest.exists():
            shutil.copy2(src_path, dest)
        item["file"] = dest_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ann",
        type=Path,
        default=PROJECT_ROOT / "data/paco/annotations/paco_lvis_v1_val.json",
        help="PACO-LVIS annotation JSON (prefer val for pilot)",
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
        default=PROJECT_ROOT / "data/paco/manifest_pilot.json",
        help="Output manifest path",
    )
    p.add_argument("--n", type=int, default=20, help="Number of images (default 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--categories",
        nargs="*",
        default=list(DEFAULT_CATEGORIES),
        help="Object category names to prioritize",
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
        "--download-missing",
        action="store_true",
        help=(
            "After sampling, download missing pilot images from the COCO CDN "
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
        default=PROJECT_ROOT / "data/paco/images",
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

    if args.image_root is not None:
        _diagnose_image_root(data, args.image_root)

    by_object = _build_candidates(
        data,
        image_root=args.image_root,
        require_image=args.require_image,
        min_area=args.min_area,
    )
    available = {k: len(v) for k, v in by_object.items() if v}
    print(f"  object categories with candidates: {len(available)}")

    selected = sample_pilot(by_object, args.categories, args.n, args.seed)
    if not selected:
        _diagnose_image_root(data, args.image_root)
        raise SystemExit(
            "No candidates found. If you only have val2017/, drop --require-image "
            "and use --download-missing (PACO val images are often in train2017)."
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

    manifest = {
        "images": [_manifest_entry(item, args.source_split) for item in selected]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(manifest['images'])} entries -> {args.output}")
    for item in selected:
        part = item.get("part") or "-"
        attrs = ",".join(item.get("attributes") or []) or "-"
        print(f"  {item['paco_category']:24s} part={part:16s} attrs={attrs}")

    print("\nNext: point configs/config.yaml at the pilot:")
    sample_dir = args.sample_dir if args.copy_images else (args.image_root or "data/paco/images")
    print(f"  data.sample_dir: {sample_dir}")
    print(f"  data.manifest_path: {args.output}")


if __name__ == "__main__":
    main()
