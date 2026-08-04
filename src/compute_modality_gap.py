"""Embedding modality gap on N=100 pairs (alignment geometry).

Distinct from occlusion vision_share (decision sensitivity):

  modality_gap      = || mean(z_img) - mean(z_txt_pos) ||
  modality_gap_all  = || mean(z_img) - mean(z_txt_pos∪neg) ||
  mean_matched_cos  = mean cos(z_img_i, z_pos_i)
  mean_matched_cos_neg = mean cos(z_img_i, z_neg_i)

  PYTHONPATH=src python src/compute_modality_gap.py \\
    --pairs-json humaneval/30jul/clip.json --all-pairs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from attribution_occlusion import _make_scorer, load_all_pairs, load_pairs_by_id
from config_loader import PROJECT_ROOT, load_config, resolve_path
from logger import get_logger

logger = get_logger(__name__)


def _mean_vec(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise ValueError("empty embedding list")
    d = len(rows[0])
    acc = [0.0] * d
    for row in rows:
        if len(row) != d:
            raise ValueError("ragged embedding rows")
        for j, v in enumerate(row):
            acc[j] += float(v)
    n = float(len(rows))
    return [v / n for v in acc]


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def modality_gap_from_stacks(
    image_embeds: Sequence[Sequence[float]],
    text_pos: Sequence[Sequence[float]],
    text_neg: Sequence[Sequence[float]] | None = None,
) -> dict[str, float]:
    """Compute gap + matched cosines from L2-normalized embed rows."""
    if len(image_embeds) == 0 or len(text_pos) == 0:
        raise ValueError("Expected non-empty image_embeds and text_pos")
    if len(image_embeds) != len(text_pos):
        raise ValueError(
            f"Length mismatch image {len(image_embeds)} vs pos {len(text_pos)}"
        )

    mean_img = _mean_vec(image_embeds)
    mean_pos = _mean_vec(text_pos)
    matched = [_dot(zi, zp) for zi, zp in zip(image_embeds, text_pos)]

    out: dict[str, float] = {
        "modality_gap": _l2(mean_img, mean_pos),
        "mean_matched_cos": sum(matched) / len(matched),
        "n_images": float(len(image_embeds)),
    }

    if text_neg is not None:
        if len(text_neg) != len(image_embeds):
            raise ValueError(
                f"Length mismatch image {len(image_embeds)} vs neg {len(text_neg)}"
            )
        all_txt = list(text_pos) + list(text_neg)
        mean_all = _mean_vec(all_txt)
        out["modality_gap_all"] = _l2(mean_img, mean_all)
        matched_neg = [_dot(zi, zn) for zi, zn in zip(image_embeds, text_neg)]
        out["mean_matched_cos_neg"] = sum(matched_neg) / len(matched_neg)
    return out


def _tensor_rows(tensors: list[Any]) -> list[list[float]]:
    return [t.detach().float().cpu().tolist() for t in tensors]


def _encode_pairs(scorer: Any, pairs: list[dict]) -> dict[str, float]:
    imgs: list[Any] = []
    poss: list[Any] = []
    negs: list[Any] = []
    for pair in pairs:
        imgs.append(scorer.encode_image(pair["image_path"]))
        poss.append(scorer.encode_text(pair["positive"]))
        negs.append(scorer.encode_text(pair["negative"]))
    return modality_gap_from_stacks(
        _tensor_rows(imgs),
        _tensor_rows(poss),
        _tensor_rows(negs),
    )


def run_modality_gap(
    pairs: list[dict],
    backends: list[str],
    out_dir: Path,
    config: dict | None = None,
    vljepa_checkpoint: str | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "metric_family": "embedding_modality_gap",
        "note": (
            "Alignment geometry (Liang-style centroid gap + matched cosines). "
            "Not the same as occlusion vision_share (decision sensitivity)."
        ),
        "num_pairs": len(pairs),
        "backends": {},
    }
    out_path = out_dir / "embedding_modality_gap.json"
    # Merge so a SigLIP-only re-run keeps CLIP / VLJEPA rows.
    if out_path.is_file():
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and isinstance(prev.get("backends"), dict):
                summary["backends"] = dict(prev["backends"])
        except (json.JSONDecodeError, OSError):
            pass

    for backend in backends:
        try:
            scorer = _make_scorer(backend, cfg, vljepa_checkpoint)
            scorer.load()
            try:
                metrics = _encode_pairs(scorer, pairs)
            finally:
                scorer.unload()
            summary["backends"][backend] = {
                "n": len(pairs),
                "modality_gap": metrics["modality_gap"],
                "modality_gap_all": metrics.get("modality_gap_all"),
                "mean_matched_cos": metrics["mean_matched_cos"],
                "mean_matched_cos_neg": metrics.get("mean_matched_cos_neg"),
            }
            logger.info(
                "%s: modality_gap=%.4f modality_gap_all=%.4f mean_matched_cos=%.4f",
                backend,
                metrics["modality_gap"],
                metrics.get("modality_gap_all", float("nan")),
                metrics["mean_matched_cos"],
            )
        except Exception as exc:  # noqa: BLE001
            summary["backends"][backend] = {"error": str(exc)}
            logger.exception("Backend %s failed", backend)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-json", required=True)
    parser.add_argument("--extra-pairs-json", nargs="+", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--image-ids", nargs="+", default=None)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="artifacts/attribution_n100",
        help="Same folder as occlusion N=100 outputs for side-by-side reporting",
    )
    parser.add_argument("--backends", nargs="+", default=["clip", "siglip", "open_vljepa"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--vljepa-checkpoint",
        default="artifacts/checkpoints/open-vljepa/best.pt",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    pairs_json = resolve_path(args.pairs_json, PROJECT_ROOT)
    if args.all_pairs:
        pairs = load_all_pairs(pairs_json)
    else:
        from attribution_occlusion import DEFAULT_IMAGE_IDS

        image_ids = args.image_ids or DEFAULT_IMAGE_IDS
        extra = (
            [resolve_path(p, PROJECT_ROOT) for p in args.extra_pairs_json]
            if args.extra_pairs_json
            else None
        )
        pairs = load_pairs_by_id(
            pairs_json, image_ids, extra_json=extra, skip_missing=args.skip_missing
        )

    out_dir = resolve_path(args.out_dir, PROJECT_ROOT)
    summary = run_modality_gap(
        pairs=pairs,
        backends=args.backends,
        out_dir=out_dir,
        config=config,
        vljepa_checkpoint=args.vljepa_checkpoint,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
