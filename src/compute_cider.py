"""
CIDEr scores for affordance caption ablations.

Without separate human reference captions for N=100, primary metric is
**hard-negative closeness**: CIDEr(negative | refs=[positive]).
Higher ⇒ negative is n-gram-closer to positive (harder lexical confound).

Human-eval ablations (N=20 pilot):
  1. neg_vs_pos on raw Qwen captions
  2. neg_vs_pos on human-filtered captions
  3. qwen_pos_vs_human_pos  (hypothesis=Qwen pos, ref=human pos)
  4. qwen_neg_vs_human_neg

Usage:
  PYTHONPATH=src python src/compute_cider.py \\
    --eval-json humaneval/30jul/clip.json \\
    --raw-captions humaneval/26jul/filtered.json \\
    --human-captions humaneval/26jul/human_filtered.json \\
    --output artifacts/eval/cider_ablations.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config_loader import PROJECT_ROOT, resolve_path
from logger import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenize(text: str) -> str:
    """Whitespace PTB-ish fallback (no Java/Stanford dependency)."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _load_objects(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", data)
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {path}")
    return objects


def _pairs_from_captions(objects: list[dict[str, Any]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for rec in objects:
        tiers = rec.get("affordance_tiers") or {}
        pos_list = [str(p).strip() for p in (tiers.get("most_probable") or []) if str(p).strip()]
        neg_list = [str(n).strip() for n in (tiers.get("negative") or []) if str(n).strip()]
        if not pos_list or not neg_list:
            continue
        pairs.append(
            {
                "image_id": str(rec.get("image_id", "")),
                "positive": pos_list[0],
                "negative": neg_list[0],
            }
        )
    return pairs


def _pairs_from_eval(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # summary.json style: {"results": {"clip": {"pairs": [...]}}}
    if "pairs" in data:
        raw_pairs = data["pairs"]
    elif "results" in data:
        raw_pairs = []
        for block in data["results"].values():
            raw_pairs.extend(block.get("pairs") or [])
    else:
        raise ValueError(f"Unrecognized eval JSON: {path}")

    pairs: list[dict[str, str]] = []
    for p in raw_pairs:
        pos = str(p.get("positive", "")).strip()
        neg = str(p.get("negative", "")).strip()
        if pos and neg:
            pairs.append(
                {
                    "image_id": str(p.get("image_id", "")),
                    "positive": pos,
                    "negative": neg,
                }
            )
    return pairs


def _cider_score(
    hypotheses: list[str],
    references: list[list[str]],
) -> tuple[float, list[float]]:
    """
    Mean CIDEr-D via pycocoevalcap.
    hypotheses[i] scored against references[i] (list of ref strings).
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as e:
        raise SystemExit(
            "pycocoevalcap is required. Install with:\n"
            "  pip install pycocoevalcap\n"
            f"Original error: {e}"
        ) from e

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references length mismatch")
    if not hypotheses:
        return 0.0, []

    # Cider.compute_score expects tokenized strings (not COCO caption dicts).
    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}
    for i, (hyp, refs) in enumerate(zip(hypotheses, references)):
        gts[i] = [_tokenize(r) for r in refs]
        res[i] = [_tokenize(hyp)]

    scorer = Cider()
    mean_score, scores = scorer.compute_score(gts, res)
    # scores may be numpy array
    per = [float(s) for s in scores]
    return float(mean_score), per


def score_neg_vs_pos(pairs: list[dict[str, str]]) -> dict[str, Any]:
    hyps = [p["negative"] for p in pairs]
    refs = [[p["positive"]] for p in pairs]
    mean, per = _cider_score(hyps, refs)
    rows = [
        {
            "image_id": p["image_id"],
            "positive": p["positive"],
            "negative": p["negative"],
            "cider": per[i],
        }
        for i, p in enumerate(pairs)
    ]
    return {
        "name": "neg_vs_pos",
        "description": "CIDEr(negative | ref=positive); higher = lexically closer hard negative",
        "num_pairs": len(pairs),
        "mean_cider": mean,
        "pairs": rows,
    }


def score_hyp_vs_ref(
    name: str,
    description: str,
    hyps: list[str],
    refs: list[str],
    image_ids: list[str],
) -> dict[str, Any]:
    mean, per = _cider_score(hyps, [[r] for r in refs])
    rows = [
        {
            "image_id": image_ids[i],
            "hypothesis": hyps[i],
            "reference": refs[i],
            "cider": per[i],
        }
        for i in range(len(hyps))
    ]
    return {
        "name": name,
        "description": description,
        "num_pairs": len(hyps),
        "mean_cider": mean,
        "pairs": rows,
    }


def _align_caption_sets(
    raw_objs: list[dict[str, Any]],
    human_objs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    human_by_id = {str(o["image_id"]): o for o in human_objs}
    aligned: list[dict[str, str]] = []
    for rec in raw_objs:
        hid = str(rec.get("image_id", ""))
        if hid not in human_by_id:
            continue
        raw_p = _pairs_from_captions([rec])
        hum_p = _pairs_from_captions([human_by_id[hid]])
        if not raw_p or not hum_p:
            continue
        aligned.append(
            {
                "image_id": hid,
                "qwen_pos": raw_p[0]["positive"],
                "qwen_neg": raw_p[0]["negative"],
                "human_pos": hum_p[0]["positive"],
                "human_neg": hum_p[0]["negative"],
            }
        )
    return aligned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Eval JSON with pairs (e.g. humaneval/30jul/clip.json) for N=100 CIDEr",
    )
    p.add_argument(
        "--captions-json",
        type=Path,
        default=None,
        help="Optional captions JSON (objects/affordance_tiers) instead of eval JSON",
    )
    p.add_argument(
        "--raw-captions",
        type=Path,
        default=None,
        help="Raw/Qwen captions for human-eval ablations",
    )
    p.add_argument(
        "--human-captions",
        type=Path,
        default=None,
        help="Human-filtered captions for human-eval ablations",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/eval/cider_ablations.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"ablations": {}}

    # --- N=100 / scale-up set ---
    scale_pairs: list[dict[str, str]] | None = None
    if args.eval_json is not None:
        path = resolve_path(str(args.eval_json), PROJECT_ROOT)
        scale_pairs = _pairs_from_eval(path)
        report["scale_up_source"] = str(path)
    elif args.captions_json is not None:
        path = resolve_path(str(args.captions_json), PROJECT_ROOT)
        scale_pairs = _pairs_from_captions(_load_objects(path))
        report["scale_up_source"] = str(path)

    if scale_pairs:
        logger.info("Scale-up set: %d pairs", len(scale_pairs))
        block = score_neg_vs_pos(scale_pairs)
        report["ablations"]["scale_up_neg_vs_pos"] = block
        print(
            f"[scale-up] neg_vs_pos  n={block['num_pairs']}  "
            f"mean_CIDEr={block['mean_cider']:.4f}"
        )

    # --- Human eval pilot ---
    if args.raw_captions and args.human_captions:
        raw_path = resolve_path(str(args.raw_captions), PROJECT_ROOT)
        hum_path = resolve_path(str(args.human_captions), PROJECT_ROOT)
        raw_objs = _load_objects(raw_path)
        hum_objs = _load_objects(hum_path)
        report["human_eval_raw_source"] = str(raw_path)
        report["human_eval_human_source"] = str(hum_path)

        raw_pairs = _pairs_from_captions(raw_objs)
        hum_pairs = _pairs_from_captions(hum_objs)

        raw_block = score_neg_vs_pos(raw_pairs)
        hum_block = score_neg_vs_pos(hum_pairs)
        report["ablations"]["pilot_raw_neg_vs_pos"] = raw_block
        report["ablations"]["pilot_human_neg_vs_pos"] = hum_block
        print(
            f"[pilot raw]    neg_vs_pos  n={raw_block['num_pairs']}  "
            f"mean_CIDEr={raw_block['mean_cider']:.4f}"
        )
        print(
            f"[pilot human]  neg_vs_pos  n={hum_block['num_pairs']}  "
            f"mean_CIDEr={hum_block['mean_cider']:.4f}"
        )

        aligned = _align_caption_sets(raw_objs, hum_objs)
        if aligned:
            pos_block = score_hyp_vs_ref(
                "qwen_pos_vs_human_pos",
                "CIDEr(Qwen positive | ref=human positive)",
                [a["qwen_pos"] for a in aligned],
                [a["human_pos"] for a in aligned],
                [a["image_id"] for a in aligned],
            )
            neg_block = score_hyp_vs_ref(
                "qwen_neg_vs_human_neg",
                "CIDEr(Qwen negative | ref=human negative)",
                [a["qwen_neg"] for a in aligned],
                [a["human_neg"] for a in aligned],
                [a["image_id"] for a in aligned],
            )
            report["ablations"]["pilot_qwen_pos_vs_human_pos"] = pos_block
            report["ablations"]["pilot_qwen_neg_vs_human_neg"] = neg_block
            print(
                f"[pilot] qwen_pos vs human_pos  n={pos_block['num_pairs']}  "
                f"mean_CIDEr={pos_block['mean_cider']:.4f}"
            )
            print(
                f"[pilot] qwen_neg vs human_neg  n={neg_block['num_pairs']}  "
                f"mean_CIDEr={neg_block['mean_cider']:.4f}"
            )

    if not report["ablations"]:
        raise SystemExit(
            "No inputs. Provide --eval-json and/or --raw-captions + --human-captions."
        )

    # Compact summary for the report table
    report["summary"] = {
        name: {"num_pairs": block["num_pairs"], "mean_cider": block["mean_cider"]}
        for name, block in report["ablations"].items()
    }

    out = resolve_path(str(args.output), PROJECT_ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Store per-pair only in ablations; keep file readable
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
