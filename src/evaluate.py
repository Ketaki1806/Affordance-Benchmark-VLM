"""
Stage 4 evaluation: frozen CLIP vs Open-VLJEPA on affordance caption pairs.

For each (most_probable, negative) pair, the backend with higher cosine
similarity wins. Outputs per-pair results and aggregate metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from clip_scorer import CLIPScorer
from config_loader import PROJECT_ROOT, load_config, resolve_path
from logger import get_logger
from open_vljepa_scorer import OpenVLJEPAScorer

logger = get_logger(__name__)


class AffordanceScorer(Protocol):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def score(self, image_path: str, text: str) -> float: ...


def _load_filtered_captions(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", data)
    if not isinstance(objects, list):
        raise ValueError(f"Expected 'objects' list in {path}")
    return objects


def _iter_pairs(record: dict[str, Any]) -> list[tuple[str, str]]:
    tiers = record.get("affordance_tiers", {})
    positives = tiers.get("most_probable", [])
    negatives = tiers.get("negative", [])
    pairs: list[tuple[str, str]] = []
    for pos in positives:
        for neg in negatives:
            pairs.append((str(pos).strip(), str(neg).strip()))
    return pairs


def _evaluate_pair(
    scorer: AffordanceScorer,
    backend: str,
    image_path: str,
    positive: str,
    negative: str,
) -> dict[str, Any]:
    pos_score = scorer.score(image_path, positive)
    neg_score = scorer.score(image_path, negative)
    chosen = "most_probable" if pos_score >= neg_score else "negative"
    correct = chosen == "most_probable"
    gap = abs(pos_score - neg_score)
    return {
        "positive": positive,
        "negative": negative,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "confidence_gap": gap,
        "model_choice": chosen,
        "correct": correct,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_pairs": 0,
            "binary_accuracy": 0.0,
            "mean_confidence_gap": 0.0,
        }
    correct = sum(1 for r in results if r["correct"])
    gaps = [r["confidence_gap"] for r in results]
    return {
        "num_pairs": len(results),
        "binary_accuracy": correct / len(results),
        "mean_confidence_gap": sum(gaps) / len(gaps),
        "num_correct": correct,
        "num_wrong": len(results) - correct,
    }


def _evaluate_backend(
    backend: str,
    scorer: AffordanceScorer,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    logger.info("Evaluating backend: %s", backend)
    scorer.load()
    pair_results: list[dict[str, Any]] = []
    try:
        for record in records:
            image_path = record.get("image_path") or record.get("file")
            if not image_path:
                raise ValueError(f"Missing image_path for {record.get('image_id')}")
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            for positive, negative in _iter_pairs(record):
                pair = _evaluate_pair(scorer, backend, image_path, positive, negative)
                pair_results.append(
                    {
                        "image_id": record.get("image_id"),
                        "object": record.get("object"),
                        "image_path": image_path,
                        **pair,
                    }
                )
                logger.info(
                    "%s %s: pos=%.4f neg=%.4f correct=%s",
                    backend,
                    record.get("image_id"),
                    pair["pos_score"],
                    pair["neg_score"],
                    pair["correct"],
                )
    finally:
        scorer.unload()

    return {
        "backend": backend,
        "summary": _summarize(pair_results),
        "pairs": pair_results,
    }


def run_evaluation(
    filtered_path: Path | str | None = None,
    backends: list[str] | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    path = resolve_path(
        filtered_path or cfg["output"]["filtered_captions"],
        PROJECT_ROOT,
    )
    records = _load_filtered_captions(path)

    eval_cfg = cfg.get("eval", {})
    selected = list(backends or eval_cfg.get("backends", ["clip"]))
    output: dict[str, Any] = {
        "source": str(path),
        "num_images": len(records),
        "results": {},
    }

    for backend in selected:
        if backend == "clip":
            scorer: AffordanceScorer = CLIPScorer(cfg)
            result = _evaluate_backend("clip", scorer, records)
            out_path = resolve_path(cfg["output"]["eval_clip"], PROJECT_ROOT)
        elif backend == "open_vljepa":
            if not cfg.get("models", {}).get("open_vljepa", {}).get("enabled", False):
                logger.warning("Skipping open_vljepa (models.open_vljepa.enabled=false)")
                continue
            scorer = OpenVLJEPAScorer(cfg)
            result = _evaluate_backend("open_vljepa", scorer, records)
            out_path = resolve_path(cfg["output"]["eval_vljepa"], PROJECT_ROOT)
        else:
            raise ValueError(f"Unknown eval backend: {backend}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.info("Saved %s eval: %s", backend, out_path)
        output["results"][backend] = {
            "path": str(out_path),
            "summary": result["summary"],
        }

    summary_path = resolve_path(cfg["output"]["eval_summary"], PROJECT_ROOT)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Saved eval summary: %s", summary_path)
    return output


if __name__ == "__main__":
    summary = run_evaluation()
    print(json.dumps(summary, indent=2))
