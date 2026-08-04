"""Pure helpers + pair-level runner/CLI for occlusion-based post-hoc attribution."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from config_loader import PROJECT_ROOT, load_config, resolve_path

DEFAULT_IMAGE_IDS = [
    "lvis_486018",
    "lvis_75183",
    "lvis_258649",
    "lvis_446014",
    "lvis_479944",
    "lvis_512070",
    "lvis_52835",
    "lvis_88609",
]

PART_LEXICON = frozenset(
    {
        "lid",
        "handle",
        "cap",
        "door",
        "blade",
        "button",
        "frame",
        "side",
        "rim",
        "tab",
        "touchpad",
        "visor",
        "armrest",
        "leg",
        "screen",
        "cuff",
    }
)

VERB_HINTS = frozenset(
    {
        "open",
        "close",
        "push",
        "pull",
        "twist",
        "turn",
        "lift",
        "press",
        "hold",
        "grab",
        "slide",
        "flip",
        "rotate",
        "squeeze",
        "tap",
        "touch",
        "move",
        "raise",
        "lower",
        "drop",
        "place",
        "set",
        "use",
        "wear",
        "fill",
        "pour",
        "cut",
        "slice",
        "wipe",
        "clean",
        "wash",
        "dry",
        "heat",
        "cool",
        "store",
        "serve",
        "blend",
        "mix",
        "stir",
        "shake",
        "hang",
        "pick",
        "carry",
        "grip",
        "insert",
        "remove",
        "secure",
        "release",
        "attach",
        "detach",
    }
)


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def leave_one_out(text: str, index: int) -> str:
    tokens = tokenize_words(text)
    remaining = [tok for i, tok in enumerate(tokens) if i != index]
    return " ".join(remaining)


def role_labels(tokens: list[str]) -> list[str | None]:
    labels: list[str | None] = [None] * len(tokens)

    if tokens and tokens[0].lower() in VERB_HINTS:
        labels[0] = "verb"

    for i, tok in enumerate(tokens):
        if tok.lower() in PART_LEXICON:
            labels[i] = "part"

    after_to = False
    for i, tok in enumerate(tokens):
        if tok.lower() == "to":
            after_to = True
            continue
        if after_to and labels[i] is None:
            labels[i] = "purpose"

    # POS-free fallback (spec §3): if the first token still has no role, treat it as the verb.
    if tokens and labels[0] is None:
        labels[0] = "verb"

    return labels


def ablate_role(text: str, role: str) -> str:
    tokens = tokenize_words(text)
    labels = role_labels(tokens)
    remaining = [tok for tok, label in zip(tokens, labels) if label != role]
    return " ".join(remaining)


def blackout_grid_cell(
    image: Image.Image, row: int, col: int, grid: int = 3
) -> Image.Image:
    out = image.copy()
    width, height = out.size
    x0 = col * width // grid
    x1 = (col + 1) * width // grid
    y0 = row * height // grid
    y1 = (row + 1) * height // grid
    if x1 > x0 and y1 > y0:
        draw = ImageDraw.Draw(out)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(0, 0, 0))
    return out


def delta(pos: float, neg: float) -> float:
    return pos - neg


def _index_pairs_by_id(eval_json: Path) -> dict[str, dict]:
    with open(eval_json, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", []) if isinstance(data, dict) else data
    return {p.get("image_id"): p for p in pairs}


def load_all_pairs(eval_json: Path) -> list[dict]:
    """Load every pair from an eval JSON (N=100 scale-up). Order preserved."""
    with open(eval_json, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", []) if isinstance(data, dict) else data
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"No pairs found in {eval_json}")
    return list(pairs)


def load_pairs_by_id(
    eval_json: Path,
    image_ids: list[str],
    extra_json: list[Path] | None = None,
    skip_missing: bool = False,
) -> list[dict]:
    """Load pairs by image_id from a primary eval JSON, with optional fallback sources.

    Any `image_id` absent from `eval_json` is looked up in each of `extra_json`, in order
    (first match wins). If still missing: raise `ValueError`, unless `skip_missing` is True,
    in which case a warning is printed and that id is dropped from the result.
    """
    by_id = _index_pairs_by_id(eval_json)
    extra_by_id = [_index_pairs_by_id(path) for path in (extra_json or [])]

    result: list[dict] = []
    missing: list[str] = []
    for image_id in image_ids:
        pair = by_id.get(image_id)
        if pair is None:
            for extra in extra_by_id:
                if image_id in extra:
                    pair = extra[image_id]
                    break
        if pair is None:
            missing.append(image_id)
            continue
        result.append(pair)

    if missing:
        if skip_missing:
            print(
                f"Warning: image_id(s) not found in {eval_json} or extra sources, "
                f"skipping: {missing}"
            )
        else:
            raise ValueError(
                f"image_id(s) not found in {eval_json} or extra sources: {missing}"
            )

    return result


def _text_leave_one_out(
    scorer: Any,
    image_path: str,
    text: str,
    other_score: float,
    text_is_positive: bool,
    base_score: float,
    base_delta: float,
) -> list[dict[str, Any]]:
    tokens = tokenize_words(text)
    labels = role_labels(tokens)
    records: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        loo_text = leave_one_out(text, i)
        s_new = scorer.score(image_path, loo_text)
        new_delta = delta(s_new, other_score) if text_is_positive else delta(other_score, s_new)
        records.append(
            {
                "token": token,
                "role": labels[i],
                "ds": s_new - base_score,
                "d_delta": new_delta - base_delta,
            }
        )
    return records


def _role_ablations(
    scorer: Any,
    image_path: str,
    text: str,
    other_score: float,
    text_is_positive: bool,
    base_score: float,
    base_delta: float,
) -> dict[str, dict[str, Any]]:
    tokens = tokenize_words(text)
    labels = role_labels(tokens)
    result: dict[str, dict[str, Any]] = {}
    for role in ("verb", "part", "purpose"):
        if role not in labels:
            continue
        ablated = ablate_role(text, role)
        s_new = scorer.score(image_path, ablated)
        new_delta = delta(s_new, other_score) if text_is_positive else delta(other_score, s_new)
        result[role] = {
            "text": ablated,
            "ds": s_new - base_score,
            "d_delta": new_delta - base_delta,
        }
    return result


def _image_occlusion_delta_drop(
    scorer: Any,
    image: Image.Image,
    positive: str,
    negative: str,
    grid: int,
    base_delta: float,
) -> list[list[float]]:
    delta_drop = [[0.0] * grid for _ in range(grid)]
    for row, col in itertools.product(range(grid), repeat=2):
        occluded = blackout_grid_cell(image, row, col, grid)
        fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            occluded.convert("RGB").save(tmp_path, format="JPEG")
            tmp_str = str(tmp_path)
            s_pos = scorer.score(tmp_str, positive)
            s_neg = scorer.score(tmp_str, negative)
            delta_drop[row][col] = base_delta - delta(s_pos, s_neg)
        finally:
            tmp_path.unlink(missing_ok=True)
    return delta_drop


def save_grid_overlay(base: Image.Image, delta_drop: list[list[float]], path: Path) -> None:
    """Alpha-blend a red heatmap (proportional to positive delta_drop) over base, with grid lines."""
    base_rgba = base.convert("RGBA")
    width, height = base_rgba.size
    grid = len(delta_drop)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    flat = [value for row in delta_drop for value in row]
    max_abs = max((abs(value) for value in flat), default=0.0)
    if max_abs <= 0:
        max_abs = 1.0

    for row in range(grid):
        for col in range(grid):
            drop = delta_drop[row][col]
            intensity = max(0.0, drop) / max_abs
            alpha = int(180 * intensity)
            x0 = col * width // grid
            x1 = (col + 1) * width // grid
            y0 = row * height // grid
            y1 = (row + 1) * height // grid
            if x1 > x0 and y1 > y0:
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(255, 0, 0, alpha))

    for i in range(1, grid):
        x = i * width // grid
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 160), width=1)
        y = i * height // grid
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 160), width=1)

    composited = Image.alpha_composite(base_rgba, overlay).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    composited.save(path, format="PNG")


def modality_sensitivity(result: dict[str, Any]) -> dict[str, float]:
    """Peak vision vs text sensitivity for one attributed pair.

    vision_share = max|grid Δ-drop| / (max|grid| + max|text dΔ|).
    """
    text_effects: list[float] = []
    text_occ = result.get("text_occlusion", {})
    for side in ("positive", "negative"):
        for rec in text_occ.get(side, []):
            text_effects.append(float(rec["d_delta"]))
    grid = result.get("image_occlusion", {}).get("delta_drop") or []
    flat = [float(x) for row in grid for x in row]
    max_abs_text = max((abs(x) for x in text_effects), default=0.0)
    max_abs_grid = max((abs(x) for x in flat), default=0.0)
    denom = max_abs_grid + max_abs_text
    vision_share = (max_abs_grid / denom) if denom > 0 else 0.0
    return {
        "max_abs_text": max_abs_text,
        "max_abs_grid": max_abs_grid,
        "vision_share": vision_share,
    }


def attribute_pair(
    scorer: Any,
    backend: str,
    pair: dict,
    grid: int,
    out_dir: Path,
    *,
    save_overlays: bool = True,
) -> dict[str, Any]:
    """Run text leave-one-out, role ablations, and image blackout attribution for one pair."""
    image_path = pair["image_path"]
    positive = pair["positive"]
    negative = pair["negative"]
    image_id = pair["image_id"]

    s_pos = scorer.score(image_path, positive)
    s_neg = scorer.score(image_path, negative)
    d0 = delta(s_pos, s_neg)

    pos_records = _text_leave_one_out(scorer, image_path, positive, s_neg, True, s_pos, d0)
    neg_records = _text_leave_one_out(scorer, image_path, negative, s_pos, False, s_neg, d0)

    role_ablations = {
        "positive": _role_ablations(scorer, image_path, positive, s_neg, True, s_pos, d0),
        "negative": _role_ablations(scorer, image_path, negative, s_pos, False, s_neg, d0),
    }

    base_image = Image.open(image_path).convert("RGB")
    delta_drop = _image_occlusion_delta_drop(scorer, base_image, positive, negative, grid, d0)

    model_choice = "most_probable" if s_pos >= s_neg else "negative"
    baseline: dict[str, Any] = {
        "pos_score": s_pos,
        "neg_score": s_neg,
        "delta": d0,
        "model_choice": model_choice,
        # Always derived from *this* backend's scores (matches evaluate.py's rule);
        # never copied from the source pairs JSON, which may reflect a different backend.
        "correct": model_choice == "most_probable",
    }

    result: dict[str, Any] = {
        "image_id": image_id,
        "backend": backend,
        "object": pair.get("object"),
        "image_path": image_path,
        "positive": positive,
        "negative": negative,
        "baseline": baseline,
        "text_occlusion": {
            "positive": pos_records,
            "negative": neg_records,
            "role_ablations": role_ablations,
        },
        "image_occlusion": {"grid": grid, "delta_drop": delta_drop},
        "modality": modality_sensitivity(
            {
                "text_occlusion": {
                    "positive": pos_records,
                    "negative": neg_records,
                },
                "image_occlusion": {"delta_drop": delta_drop},
            }
        ),
    }

    backend_dir = Path(out_dir) / backend
    backend_dir.mkdir(parents=True, exist_ok=True)

    json_path = backend_dir / f"{image_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if save_overlays:
        grid_path = backend_dir / f"{image_id}_grid.png"
        save_grid_overlay(base_image, delta_drop, grid_path)

    return result


def _top_text_drops(result: dict[str, Any], top_n: int = 3) -> list[dict[str, Any]]:
    """Top-N leave-one-out records by abs(d_delta), across positive+negative sides."""
    candidates: list[dict[str, Any]] = []
    text_occlusion = result.get("text_occlusion", {})
    for side in ("positive", "negative"):
        for record in text_occlusion.get(side, []):
            candidates.append(
                {
                    "side": side,
                    "token": record["token"],
                    "d_delta": record["d_delta"],
                    "ds": record["ds"],
                }
            )
    candidates.sort(key=lambda r: abs(r["d_delta"]), reverse=True)
    return candidates[:top_n]


def _make_scorer(backend: str, config: dict, vljepa_checkpoint: str | None) -> Any:
    models = dict(config.get("models", {}))
    cfg = dict(config)

    if backend == "clip":
        from clip_scorer import CLIPScorer

        clip_models = dict(models)
        clip_models.pop("clip_checkpoint", None)
        cfg["models"] = clip_models
        return CLIPScorer(cfg)

    if backend == "siglip":
        from siglip_scorer import SigLIPScorer

        return SigLIPScorer(cfg)

    if backend == "open_vljepa":
        from open_vljepa_scorer import OpenVLJEPAScorer

        vljepa_models = dict(models)
        vljepa_cfg = dict(vljepa_models.get("open_vljepa", {}))
        if vljepa_checkpoint:
            vljepa_cfg["checkpoint"] = vljepa_checkpoint
        vljepa_models["open_vljepa"] = vljepa_cfg
        cfg["models"] = vljepa_models
        return OpenVLJEPAScorer(cfg)

    raise ValueError(f"Unknown backend: {backend}")


def run_attribution(
    pairs: list[dict],
    backends: list[str],
    out_dir: Path,
    grid: int = 3,
    config: dict | None = None,
    vljepa_checkpoint: str | None = None,
    *,
    save_overlays: bool = True,
) -> dict[str, Any]:
    cfg = config or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary: dict[str, Any] = {
        "num_pairs": len(pairs),
        "grid": grid,
        "save_overlays": save_overlays,
        "backends": {},
    }
    # Merge into an existing summary so SigLIP-only re-runs keep CLIP / VLJEPA.
    if summary_path.is_file():
        try:
            with open(summary_path, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and isinstance(prev.get("backends"), dict):
                summary["backends"] = dict(prev["backends"])
        except (json.JSONDecodeError, OSError):
            pass

    def _write_summary() -> None:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

    for backend in backends:
        try:
            scorer = _make_scorer(backend, cfg, vljepa_checkpoint)
            scorer.load()
            pair_summaries = []
            try:
                for pair in pairs:
                    result = attribute_pair(
                        scorer,
                        backend,
                        pair,
                        grid,
                        out_dir,
                        save_overlays=save_overlays,
                    )
                    pair_summaries.append(
                        {
                            "image_id": result["image_id"],
                            "baseline": result["baseline"],
                            "top_text_drops": _top_text_drops(result),
                            "image_occlusion": {
                                "delta_drop": result["image_occlusion"]["delta_drop"]
                            },
                            "modality": result["modality"],
                        }
                    )
            finally:
                scorer.unload()
            summary["backends"][backend] = {
                "num_pairs": len(pair_summaries),
                "pairs": pair_summaries,
            }
        except Exception as exc:  # noqa: BLE001 - one bad backend must not void the run
            summary["backends"][backend] = {"error": str(exc)}
        # Write after every backend (not just at the end) so a later failure/preemption
        # does not discard summaries for backends that already finished.
        _write_summary()

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-json", required=True, help="Primary eval JSON with top-level 'pairs'")
    parser.add_argument(
        "--extra-pairs-json",
        nargs="+",
        default=None,
        help="Additional eval JSON(s) consulted (in order) for image_ids missing from --pairs-json",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Warn and continue (instead of raising) if an image_id is missing from all sources",
    )
    parser.add_argument("--out-dir", default="artifacts/attribution")
    parser.add_argument(
        "--backends", nargs="+", default=["clip", "siglip", "open_vljepa"]
    )
    parser.add_argument(
        "--image-ids",
        nargs="+",
        default=None,
        help="Defaults to DEFAULT_IMAGE_IDS (ignored if --all-pairs)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Attribute every pair in --pairs-json (N=100 scale-up); ignores --image-ids",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip writing *_grid.png overlays (faster for full-N runs)",
    )
    def _positive_int(value: str) -> int:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"--grid must be positive, got {ivalue}")
        return ivalue

    parser.add_argument("--grid", type=_positive_int, default=3)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--vljepa-checkpoint",
        default="artifacts/checkpoints/open-vljepa/best.pt",
        help="Written into config models.open_vljepa.checkpoint (forces ZS checkpoint)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    pairs_json = resolve_path(args.pairs_json, PROJECT_ROOT)
    if args.all_pairs:
        pairs = load_all_pairs(pairs_json)
    else:
        image_ids = args.image_ids or DEFAULT_IMAGE_IDS
        extra_pairs_json = (
            [resolve_path(p, PROJECT_ROOT) for p in args.extra_pairs_json]
            if args.extra_pairs_json
            else None
        )
        pairs = load_pairs_by_id(
            pairs_json,
            image_ids,
            extra_json=extra_pairs_json,
            skip_missing=args.skip_missing,
        )
    out_dir = resolve_path(args.out_dir, PROJECT_ROOT)

    summary = run_attribution(
        pairs=pairs,
        backends=args.backends,
        out_dir=out_dir,
        grid=args.grid,
        config=config,
        vljepa_checkpoint=args.vljepa_checkpoint,
        save_overlays=not args.no_overlays,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
