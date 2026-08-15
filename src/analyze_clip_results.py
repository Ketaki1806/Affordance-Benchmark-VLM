"""
Analyze and visualize CLIP evaluation results from artifacts/eval/clip.json.

Usage (from repo root):
  python src/analyze_clip_results.py
  python src/analyze_clip_results.py --clip-json artifacts/eval/clip.json --out-dir artifacts/eval/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config_loader import PROJECT_ROOT, load_config, resolve_path

# Words that are likely to be part of a spatial relationship used in PACO captions
PART_WORDS = {
    "cap", "rim", "handle", "lid", "drawer", "blade", "lever", "knob",
    "spout", "body", "base", "shaft", "head", "spatula",
}

# Load the clip.json file and return a dictionary of the results
def load_clip_results(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Extract the parts of the caption that are likely to be part of a spatial relationship
def extract_parts(caption: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", caption.lower()))
    return words & PART_WORDS

# Heuristic failure tag for wrong CLIP choices
def tag_failure(positive: str, negative: str) -> str:
    """Heuristic failure tag for wrong CLIP choices."""
    pos_parts = extract_parts(positive)
    neg_parts = extract_parts(negative)
    if pos_parts and neg_parts and pos_parts != neg_parts:
        return "spatial"
    return "attribute"

# Enrich the pairs with the margin and failure tag
def enrich_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in pairs:
        pos = float(p["pos_score"])
        neg = float(p["neg_score"])
        margin = pos - neg
        row = {
            **p,
            "margin": margin,
            "failure_tag": None if p["correct"] else tag_failure(p["positive"], p["negative"]),
        }
        rows.append(row)
    return rows

# Print the summary of the analysis
def print_summary(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("=== CLIP evaluation summary ===")
    print(f"Pairs:              {summary.get('num_pairs', len(rows))}")
    print(f"Binary accuracy:    {summary.get('binary_accuracy', 0):.1%}")
    print(f"Mean conf. gap:     {summary.get('mean_confidence_gap', 0):.4f}")
    print(f"Correct / wrong:    {summary.get('num_correct')} / {summary.get('num_wrong')}")
    print()

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print("=== Failures ===")
        for r in wrong:
            print(f"  {r['image_id']} ({r.get('object', '')})")
            print(f"    tag:    {r['failure_tag']}")
            print(f"    margin: {r['margin']:.4f}")
            print(f"    pos:    {r['positive']}")
            print(f"    neg:    {r['negative']}")
        print()

    print("=== Per-image margins (pos - neg) ===")
    for r in sorted(rows, key=lambda x: x["margin"]):
        mark = "OK" if r["correct"] else "WRONG"
        print(
            f"  {r['image_id']:12} {mark:5}  margin={r['margin']:+.4f}  "
            f"gap={r['confidence_gap']:.4f}"
        )

# Save the analysis to a JSON file
def save_analysis_json(rows: list[dict[str, Any]], out_path: Path) -> None:
    failures = [r for r in rows if not r["correct"]]
    by_tag: dict[str, int] = {}
    for r in failures:
        tag = r["failure_tag"] or "unknown"
        by_tag[tag] = by_tag.get(tag, 0) + 1

    payload = {
        "num_pairs": len(rows),
        "num_correct": sum(1 for r in rows if r["correct"]),
        "num_wrong": len(failures),
        "binary_accuracy": sum(1 for r in rows if r["correct"]) / len(rows) if rows else 0,
        "mean_confidence_gap": sum(r["confidence_gap"] for r in rows) / len(rows) if rows else 0,
        "mean_margin": sum(r["margin"] for r in rows) / len(rows) if rows else 0,
        "failure_breakdown": by_tag,
        "pairs": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved analysis: {out_path}")

# Plot the results using matplotlib
def plot_results(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["image_id"].replace("sample_", "") for r in rows]
    pos_scores = [r["pos_score"] for r in rows]
    neg_scores = [r["neg_score"] for r in rows]
    margins = [r["margin"] for r in rows]
    gaps = [r["confidence_gap"] for r in rows]
    colors = ["#2ca02c" if r["correct"] else "#d62728" for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(rows))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pos_scores, width, label="Positive", color="#2ca02c")
    ax.bar([i + width / 2 for i in x], neg_scores, width, label="Negative", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("CLIP cosine similarity")
    ax.set_title("CLIP image-text similarity by affordance pair")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path1 = out_dir / "clip_similarity_by_pair.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path1}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, margins, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Margin (sim_pos - sim_neg)")
    ax.set_title("Signed margin per pair")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path2 = out_dir / "clip_signed_margin.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path2}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels, gaps, color=colors)
    ax.set_xlabel("Confidence gap")
    ax.set_title("Confidence gap (smaller = harder)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path3 = out_dir / "clip_confidence_gap.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path3}")

    correct = sum(1 for r in rows if r["correct"])
    wrong = len(rows) - correct
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [correct, wrong],
        labels=[f"Correct ({correct})", f"Wrong ({wrong})"],
        colors=["#2ca02c", "#d62728"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title("CLIP binary accuracy")
    path4 = out_dir / "clip_outcome_pie.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path4}")


# Parse the arguments
def parse_args() -> argparse.Namespace:
    cfg = load_config()
    default_clip = resolve_path(cfg["output"]["eval_clip"], PROJECT_ROOT)
    default_out = resolve_path("artifacts/eval/figures", PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Analyze CLIP eval JSON and plot results.")
    parser.add_argument("--clip-json", type=Path, default=default_clip)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--analysis-json", type=Path, default=None, help="Optional enriched JSON output")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_clip_results(args.clip_json)
    pairs = data.get("pairs", [])
    if not pairs:
        raise SystemExit(f"No pairs found in {args.clip_json}")

    rows = enrich_pairs(pairs)
    summary = data.get("summary", {})
    print_summary(rows, summary)

    analysis_path = args.analysis_json or (args.out_dir.parent / "clip_analysis.json")
    save_analysis_json(rows, analysis_path)

    if not args.no_plots:
        plot_results(rows, args.out_dir)


if __name__ == "__main__":
    main()
