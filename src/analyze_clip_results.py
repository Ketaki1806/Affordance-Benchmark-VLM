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

PART_WORDS = {
    "cap", "rim", "handle", "lid", "drawer", "blade", "lever", "knob",
    "spout", "body", "base", "shaft", "head", "spatula",
}


def load_clip_results(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_parts(caption: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", caption.lower()))
    return words & PART_WORDS


def tag_failure(positive: str, negative: str) -> str:
    """Heuristic failure tag for wrong CLIP choices."""
    pos_parts = extract_parts(positive)
    neg_parts = extract_parts(negative)
    if pos_parts and neg_parts and pos_parts != neg_parts:
        return "spatial"
    return "attribute"


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


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        f"{body}\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
    print(f"Saved plot: {path}")


def _plot_grouped_bars_svg(
    path: Path,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    y_label: str,
) -> None:
    width, height = 900, 420
    margin = {"l": 70, "r": 20, "t": 50, "b": 80}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    all_vals = [v for _, vals, _ in series for v in vals]
    y_min, y_max = 0.0, max(all_vals) * 1.1 if all_vals else 1.0

    def y_px(v: float) -> float:
        return margin["t"] + plot_h * (1 - (v - y_min) / (y_max - y_min))

    n = len(labels)
    group_w = plot_w / max(n, 1)
    bar_w = group_w / (len(series) + 1)
    parts = [
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16" font-family="sans-serif">{_svg_escape(title)}</text>',
        f'<text x="20" y="{margin["t"] + plot_h/2}" transform="rotate(-90 20 {margin["t"] + plot_h/2})" text-anchor="middle" font-size="12" font-family="sans-serif">{_svg_escape(y_label)}</text>',
        f'<line x1="{margin["l"]}" y1="{margin["t"] + plot_h}" x2="{margin["l"] + plot_w}" y2="{margin["t"] + plot_h}" stroke="#333"/>',
    ]
    for i, label in enumerate(labels):
        cx = margin["l"] + group_w * i + group_w / 2
        parts.append(
            f'<text x="{cx}" y="{height - 20}" text-anchor="middle" font-size="11" font-family="sans-serif">{_svg_escape(label)}</text>'
        )
    for s_idx, (name, values, color) in enumerate(series):
        for i, val in enumerate(values):
            x = margin["l"] + group_w * i + bar_w * (s_idx + 0.5)
            y_top = y_px(val)
            y_base = y_px(y_min)
            parts.append(
                f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w * 0.9:.1f}" '
                f'height="{y_base - y_top:.1f}" fill="{color}"/>'
            )
    legend_x = margin["l"]
    for s_idx, (name, _, color) in enumerate(series):
        ly = 42 + s_idx * 16
        parts.append(f'<rect x="{legend_x}" y="{ly}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{ly + 11}" font-size="11" font-family="sans-serif">{_svg_escape(name)}</text>'
        )
    _write_svg(path, width, height, "\n".join(parts))


def _plot_vertical_bars_svg(
    path: Path,
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    y_label: str,
    zero_line: bool = False,
) -> None:
    width, height = 900, 360
    margin = {"l": 70, "r": 20, "t": 50, "b": 70}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    y_min = min(0.0, min(values)) if zero_line else 0.0
    y_max = max(values) * 1.1 if values else 1.0
    if zero_line:
        y_min = min(y_min, min(values) * 1.2)
        y_max = max(y_max, max(values) * 1.1, 0.01)

    def y_px(v: float) -> float:
        return margin["t"] + plot_h * (1 - (v - y_min) / (y_max - y_min))

    bar_w = plot_w / max(len(labels), 1) * 0.7
    parts = [
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16" font-family="sans-serif">{_svg_escape(title)}</text>',
        f'<text x="20" y="{margin["t"] + plot_h/2}" transform="rotate(-90 20 {margin["t"] + plot_h/2})" text-anchor="middle" font-size="12" font-family="sans-serif">{_svg_escape(y_label)}</text>',
        f'<line x1="{margin["l"]}" y1="{y_px(0) if zero_line else margin["t"] + plot_h}" x2="{margin["l"] + plot_w}" y2="{y_px(0) if zero_line else margin["t"] + plot_h}" stroke="#333"/>',
    ]
    if zero_line:
        parts.append(
            f'<line x1="{margin["l"]}" y1="{y_px(0):.1f}" x2="{margin["l"] + plot_w}" y2="{y_px(0):.1f}" stroke="#999" stroke-dasharray="4,4"/>'
        )
    for i, (label, val, color) in enumerate(zip(labels, values, colors, strict=True)):
        cx = margin["l"] + (plot_w / len(labels)) * i + (plot_w / len(labels)) / 2
        y_top = y_px(val)
        y_base = y_px(0 if zero_line else y_min)
        parts.append(
            f'<rect x="{cx - bar_w/2:.1f}" y="{min(y_top, y_base):.1f}" width="{bar_w:.1f}" '
            f'height="{abs(y_base - y_top):.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 18}" text-anchor="middle" font-size="11" font-family="sans-serif">{_svg_escape(label)}</text>'
        )
    _write_svg(path, width, height, "\n".join(parts))


def _plot_horizontal_bars_svg(
    path: Path,
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    x_label: str,
) -> None:
    width, height = 900, 360
    margin = {"l": 90, "r": 20, "t": 50, "b": 50}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    x_max = max(values) * 1.15 if values else 1.0
    bar_h = plot_h / max(len(labels), 1) * 0.65
    parts = [
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16" font-family="sans-serif">{_svg_escape(title)}</text>',
        f'<text x="{width/2}" y="{height - 12}" text-anchor="middle" font-size="12" font-family="sans-serif">{_svg_escape(x_label)}</text>',
    ]
    for i, (label, val, color) in enumerate(zip(labels, values, colors, strict=True)):
        y = margin["t"] + (plot_h / len(labels)) * i + (plot_h / len(labels) - bar_h) / 2
        w = plot_w * (val / x_max)
        parts.append(
            f'<text x="{margin["l"] - 8}" y="{y + bar_h/2 + 4}" text-anchor="end" font-size="11" font-family="sans-serif">{_svg_escape(label)}</text>'
        )
        parts.append(f'<rect x="{margin["l"]}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
    _write_svg(path, width, height, "\n".join(parts))


def _plot_outcome_svg(path: Path, correct: int, wrong: int) -> None:
    total = correct + wrong
    acc = correct / total if total else 0
    width, height = 420, 200
    bar_w = 280
    cx = 70
    parts = [
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16" font-family="sans-serif">CLIP binary accuracy ({acc:.0%})</text>',
        f'<rect x="{cx}" y="70" width="{bar_w * acc:.1f}" height="36" fill="#2ca02c"/>',
        f'<rect x="{cx + bar_w * acc:.1f}" y="70" width="{bar_w * (1 - acc):.1f}" height="36" fill="#d62728"/>',
        f'<text x="{cx}" y="130" font-size="13" font-family="sans-serif">Correct: {correct}</text>',
        f'<text x="{cx + 140}" y="130" font-size="13" font-family="sans-serif">Wrong: {wrong}</text>',
    ]
    _write_svg(path, width, height, "\n".join(parts))


def plot_results_svg(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["image_id"].replace("sample_", "") for r in rows]
    colors = ["#2ca02c" if r["correct"] else "#d62728" for r in rows]

    _plot_grouped_bars_svg(
        out_dir / "clip_similarity_by_pair.svg",
        labels,
        [
            ("Positive", [r["pos_score"] for r in rows], "#2ca02c"),
            ("Negative", [r["neg_score"] for r in rows], "#d62728"),
        ],
        "CLIP image-text similarity by affordance pair",
        "Cosine similarity",
    )
    _plot_vertical_bars_svg(
        out_dir / "clip_signed_margin.svg",
        labels,
        [r["margin"] for r in rows],
        colors,
        "Signed margin (sim_pos - sim_neg)",
        "Margin",
        zero_line=True,
    )
    _plot_horizontal_bars_svg(
        out_dir / "clip_confidence_gap.svg",
        labels,
        [r["confidence_gap"] for r in rows],
        colors,
        "Confidence gap",
        "|sim_pos - sim_neg|",
    )
    correct = sum(1 for r in rows if r["correct"])
    _plot_outcome_svg(out_dir / "clip_outcome.svg", correct, len(rows) - correct)


def plot_results_matplotlib(rows: list[dict[str, Any]], out_dir: Path) -> None:
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


def plot_results(rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        plot_results_matplotlib(rows, out_dir)
    except ImportError:
        print("matplotlib not available; writing SVG plots instead.")
        plot_results_svg(rows, out_dir)


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
