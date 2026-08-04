"""Plot occlusion vision_share and (optional) embedding modality gap.

Two distinct metrics — do not conflate:

  occlusion vision_share  = decision sensitivity (leave-one-out + grid blackout)
  embedding modality_gap  = alignment geometry (||mean z_img - mean z_txt||)

Reads:
  artifacts/attribution_n100/summary.json
  artifacts/attribution_n100/embedding_modality_gap.json (optional)

Writes:
  artifacts/attribution_n100/modality_sensitivity.csv
  artifacts/figures/modality_sensitivity_n100.svg   # occlusion vision share
  artifacts/figures/embedding_modality_gap_n100.svg  # gap + matched cos (if gap JSON present)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts" / "attribution_n100" / "summary.json"
DEFAULT_GAP = ROOT / "artifacts" / "attribution_n100" / "embedding_modality_gap.json"
DEFAULT_CSV = ROOT / "artifacts" / "attribution_n100" / "modality_sensitivity.csv"
DEFAULT_SVG = ROOT / "artifacts" / "figures" / "modality_sensitivity_n100.svg"
DEFAULT_GAP_SVG = ROOT / "artifacts" / "figures" / "embedding_modality_gap_n100.svg"

BACKEND_LABELS = {
    "clip": "CLIP",
    "siglip": "SigLIP",
    "open_vljepa": "Open-VLJEPA",
}


def _mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def aggregate(summary: dict) -> tuple[list[dict], list[dict]]:
    """Return (per-pair rows, per-backend aggregate rows)."""
    rows: list[dict] = []
    aggs: list[dict] = []
    backends = summary.get("backends", {})
    for backend, block in backends.items():
        if "error" in block or "pairs" not in block:
            continue
        shares_all: list[float] = []
        shares_ok: list[float] = []
        shares_wrong: list[float] = []
        max_text: list[float] = []
        max_grid: list[float] = []
        for p in block["pairs"]:
            mod = p.get("modality") or {}
            vs = float(mod.get("vision_share", 0.0))
            mt = float(mod.get("max_abs_text", 0.0))
            mg = float(mod.get("max_abs_grid", 0.0))
            correct = bool(p.get("baseline", {}).get("correct", False))
            rows.append(
                {
                    "backend": backend,
                    "image_id": p.get("image_id"),
                    "correct": correct,
                    "vision_share": vs,
                    "max_abs_text": mt,
                    "max_abs_grid": mg,
                    "delta": p.get("baseline", {}).get("delta"),
                }
            )
            shares_all.append(vs)
            max_text.append(mt)
            max_grid.append(mg)
            (shares_ok if correct else shares_wrong).append(vs)
        aggs.append(
            {
                "backend": backend,
                "label": BACKEND_LABELS.get(backend, backend),
                "n": len(shares_all),
                "mean_vision_share": _mean(shares_all),
                "mean_vision_share_correct": _mean(shares_ok),
                "mean_vision_share_wrong": _mean(shares_wrong),
                "n_correct": len(shares_ok),
                "n_wrong": len(shares_wrong),
                "mean_max_abs_text": _mean(max_text),
                "mean_max_abs_grid": _mean(max_grid),
            }
        )
    return rows, aggs


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "image_id",
        "correct",
        "vision_share",
        "max_abs_text",
        "max_abs_grid",
        "delta",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_svg(aggs: list[dict], path: Path) -> None:
    """Grouped bar chart: mean vision_share overall / correct / wrong per backend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not aggs:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80">'
            '<text x="12" y="40">No backend data</text></svg>\n',
            encoding="utf-8",
        )
        return

    W, H = 720, 340
    margin_l, margin_r, margin_t, margin_b = 56, 24, 48, 64
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    n = len(aggs)
    group_w = plot_w / n
    bar_w = group_w / 4.2
    ymax = max(
        0.35,
        max(
            a["mean_vision_share"]
            for a in aggs
        ),
        max(a["mean_vision_share_correct"] for a in aggs),
        max(a["mean_vision_share_wrong"] for a in aggs),
    )
    ymax = min(1.0, ymax * 1.15)

    colors = {
        "all": "#2c5f8a",
        "correct": "#3d8b6e",
        "wrong": "#c45c26",
    }

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#faf9f7"/>',
        '<text x="24" y="28" font-family="Georgia, serif" font-size="16" fill="#1a1a1a">'
        "Occlusion vision share (decision sensitivity, N=100)</text>",
        f'<text x="24" y="46" font-family="system-ui,sans-serif" font-size="11" fill="#555">'
        f"Not embedding modality gap. vision_share = max|grid|/(max|grid|+max|text|); "
        f"ymax={ymax:.2f}</text>",
    ]

    # y-axis ticks
    for t in (0.0, 0.25 * ymax, 0.5 * ymax, 0.75 * ymax, ymax):
        y = margin_t + plot_h * (1 - t / ymax)
        parts.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{W - margin_r}" y2="{y:.1f}" '
            f'stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="system-ui,sans-serif" font-size="11" fill="#444">'
            f"{t:.2f}</text>"
        )

    for i, a in enumerate(aggs):
        gx = margin_l + i * group_w + group_w * 0.12
        series = [
            ("all", a["mean_vision_share"]),
            ("correct", a["mean_vision_share_correct"]),
            ("wrong", a["mean_vision_share_wrong"]),
        ]
        for j, (key, val) in enumerate(series):
            h = plot_h * (val / ymax) if ymax > 0 else 0
            x = gx + j * (bar_w + 4)
            y = margin_t + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{colors[key]}"/>'
            )
            if val > 0:
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                    f'font-family="system-ui,sans-serif" font-size="10" fill="#333">'
                    f"{val:.2f}</text>"
                )
        cx = margin_l + (i + 0.5) * group_w
        label = escape(a["label"])
        parts.append(
            f'<text x="{cx:.1f}" y="{H - 36}" text-anchor="middle" '
            f'font-family="system-ui,sans-serif" font-size="13" fill="#1a1a1a">'
            f"{label}</text>"
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{H - 20}" text-anchor="middle" '
            f'font-family="system-ui,sans-serif" font-size="10" fill="#666">'
            f'n={a["n"]} (ok {a["n_correct"]} / wrong {a["n_wrong"]})</text>'
        )

    # legend
    lx = W - margin_r - 200
    ly = margin_t - 4
    for j, (key, lab) in enumerate(
        [("all", "all"), ("correct", "correct"), ("wrong", "wrong")]
    ):
        x = lx + j * 68
        parts.append(
            f'<rect x="{x}" y="{ly - 10}" width="12" height="12" fill="{colors[key]}"/>'
        )
        parts.append(
            f'<text x="{x + 16}" y="{ly}" font-family="system-ui,sans-serif" '
            f'font-size="11" fill="#333">{escape(lab)}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_gap_svg(gap_summary: dict, path: Path) -> None:
    """Bar chart: modality_gap / modality_gap_all + matched cosines per backend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for backend, block in (gap_summary.get("backends") or {}).items():
        if "error" in block or "modality_gap" not in block:
            continue
        rows.append(
            {
                "backend": backend,
                "label": BACKEND_LABELS.get(backend, backend),
                "modality_gap": float(block["modality_gap"]),
                "modality_gap_all": float(block.get("modality_gap_all") or 0.0),
                "mean_matched_cos": float(block.get("mean_matched_cos") or 0.0),
                "mean_matched_cos_neg": float(block.get("mean_matched_cos_neg") or 0.0),
            }
        )
    if not rows:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80">'
            '<text x="12" y="40">No embedding modality gap data</text></svg>\n',
            encoding="utf-8",
        )
        return

    W, H = 760, 360
    margin_l, margin_r, margin_t, margin_b = 56, 24, 56, 72
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    n = len(rows)
    group_w = plot_w / n
    bar_w = group_w / 5.5
    ymax = max(
        0.2,
        max(r["modality_gap"] for r in rows),
        max(r["modality_gap_all"] for r in rows),
        max(r["mean_matched_cos"] for r in rows),
        max(r["mean_matched_cos_neg"] for r in rows),
    )
    ymax = ymax * 1.15

    colors = {
        "gap": "#5b4a8a",
        "gap_all": "#8a6bb5",
        "cos": "#2c7a6b",
        "cos_neg": "#c47a26",
    }
    series_keys = [
        ("gap", "modality_gap", "gap (pos)"),
        ("gap_all", "modality_gap_all", "gap (all)"),
        ("cos", "mean_matched_cos", "cos(img,pos)"),
        ("cos_neg", "mean_matched_cos_neg", "cos(img,neg)"),
    ]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#faf9f7"/>',
        '<text x="24" y="26" font-family="Georgia, serif" font-size="16" fill="#1a1a1a">'
        "Embedding modality gap (alignment geometry, N=100)</text>",
        '<text x="24" y="44" font-family="system-ui,sans-serif" font-size="11" fill="#555">'
        "Not occlusion vision_share. gap = ||mean z_img − mean z_txt||; "
        "matched cos = mean cos(z_img, z_caption)</text>",
    ]

    for t in (0.0, 0.25 * ymax, 0.5 * ymax, 0.75 * ymax, ymax):
        y = margin_t + plot_h * (1 - t / ymax)
        parts.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{W - margin_r}" y2="{y:.1f}" '
            f'stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="system-ui,sans-serif" font-size="11" fill="#444">'
            f"{t:.2f}</text>"
        )

    for i, row in enumerate(rows):
        gx = margin_l + i * group_w + group_w * 0.08
        for j, (ckey, fkey, _) in enumerate(series_keys):
            val = row[fkey]
            h = plot_h * (val / ymax) if ymax > 0 else 0
            x = gx + j * (bar_w + 3)
            y = margin_t + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{colors[ckey]}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                f'font-family="system-ui,sans-serif" font-size="9" fill="#333">'
                f"{val:.3f}</text>"
            )
        cx = margin_l + (i + 0.5) * group_w
        parts.append(
            f'<text x="{cx:.1f}" y="{H - 40}" text-anchor="middle" '
            f'font-family="system-ui,sans-serif" font-size="13" fill="#1a1a1a">'
            f'{escape(row["label"])}</text>'
        )

    lx = 24
    ly = H - 18
    for j, (ckey, _, lab) in enumerate(series_keys):
        x = lx + j * 175
        parts.append(
            f'<rect x="{x}" y="{ly - 10}" width="12" height="12" fill="{colors[ckey]}"/>'
        )
        parts.append(
            f'<text x="{x + 16}" y="{ly}" font-family="system-ui,sans-serif" '
            f'font-size="11" fill="#333">{escape(lab)}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_aggregate_json(aggs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_family": "occlusion_vision_share",
                "note": "Decision sensitivity; distinct from embedding_modality_gap.",
                "backends": aggs,
            },
            f,
            indent=2,
        )
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--gap-json", type=Path, default=DEFAULT_GAP)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--gap-svg", type=Path, default=DEFAULT_GAP_SVG)
    parser.add_argument(
        "--agg-json",
        type=Path,
        default=ROOT / "artifacts" / "attribution_n100" / "modality_sensitivity.json",
    )
    args = parser.parse_args(argv)

    wrote: list[str] = []
    if args.summary.is_file():
        with open(args.summary, encoding="utf-8") as f:
            summary = json.load(f)
        rows, aggs = aggregate(summary)
        write_csv(rows, args.csv)
        write_svg(aggs, args.svg)
        write_aggregate_json(aggs, args.agg_json)
        print(json.dumps({"occlusion_vision_share": {"n_rows": len(rows), "backends": aggs}}, indent=2))
        wrote.extend([str(args.csv), str(args.svg), str(args.agg_json)])
    else:
        print(f"Skip occlusion vision share plot (missing {args.summary})")

    if args.gap_json.is_file():
        with open(args.gap_json, encoding="utf-8") as f:
            gap = json.load(f)
        write_gap_svg(gap, args.gap_svg)
        print(json.dumps({"embedding_modality_gap": gap.get("backends")}, indent=2))
        wrote.append(str(args.gap_svg))
    else:
        print(f"Skip embedding modality gap plot (missing {args.gap_json})")

    if not wrote:
        raise SystemExit("Nothing to plot: need summary.json and/or embedding_modality_gap.json")
    for p in wrote:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
