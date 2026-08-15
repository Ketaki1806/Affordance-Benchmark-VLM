"""FT improvement broken down by failure tag (frozen wrong → FT correct).

Compares ZS/frozen vs FT train_500 on the same val N=100 pairs.
Tags reuse scripts/plot_error_tags.py heuristics.

Outputs:
  artifacts/figures/ft_improvement_by_tag.png
  artifacts/figures/ft_improvement_by_tag.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_error_tags import TAGS, tag  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "figures"
COMPARISONS = {
    "CLIP": (
        ROOT / "humaneval" / "30jul" / "clip.json",
        ROOT / "humaneval" / "1aug" / "clip_ft.json",
    ),
    "Open-VLJEPA": (
        ROOT / "humaneval" / "31jul" / "open_vljepa.json",
        ROOT / "humaneval" / "1aug" / "open_vljepa_ft.json",
    ),
}
TAG_COLORS = {
    "attribute": (59, 110, 165),
    "spatial": (196, 138, 42),
    "fluency": (176, 74, 74),
}


def _font(size: int):
    for name in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ):
        p = Path(name)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _load_pairs(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["image_id"]: p for p in data["pairs"]}


def analyze(before: Path, after: Path) -> dict:
    a = _load_pairs(before)
    b = _load_pairs(after)
    fixed: Counter = Counter()
    remain: Counter = Counter()
    new_err: Counter = Counter()
    for iid, pa in a.items():
        pb = b[iid]
        if (not pa["correct"]) and pb["correct"]:
            fixed[tag(pa)] += 1
        elif (not pa["correct"]) and (not pb["correct"]):
            remain[tag(pa)] += 1
        elif pa["correct"] and (not pb["correct"]):
            new_err[tag(pb)] += 1

    by_tag = {}
    for t in TAGS:
        base = fixed[t] + remain[t]
        by_tag[t] = {
            "frozen_wrong": base,
            "fixed": fixed[t],
            "remain_wrong": remain[t],
            "fix_rate": (fixed[t] / base) if base else 0.0,
            "new_errors": new_err[t],
        }
    return {
        "frozen_wrong": sum(fixed.values()) + sum(remain.values()),
        "ft_wrong": sum(1 for p in b.values() if not p["correct"]),
        "fixed_total": sum(fixed.values()),
        "remain_total": sum(remain.values()),
        "new_errors_total": sum(new_err.values()),
        "by_tag": by_tag,
    }


def plot(results: dict[str, dict], path: Path) -> None:
    W, H = 1100, 560
    img = Image.new("RGB", (W, H), (250, 249, 247))
    draw = ImageDraw.Draw(img)
    title_f, small_f, tick_f = _font(20), _font(13), _font(12)

    draw.text((28, 16), "Which failure types does ranking FT fix?", font=title_f, fill=(26, 26, 26))
    draw.text(
        (28, 44),
        "Among frozen wrongs on val N=100: fixed = wrong→correct after FT (train_500). Same tag heuristics as error analysis.",
        font=small_f,
        fill=(90, 90, 90),
    )

    models = list(results)
    # Left: fix-rate % by tag
    ml, mr, mt, mb = 70, 30, 100, 70
    left_w = 500
    plot_h = H - mt - mb
    plot_w = left_w - ml - 20
    ymax = 1.0

    draw.text((ml + plot_w / 2 - 80, mt - 22), "Fix rate by tag", font=small_f, fill=(30, 30, 30))
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = mt + plot_h * (1 - t)
        draw.line([(ml, y), (ml + plot_w, y)], fill=(220, 220, 220), width=1)
        draw.text((8, y - 8), f"{t:.0%}", font=tick_f, fill=(70, 70, 70))

    n_tags = len(TAGS)
    group_w = plot_w / n_tags
    bar_w = group_w / (len(models) + 1.4)
    model_colors = {"CLIP": (44, 95, 138), "Open-VLJEPA": (61, 139, 110)}

    for i, tname in enumerate(TAGS):
        for j, m in enumerate(models):
            rate = results[m]["by_tag"][tname]["fix_rate"]
            h = plot_h * rate
            x = ml + i * group_w + group_w * 0.18 + j * (bar_w + 6)
            y = mt + plot_h - h
            draw.rectangle([x, y, x + bar_w, mt + plot_h], fill=model_colors[m])
            draw.text((x, y - 16), f"{rate:.0%}", font=tick_f, fill=(40, 40, 40))
            base = results[m]["by_tag"][tname]["frozen_wrong"]
            fixed = results[m]["by_tag"][tname]["fixed"]
            draw.text((x - 2, mt + plot_h + 4), f"{fixed}/{base}", font=tick_f, fill=(80, 80, 80))
        cx = ml + (i + 0.5) * group_w
        bbox = draw.textbbox((0, 0), tname, font=small_f)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw / 2, H - 42), tname, font=small_f, fill=(30, 30, 30))

    # Right: stacked fixed vs remain counts
    rx0 = 560
    r_plot_w = W - rx0 - mr - 20
    draw.text((rx0 + r_plot_w / 2 - 110, mt - 22), "Frozen wrongs: fixed vs remain", font=small_f, fill=(30, 30, 30))
    ymax_c = max(
        max(results[m]["by_tag"][t]["frozen_wrong"] for t in TAGS) for m in models
    ) + 4
    for tick in range(0, int(ymax_c) + 1, 5):
        y = mt + plot_h * (1 - tick / ymax_c)
        draw.line([(rx0, y), (rx0 + r_plot_w, y)], fill=(220, 220, 220), width=1)
        draw.text((rx0 - 28, y - 8), str(tick), font=tick_f, fill=(70, 70, 70))

    group_w = r_plot_w / n_tags
    bar_w = group_w / (len(models) + 1.4)
    for i, tname in enumerate(TAGS):
        for j, m in enumerate(models):
            row = results[m]["by_tag"][tname]
            fixed_n = row["fixed"]
            remain_n = row["remain_wrong"]
            x = rx0 + i * group_w + group_w * 0.18 + j * (bar_w + 6)
            # remain (top of stack visually = above fixed)
            h_f = plot_h * (fixed_n / ymax_c)
            h_r = plot_h * (remain_n / ymax_c)
            y_base = mt + plot_h
            draw.rectangle([x, y_base - h_f, x + bar_w, y_base], fill=TAG_COLORS[tname])
            draw.rectangle(
                [x, y_base - h_f - h_r, x + bar_w, y_base - h_f],
                fill=(210, 210, 210),
            )
        cx = rx0 + (i + 0.5) * group_w
        bbox = draw.textbbox((0, 0), tname, font=small_f)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw / 2, H - 42), tname, font=small_f, fill=(30, 30, 30))

    # legends
    lx = 70
    for j, m in enumerate(models):
        x = lx + j * 160
        draw.rectangle([x, H - 22, x + 14, H - 8], fill=model_colors[m])
        draw.text((x + 20, H - 24), m, font=small_f, fill=(40, 40, 40))
    draw.rectangle([440, H - 22, 454, H - 8], fill=(59, 110, 165))
    draw.text((460, H - 24), "fixed (colored)", font=small_f, fill=(40, 40, 40))
    draw.rectangle([600, H - 22, 614, H - 8], fill=(210, 210, 210))
    draw.text((620, H - 24), "remain wrong", font=small_f, fill=(40, 40, 40))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    print(f"Wrote {path}")


def main() -> None:
    results = {}
    for name, (before, after) in COMPARISONS.items():
        results[name] = analyze(before, after)
        print(name, json.dumps(results[name], indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(results, OUT_DIR / "ft_improvement_by_tag.png")
    payload = {
        "note": "FT = train_500 ranking FT on val N=100; train_1000 pair dumps not required for this comparison",
        "models": results,
    }
    out_json = OUT_DIR / "ft_improvement_by_tag.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
