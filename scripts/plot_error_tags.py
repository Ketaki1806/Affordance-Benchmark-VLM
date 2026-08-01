"""Write failure-tag mix figure (SVG, no matplotlib required).

Output: artifacts/figures/error_tags_n100.svg
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "figures"
EVALS = {
    "CLIP": ROOT / "humaneval" / "30jul" / "clip.json",
    "SigLIP": ROOT / "humaneval" / "1aug" / "siglip.json",
    "VLJEPA ZS": ROOT / "humaneval" / "31jul" / "open_vljepa.json",
}

LOC_WORDS = {
    "wall",
    "ceiling",
    "floor",
    "table",
    "rack",
    "sink",
    "cup",
    "drawer",
    "toilet",
    "ground",
    "leash",
    "belly",
    "face",
    "teeth",
    "screen",
    "logo",
}
PARTISH = {
    "handle",
    "lid",
    "cap",
    "button",
    "blade",
    "door",
    "leg",
    "armrest",
    "touchpad",
    "visor",
    "frame",
    "rim",
    "cuff",
    "screen",
    "strings",
    "tab",
}
COLORS = {
    "attribute": "#3B6EA5",
    "spatial": "#C48A2A",
    "fluency": "#B04A4A",
}
TAGS = ("attribute", "spatial", "fluency")


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def tag(pair: dict) -> str:
    pos, neg = pair["positive"], pair["negative"]
    tp, tn = tokens(pos), tokens(neg)
    obj = pair["object"].lower()

    if "dry the towel with the towel" in neg.lower():
        return "fluency"
    if "press the button to activate the lamp" in pos.lower() and "pillow" in obj:
        return "fluency"
    if "hold the phone" in pos.lower() and "ladder" in obj:
        return "fluency"
    if "brush the teeth" in pos.lower() and "dog" in obj:
        return "fluency"
    if "squat with legs" in pos.lower():
        return "fluency"
    if "drill the handle" in pos.lower():
        return "fluency"
    if "spoon spoon" in neg.lower():
        return "fluency"
    if "pour from the lid if you need to" in pos.lower():
        return "fluency"

    loc_p, loc_n = tp & LOC_WORDS, tn & LOC_WORDS
    if loc_p != loc_n and (loc_p or loc_n):
        return "spatial"
    part_p, part_n = tp & PARTISH, tn & PARTISH
    if part_p != part_n and part_p and part_n:
        return "spatial"
    return "attribute"


def counts_for(path: Path) -> Counter:
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    c: Counter = Counter()
    for p in pairs:
        if not p["correct"]:
            c[tag(p)] += 1
    return c


def main() -> None:
    models = list(EVALS)
    data = {m: counts_for(EVALS[m]) for m in models}
    wrong_n = {m: sum(data[m].values()) for m in models}

    # layout
    W, H = 920, 420
    left_plot = (70, 70, 360, 280)  # x,y,w,h
    right_plot = (500, 70, 360, 280)
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="28" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" '
        'font-size="16" font-weight="700" fill="#222">'
        "Error analysis — frozen N=100 (Qwen captions)</text>",
        '<text x="460" y="48" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" '
        'font-size="11" fill="#666">'
        "Wrong-pair tags: attribute / spatial / fluency</text>",
    ]

    # legend
    lx = 70
    for i, t in enumerate(TAGS):
        parts.append(
            f'<rect x="{lx + i * 120}" y="395" width="12" height="12" fill="{COLORS[t]}"/>'
        )
        parts.append(
            f'<text x="{lx + i * 120 + 18}" y="405" font-family="Segoe UI, Helvetica, Arial, sans-serif" '
            f'font-size="12" fill="#333">{escape(t.capitalize())}</text>'
        )

    # --- left: stacked % ---
    x0, y0, pw, ph = left_plot
    parts.append(
        f'<text x="{x0 + pw/2}" y="{y0 - 12}" text-anchor="middle" '
        'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="600" fill="#222">'
        "Failure-tag mix (normalized)</text>"
    )
    # y axis
    for pct in (0, 25, 50, 75, 100):
        yy = y0 + ph - ph * pct / 100
        parts.append(
            f'<line x1="{x0}" y1="{yy}" x2="{x0 + pw}" y2="{yy}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8}" y="{yy + 4}" text-anchor="end" '
            f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="10" fill="#666">{pct}%</text>'
        )
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + ph}" stroke="#ccc"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{y0 + ph}" x2="{x0 + pw}" y2="{y0 + ph}" stroke="#ccc"/>'
    )

    n = len(models)
    bar_w = pw / (n + 1)
    for i, m in enumerate(models):
        cx = x0 + (i + 0.75) * (pw / n)
        bottom = y0 + ph
        for t in TAGS:
            share = data[m][t] / wrong_n[m] if wrong_n[m] else 0
            h = ph * share
            top = bottom - h
            parts.append(
                f'<rect x="{cx - bar_w/2}" y="{top}" width="{bar_w}" height="{h}" fill="{COLORS[t]}"/>'
            )
            if share >= 0.08:
                parts.append(
                    f'<text x="{cx}" y="{(top + bottom)/2 + 4}" text-anchor="middle" '
                    f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" '
                    f'font-weight="700" fill="white">{share*100:.0f}%</text>'
                )
            bottom = top
        parts.append(
            f'<text x="{cx}" y="{y0 + ph + 18}" text-anchor="middle" '
            f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#222">'
            f"{escape(m)}</text>"
        )

    parts.append(
        f'<text x="{x0 - 45}" y="{y0 + ph/2}" text-anchor="middle" '
        'transform="rotate(-90 ' + f"{x0 - 45},{y0 + ph/2}" + ')" '
        'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#555">'
        "Share of wrong pairs (%)</text>"
    )

    # --- right: absolute grouped ---
    x0, y0, pw, ph = right_plot
    parts.append(
        f'<text x="{x0 + pw/2}" y="{y0 - 12}" text-anchor="middle" '
        'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="600" fill="#222">'
        "Absolute wrong counts by tag</text>"
    )
    ymax = max(max(data[m][t] for t in TAGS) for m in models) + 4
    for tick in range(0, int(ymax) + 1, 5):
        yy = y0 + ph - ph * tick / ymax
        parts.append(
            f'<line x1="{x0}" y1="{yy}" x2="{x0 + pw}" y2="{yy}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8}" y="{yy + 4}" text-anchor="end" '
            f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="10" fill="#666">{tick}</text>'
        )
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + ph}" stroke="#ccc"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{y0 + ph}" x2="{x0 + pw}" y2="{y0 + ph}" stroke="#ccc"/>'
    )

    group_w = pw / n
    sub_w = group_w / (len(TAGS) + 1.5)
    for i, m in enumerate(models):
        gx = x0 + i * group_w + group_w / 2
        for j, t in enumerate(TAGS):
            v = data[m][t]
            h = ph * v / ymax
            bx = gx + (j - 1) * sub_w - sub_w / 2
            by = y0 + ph - h
            parts.append(
                f'<rect x="{bx}" y="{by}" width="{sub_w * 0.9}" height="{h}" fill="{COLORS[t]}"/>'
            )
            parts.append(
                f'<text x="{bx + sub_w * 0.45}" y="{by - 4}" text-anchor="middle" '
                f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="10" fill="#333">{v}</text>'
            )
        parts.append(
            f'<text x="{gx}" y="{y0 + ph + 18}" text-anchor="middle" '
            f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#222">'
            f"{escape(m)}</text>"
        )
        parts.append(
            f'<text x="{gx}" y="{y0 + ph + 34}" text-anchor="middle" '
            f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="10" fill="#666">'
            f"(n_wrong={wrong_n[m]})</text>"
        )

    parts.append(
        f'<text x="{x0 - 45}" y="{y0 + ph/2}" text-anchor="middle" '
        'transform="rotate(-90 ' + f"{x0 - 45},{y0 + ph/2}" + ')" '
        'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#555">'
        "Wrong pairs (count)</text>"
    )

    parts.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "error_tags_n100.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    for m in models:
        c = data[m]
        n_w = wrong_n[m]
        print(
            f"{m}: wrong={n_w}  attr={c['attribute']} "
            f"spatial={c['spatial']} fluency={c['fluency']}"
        )


if __name__ == "__main__":
    main()
