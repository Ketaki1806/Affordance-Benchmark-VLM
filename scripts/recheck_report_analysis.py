"""Recheck figures + analysis consistency. Run: py -3 scripts/recheck_report_analysis.py"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

# Same heuristics as plot_error_tags.py (keep in sync for audit)
LOC_WORDS = {
    "wall", "ceiling", "floor", "table", "rack", "sink", "cup", "drawer",
    "toilet", "ground", "leash", "belly", "face", "teeth", "screen", "logo",
}
PARTISH = {
    "handle", "lid", "door", "button", "lever", "knob", "cap", "spout",
    "blade", "wheel", "pedal", "trigger", "switch", "cord", "plug", "hose",
}
ATTR_WORDS = {
    "open", "close", "carry", "empty", "fill", "pour", "push", "pull",
    "twist", "press", "lift", "hang", "store", "blend", "serve", "wash",
    "cut", "peel", "power", "answer", "off", "on",
}


def tokenize(s: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z']+", s.lower())


def tag_pair(pos: str, neg: str) -> str:
    """Rough contrast tag; mirrors report intent (not identical to plot script)."""
    pt, nt = set(tokenize(pos)), set(tokenize(neg))
    pos_only, neg_only = pt - nt, nt - pt
    # fluency: very little overlap or odd length
    if len(pt & nt) < 2 and (len(pos_only) > 4 or len(neg_only) > 4):
        # weak heuristic — check later against official plot
        pass
    if (pos_only | neg_only) & LOC_WORDS:
        return "spatial"
    if (pos_only | neg_only) & ATTR_WORDS or (pos_only | neg_only) & PARTISH:
        # purpose/action swap more attribute
        if (pos_only | neg_only) & ATTR_WORDS:
            return "attribute"
        return "spatial" if (pos_only | neg_only) & LOC_WORDS else "attribute"
    return "attribute"


def load_eval(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pairs", data)


def main() -> None:
    issues: list[str] = []
    notes: list[str] = []

    # --- 1. Attribution N=100 ---
    summary_path = ROOT / "artifacts/attribution_n100/summary.json"
    gap_path = ROOT / "artifacts/attribution_n100/embedding_modality_gap.json"
    ms_path = ROOT / "artifacts/attribution_n100/modality_sensitivity.json"
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    g = json.loads(gap_path.read_text(encoding="utf-8"))
    ms = json.loads(ms_path.read_text(encoding="utf-8"))

    print("=== Occlusion vision share (from summary.json) ===")
    recomputed = {}
    for b, block in s["backends"].items():
        if "error" in block:
            issues.append(f"summary {b} still has error: {block['error'][:120]}")
            print(f"  {b}: ERROR")
            continue
        pairs = block["pairs"]
        if len(pairs) != 100:
            issues.append(f"{b}: expected 100 pairs, got {len(pairs)}")
        vs = [float(p["modality"]["vision_share"]) for p in pairs]
        ok = [float(p["modality"]["vision_share"]) for p in pairs if p["baseline"]["correct"]]
        wr = [float(p["modality"]["vision_share"]) for p in pairs if not p["baseline"]["correct"]]
        n_ok = len(ok)
        row = {
            "n": len(pairs),
            "acc": n_ok / len(pairs),
            "vs": mean(vs),
            "vs_ok": mean(ok) if ok else float("nan"),
            "vs_wr": mean(wr) if wr else float("nan"),
            "n_ok": n_ok,
            "n_wr": len(wr),
        }
        recomputed[b] = row
        print(
            f"  {b}: acc={n_ok}/100={row['acc']:.2f}  "
            f"vs={row['vs']:.3f} ok={row['vs_ok']:.3f} wrong={row['vs_wr']:.3f}"
        )

        # vision_share formula sanity on a few pairs
        for p in pairs[:3]:
            mt = p["modality"]["max_abs_text"]
            mg = p["modality"]["max_abs_grid"]
            vs_ = p["modality"]["vision_share"]
            denom = mt + mg
            expect = (mg / denom) if denom > 0 else 0.0
            if abs(vs_ - expect) > 1e-6:
                issues.append(f"{b} {p['image_id']}: vision_share mismatch {vs_} vs {expect}")

    print("\n=== modality_sensitivity.json vs summary ===")
    for row in ms["backends"]:
        b = row["backend"]
        r = recomputed.get(b)
        if not r:
            issues.append(f"modality_sensitivity has {b} missing from summary")
            continue
        for key, a, bval in [
            ("mean_vision_share", row["mean_vision_share"], r["vs"]),
            ("mean_vision_share_correct", row["mean_vision_share_correct"], r["vs_ok"]),
            ("mean_vision_share_wrong", row["mean_vision_share_wrong"], r["vs_wr"]),
        ]:
            if abs(a - bval) > 1e-9:
                issues.append(f"{b} {key}: plot agg {a} != summary {bval}")
        print(f"  {b}: plot OK match summary")

    # Report table rounding check
    report_vs = {"clip": (0.26, 0.28, 0.24), "siglip": (0.23, 0.26, 0.19), "open_vljepa": (0.24, 0.24, 0.24)}
    print("\n=== Report table rounding (vision share) ===")
    for b, (a, o, w) in report_vs.items():
        r = recomputed[b]
        ok_r = abs(round(r["vs"], 2) - a) < 0.011 and abs(round(r["vs_ok"], 2) - o) < 0.011
        print(f"  {b}: report {a}/{o}/{w}  actual {r['vs']:.3f}/{r['vs_ok']:.3f}/{r['vs_wr']:.3f}  "
              f"{'OK' if ok_r else 'CHECK'}")
        if abs(r["vs"] - a) > 0.015:
            issues.append(f"report vision_share {b} rounded oddly: {a} vs {r['vs']:.3f}")

    print("\n=== Embedding modality gap ===")
    report_gap = {
        "clip": (0.97, 0.97, 0.18, 0.17),
        "siglip": (1.04, 1.04, 0.07, 0.06),
        "open_vljepa": (0.27, 0.33, 0.57, 0.54),
    }
    for b, v in g["backends"].items():
        if "error" in v:
            issues.append(f"gap {b} error: {v['error']}")
            continue
        print(
            f"  {b}: gap={v['modality_gap']:.4f} all={v['modality_gap_all']:.4f} "
            f"cos+={v['mean_matched_cos']:.4f} cos-={v['mean_matched_cos_neg']:.4f}"
        )
        if b in report_gap:
            rg = report_gap[b]
            actual = (
                round(v["modality_gap"], 2),
                round(v["modality_gap_all"], 2),
                round(v["mean_matched_cos"], 2),
                round(v["mean_matched_cos_neg"], 2),
            )
            if actual != rg:
                # allow 0.01 float
                if any(abs(a - e) > 0.01 for a, e in zip(actual, rg)):
                    issues.append(f"report gap row {b}: report {rg} vs rounded {actual}")
                else:
                    notes.append(f"gap {b}: minor rounding OK {rg} ~ {actual}")

    # Cross-check: VLJEPA better alignment but worse/similar acc
    if recomputed["open_vljepa"]["acc"] < recomputed["clip"]["acc"] and g["backends"]["open_vljepa"]["modality_gap"] < g["backends"]["clip"]["modality_gap"]:
        notes.append("CLAIM OK: VLJEPA smaller modality gap but lower acc than CLIP")
    else:
        issues.append("Unexpected: VLJEPA gap/acc relationship vs claim")

    # Vision share all < 0.35 → text dominates claim
    if all(recomputed[b]["vs"] < 0.35 for b in recomputed):
        notes.append("CLAIM OK: vision share ~0.23-0.26, text-dominated decisions")
    else:
        issues.append("Vision share higher than reported narrative")

    # --- 2. Eval JSON accuracies vs attribution baselines ---
    print("\n=== Eval JSON accuracies (humaneval) vs attribution baselines ===")
    evals = {
        "clip": ROOT / "humaneval/30jul/clip.json",
        "siglip": ROOT / "humaneval/1aug/siglip.json",
        "open_vljepa": ROOT / "humaneval/31jul/open_vljepa.json",
    }
    for b, path in evals.items():
        if not path.is_file():
            notes.append(f"missing local eval {path} (cluster-only?)")
            continue
        pairs = load_eval(path)
        n = len(pairs)
        # prefer explicit correct if present
        if pairs and "correct" in pairs[0]:
            n_ok = sum(1 for p in pairs if p.get("correct"))
        else:
            n_ok = sum(
                1
                for p in pairs
                if float(p.get("pos_score", 0)) >= float(p.get("neg_score", 0))
            )
        print(f"  {path.name}: {n_ok}/{n}={n_ok/n:.2f}")
        if b in recomputed and abs(recomputed[b]["acc"] - n_ok / n) > 0.02:
            # attribution re-scores; may differ slightly
            notes.append(
                f"{b}: eval acc {n_ok/n:.2f} vs attribution baseline {recomputed[b]['acc']:.2f} "
                f"(re-score; small drift possible)"
            )

    # --- 3. Error tag counts via plot script logic if available ---
    print("\n=== Error tags (plot_error_tags.tag) ===")
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import plot_error_tags as pet

        reported = {
            "CLIP": (37, 24, 8, 5),
            "SigLIP": (40, 30, 6, 4),
            "VLJEPA ZS": (46, 38, 5, 3),
        }
        for name, path in pet.EVALS.items():
            if not path.is_file():
                issues.append(f"error-tag eval missing: {path}")
                continue
            c = pet.counts_for(path)
            wrong = sum(c.values())
            a, sp, f = c.get("attribute", 0), c.get("spatial", 0), c.get("fluency", 0)
            print(f"  {name}: wrong={wrong} attribute={a} spatial={sp} fluency={f}")
            if name in reported:
                rw, ra, rsp, rf = reported[name]
                if (wrong, a, sp, f) != (rw, ra, rsp, rf):
                    issues.append(
                        f"error tags {name}: report {(rw, ra, rsp, rf)} vs recomputed {(wrong, a, sp, f)}"
                    )
                else:
                    notes.append(f"error tags exact match for {name}")
            # percentages in report
            if wrong:
                print(
                    f"    pct attribute={100*a/wrong:.0f}% spatial={100*sp/wrong:.0f}% fluency={100*f/wrong:.0f}%"
                )
    except Exception as e:
        notes.append(f"Could not re-run error tag script: {e}")
        print(f"  (skip detailed tag recompute: {e})")
        for label, (w, a, sp, f) in {
            "CLIP": (37, 24, 8, 5),
            "SigLIP": (40, 30, 6, 4),
            "VLJEPA": (46, 38, 5, 3),
        }.items():
            if a + sp + f != w:
                issues.append(f"error tag counts don't sum for {label}: {a}+{sp}+{f}!={w}")
            else:
                notes.append(f"error tag arithmetic OK for {label}")

    # --- 4. Files ---
    print("\n=== Figure files ===")
    xai = list((ROOT / "artifacts/report_figures/xai").glob("*.png"))
    print(f"  XAI pack: {len(xai)} PNGs (expect 24)")
    if len(xai) != 24:
        issues.append(f"XAI pack expected 24, got {len(xai)}")
    for name in [
        "01_spatial_clip_wall_GOOD.png",
        "02_spatial_clip_ceiling_MISGROUND.png",
        "06_object_clip_microwave_MISGROUND.png",
        "09_attr_clip_microwave_grid.png",
        "10_attr_vljepa_microwave_grid.png",
    ]:
        p = ROOT / "artifacts/report_figures/xai" / name
        if not p.is_file() or p.stat().st_size < 10_000:
            issues.append(f"bad/missing figure {name}")
        else:
            print(f"  OK {name} ({p.stat().st_size/1024:.0f} KB)")

    for name in [
        "modality_sensitivity_n100.svg",
        "embedding_modality_gap_n100.svg",
        "error_tags_n100.svg",
    ]:
        p = ROOT / "artifacts/figures" / name
        if not p.is_file():
            issues.append(f"missing SVG {name}")
        else:
            print(f"  OK {name}")

    # DOCX embeds
    import zipfile
    docx = ROOT / "artifacts/reports/Affordance_Benchmark_Seminar_Report.docx"
    if docx.is_file():
        with zipfile.ZipFile(docx) as z:
            media = [n for n in z.namelist() if n.startswith("word/media/")]
            print(f"  DOCX media entries: {len(media)}")
            if len(media) < 10:
                issues.append(f"DOCX has few embeds: {len(media)}")
            # check broken zero-size
            for n in media:
                if z.getinfo(n).file_size < 1000:
                    issues.append(f"DOCX tiny media {n}")
    else:
        issues.append("DOCX missing")

    # SigLIP folder count
    for backend in ("clip", "siglip", "open_vljepa"):
        d = ROOT / "artifacts/attribution_n100" / backend
        if d.is_dir():
            n = len(list(d.glob("*.json")))
            print(f"  attribution_n100/{backend}: {n} json")
            if n != 100:
                issues.append(f"{backend} pair json count {n} != 100")
        else:
            issues.append(f"missing dir attribution_n100/{backend}")

    # Acc in report for attribution matches
    report_acc = {"clip": 0.63, "siglip": 0.60, "open_vljepa": 0.54}
    print("\n=== Attribution acc vs report headline N=100 frozen ===")
    for b, target in report_acc.items():
        a = recomputed[b]["acc"]
        # attribution re-score: report uses original eval; check drift
        print(f"  {b}: attribution {a:.2f}  report headline {target:.2f}")
        if abs(a - target) > 0.011:
            notes.append(
                f"NOTE: {b} attribution-derived acc {a:.2f} vs report eval headline {target:.2f} "
                f"— report uses original humaneval dumps; attribution re-scores. Prefer citing eval for acc."
            )

    print("\n======== SUMMARY ========")
    if issues:
        print(f"ISSUES ({len(issues)}):")
        for x in issues:
            print("  !", x)
    else:
        print("No hard issues found.")
    if notes:
        print(f"NOTES ({len(notes)}):")
        for x in notes:
            print("  -", x)


if __name__ == "__main__":
    main()
