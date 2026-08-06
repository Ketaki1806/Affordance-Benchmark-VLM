"""Build seminar report as .docx with figures (no python-docx).

Uses first-person singular (solo project). Embeds curated XAI PNGs plus
generated pipeline / bar-chart figures.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "reports" / "Affordance_Benchmark_Seminar_Report.docx"
MEDIA_DIR = ROOT / "artifacts" / "reports" / "_report_media"
XAI = ROOT / "artifacts" / "report_figures" / "xai"
FIG = ROOT / "artifacts" / "figures"
ATTR = ROOT / "artifacts" / "attribution_n100"


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


def make_pipeline_png(path: Path) -> None:
    """Horizontal project pipeline diagram."""
    W, H = 1400, 320
    img = Image.new("RGB", (W, H), (250, 249, 247))
    draw = ImageDraw.Draw(img)
    title_f = _font(22)
    box_f = _font(16)
    small_f = _font(13)
    draw.text((40, 18), "Project pipeline", font=title_f, fill=(26, 26, 26))

    stages = [
        ("PACO-LVIS\nparts + images", (44, 92, 130)),
        ("Qwen2.5-VL\npos + hard neg", (61, 120, 90)),
        ("Human check\n(pilot N=20)", (140, 100, 50)),
        ("Score Δ\nCLIP / SigLIP /\nOpen-VLJEPA", (90, 70, 130)),
        ("Error tags +\nXAI + gap", (130, 70, 70)),
    ]
    n = len(stages)
    margin = 36
    gap = 28
    box_w = (W - 2 * margin - (n - 1) * gap) // n
    box_h = 150
    y0 = 70
    for i, (label, color) in enumerate(stages):
        x0 = margin + i * (box_w + gap)
        x1, y1 = x0 + box_w, y0 + box_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=color)
        # multiline center
        lines = label.split("\n")
        total_h = len(lines) * 22
        ty = y0 + (box_h - total_h) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=box_f)
            tw = bbox[2] - bbox[0]
            draw.text((x0 + (box_w - tw) // 2, ty), line, font=box_f, fill=(255, 255, 255))
            ty += 22
        if i < n - 1:
            ax0 = x1 + 4
            ax1 = x1 + gap - 4
            mid = y0 + box_h // 2
            draw.line([(ax0, mid), (ax1, mid)], fill=(80, 80, 80), width=3)
            draw.polygon([(ax1, mid), (ax1 - 10, mid - 7), (ax1 - 10, mid + 7)], fill=(80, 80, 80))

    draw.text(
        (40, H - 36),
        "Then: N=100 automatic eval · FT on ~500 pairs · occlusion vision share · embedding modality gap · word heatmaps",
        font=small_f,
        fill=(80, 80, 80),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def make_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    ymax: float | None = None,
) -> None:
    W, H = 1100, 520
    img = Image.new("RGB", (W, H), (250, 249, 247))
    draw = ImageDraw.Draw(img)
    title_f = _font(20)
    small_f = _font(13)
    tick_f = _font(12)
    draw.text((28, 18), title, font=title_f, fill=(26, 26, 26))
    draw.text((28, 48), subtitle, font=small_f, fill=(90, 90, 90))

    ml, mr, mt, mb = 70, 30, 90, 90
    plot_w, plot_h = W - ml - mr, H - mt - mb
    vals = [v for _, xs, _ in series for v in xs]
    ymax = ymax if ymax is not None else max(vals) * 1.15 if vals else 1.0
    if ymax <= 0:
        ymax = 1.0

    n = len(labels)
    group_w = plot_w / max(n, 1)
    n_series = len(series)
    bar_w = group_w / (n_series + 1.6)

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = mt + plot_h * (1 - t)
        val = ymax * t
        draw.line([(ml, y), (W - mr, y)], fill=(220, 220, 220), width=1)
        draw.text((8, y - 8), f"{val:.2f}", font=tick_f, fill=(70, 70, 70))

    for i, lab in enumerate(labels):
        for j, (sname, xs, color) in enumerate(series):
            v = xs[i]
            h = plot_h * (v / ymax)
            x = ml + i * group_w + group_w * 0.18 + j * (bar_w + 4)
            y = mt + plot_h - h
            draw.rectangle([x, y, x + bar_w, mt + plot_h], fill=color)
            draw.text((x, y - 16), f"{v:.2f}", font=tick_f, fill=(40, 40, 40))
        cx = ml + (i + 0.5) * group_w
        bbox = draw.textbbox((0, 0), lab, font=small_f)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw / 2, H - 58), lab, font=small_f, fill=(30, 30, 30))

    lx = ml
    for j, (sname, _, color) in enumerate(series):
        x = lx + j * 160
        draw.rectangle([x, H - 28, x + 14, H - 14], fill=color)
        draw.text((x + 20, H - 30), sname, font=small_f, fill=(40, 40, 40))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def make_charts_from_json() -> dict[str, Path]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    pipeline = MEDIA_DIR / "pipeline.png"
    make_pipeline_png(pipeline)
    paths["pipeline"] = pipeline

    sens = ATTR / "modality_sensitivity.json"
    gap = ATTR / "embedding_modality_gap.json"
    if sens.is_file():
        data = json.loads(sens.read_text(encoding="utf-8"))
        backends = data.get("backends", [])
        labels = [b["label"] for b in backends]
        make_bar_chart(
            MEDIA_DIR / "occlusion_vision_share.png",
            "Occlusion vision share (decision sensitivity, N=100)",
            "Not the embedding modality gap. vision_share = max|grid| / (max|grid| + max|text|)",
            labels,
            [
                ("all", [b["mean_vision_share"] for b in backends], (44, 95, 138)),
                ("correct", [b["mean_vision_share_correct"] for b in backends], (61, 139, 110)),
                ("wrong", [b["mean_vision_share_wrong"] for b in backends], (196, 92, 38)),
            ],
            ymax=0.4,
        )
        paths["vision_share"] = MEDIA_DIR / "occlusion_vision_share.png"

    if gap.is_file():
        data = json.loads(gap.read_text(encoding="utf-8"))
        order = ["clip", "siglip", "open_vljepa"]
        label_map = {"clip": "CLIP", "siglip": "SigLIP", "open_vljepa": "Open-VLJEPA"}
        rows = []
        for k in order:
            b = data.get("backends", {}).get(k)
            if b and "modality_gap" in b:
                rows.append((label_map[k], b))
        if rows:
            labels = [r[0] for r in rows]
            make_bar_chart(
                MEDIA_DIR / "embedding_modality_gap.png",
                "Embedding modality gap (alignment geometry, N=100)",
                "Not occlusion vision share. gap = ||mean z_img - mean z_txt||; matched cosines",
                labels,
                [
                    ("gap (pos)", [r[1]["modality_gap"] for r in rows], (91, 74, 138)),
                    ("gap (all)", [r[1]["modality_gap_all"] for r in rows], (138, 107, 181)),
                    ("cos(img,pos)", [r[1]["mean_matched_cos"] for r in rows], (44, 122, 107)),
                    ("cos(img,neg)", [r[1]["mean_matched_cos_neg"] for r in rows], (196, 122, 38)),
                ],
            )
            paths["modality_gap"] = MEDIA_DIR / "embedding_modality_gap.png"

    # Error tags simple PNG
    err = MEDIA_DIR / "error_tags.png"
    make_bar_chart(
        err,
        "Failure-tag mix on frozen wrongs (N=100)",
        "Attribute = purpose/action swap; spatial = location/part; fluency = caption noise",
        ["CLIP", "SigLIP", "Open-VLJEPA"],
        [
            ("attribute", [65, 75, 83], (44, 95, 138)),
            ("spatial", [22, 15, 11], (61, 139, 110)),
            ("fluency", [14, 10, 7], (196, 92, 38)),
        ],
        ymax=100,
    )
    paths["error_tags"] = err
    return paths


# ---------- OOXML helpers ----------

def p(text: str, *, bold: bool = False, center: bool = False, size: int = 22) -> str:
    inner = escape(text)
    rpr = f'<w:rPr>{"<w:b/>" if bold else ""}<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
    jc = '<w:jc w:val="center"/>' if center else ""
    return (
        f'<w:p><w:pPr><w:spacing w:after="140" w:line="276" w:lineRule="auto"/>{jc}</w:pPr>'
        f"<w:r>{rpr}<w:t xml:space=\"preserve\">{inner}</w:t></w:r></w:p>"
    )


def h(text: str, level: int = 1) -> str:
    size = {1: 32, 2: 26, 3: 24}.get(level, 24)
    before = {1: 320, 2: 240, 3: 180}.get(level, 180)
    return (
        f'<w:p><w:pPr><w:spacing w:before="{before}" w:after="100"/></w:pPr>'
        f'<w:r><w:rPr><w:b/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
        f"<w:t>{escape(text)}</w:t></w:r></w:p>"
    )


def blank() -> str:
    return '<w:p><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>'


def caption(text: str) -> str:
    return p(text, center=True, size=18)


def table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(text: str, header: bool = False) -> str:
        weight = "<w:b/>" if header else ""
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"1800\" w:type=\"dxa\"/></w:tcPr>"
            f'<w:p><w:pPr><w:spacing w:after="40"/></w:pPr>'
            f'<w:r><w:rPr>{weight}<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
            f"<w:t>{escape(text)}</w:t></w:r></w:p></w:tc>"
        )

    body = [
        "<w:tbl>",
        "<w:tblPr><w:tblW w:w=\"9000\" w:type=\"dxa\"/><w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:color="999999"/>'
        '<w:left w:val="single" w:sz="4" w:color="999999"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="999999"/>'
        '<w:right w:val="single" w:sz="4" w:color="999999"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="CCCCCC"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="CCCCCC"/>'
        "</w:tblBorders></w:tblPr>",
    ]
    body.append("<w:tr>" + "".join(cell(x, True) for x in headers) + "</w:tr>")
    for row in rows:
        body.append("<w:tr>" + "".join(cell(c) for c in row) + "</w:tr>")
    body.append("</w:tbl>")
    body.append(blank())
    return "".join(body)


class DocBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.images: list[tuple[str, Path]] = []  # (rId, path)
        self._img_i = 0

    def add(self, xml: str) -> None:
        self.parts.append(xml)

    def add_image(self, path: Path, *, width_in: float = 5.8, max_height_in: float = 3.8) -> None:
        if not path.is_file():
            self.add(p(f"[Missing figure: {path.name}]", size=18))
            return
        with Image.open(path) as im:
            w_px, h_px = im.size
        aspect = h_px / max(w_px, 1)
        w_in = width_in
        h_in = w_in * aspect
        if h_in > max_height_in:
            h_in = max_height_in
            w_in = h_in / aspect
        cx = int(w_in * 914400)
        cy = int(h_in * 914400)
        self._img_i += 1
        r_id = f"rId{self._img_i}"
        doc_pr_id = self._img_i
        self.images.append((r_id, path))
        self.parts.append(
            f"""<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="60"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
                 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
                 distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{doc_pr_id}" name="{escape(path.name)}"/>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{escape(path.name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{r_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>"""
        )

    def document_xml(self) -> str:
        body = "".join(self.parts)
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    def rels_xml(self) -> str:
        rels = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]
        for r_id, path in self.images:
            target = f"media/{path.name}"
            # uniquify media names if collision
            rels.append(
                f'<Relationship Id="{r_id}" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="{escape(target)}"/>'
            )
        rels.append("</Relationships>")
        return "\n".join(rels)


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def build() -> Path:
    charts = make_charts_from_json()
    d = DocBuilder()

    d.add(p("Affordance Caption Benchmark for Vision–Language Models", bold=True, center=True, size=36))
    d.add(p("Seminar Project Report — AVPR / SS2026", center=True, size=22))
    d.add(p("Solo project · findings through August 2026", center=True, size=20))
    d.add(blank())

    # Voice note for the student
    d.add(h("Note on voice", 1))
    d.add(p(
        "I carried out this project on my own, so I write in the first person (I / my) rather than "
        "we. In multi-author papers we is conventional; for a solo seminar report I is clearer and "
        "honest about ownership."
    ))

    d.add(h("1. What I set out to do", 1))
    d.add(p(
        "The question is simple on paper and annoying in practice: given an image of an object "
        "(with a part visible, from PACO-LVIS), can a vision–language embedding pick the caption "
        "that describes a valid affordance over a hard negative that looks almost the same?"
    ))
    d.add(p(
        "Hard negatives are not random wrong sentences. They keep the same part and often the same "
        "verb family, but flip the purpose or action — open vs close the microwave by the door "
        "handle, hang the mirror on the wall vs on the ceiling. That follows the SugarCrepe idea "
        "of fluent, confusable distractors, specialized here to affordance and purpose swaps."
    ))
    d.add(p(
        "I draft captions with Qwen2.5-VL-7B-Instruct in a [Verb] the [part] to [purpose] style, "
        "then score image–caption cosine similarity with CLIP, SigLIP, and Open-VLJEPA (zero-shot "
        "and fine-tuned where available). I human-checked a pilot of 20 pairs; the larger N=100 "
        "set is mostly automatic and should be read with that noise floor in mind."
    ))

    d.add(h("1.1 Why Qwen2.5-VL-7B-Instruct", 2))
    d.add(p(
        "Caption generation uses Hugging Face id Qwen/Qwen2.5-VL-7B-Instruct (float16 on one GPU). "
        "I did not run a full ablation across every Qwen size; the choice is a practical trade-off "
        "for this seminar stack on LST Condor, with a few deliberate axes:"
    ))
    d.add(p(
        "Family. Qwen2.5-VL is the current open vision–language line with solid instruction "
        "following for structured JSON-style captions, and first-class support in transformers "
        "(Qwen2_5_VLForConditionalGeneration + qwen-vl-utils). That mattered more here than using "
        "an older Qwen2-VL checkpoint or a closed API model that would complicate reproducibility."
    ))
    d.add(p(
        "Size. 7B Instruct is the middle ground: clearly stronger than the 3B class for fluent "
        "affordance wording and hard negatives, while still fitting a single Condor GPU under "
        "float16 with the project’s min/max pixel settings. 32B/72B would improve quality in "
        "principle but was not worth the queue time, memory, and cost for generating N=100 plus "
        "the ~500 FT caption pool."
    ))
    d.add(p(
        "Instruct variant. The Instruct checkpoint matches the chat template I use (one user turn "
        "with image + text) and is better suited to “write one positive and one hard negative” "
        "than a raw base LM. Captions remain candidates only — the human pilot (keep / edit / "
        "drop) is what makes the small eval trustworthy."
    ))
    d.add(p(
        "What I am not claiming. I am not claiming that 7B is optimal for affordance captioning "
        "in general, only that it was a reproducible, GPU-feasible generator for this benchmark "
        "pipeline. Fluency errors on raw N=100 are partly a reminder of that limit."
    ))

    d.add(h("2. Project pipeline", 1))
    d.add(p(
        "End to end, the work runs as a pipeline from PACO parts to scored pairs, then into error "
        "tags and two separate XAI-style analyses (occlusion decision sensitivity vs embedding "
        "alignment geometry)."
    ))
    d.add_image(charts["pipeline"], width_in=6.2, max_height_in=2.0)
    d.add(caption("Figure 1. Project pipeline from PACO images to evaluation and analysis."))
    d.add(p(
        "In practice: build a preferred-part manifest → generate pos/neg with Qwen → keep / edit / "
        "drop on the pilot → evaluate frozen and fine-tuned scorers on Condor → tag wrongs → run "
        "word heatmaps, N=100 occlusion attribution, and embedding modality gap → curate figures."
    ))

    d.add(h("3. Related work and existing datasets", 1))
    d.add(p(
        "AffordanceCLIP (Cuttano et al., 2024) asks what CLIP knows about actions like peeling a "
        "banana, mainly through localization on AGD20K. My setup is narrower and easier to score: "
        "binary caption choice, no pixel masks, open vocabulary text. I inherit the weakness of "
        "global embeddings on spatial detail, but I get a direct read on whether purpose polarity "
        "survives in the shared embedding space."
    ))
    d.add(p(
        "Open-VLJEPA needs a careful caveat. Architecturally it mirrors Meta’s VL-JEPA "
        "(frozen V-JEPA 2, Llama-3.2 predictor, EmbeddingGemma Y-encoder, InfoNCE). The checkpoint "
        "I run is a community re-implementation trained at far smaller scale — not Meta’s full "
        "system. My numbers bound the open model, not the closed-scale paper model."
    ))

    d.add(h("3.1 Is there already a dataset like mine?", 2))
    d.add(p(
        "Short answer: not as a single public benchmark. Pieces of the idea exist in two families—"
        "VL hard-negative ranking suites, and affordance / part datasets—but I did not find a "
        "released set that combines PACO part–tied affordance purpose swaps with SugarCrepe-style "
        "fluent hard negatives under binary image–caption ranking the way this project does."
    ))
    d.add(p(
        "Closest by protocol (image + positive vs fluent hard-negative caption): SugarCrepe and "
        "SugarCrepe++ (compositionality hard negatives), older template suites such as ARO, CREPE, "
        "and VL-CheckList, Winoground (small human two-image / two-caption set), and HNC (hard "
        "negative captions for fine-grained image–text matching). These share the ranking shape; "
        "they are not about affordance or purpose polarity on object parts."
    ))
    d.add(p(
        "Closest by affordance or parts: AGD20K (and AffordanceCLIP) for pixel affordance "
        "grounding; PACO itself for parts and appearance attributes on LVIS/Ego4D (my image "
        "source); FG-OVD-style PACO fine-grained caption benchmarks that swap attributes such as "
        "color or material into hard negatives; classical affordance mask sets (IIT-AFF, UMD, "
        "PAD, and similar); and HOI datasets (HICO, HAKE). FG-OVD is mechanically close—PACO plus "
        "hard-neg text—but the content is appearance attributes, not “open vs close to …” "
        "affordance polarity. AGD20K is affordance, but segmentation rather than caption choice."
    ))
    d.add(table(
        ["Dataset / suite", "Focus", "How it differs from this project"],
        [
            ["SugarCrepe / ++", "Fluent VL hard negs", "Compositionality, not affordance purpose"],
            ["ARO / CREPE / VL-CheckList", "Template hard negs", "Attributes/relations; often less fluent"],
            ["Winoground", "Tiny human set", "Two images × two captions; broader skills"],
            ["AGD20K / AffordanceCLIP", "Affordance heatmaps", "Localization, not binary captions"],
            ["PACO", "Parts + attributes", "No affordance purpose ranking task"],
            ["FG-OVD (PACO captions)", "Attribute hard negs", "Color/material swaps, not affordance"],
            ["IIT-AFF / UMD / PAD", "Affordance masks", "Classical CV / robotics labels"],
            ["HICO / HAKE", "Human–object interaction", "People–object actions, different task"],
        ],
    ))
    d.add(p(
        "For the report I therefore position the contribution as: existing VL hard-negative "
        "suites test compositionality; affordance datasets test localization; PACO tests parts "
        "and attributes. I combine PACO parts with affordance purpose hard negatives in a "
        "SugarCrepe-style binary ranking setup. That specific slice is not, to my knowledge, an "
        "existing large public benchmark. With N=100 and mostly automatic Qwen captions it is "
        "best described as a diagnostic set for this seminar, not yet a released dataset of "
        "SugarCrepe scale."
    ))

    d.add(h("4. Main quantitative results", 1))
    d.add(h("4.1 Pilot (N=20)", 2))
    d.add(p(
        "On raw Qwen captions, frozen CLIP reached 0.70. After my human filtering that dropped to "
        "0.50 — a reminder that format-valid JSON is not the same as a grounded affordance pair. "
        "Several pilot items needed edit or drop (wrong object, weak negatives, missing handle)."
    ))

    d.add(h("4.2 Scale-up N=100 (Qwen captions, no full human filter)", 2))
    d.add(table(
        ["Model", "Accuracy", "Notes"],
        [
            ["CLIP frozen", "0.63", "Mean confidence gap ~0.021"],
            ["SigLIP", "0.60", "Same dual-encoder family as CLIP"],
            ["Open-VLJEPA ZS", "0.54", "Open re-impl, small pretrain"],
            ["Open-VLJEPA FT", "0.64", "Ranking FT on ~500 held-out pairs"],
            ["CLIP FT", "0.87", "Same FT setup; may partly match caption style"],
        ],
    ))
    d.add(p(
        "SigLIP sitting next to CLIP is useful: the struggle is not a CLIP-only quirk. Fine-tuning "
        "helps a lot for CLIP and only modestly for Open-VLJEPA. These FT numbers are fair among "
        "models I actually ran; they are not a claim about Meta VL-JEPA."
    ))

    d.add(h("4.3 Human pilot after SugarCrepe-style edits (N=20, FT models)", 2))
    d.add(table(
        ["Model", "Accuracy", "Comment"],
        [
            ["CLIP FT", "0.90 (18/20)", "Strong; 2 residual attribute fails"],
            ["VLJEPA FT", "0.45 (9/20)", "Below chance after cleaner negatives"],
        ],
    ))
    d.add(p(
        "Before the edits, VLJEPA-FT was around 0.55. Cleaning the negatives made it worse, which "
        "is exactly what a hard-negative protocol should do when a model was leaning on leaky cues. "
        "CLIP-FT stayed high."
    ))

    d.add(h("4.4 Y-space (text-only EmbeddingGemma on the N=100 captions)", 2))
    d.add(p(
        "Mean cos(pos, neg) is 0.815 (median 0.809); 55% of pairs sit above 0.8. Hard negatives "
        "really are close in text embedding space. CLIP wrongs are only slightly higher on average "
        "(0.827 vs 0.807 when correct) — a weak signal, not a full explanation."
    ))

    d.add(h("5. Error analysis", 1))
    d.add(p(
        "For every frozen wrong on N=100, I tagged the dominant pos/neg contrast: attribute "
        "(same part, wrong action/purpose), spatial (location or part mismatch), or fluency "
        "(junk / wrong-object / ungrounded Qwen text). Tags are contrast heuristics plus spot "
        "checks — good for structure, not gospel."
    ))
    d.add_image(charts["error_tags"], width_in=5.8, max_height_in=3.2)
    d.add(caption("Figure 2. Share of frozen wrongs by failure tag (percent within each model’s wrongs)."))
    d.add(table(
        ["Model", "Wrong", "Attribute", "Spatial", "Fluency"],
        [
            ["CLIP frozen", "37", "24 (65%)", "8 (22%)", "5 (14%)"],
            ["SigLIP", "40", "30 (75%)", "6 (15%)", "4 (10%)"],
            ["Open-VLJEPA ZS", "46", "38 (83%)", "5 (11%)", "3 (7%)"],
        ],
    ))
    d.add(p(
        "The intended hard-negative type — attribute / purpose swaps — is also where the models "
        "fail most. Spatial errors are fewer but recurring. Fluency errors are the caption-noise "
        "floor (~7–14%). Fourteen image IDs are wrong for all three frozen backends."
    ))

    d.add(h("6. Worked examples (with figures)", 1))

    d.add(h("6.1 Spatial hard neg — mirror wall vs ceiling", 2))
    d.add(p(
        "Image lvis_75183. Positive: “Hang the mirror on the wall.” Negative: “Hang the mirror "
        "on the ceiling.” CLIP and SigLIP prefer the negative; this is the spatial story."
    ))
    d.add_image(XAI / "01_spatial_clip_wall_GOOD.png", width_in=5.6, max_height_in=3.4)
    d.add(caption("Figure 3. CLIP word heatmap for “wall” — heat lands on wall surfaces (good grounding)."))
    d.add_image(XAI / "02_spatial_clip_ceiling_MISGROUND.png", width_in=5.6, max_height_in=3.4)
    d.add(caption(
        "Figure 4. CLIP heatmap for “ceiling” on the same scene — hot spots on wall/mantel, not "
        "the ceiling (misground; matches the spatial hard-negative failure mode)."
    ))

    d.add(h("6.2 Attribute hard neg — microwave open vs close", 2))
    d.add(p(
        "Image lvis_486018. Positive: open by pulling the door handle. Negative: close by pulling "
        "the door handle. Same part and verb, flipped purpose. CLIP, SigLIP, and Open-VLJEPA all "
        "prefer close on the frozen N=100 run."
    ))
    d.add_image(XAI / "06_object_clip_microwave_MISGROUND.png", width_in=5.6, max_height_in=3.4)
    d.add(caption("Figure 5. CLIP “microwave” word map — heat on blank wall, not the appliance (object noun fail)."))
    d.add_image(XAI / "07_part_siglip_door.png", width_in=5.6, max_height_in=3.4)
    d.add(caption("Figure 6. SigLIP “door” map on the microwave — modest focus near door/frame (mixed)."))
    d.add_image(XAI / "09_attr_clip_microwave_grid.png", width_in=5.2, max_height_in=3.2)
    d.add(caption("Figure 7. CLIP 3×3 occlusion grid (Δ-drop) on the microwave pair — vision mass often off-object."))
    d.add_image(XAI / "10_attr_vljepa_microwave_grid.png", width_in=5.2, max_height_in=3.2)
    d.add(caption(
        "Figure 8. Open-VLJEPA occlusion grid on the same pair — stronger vision response, still a failure case."
    ))

    d.add(h("6.3 Part / purpose residue — blender lid", 2))
    d.add(p(
        "Image lvis_258649 (also in the human pilot). Positive: push down the lid to blend. "
        "Negative variants flip purpose (serve) or action. Even after FT, purpose on the same "
        "part can remain confusable."
    ))
    d.add_image(XAI / "08_part_clip_lid_MISGROUND.png", width_in=5.6, max_height_in=3.4)
    d.add(caption("Figure 9. CLIP “lid” heatmap — heat on person/shirt rather than the blender lid."))
    d.add_image(XAI / "21_neg_clip_serve_MISGROUND.png", width_in=5.6, max_height_in=3.4)
    d.add(caption("Figure 10. CLIP “serve” on the negative caption — face/shirt/edge, not the action."))

    d.add(h("6.4 One fluency caveat", 2))
    d.add(p(
        "Pillow image with lamp captions (lvis_422959) shows up as a CLIP “wrong” under automatic "
        "tags. Calling that a model failure is unfair — it is caption noise, and it is why I treat "
        "raw Qwen N=100 as an upper-noise bound rather than a clean gold set."
    ))

    d.add(h("7. Explainability: two metrics I keep separate", 1))
    d.add(p(
        "I report two different things on purpose. Occlusion vision share asks how much the binary "
        "decision moves when I ablate text or black out image regions. Embedding modality gap asks "
        "how far image and text clouds sit in the shared space. One is decision sensitivity; the "
        "other is alignment geometry. They should not be mixed under one name."
    ))

    d.add(h("7.1 Occlusion vision share (decision sensitivity)", 2))
    d.add(p(
        "Protocol: leave-one-out on caption words, plus a 3×3 blackout grid, measured on "
        "Δ = s_pos − s_neg. I first ran eight failure-focused pairs, then scaled to all N=100."
    ))
    d.add_image(charts["vision_share"], width_in=5.8, max_height_in=3.3)
    d.add(caption("Figure 11. Mean occlusion vision share on N=100, by backend and correct/wrong."))
    d.add(table(
        ["Backend", "Acc", "Mean vision share", "Correct", "Wrong"],
        [
            ["CLIP", "63%", "0.26", "0.28", "0.24"],
            ["SigLIP", "60%", "0.23", "0.26", "0.19"],
            ["Open-VLJEPA", "54%", "0.24", "0.24", "0.24"],
        ],
    ))
    d.add(p(
        "Vision accounts for roughly a quarter of peak sensitivity; text leave-one-out still "
        "dominates. This is an occlusion proxy, not Grad-CAM."
    ))

    d.add(h("7.2 Embedding modality gap (alignment geometry)", 2))
    d.add(p(
        "On the same N=100 pairs I encode images and captions once per backend and report "
        "||mean(z_img) − mean(z_txt)|| after L2-normalization (positives only, and pos∪neg), "
        "plus mean matched cosines."
    ))
    d.add_image(charts["modality_gap"], width_in=5.8, max_height_in=3.3)
    d.add(caption("Figure 12. Embedding modality gap and matched cosines on N=100."))
    d.add(table(
        ["Backend", "gap (pos)", "gap (all)", "cos(img,pos)", "cos(img,neg)"],
        [
            ["CLIP", "0.97", "0.97", "0.18", "0.17"],
            ["SigLIP", "1.04", "1.04", "0.07", "0.06"],
            ["Open-VLJEPA", "0.27", "0.33", "0.57", "0.54"],
        ],
    ))
    d.add(p(
        "Open-VLJEPA looks much more aligned in embedding space. CLIP and SigLIP show a classic "
        "large modality gap. That nicer geometry does not buy Open-VLJEPA better affordance "
        "ranking here — which is exactly why I insist on naming the two metrics differently."
    ))

    d.add(h("8. Takeaways", 1))
    d.add(p(
        "1. Frozen dual encoders sit around 60–63% on N=100 Qwen affordance pairs; Open-VLJEPA "
        "ZS is weaker (0.54), FT recovers to 0.64, CLIP FT jumps to 0.87 (with style-match caveats)."
    ))
    d.add(p(
        "2. Failures are mostly attribute / purpose polarity — the hard-negative design. Spatial "
        "misses exist; fluency misses are caption noise."
    ))
    d.add(p(
        "3. On a cleaned human pilot, CLIP-FT stays strong (0.90); VLJEPA-FT falls to 0.45 when "
        "negatives get less leaky."
    ))
    d.add(p(
        "4. Occlusion analysis: the binary choice is mostly text-explained (~0.23–0.26 vision share)."
    ))
    d.add(p(
        "5. Embedding modality gap: Open-VLJEPA is far more aligned than CLIP/SigLIP, yet that "
        "alignment does not translate into better affordance ranking on this task."
    ))

    d.add(h("9. Limits", 1))
    d.add(p(
        "N=100 with mostly automatic captions is a noisy ceiling. Human validation covers the "
        "pilot thoroughly, not the full hundred. Fine-tuning on Qwen-style pairs can inflate "
        "CLIP-FT if the model learns caption mannerisms. Open-VLJEPA is architecture-close to "
        "Meta VL-JEPA and scale-far from it. Occlusion grids are coarse; word heatmaps are "
        "illustrative dual-encoder maps, not causal proof."
    ))

    d.add(h("10. What I would do next", 1))
    d.add(p(
        "Human-edit or validate the full N=100 (or grow to a few hundred), add one or two "
        "external VLMs under the same protocol, and keep the two-metric XAI story as analysis "
        "rather than as a method claim. That is the shortest path from this seminar report toward "
        "a workshop-style write-up."
    ))

    d.add(h("11. Key artifacts", 1))
    d.add(p(
        "Eval dumps: humaneval/30jul/clip.json, 1aug/siglip.json, 31jul/open_vljepa.json. "
        "N=100 occlusion + gap: artifacts/attribution_n100/. "
        "Curated XAI pack: artifacts/report_figures/xai/. "
        "Working notes: docs/project_notes.md. "
        "This document: artifacts/reports/Affordance_Benchmark_Seminar_Report.docx."
    ))
    d.add(blank())
    d.add(p("— End of report.", size=20))

    # Write zip; uniquify media filenames
    OUT.parent.mkdir(parents=True, exist_ok=True)
    used_names: dict[str, int] = {}
    media_entries: list[tuple[str, Path, bytes]] = []
    for r_id, path in d.images:
        name = path.name
        if name in used_names:
            used_names[name] += 1
            stem = path.stem
            name = f"{stem}_{used_names[path.name]}{path.suffix}"
        else:
            used_names[name] = 0
        data = path.read_bytes()
        media_entries.append((r_id, Path(name), data))

    # rebuild rels with uniquified names
    rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for r_id, name, _ in media_entries:
        rels.append(
            f'<Relationship Id="{r_id}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{name.as_posix()}"/>'
        )
    rels.append("</Relationships>")

    # Patch document r:embed already uses rIds — OK. Media paths use uniquified names in zip.
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("word/document.xml", d.document_xml())
        zf.writestr("word/_rels/document.xml.rels", "\n".join(rels))
        for _, name, data in media_entries:
            zf.writestr(f"word/media/{name.as_posix()}", data)

    return OUT


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out}")
    print(f"Embedded images under {MEDIA_DIR} plus curated XAI PNGs.")
