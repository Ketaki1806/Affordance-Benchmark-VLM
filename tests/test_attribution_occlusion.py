import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribution_occlusion import (
    ablate_role,
    blackout_grid_cell,
    delta,
    leave_one_out,
    load_pairs_by_id,
    main,
    role_labels,
    tokenize_words,
)
from PIL import Image


def test_tokenize_words_strips_punct():
    assert tokenize_words("Open the lid.") == ["Open", "the", "lid"]


def test_leave_one_out():
    assert leave_one_out("Open the lid", 0) == "the lid"
    assert leave_one_out("Open the lid", 1) == "Open lid"


def test_role_labels_marks_part_and_purpose():
    toks = tokenize_words("Twist the lid to open the jar")
    labels = role_labels(toks)
    assert "part" in labels
    assert "purpose" in labels


def test_ablate_role_removes_part():
    out = ablate_role("Twist the lid to open the jar", "part")
    assert "lid" not in out.lower()


def test_delta():
    assert delta(0.5, 0.3) == 0.2


def test_blackout_grid_cell_zeros_corner():
    img = Image.new("RGB", (90, 90), color=(255, 255, 255))
    out = blackout_grid_cell(img, 0, 0, grid=3)
    assert out.getpixel((0, 0)) == (0, 0, 0)
    assert out.getpixel((29, 29)) == (0, 0, 0)
    assert out.getpixel((30, 0)) == (255, 255, 255)
    assert out.getpixel((0, 30)) == (255, 255, 255)
    assert out.getpixel((89, 89)) == (255, 255, 255)


class FakeScorer:
    def load(self):
        pass

    def unload(self):
        pass

    def score(self, image_path: str, text: str) -> float:
        return 0.1 * len(text.split())


def test_attribute_pair_runs(tmp_path):
    from attribution_occlusion import attribute_pair

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (60, 60), (128, 128, 128)).save(img_path)
    pair = {
        "image_id": "lvis_test",
        "object": "box",
        "image_path": str(img_path),
        "positive": "Open the lid to access food",
        "negative": "Close the lid to store food",
        "correct": True,
    }
    out = attribute_pair(FakeScorer(), "fake", pair, grid=3, out_dir=tmp_path)
    assert out["baseline"]["delta"] == out["baseline"]["pos_score"] - out["baseline"]["neg_score"]
    assert len(out["image_occlusion"]["delta_drop"]) == 3
    assert (tmp_path / "fake" / "lvis_test.json").is_file()
    assert (tmp_path / "fake" / "lvis_test_grid.png").is_file()


def test_attribute_pair_baseline_model_choice_and_correct(tmp_path):
    """baseline.model_choice mirrors evaluate.py's pos_score >= neg_score rule; correct passes through."""
    from attribution_occlusion import attribute_pair

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (60, 60), (128, 128, 128)).save(img_path)

    # positive has more tokens than negative -> FakeScorer gives it a higher score -> most_probable.
    pair_pos_wins = {
        "image_id": "lvis_pos",
        "image_path": str(img_path),
        "positive": "Open the lid to access the food quickly",
        "negative": "Close it",
        "correct": True,
    }
    out = attribute_pair(FakeScorer(), "fake", pair_pos_wins, grid=3, out_dir=tmp_path)
    assert out["baseline"]["model_choice"] == "most_probable"
    assert out["baseline"]["correct"] is True

    # negative has more tokens -> higher score -> not most_probable ("negative").
    # correct is always *derived* from this run's model_choice, never copied from the
    # input pair (there is none here), so it must be False, not omitted.
    pair_neg_wins = {
        "image_id": "lvis_neg",
        "image_path": str(img_path),
        "positive": "Open it",
        "negative": "Close the lid to store the food safely",
    }
    out2 = attribute_pair(FakeScorer(), "fake", pair_neg_wins, grid=3, out_dir=tmp_path)
    assert out2["baseline"]["model_choice"] == "negative"
    assert out2["baseline"]["correct"] is False

    # A pair whose source "correct" disagrees with this backend's own scores must NOT be
    # copied through: correct is derived from model_choice, ignoring pair["correct"].
    pair_source_correct_disagrees = {
        "image_id": "lvis_disagree",
        "image_path": str(img_path),
        "positive": "Open it",
        "negative": "Close the lid to store the food safely",
        "correct": True,
    }
    out_disagree = attribute_pair(
        FakeScorer(), "fake", pair_source_correct_disagrees, grid=3, out_dir=tmp_path
    )
    assert out_disagree["baseline"]["model_choice"] == "negative"
    assert out_disagree["baseline"]["correct"] is False

    # equal scores (tie) -> most_probable, matching evaluate.py's ">=" rule.
    pair_tie = {
        "image_id": "lvis_tie",
        "image_path": str(img_path),
        "positive": "Open the lid",
        "negative": "Close the box",
    }
    out3 = attribute_pair(FakeScorer(), "fake", pair_tie, grid=3, out_dir=tmp_path)
    assert out3["baseline"]["pos_score"] == out3["baseline"]["neg_score"]
    assert out3["baseline"]["model_choice"] == "most_probable"


def test_run_attribution_summary_enriched(tmp_path):
    from attribution_occlusion import run_attribution

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (60, 60), (128, 128, 128)).save(img_path)
    pairs = [
        {
            "image_id": "lvis_test",
            "image_path": str(img_path),
            "positive": "Open the lid to access food",
            "negative": "Close the lid to store food",
        }
    ]

    class _FakeScorerNoop(FakeScorer):
        pass

    import attribution_occlusion as mod

    original_make_scorer = mod._make_scorer
    mod._make_scorer = lambda backend, config, vljepa_checkpoint: _FakeScorerNoop()
    try:
        summary = run_attribution(pairs, backends=["fake"], out_dir=tmp_path, grid=3)
    finally:
        mod._make_scorer = original_make_scorer

    pair_summary = summary["backends"]["fake"]["pairs"][0]
    assert pair_summary["image_id"] == "lvis_test"
    assert "baseline" in pair_summary
    assert pair_summary["baseline"]["model_choice"] in ("most_probable", "negative")

    top_drops = pair_summary["top_text_drops"]
    assert len(top_drops) <= 3
    for record in top_drops:
        assert record["side"] in ("positive", "negative")
        assert "token" in record
        assert "d_delta" in record
        assert "ds" in record
    # sorted by abs(d_delta) descending
    abs_deltas = [abs(r["d_delta"]) for r in top_drops]
    assert abs_deltas == sorted(abs_deltas, reverse=True)

    assert "image_occlusion" in pair_summary
    delta_drop = pair_summary["image_occlusion"]["delta_drop"]
    assert len(delta_drop) == 3
    assert all(len(row) == 3 for row in delta_drop)

    # summary.json on disk must round-trip the same enrichment.
    with open(tmp_path / "summary.json", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["backends"]["fake"]["pairs"][0]["top_text_drops"] == top_drops


class OcclusionSensitiveScorer:
    """Scorer whose score depends on whether the (occluded) image path/content signals occlusion.

    Used to exercise the delta_drop formula (baseline_delta - new_delta) with a scorer that
    actually reacts to blacked-out grid cells, rather than always returning a constant.
    """

    def load(self):
        pass

    def unload(self):
        pass

    def score(self, image_path: str, text: str) -> float:
        # Occluded temp images are always freshly written JPEGs; detect occlusion by
        # checking whether the image content contains any pure-black pixels (the blackout fill).
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            has_black = any(px == (0, 0, 0) for px in rgb.getdata())
        base = 0.1 * len(text.split())
        return base - 0.5 if has_black else base


def test_image_occlusion_delta_drop_structure_and_formula(tmp_path):
    from attribution_occlusion import _image_occlusion_delta_drop, delta

    base_image = Image.new("RGB", (60, 60), (200, 200, 200))
    positive = "Open the lid"
    negative = "Close the lid"
    scorer = OcclusionSensitiveScorer()
    base_path = tmp_path / "base.jpg"
    base_image.save(base_path)
    s_pos = scorer.score(str(base_path), positive)
    s_neg = scorer.score(str(base_path), negative)
    base_delta = delta(s_pos, s_neg)

    grid = 3
    delta_drop = _image_occlusion_delta_drop(
        scorer, base_image, positive, negative, grid, base_delta
    )

    assert len(delta_drop) == grid
    assert all(len(row) == grid for row in delta_drop)
    # Both positive and negative scores drop by the same 0.5 when occluded, so the delta
    # (pos - neg) is unchanged -> baseline_delta - new_delta should be ~0 for every cell.
    for row in delta_drop:
        for value in row:
            assert value == pytest.approx(0.0)


def test_load_pairs_by_id_filters_and_orders(tmp_path):
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "pairs": [
                    {"image_id": "a", "positive": "p_a", "negative": "n_a"},
                    {"image_id": "b", "positive": "p_b", "negative": "n_b"},
                    {"image_id": "c", "positive": "p_c", "negative": "n_c"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = load_pairs_by_id(eval_json, ["c", "a"])
    assert [p["image_id"] for p in result] == ["c", "a"]
    assert result[0]["positive"] == "p_c"


def test_load_pairs_by_id_raises_on_missing_id(tmp_path):
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps({"pairs": [{"image_id": "a", "positive": "p", "negative": "n"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing"):
        load_pairs_by_id(eval_json, ["missing"])


def test_load_pairs_by_id_falls_back_to_extra_json(tmp_path):
    """C1: an id missing from the primary source is resolved from --extra-pairs-json."""
    primary_json = tmp_path / "primary.json"
    primary_json.write_text(
        json.dumps({"pairs": [{"image_id": "a", "positive": "p_a", "negative": "n_a"}]}),
        encoding="utf-8",
    )
    extra_json = tmp_path / "extra.json"
    extra_json.write_text(
        json.dumps(
            {
                "pairs": [
                    {"image_id": "blender", "positive": "p_blender", "negative": "n_blender"}
                ]
            }
        ),
        encoding="utf-8",
    )

    result = load_pairs_by_id(primary_json, ["a", "blender"], extra_json=[extra_json])
    assert [p["image_id"] for p in result] == ["a", "blender"]
    assert result[1]["positive"] == "p_blender"


def test_load_pairs_by_id_skip_missing_warns_and_drops(tmp_path, capsys):
    """--skip-missing continues (dropping the id) instead of raising when unresolvable."""
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps({"pairs": [{"image_id": "a", "positive": "p", "negative": "n"}]}),
        encoding="utf-8",
    )

    result = load_pairs_by_id(eval_json, ["a", "missing"], skip_missing=True)
    assert [p["image_id"] for p in result] == ["a"]
    assert "missing" in capsys.readouterr().out


def test_make_scorer_clip_does_not_mutate_original_config(monkeypatch):
    """_make_scorer('clip', cfg) must force a frozen (checkpointless) CLIP without mutating cfg.

    Stubs out the real (torch-backed) clip_scorer module with a FakeScorer-style stand-in so
    this test stays fast and works without torch/transformers installed.
    """
    import sys
    import types

    from attribution_occlusion import _make_scorer

    fake_module = types.ModuleType("clip_scorer")

    class FakeCLIPScorer:
        def __init__(self, config):
            self.config = config
            self.checkpoint = config["models"].get("clip_checkpoint")

    fake_module.CLIPScorer = FakeCLIPScorer
    monkeypatch.setitem(sys.modules, "clip_scorer", fake_module)

    cfg = {
        "models": {
            "clip": "openai/clip-vit-base-patch32",
            "clip_checkpoint": "artifacts/checkpoints/clip/best.pt",
            "clip_device": "cpu",
        }
    }
    original_checkpoint = cfg["models"]["clip_checkpoint"]

    scorer = _make_scorer("clip", cfg, vljepa_checkpoint=None)

    # Original config dict must still have its clip_checkpoint key untouched.
    assert cfg["models"]["clip_checkpoint"] == original_checkpoint
    # The scorer built from a copy without clip_checkpoint must be "frozen" (no checkpoint).
    assert scorer.checkpoint is None


def test_cli_grid_rejects_non_positive(tmp_path, capsys):
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            {"pairs": [{"image_id": "a", "positive": "p", "negative": "n", "image_path": "x.jpg"}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        main(["--pairs-json", str(eval_json), "--grid", "0"])

    with pytest.raises(SystemExit):
        main(["--pairs-json", str(eval_json), "--grid", "-1"])
