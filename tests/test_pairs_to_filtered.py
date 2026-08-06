"""Tests for pairs_to_filtered converter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_to_filtered import pairs_to_filtered


def test_pairs_to_filtered_shape(tmp_path):
    pairs = [
        {
            "image_id": "lvis_1",
            "object": "cup",
            "image_path": "/tmp/a.jpg",
            "positive": "Lift the handle to drink.",
            "negative": "Lift the handle to pour.",
        }
    ]
    out = pairs_to_filtered(pairs)
    assert len(out["objects"]) == 1
    obj = out["objects"][0]
    assert obj["affordance_tiers"]["most_probable"] == ["Lift the handle to drink."]
    assert obj["affordance_tiers"]["negative"] == ["Lift the handle to pour."]


def test_pairs_to_filtered_skips_incomplete():
    out = pairs_to_filtered(
        [{"image_id": "x", "positive": "only pos", "image_path": "/a.jpg"}]
    )
    assert out["objects"] == []
