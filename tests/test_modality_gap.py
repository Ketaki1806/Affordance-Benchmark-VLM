"""Unit tests for embedding modality gap (alignment geometry)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compute_modality_gap import modality_gap_from_stacks


def _norm_rows(rows: list[list[float]]) -> list[list[float]]:
    out = []
    for row in rows:
        n = math.sqrt(sum(v * v for v in row))
        out.append([v / n for v in row])
    return out


def test_modality_gap_identical_clouds_is_zero():
    z = _norm_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = modality_gap_from_stacks(z, [r[:] for r in z], [r[:] for r in z])
    assert out["modality_gap"] == pytest.approx(0.0, abs=1e-6)
    assert out["modality_gap_all"] == pytest.approx(0.0, abs=1e-6)
    assert out["mean_matched_cos"] == pytest.approx(1.0, abs=1e-5)


def test_modality_gap_orthogonal_means():
    imgs = [[1.0, 0.0], [1.0, 0.0]]
    txt = [[0.0, 1.0], [0.0, 1.0]]
    out = modality_gap_from_stacks(imgs, txt, txt)
    assert out["modality_gap"] == pytest.approx(math.sqrt(2), abs=1e-5)
    assert out["mean_matched_cos"] == pytest.approx(0.0, abs=1e-5)


def test_modality_gap_all_uses_pos_and_neg():
    imgs = [[1.0, 0.0], [1.0, 0.0]]
    pos = [[1.0, 0.0], [1.0, 0.0]]
    neg = [[-1.0, 0.0], [-1.0, 0.0]]
    out = modality_gap_from_stacks(imgs, pos, neg)
    assert out["modality_gap"] == pytest.approx(0.0, abs=1e-6)
    assert out["modality_gap_all"] == pytest.approx(1.0, abs=1e-5)
    assert out["mean_matched_cos"] == pytest.approx(1.0, abs=1e-5)
    assert out["mean_matched_cos_neg"] == pytest.approx(-1.0, abs=1e-5)
