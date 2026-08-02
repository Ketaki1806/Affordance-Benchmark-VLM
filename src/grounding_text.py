"""Pure helpers for word–region grounding (no torch)."""

from __future__ import annotations

import re

from attribution_occlusion import tokenize_words

_STOP = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "on",
    "in",
    "for",
    "with",
    "from",
    "by",
    "and",
    "or",
    "your",
    "you",
    "it",
    "into",
}


def content_words(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokenize_words(text):
        low = tok.lower()
        if low in _STOP or len(low) < 2:
            continue
        if low not in seen:
            seen.add(low)
            out.append(tok)
    return out


def safe_name(word: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", word).strip("_").lower() or "word"
