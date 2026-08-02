from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grounding_text import content_words, safe_name


def test_content_words_skips_stopwords():
    assert content_words("Hang the mirror on the wall.") == ["Hang", "mirror", "wall"]


def test_safe_name():
    assert safe_name("wall") == "wall"
    assert safe_name("door-handle") == "door_handle"
