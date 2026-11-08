"""Regression tests for WenetSpeech MER tokenization (#1636)."""

import pytest

from lmms_eval.tasks.wenet_speech.utils import compute_mer, tokenize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello世界", ["hello", "世", "界"]),
        ("hello 世界", ["hello", "世", "界"]),
        ("HELLO世界", ["HELLO", "世", "界"]),
        ("don't", ["don't"]),
        ("hello123", ["hello123"]),
        ("hello_world", ["hello_world"]),
        ("你好世界", ["你", "好", "世", "界"]),
        ("hello", ["hello"]),
        ("", []),
        ("...", []),
    ],
)
def test_tokenize_english_word_chinese_char_boundary(text, expected):
    assert tokenize(text) == expected


def test_mer_whitespace_only_english_chinese_boundary_is_zero():
    assert compute_mer("hello世界", "hello 世界") == pytest.approx(0.0)


def test_mer_single_chinese_char_substitution():
    # hello / 世 / 界 vs hello / 世 / 介 -> 1/3
    assert compute_mer("hello世界", "hello世介") == pytest.approx(1 / 3)


def test_mer_identical_mixed_text_is_zero():
    assert compute_mer("hello世界", "hello世界") == pytest.approx(0.0)
