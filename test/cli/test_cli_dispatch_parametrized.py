"""Parametrized versions of CLI dispatch tests.

Demonstrates pytest.mark.parametrize for cases that were individual methods
in test_cli_dispatch.py. Both files can coexist — this one covers the same
logic more concisely and is easier to extend.
"""

import pytest

from lmms_eval.cli.dispatch import _is_eval_wizard, _is_legacy_invocation
from lmms_eval.cli.models_cmd import _col

# ---------------------------------------------------------------------------
# _is_legacy_invocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], False),
        (["--model", "openai", "--tasks", "mme"], True),
        (["--help"], True),
        (["-h"], True),
        (["tasks"], False),
        (["foobar"], False),
        (["eval", "--model", "openai"], False),
        (["--tasks", "list"], True),
    ],
    ids=[
        "empty",
        "long_flags",
        "help_flag",
        "short_flag",
        "subcommand",
        "unknown_bare_word",
        "eval_prefix_with_flags",
        "tasks_list_flag",
    ],
)
def test_is_legacy_invocation(argv, expected):
    assert _is_legacy_invocation(argv) is expected


# ---------------------------------------------------------------------------
# _is_eval_wizard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], False),
        (["eval"], True),
        (["eval", "--model", "openai"], False),
        (["tasks"], False),
    ],
    ids=["empty", "eval_alone", "eval_with_flags", "non_eval"],
)
def test_is_eval_wizard(argv, expected):
    assert _is_eval_wizard(argv) is expected


# ---------------------------------------------------------------------------
# _col helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, width, expected_len",
    [
        ("hi", 10, 10),
        ("abcdefghij", 5, 5),
        ("abc", 3, 3),
    ],
    ids=["short_padded", "long_truncated", "exact_width"],
)
def test_col_length(text, width, expected_len):
    result = _col(text, width)
    assert len(result) == expected_len


def test_col_short_text_starts_with_input():
    result = _col("hi", 10)
    assert result.startswith("hi")


def test_col_long_text_truncated_content():
    result = _col("abcdefghij", 5)
    assert result == "abcde"
