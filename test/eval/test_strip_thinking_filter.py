"""End-to-end tests for chaining StripThinkingFilter into task filter pipelines.

When a task sets ``auto_strip_thinking``, the evaluator prepends a
``StripThinkingFilter`` to the FRONT of each existing ``FilterEnsemble``'s chain
(it does NOT add a sibling ensemble). These tests exercise that wiring at the
ensemble level and assert the two properties the integration must guarantee:

  (a) the scored/default filter key holds the STRIPPED string that the
      extraction filters selected -- not a list, and not the un-stripped text; and
  (b) no extra ``"strip_thinking"`` filter key is created (a sibling ensemble
      would have produced one, holding an unselected list that later crashes
      string-based ``process_results``).

See ``lmms_eval/evaluator.py`` (auto_strip_thinking wiring) and
``lmms_eval/filters/transformation.py`` (StripThinkingFilter).
"""

from lmms_eval.api.instance import Instance
from lmms_eval.filters import build_filter_ensemble
from lmms_eval.filters.transformation import StripThinkingFilter


def _make_instance(resps, idx=0, doc_id=0):
    inst = Instance(
        request_type="generate_until",
        arguments=("prompt", {}, None, doc_id, "test_task", "test"),
        idx=idx,
        metadata={"task": "test_task", "doc_id": doc_id, "repeats": 1},
    )
    inst.resps = resps
    return inst


def _prepend_strip(ensemble):
    """Mirror the evaluator's auto_strip_thinking wiring: strip at the front."""
    ensemble.filters.insert(0, StripThinkingFilter())
    return ensemble


def test_strip_thinking_chained_before_take_first_yields_stripped_string():
    # Minimal task pipeline: a single take_first selection step named "default".
    ensemble = _prepend_strip(build_filter_ensemble("default", [["take_first", None]]))

    inst = _make_instance(["<think>long chain of reasoning</think>\n\nParis"])
    ensemble.apply([inst], docs=[None])

    # (a) The scored/default key holds the STRIPPED string -- not a list, and not
    #     the un-stripped "<think>...</think>Paris".
    assert inst.filtered_resps["default"] == "Paris"
    assert isinstance(inst.filtered_resps["default"], str)

    # (b) No sibling "strip_thinking" key is created; "default" is the only key.
    assert "strip_thinking" not in inst.filtered_resps
    assert list(inst.filtered_resps.keys()) == ["default"]


def test_strip_thinking_runs_before_regex_extraction():
    # Realistic flexible-extract pipeline: regex extraction, then take_first.
    ensemble = _prepend_strip(
        build_filter_ensemble(
            "flexible-extract",
            [
                ["regex", {"regex_pattern": r"answer is \(?([A-D])\)?"}],
                ["take_first", None],
            ],
        )
    )

    # The reasoning block holds a DECOY "answer is (A)" that must be stripped
    # before the regex runs; the real answer after </think> is (C). If the strip
    # did not run first (the old sibling-ensemble bug), regex would see both and
    # extract the decoy "A".
    inst = _make_instance(["<think>maybe the answer is (A)?</think> The answer is (C)"])
    ensemble.apply([inst], docs=[None])

    assert inst.filtered_resps["flexible-extract"] == "C"
    assert "strip_thinking" not in inst.filtered_resps


def test_strip_thinking_is_noop_on_plain_text():
    # Non-reasoning output passes through unchanged (aside from take_first selection),
    # so the filter is safe to prepend for non-reasoning models.
    ensemble = _prepend_strip(build_filter_ensemble("default", [["take_first", None]]))

    inst = _make_instance(["Just a plain answer."])
    ensemble.apply([inst], docs=[None])

    assert inst.filtered_resps["default"] == "Just a plain answer."
    assert "strip_thinking" not in inst.filtered_resps
