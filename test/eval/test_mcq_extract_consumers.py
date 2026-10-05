"""Hermetic integration checks for consumers of the shared MCQ extractor.

These exercise actual scoring functions using small synthetic documents. They
do not download data, resolve media, run models, or constitute model E2E evidence.
"""

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from lmms_eval.verifiers import build_verification_pipeline
from lmms_eval.verifiers.extractors import MCQExtractor

_CASES = json.loads((Path(__file__).with_name("mcq_extract") / "cases.json").read_text(encoding="utf-8"))["cases"]
_SCORING_IDS = {
    "regression_lowercase_explanation_dog",
    "regression_lowercase_because",
    "regression_closed_lowercase_tag",
    "regression_cubic_period",
    "regression_data_period",
    "regression_a_explanation",
    "container_tag_invalid_final",
    "container_mixed_box_then_tag",
    "correction_later_answer",
    "correction_last_invalid",
    "ambiguous_full_option_list",
    "ambiguous_list_with_final_answer",
    "phrase_out_of_alphabet",
    "multilingual_chinese_is",
    "multilingual_korean_colon",
    "multilingual_japanese_lower",
    "boundary_uppercase_variable_end",
    "boundary_word_right_paren",
    "article_lower_noun",
    "model_reply_option_restatement",
}
_SCORING_CASES = [case for case in _CASES if case["id"] in _SCORING_IDS]
assert {case["id"] for case in _SCORING_CASES} == _SCORING_IDS


@pytest.mark.parametrize(
    "constructor_choices,override,response,expected",
    [
        (None, None, "h", "H"),
        (["A", "B"], None, "Answer: C", ""),
        (["A", "B"], ["A", "B", "C", "D"], "Answer: C", "C"),
        (["A", "B", "C", "D"], ["A", "B"], "Answer: C", ""),
        (["a", "b"], None, "Answer: B", "B"),
        (["A", "B"], [], "h", "H"),
        (["A", "B"], ["A", "C", "F"], "I choose f.", "F"),
        (["A", "B", "C", "D"], None, "Answer: B. Final answer: E", ""),
    ],
)
def test_mcq_extractor_choice_configuration(constructor_choices: list[str] | None, override: list[str] | None, response: str, expected: str) -> None:
    """Constructor choices and per-call overrides obey the same public contract."""
    extractor = MCQExtractor(choices=constructor_choices)
    original = constructor_choices.copy() if constructor_choices is not None else None
    kwargs = {} if override is None else {"choices": override}
    assert extractor.extract(response, **kwargs) == expected
    assert extractor(response, **kwargs) == expected
    assert constructor_choices == original


def test_explicit_none_override_uses_default_alphabet() -> None:
    """Passing choices=None explicitly overrides a restricted constructor value."""
    extractor = MCQExtractor(choices=["A", "B"])
    assert extractor.extract("H") == ""
    assert extractor.extract("H", choices=None) == "H"
    assert extractor.extract("H") == ""


def test_per_call_override_does_not_persist() -> None:
    """An override for one sample must not alter the next sample's alphabet."""
    extractor = MCQExtractor(choices=["A", "B"])
    assert extractor("Answer: C", choices=["A", "B", "C"]) == "C"
    assert extractor("Answer: C") == ""
    assert extractor("Answer: b") == "B"


@pytest.mark.parametrize("case", _SCORING_CASES, ids=lambda case: case["id"])
def test_verification_pipeline_scores_extracted_answer(case: dict[str, Any]) -> None:
    """The real pipeline receives the reviewed answer and preserves provenance."""
    pipeline = build_verification_pipeline({"extractors": [{"type": "mcq", "choices": list("ABCD")}], "verifier": {"type": "mcq_match"}})
    expected = case["expected"]
    ground_truth = expected or "A"
    result = pipeline("Which option fits?", case["response"], ground_truth)
    assert result.metadata["raw_prediction"] == case["response"]
    assert result.metadata["extracted_prediction"] == expected
    assert result.is_correct is bool(expected)
    assert result.score == float(bool(expected))
    assert result.metadata["confident"] is bool(expected)
    if expected:
        other_answer = next(letter for letter in "ABCD" if letter != expected)
        incorrect = pipeline.verify("Which option fits?", case["response"], other_answer)
        assert incorrect.score == 0.0
        assert not incorrect.is_correct
        assert incorrect.metadata["confident"] is True


def test_pipeline_receives_per_sample_choices() -> None:
    """The pipeline forwards alphabet overrides to its MCQ extraction step."""
    pipeline = build_verification_pipeline({"extractors": [{"type": "mcq", "choices": list("AB")}], "verifier": {"type": "mcq_match"}})
    result = pipeline("Four-option question", "Answer: d", "D", choices=list("ABCD"))
    assert result.is_correct
    assert result.metadata["extracted_prediction"] == "D"
    assert pipeline("Two-option question", "Answer: d", "D").score == 0.0


def test_reasoning_stripping_stays_an_upstream_pipeline_step() -> None:
    """A strip-reasoning step removes conflicting explicit declarations first."""
    pipeline = build_verification_pipeline({"extractors": [{"type": "strip_reasoning"}, {"type": "mcq", "choices": list("ABCD")}], "verifier": {"type": "mcq_match"}})
    response = "<think>The answer is A. I should verify that guess.</think>\n(b)"
    result = pipeline("Which option fits?", response, "B")
    assert result.is_correct
    assert result.metadata["raw_prediction"] == response
    assert result.metadata["extracted_prediction"] == "B"


@pytest.mark.parametrize("case", _SCORING_CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("scorer_name", ["mmstar_process_results", "mmstar_process_results_ko"])
def test_mmstar_actual_scoring(case: dict[str, Any], scorer_name: str) -> None:
    """Plain and Korean MMStar score positive answers and abstentions correctly."""
    module = importlib.import_module("lmms_eval.tasks.mmstar.utils")
    scorer = getattr(module, scorer_name)
    expected = case["expected"]
    doc = {"index": 37, "answer": expected or "A", "category": "coarse perception", "l2_category": "image scene and topic"}
    result = scorer(doc, [case["response"]])
    metric_record = {"question_id": 37, "l2_category": "image scene and topic", "score": float(bool(expected))}
    assert result == {"coarse perception": metric_record, "average": metric_record}
    if expected:
        doc["answer"] = next(letter for letter in "ABCD" if letter != expected)
        assert scorer(doc, [case["response"]])["average"]["score"] == 0.0


@pytest.mark.parametrize("benchmark", ["lvbench", "mlvu", "videomme"])
@pytest.mark.parametrize("case", _SCORING_CASES, ids=lambda case: case["id"])
def test_video_benchmark_four_choice_wrappers(benchmark: str, case: dict[str, Any]) -> None:
    """Existing video-task wrappers share the generic contract without media I/O."""
    module = importlib.import_module(f"lmms_eval.tasks.{benchmark}.utils")
    assert module.extract_characters_regex(case["response"]) == case["expected"]


@pytest.mark.parametrize("case", _SCORING_CASES, ids=lambda case: case["id"])
def test_lvbench_actual_boolean_score(case: dict[str, Any]) -> None:
    """LVBench's public scorer consumes the extraction result, including negatives."""
    module = importlib.import_module("lmms_eval.tasks.lvbench.utils")
    result = module.lvbench_process_results({"answer": case["expected"] or "A"}, [case["response"]])
    assert result == {"lvbench_score": bool(case["expected"])}


@pytest.mark.parametrize(
    "response,expected",
    [("I", "I"), ("The answer is n.", "N"), ("<answer>k</answer>", "K"), ("Answer: O", ""), ("The ID_B. appears in the title.", ""), ("I think the image is too blurry.", "")],
)
def test_cgbench_fourteen_choice_score(response: str, expected: str) -> None:
    """CG-Bench's A-N alphabet reaches its actual prediction and score record."""
    module = importlib.import_module("lmms_eval.tasks.cgbench.utils")
    doc = {"right_answer": expected or "A", "qid": "synthetic-score-doc", "video_uid": "not-loaded", "duration": "short", "domain": "test", "sub_category": "test"}
    result = module.cgbench_process_results(doc, [response])["cgbench_accuracy"]
    assert result["pred_answer"] == expected
    assert result["score"] == float(bool(expected))
    assert result["question_id"] == doc["qid"]


@pytest.mark.parametrize(
    "response,expected",
    [("The answer is c.", "C"), ("I choose f.", "F"), ("Answer: B", ""), ("Answer: A or F", ""), ("<think>Answer: C</think>\nThe answer is F", "F")],
)
def test_mmr_v_noncontiguous_choices(response: str, expected: str) -> None:
    """MMR-V preserves the offered letters instead of inferring them from a count."""
    module = importlib.import_module("lmms_eval.tasks.mmr_v.utils")
    doc = {"options": ["(A) first", "(C) second", "(F) third"]}
    assert module.extract_mmr_v_answer(response, module._choices(doc)) == expected


@pytest.mark.parametrize(
    "response,num_options,expected",
    [("The answer is b.", 2, 1), ("The answer is c.", 3, 2), ("I choose e.", 5, 4), ("Answer: F", 5, None), ("The shape is cubic. I think so.", 4, None)],
)
def test_kaleidoscope_lenient_fallback(response: str, num_options: int, expected: int | None) -> None:
    """Kaleidoscope maps the generic fallback letter to an index at the actual count."""
    module = importlib.import_module("lmms_eval.tasks.kaleidoscope.utils")
    assert module.extract_choice(response, num_options, prompt_type="direct", lenient=True) == expected
    assert module.extract_choice(response, num_options, prompt_type="direct", lenient=False) is None
