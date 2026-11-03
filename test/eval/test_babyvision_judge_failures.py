import math
from types import SimpleNamespace

import pytest

from lmms_eval.tasks.babyvision import utils as und_utils
from lmms_eval.tasks.babyvision_gen import utils as gen_utils


@pytest.mark.parametrize("utils", [und_utils, gen_utils], ids=["und", "gen"])
@pytest.mark.parametrize(
    "response,expected",
    [
        ("True", True),
        ("false", False),
        ("False. The prediction is not true.", False),
        ("untrue", None),
        ("I cannot tell", None),
        ("", None),
    ],
)
def test_parse_bool_response(utils, response, expected):
    assert utils.parse_bool_response(response) == expected


def _und_doc(answer="(4,7)"):
    return {"type": "t", "subtype": "s", "question": "q", "ansType": "blank", "blankAns": answer, "choiceAns": None}


def _judge(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _rate_limited(**kwargs):
    raise RuntimeError("429 Too Many Requests")


def test_und_judge_failure_is_not_scored_as_wrong(monkeypatch):
    monkeypatch.setattr(und_utils, "_get_openai_client", lambda: _judge(_rate_limited))

    result = und_utils.babyvision_process_results(_und_doc(), ["\\boxed{(4,7)}"])["babyvision_overall_accuracy"]

    assert result["correct"] is None
    assert "429" in result["error"]


def test_und_unparseable_verdict_is_not_scored(monkeypatch):
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="I cannot tell"))])
    monkeypatch.setattr(und_utils, "_get_openai_client", lambda: _judge(lambda **kwargs: reply))

    result = und_utils.babyvision_process_results(_und_doc(), ["\\boxed{(4,7)}"])["babyvision_overall_accuracy"]

    assert result["correct"] is None
    assert result["error"] == "Could not parse judge response"


def test_und_missing_api_key_is_not_scored_as_wrong(monkeypatch):
    monkeypatch.delenv("BABYVISION_API_KEY", raising=False)

    result = und_utils.babyvision_process_results(_und_doc(), ["\\boxed{(4,7)}"])["babyvision_overall_accuracy"]

    assert result["correct"] is None


@pytest.mark.parametrize("utils", [und_utils, gen_utils], ids=["und", "gen"])
def test_aggregate_leaves_out_unjudged_samples(utils):
    results = [
        {"task_type": "t", "subtype": "s", "correct": True},
        {"task_type": "t", "subtype": "s", "correct": False},
        {"task_type": "t", "subtype": "s", "correct": None, "error": "Connection error."},
    ]

    # 1 of the 2 judged samples is correct
    assert utils.babyvision_aggregate_results(results) == 0.5


@pytest.mark.parametrize("utils", [und_utils, gen_utils], ids=["und", "gen"])
def test_aggregate_is_nan_when_nothing_was_judged(utils):
    results = [{"task_type": "t", "subtype": "s", "correct": None, "error": "No API client"}] * 3

    assert math.isnan(utils.babyvision_aggregate_results(results))


def _gen_doc():
    return {"taskId": 1, "type": "t", "subtype": "s", "generationPrompt": "draw"}


def test_gen_failures_keep_task_type_for_aggregation(monkeypatch):
    monkeypatch.setattr(gen_utils, "BABYVISION_API_KEY", None)
    no_client = gen_utils.babyvision_process_results(_gen_doc(), ['{"images": ["gen.png"]}'])["babyvision_overall_accuracy"]
    no_image = gen_utils.babyvision_process_results(_gen_doc(), ['{"images": []}'])["babyvision_overall_accuracy"]

    # no judge: not scored. no image: wrong, judge or not.
    assert no_client["correct"] is None
    assert no_image["correct"] is False
    # these used to have no task_type, and aggregation raised KeyError
    assert gen_utils.babyvision_aggregate_results([no_client, no_image]) == 0.0
