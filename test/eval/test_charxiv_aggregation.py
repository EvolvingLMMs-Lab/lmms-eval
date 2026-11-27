"""CharXiv accuracy counts failed or invalid gradings as wrong, as the official get_stats.py does."""

import importlib.util

import pytest

from lmms_eval.tasks.charxiv import utils
from lmms_eval.tasks.charxiv.descriptive_utils import build_dummy_output


def _reasoning_results(n):
    docs = [
        {
            "figure_path": f"images/{i}.jpg",
            "reasoning_q": "How many data points are above 50?",
            "reasoning_a": "3",
            "reasoning_a_type": 4,
        }
        for i in range(n)
    ]
    return [utils.charxiv_reasoning_process_results(doc, ["3"])["reasoning_acc"] for doc in docs]


def _descriptive_results(n):
    return [{"group": (f"images/{i}.jpg_0", "2", "2"), "qid": 1} for i in range(n)]


@pytest.fixture(autouse=True)
def no_client(monkeypatch):
    monkeypatch.setattr(utils, "_get_client", lambda: None)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([1, 1, 1, 1], 1.0),
        ([1, 0, 1, -1], 0.5),
        ([-1, -1, -1, -1], 0.0),
        ([1, "1", 2, 0], 0.25),
    ],
    ids=["all-correct", "one-failed", "all-failed", "non-binary"],
)
def test_reasoning_accuracy_counts_invalid_scores_as_zero(monkeypatch, scores, expected):
    replies = iter(scores)
    monkeypatch.setattr(utils, "get_reasoning_result_gpt", lambda client, prompt, model: ("3", next(replies)))

    assert utils.charxiv_reasoning_aggregate_results(_reasoning_results(len(scores))) == pytest.approx(expected)


def test_descriptive_accuracy_counts_failed_batches_as_zero(monkeypatch):
    # Six responses for one template make two grading batches (5 + 1); the first batch fails.
    replies = iter([build_dummy_output(5), {"extract_answer_T1": "2", "score_T1": 1}])
    monkeypatch.setattr(utils, "get_descriptive_result_gpt", lambda client, prompt, length, model: next(replies))

    assert utils.charxiv_descriptive_aggregate_results(_descriptive_results(6)) == pytest.approx(1 / 6)


def _load_utils_with_env(monkeypatch, **env):
    # A separate copy of the module, so the judge settings read at import time see this environment.
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_VERSION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("charxiv_utils_env", utils.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("base_url", [None, ""], ids=["unset", "empty"])
def test_judge_defaults_to_the_official_endpoint_and_model(monkeypatch, base_url):
    env = {"OPENAI_API_KEY": "sk-test"}
    if base_url is not None:
        env["OPENAI_BASE_URL"] = base_url
    fresh = _load_utils_with_env(monkeypatch, **env)

    assert fresh.MODEL_VERSION == "gpt-4o-2024-05-13"
    assert str(fresh._get_client().base_url) == "https://api.openai.com/v1/"


def test_judge_settings_come_from_the_environment(monkeypatch):
    fresh = _load_utils_with_env(monkeypatch, OPENAI_API_KEY="sk-test", OPENAI_BASE_URL="http://127.0.0.1:8000/v1", MODEL_VERSION="judge")

    assert fresh.MODEL_VERSION == "judge"
    assert str(fresh._get_client().base_url) == "http://127.0.0.1:8000/v1/"


def test_keyless_local_judge_still_works(monkeypatch):
    fresh = _load_utils_with_env(monkeypatch, OPENAI_BASE_URL="http://127.0.0.1:8000/v1", MODEL_VERSION="judge")

    assert str(fresh._get_client().base_url) == "http://127.0.0.1:8000/v1/"


def test_missing_api_key_stops_grading_instead_of_scoring_zero(monkeypatch):
    fresh = _load_utils_with_env(monkeypatch)
    results = [fresh.charxiv_reasoning_process_results({"figure_path": "images/0.jpg", "reasoning_q": "q", "reasoning_a": "3", "reasoning_a_type": 4}, ["3"])["reasoning_acc"]]

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        fresh.charxiv_reasoning_aggregate_results(results)
