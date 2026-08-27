"""Regression tests for issue #1319.

Text-only tasks (``doc_to_visual is None``) that fall back on the auto
``doc_to_messages`` wrapper must carry the full task context (description +
few-shot examples + current question) in the user turn, matching the ``ctx``
built by ``build_all_requests`` and the simple adapters (lm-eval-harness
semantics). Multimodal tasks and tasks with an explicit ``doc_to_messages``
must be unchanged.

The tasks here are constructed from in-memory datasets (``datasets.load_dataset``
is monkeypatched), so no download or network access is needed.
"""

import datasets
import pytest

import lmms_eval.api.metrics  # noqa: F401  # registers metric/aggregation fns (evaluator does this in real runs)
from lmms_eval.api.task import ConfigurableMessagesTask

TRAIN_DOCS = [
    {"subject": "math", "question": "1+1=?", "answer": "2"},
    {"subject": "math", "question": "2+2=?", "answer": "4"},
    {"subject": "math", "question": "3+3=?", "answer": "6"},
]
TEST_DOCS = [
    {"subject": "math", "question": "What is 2+2?", "answer": "4"},
]

DESCRIPTION = "The following are multiple choice questions (with answers) about math."


def _text_only_config():
    return {
        "task": "demo_text_ctx",
        "dataset_path": "lmms-lab/demo",
        "dataset_name": "default",
        "test_split": "test",
        "training_split": "train",
        "fewshot_split": "train",
        "doc_to_text": lambda doc: f"Question: {doc['question']}",
        "doc_to_target": lambda doc: doc["answer"],
        "doc_to_visual": None,
        "doc_to_messages": None,
        "description": DESCRIPTION,
        "num_fewshot": 2,
        "fewshot_config": {"sampler": "first_n"},
        "output_type": "generate_until",
    }


def _build_task(monkeypatch, tmp_path, config):
    monkeypatch.setenv("LMMS_EVAL_DATASETS_CACHE", str(tmp_path))
    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(TRAIN_DOCS),
            "test": datasets.Dataset.from_list(TEST_DOCS),
        }
    )
    monkeypatch.setattr(
        "lmms_eval.api.task.datasets.load_dataset",
        lambda *args, **kwargs: dataset,
    )
    return ConfigurableMessagesTask(config=config)


def _user_text(messages):
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    texts = [p["text"] for p in messages[0]["content"] if isinstance(p, dict) and p.get("type") == "text"]
    assert len(texts) == 1
    return texts[0]


@pytest.fixture
def text_task(monkeypatch, tmp_path):
    return _build_task(monkeypatch, tmp_path, _text_only_config())


def test_text_task_auto_messages_equal_fewshot_context(text_task):
    doc = TEST_DOCS[0]
    user_text = _user_text(text_task.doc_to_messages(doc))
    expected = text_task.fewshot_context(doc, text_task.config.num_fewshot)
    assert user_text == expected


def test_text_task_auto_messages_contain_description_fewshot_and_question(text_task):
    doc = TEST_DOCS[0]
    user_text = _user_text(text_task.doc_to_messages(doc))
    assert DESCRIPTION in user_text
    assert "Question: 1+1=?" in user_text
    assert "Question: 2+2=?" in user_text
    assert "Question: What is 2+2?" in user_text


def test_text_task_auto_messages_are_deterministic(text_task):
    doc = TEST_DOCS[0]
    assert text_task.doc_to_messages(doc) == text_task.doc_to_messages(doc)


def test_text_task_auto_messages_num_fewshot_zero_keeps_description(monkeypatch, tmp_path):
    config = _text_only_config()
    config["num_fewshot"] = 0
    task = _build_task(monkeypatch, tmp_path, config)
    doc = TEST_DOCS[0]
    user_text = _user_text(task.doc_to_messages(doc))
    assert user_text == task.fewshot_context(doc, 0)
    assert DESCRIPTION in user_text
    assert "Question: What is 2+2?" in user_text


def test_multimodal_auto_messages_unchanged(monkeypatch, tmp_path):
    config = _text_only_config()
    config["doc_to_visual"] = lambda doc: ["image1.png"]
    task = _build_task(monkeypatch, tmp_path, config)
    doc = TEST_DOCS[0]

    messages = task.doc_to_messages(doc)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0] == {"type": "image", "url": "image1.png"}
    user_text = _user_text(messages)
    assert user_text == task.doc_to_text(doc)
    assert "Question: What is 2+2?" in user_text
    assert DESCRIPTION not in user_text


def test_explicit_doc_to_messages_unchanged(monkeypatch, tmp_path):
    def explicit_doc_to_messages(doc):
        return [{"role": "user", "content": [{"type": "text", "text": f"Q only: {doc['question']}"}]}]

    config = _text_only_config()
    config["doc_to_messages"] = explicit_doc_to_messages
    task = _build_task(monkeypatch, tmp_path, config)
    doc = TEST_DOCS[0]

    messages = task.doc_to_messages(doc)
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "Q only: What is 2+2?"}]}]
