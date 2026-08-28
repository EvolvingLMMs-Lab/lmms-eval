"""Contract tests for the owner-aligned MicroVQA task."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from PIL import Image

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "microvqa"


class _YamlLoader(yaml.SafeLoader):
    pass


_YamlLoader.add_multi_constructor(
    "!",
    lambda loader, tag_suffix, node: loader.construct_scalar(node),
)


def _load_utils():
    utils_path = TASK_DIR / "utils.py"
    spec = importlib.util.spec_from_file_location("microvqa_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _doc(correct_index=1):
    return {
        "question": "Which structure is visible?",
        "choices": ["Alpha", "Beta", "Gamma"],
        "correct_index": correct_index,
    }


def test_microvqa_uses_owner_numeric_answer_contract():
    utils = _load_utils()

    assert utils.doc_to_text(_doc()) == (
        "The following is a multiple choice question (with answers).\n"
        'Think step by step and then output the answer in the format of "The '
        'answer is (X)" at the end.\n\n'
        "Which structure is visible?\n\n"
        "Options:\n"
        "  (1): Alpha\n"
        "  (2): Beta\n"
        "  (3): Gamma\n\n"
    )
    assert utils.process_results(_doc(), ["Reasoning. The answer is **(2)**"]) == {"accuracy": 1.0}
    assert utils.process_results(_doc(), ["Reasoning. The answer is (B)"]) == {"accuracy": 0.0}


def test_microvqa_places_text_before_all_images():
    utils = _load_utils()
    first = Image.new("RGB", (2, 3), color="red")
    second = Image.new("RGB", (4, 5), color="blue")
    doc = {**_doc(), "images_list": [first, second]}

    messages = utils.doc_to_messages(doc)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content] == ["text", "image", "image"]
    assert content[0]["text"] == utils.doc_to_text(doc)
    assert [item["url"].size for item in content[1:]] == [(2, 3), (4, 5)]


@pytest.mark.parametrize("images", [[], [None]])
def test_microvqa_rejects_missing_images(images):
    utils = _load_utils()
    doc = {**_doc(), "images_list": images}

    with pytest.raises(AssertionError, match="image"):
        utils.doc_to_messages(doc)


def test_microvqa_task_config_matches_owner_protocol():
    config = yaml.load(
        (TASK_DIR / "microvqa.yaml").read_text(),
        Loader=_YamlLoader,
    )

    assert config["dataset_path"] == "jmhb/microvqa"
    assert config["task"] == "microvqa"
    assert config["test_split"] == "test"
    assert config["output_type"] == "generate_until"
    assert config["doc_to_messages"] == "utils.doc_to_messages"
    assert config["generation_kwargs"] == {
        "max_new_tokens": 2048,
        "temperature": 1.0,
        "top_p": 1.0,
        "do_sample": True,
        "seed": 0,
    }
    assert config["metric_list"] == [
        {
            "metric": "accuracy",
            "aggregation": "mean",
            "higher_is_better": True,
        }
    ]
    assert config["metadata"] == {
        "version": 1.0,
        "owner_commit": "3b52dc7131c3a285c33654856b349d9073e3604b",
    }


@pytest.mark.parametrize("results", [[], ["first", "second"]])
def test_microvqa_requires_exactly_one_response(results):
    utils = _load_utils()

    with pytest.raises(AssertionError, match="one response"):
        utils.process_results(_doc(), results)
