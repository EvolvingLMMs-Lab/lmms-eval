"""Offline MET-Bench contracts: task discovery, ordered inputs, and scoring."""

import copy
import io
from pathlib import Path
from typing import Any

import chess
import pytest
from datasets import Dataset
from PIL import Image

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.metbench import utils
from lmms_eval.utils import load_yaml_config

DOMAINS = ("minecraft", "chess", "shell")
TASK_DIR = Path(utils.__file__).parent
FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def image(index: int) -> dict[str, Any]:
    """Encode a distinct image for detecting missing or reordered frames."""
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (index, 0, 0)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def example(domain: str) -> dict[str, Any]:
    """Return one compact task whose target is distinct from its inputs."""
    if domain == "minecraft":
        return {
            "metbench_domain": domain,
            "initial_state": '{"x":0}',
            "action": "Walk forward 1 block.",
            "candidate_states": ['{"x":1}', '{"x":2}', '{"x":3}', '{"x":4}'],
            "image_initial_state": image(0),
            "image_action": image(99),
            "image_candidate_states": [image(i) for i in range(1, 5)],
            "target": 2,
        }
    return {
        "metbench_domain": domain,
        "initial_state": FEN if domain == "chess" else 1,
        "actions": ["g1f3"] * 10 if domain == "chess" else ["1 swap 2"] * 10,
        "image_actions": [image(i) for i in range(10)],
        "target": FEN if domain == "chess" else 2,
    }


def test_tasks_registered_with_pinned_compact_configs() -> None:
    """All six tasks are discoverable without downloading data or loading a model."""
    manager = TaskManager(include_defaults=False, include_path=str(TASK_DIR))
    for domain in DOMAINS:
        for modality in ("text", "image"):
            name = f"metbench_{domain}_{modality}"
            assert name in manager.all_tasks
            config = load_yaml_config(str(TASK_DIR / f"{name}.yaml"), mode="full")
            assert config["tag"] == "metbench"
            assert callable(config["doc_to_messages"])
            assert callable(config["process_results"])
            assert config["test_split"] == "test"
            assert config["metadata"]["version"] == 1.0
            assert config["generation_kwargs"] == {"max_new_tokens": 4096, "temperature": 0, "do_sample": False}
            assert config["metric_list"] == [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]
            assert len(config["dataset_kwargs"]["revision"]) == 40
            assert config["dataset_name"] == ("evaluation_text_only" if modality == "text" else "evaluation")


@pytest.mark.parametrize("domain", DOMAINS)
def test_preprocessing_preserves_order_and_uses_released_target(domain: str) -> None:
    """Preprocessing attaches the target while keeping every example in place."""
    target_column = "correct_choice" if domain == "minecraft" else "final_state"
    rows = Dataset.from_dict({"example_id": [f"{domain}-test-{i}" for i in range(500)], target_column: [2] * 500})
    result = getattr(utils, f"prepare_{domain}")(rows)
    assert result["example_id"] == rows["example_id"]
    assert result["target"] == [2] * 500
    with pytest.raises(ValueError, match="500-example"):
        getattr(utils, f"prepare_{domain}")(rows.select(range(499)))


@pytest.mark.parametrize("domain", DOMAINS)
def test_image_order_and_target_exclusion(domain: str) -> None:
    """Keep all frames in order and never append an answer to model input."""
    doc = example(domain)
    messages = utils.doc_to_messages(doc, {"modality": "image"})
    assert len(messages) == 1 and messages[0]["role"] == "user"
    parts = messages[0]["content"]
    pixels = [part["url"].getpixel((0, 0))[0] for part in parts if part["type"] == "image"]
    assert pixels == list(range(5 if domain == "minecraft" else 10))
    altered = copy.deepcopy(doc)
    altered["target"] = "TARGET_MUST_NOT_APPEAR"
    altered["final_state"] = "TARGET_MUST_NOT_APPEAR"
    altered["states"] = ["TARGET_MUST_NOT_APPEAR"] * 11
    altered["correct_choice"] = "TARGET_MUST_NOT_APPEAR"
    second = utils.doc_to_messages(altered, {"modality": "image"})[0]["content"]
    assert [p["text"] for p in parts if p["type"] == "text"] == [p["text"] for p in second if p["type"] == "text"]
    if domain == "minecraft":
        assert [p["text"] for p in parts if p["type"] == "text"][2:6] == [f"\nChoice {i}:" for i in range(1, 5)]
    else:
        assert [p["type"] for p in parts] == ["text", "image"] * 10 + ["text"]


@pytest.mark.parametrize("domain", DOMAINS)
def test_text_task_needs_no_image_columns(domain: str) -> None:
    """The text-only download is sufficient for text prompt construction."""
    doc = {key: value for key, value in example(domain).items() if not key.startswith("image_")}
    messages = utils.doc_to_messages(doc, {"modality": "text"})
    assert len(messages[0]["content"]) == 1
    assert "Think step by step" in utils.doc_to_text(doc)
    with pytest.raises(ValueError, match="chat backend"):
        utils.doc_to_visual(doc)


@pytest.mark.parametrize("domain", ["minecraft", "shell"])
@pytest.mark.parametrize("response,expected", [("FINAL ANSWER: 2", 1), ("2", 1), ("FINAL ANSWER: \\boxed{2}", 1), ("FINAL ANSWER: 1\nFINAL ANSWER: 2", 1), ("FINAL ANSWER: 1 or 2", 0), ("FINAL ANSWER: 5", 0), ("", 0)])
def test_choice_scoring(domain: str, response: str, expected: float) -> None:
    """Accept a single final choice and score malformed or ambiguous answers zero."""
    assert utils.process_results(example(domain), [response]) == {"acc": expected}


def test_chess_square_accuracy_and_final_answer_parsing() -> None:
    """Moving one piece changes two square matches; Chess scoring is not exact match."""
    doc = example("chess")
    moved = chess.Board(FEN)
    moved.push_uci("e2e4")
    assert utils.process_results(doc, [f"FINAL ANSWER: {FEN}"]) == {"acc": 1.0}
    assert utils.process_results(doc, [f"FINAL ANSWER: {moved.fen()}"]) == {"acc": 62 / 64}
    for response in [FEN, "unparseable", f"FINAL ANSWER: {FEN} or {FEN}"]:
        assert utils.process_results(doc, [response]) == {"acc": 0.0}
    assert utils.process_results(doc, [f"FINAL ANSWER: {moved.fen()}\nFINAL ANSWER: `{FEN}`"]) == {"acc": 1.0}


def test_chess_uncertainty_uses_boards() -> None:
    """The native mean standard error must agree with board-level CI bounds."""
    from lmms_eval.api.metrics import mean_stderr
    from lmms_eval.tasks.metbench.metbench_core import clustered_ratio_interval

    scores = [0.0, 1.0] * 50
    lower, upper = clustered_ratio_interval([(score, 64) for score in [0, 64] * 50])
    margin = 1.959963984540054 * mean_stderr(scores)
    assert lower == pytest.approx(0.5 - margin)
    assert upper == pytest.approx(0.5 + margin)
    assert clustered_ratio_interval([(48, 64)]) == (None, None)


@pytest.mark.parametrize("domain", DOMAINS)
def test_image_text_context_uses_visual_task_wording(domain: str) -> None:
    """Image-task context includes its text captions without text-only states or moves."""
    doc = example(domain)
    visual_context = utils.doc_to_text(doc, {"modality": "image"})
    text_context = utils.doc_to_text(doc, {"modality": "text"})
    assert "Think step by step" in visual_context
    if domain == "minecraft":
        assert "Input frame:" in visual_context
        assert "Candidate next frames:" in visual_context
        assert doc["initial_state"] not in visual_context
        assert doc["initial_state"] in text_context
        for candidate in doc["candidate_states"]:
            assert candidate not in visual_context
            assert candidate in text_context
    else:
        assert "highlighted green" in visual_context or "green square" in visual_context
        assert doc["actions"][0] not in visual_context
        assert doc["actions"][0] in text_context
