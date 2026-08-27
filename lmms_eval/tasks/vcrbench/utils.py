import os
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from loguru import logger as eval_logger

from lmms_eval.api.reasoning import strip_reasoning_tags
from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

Document = Mapping[str, Any]
TaskKwargs = Mapping[str, Any]
MetricRecord = dict[str, Any]

# Official prompt from VCRBench/data/dataset.py (PROMPT). The two trailing spaces
# at the end of the first and second line are part of the released prompt; keep them.
VCRBENCH_PROMPT = """The given video consists of multiple short clips, each showing a different segment needed to complete the task: {goal}. 
These clips are randomly shuffled, and your job is to arrange them in the correct order to complete the task: {goal}. 
The clip numbers are mentioned at the beginning of each clip as Clip 1, Clip 2, and so on.
In order to solve this task, first, you should identify the activity that is performed in each clip, and then use your reasoning and common sense to arrange these clips to successfully complete the task.

The final output should be in this format:

Correct order: <mention the Clip numbers separated by a comma>
"""

# Official phrase cascade from VCRBench/process.py. The official loop stops after the
# first phrase that matches any sentence, and the empty phrase matches every sentence,
# so the empty phrase is the effective path for every non-degenerate response.
ANSWER_PHRASES = (
    "",
    "correct order is:",
    "correct order:",
    "correct order",
    "**Correct order:**",
    "*Correct order:*",
    "follow these steps in order",
    "the correct order is",
    "the correct order is",
    "the final output should be",
)

_REASONING_TAG_PAIRS = [["<think>", "</think>"], ["<thinking>", "</thinking>"]]
_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def vcrbench_doc_to_visual(doc: Document) -> list[str]:
    """Resolve the local video for one VCRBench example."""
    video_path = resolve_media_reference(
        doc["video_file"],
        media_type="video",
        cache_dir="vcrbench",
        env_vars=("VCRBENCH_VIDEO_DIR", "VCRBENCH_ROOT"),
        extra_subdirs=("videos",),
    )
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"VCRBench video not found: {video_path}. Set VCRBENCH_VIDEO_DIR to the directory holding the video_<N>.mp4 files, " "or VCRBENCH_ROOT to the directory holding the videos/ subdirectory.")
    return [video_path]


def vcrbench_doc_to_text(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Format the official VCRBench step-ordering prompt for one goal."""
    kwargs = lmms_eval_specific_kwargs or {}
    prompt = VCRBENCH_PROMPT.format(goal=doc["goal"])
    return f"{kwargs.get('pre_prompt', '')}{prompt}{kwargs.get('post_prompt', '')}"


def vcrbench_target_order(doc: Document) -> list[int]:
    """Return the ground-truth clip order as 1-indexed clip numbers."""
    return [step + 1 for step in doc["ground_truth"]]


def vcrbench_doc_to_target(doc: Document) -> str:
    """Render the ground-truth clip order the way the prompt asks for it."""
    return ", ".join(str(clip) for clip in vcrbench_target_order(doc))


def fetch_predicted_order(text: str) -> list[int]:
    """Read one comma-separated clip order, taking the first integer of every field."""
    predicted_order = []
    for part in str(text).split(","):
        match = re.search(r"\d+", part.strip())
        if match is not None:
            predicted_order.append(int(match.group()))
    return predicted_order


def extract_following_text(text: str, phrase: str) -> list[str]:
    """Return the text that follows ``phrase`` in every sentence that contains it."""
    phrase_regex = re.compile(re.escape(phrase), re.IGNORECASE)
    results = []
    for sentence in _SENTENCE_PATTERN.split(text):
        match = phrase_regex.search(sentence)
        if match:
            results.append(sentence[match.end() :].strip())
    return results


def is_consecutive_in_range(sequence: Sequence[int], start: int, end: int) -> bool:
    """Check that ``sequence`` is a permutation of every integer in ``[start, end]``."""
    return set(sequence) == set(range(start, end + 1))


def compare_lists(target: Sequence[int], prediction: Sequence[int]) -> list[int]:
    """Score the prediction position by position; a length mismatch scores all zeros."""
    if len(target) != len(prediction):
        return [0] * len(target)
    return [1 if a == b else 0 for a, b in zip(target, prediction)]


def _wrapped_answer(response: str) -> str | None:
    """Return the payload of an ``<answer>`` or ``\\boxed{}`` wrapper, if the model used one."""
    match = _ANSWER_TAG_PATTERN.search(response)
    if match:
        return match.group(1)

    boxed_start = response.rfind("\\boxed{")
    if boxed_start == -1:
        return None
    depth = 0
    for position in range(boxed_start + len("\\boxed{") - 1, len(response)):
        if response[position] == "{":
            depth += 1
        elif response[position] == "}":
            depth -= 1
            if depth == 0:
                return response[boxed_start + len("\\boxed{") : position]
    return None


def extract_predicted_order(response: str, num_steps: int) -> list[int]:
    """Extract the predicted clip order, falling back to zeros when no valid order is found.

    Reasoning models are handled first: ``<think>`` blocks are stripped and an
    ``<answer>`` or ``\\boxed{}`` wrapper is read directly. Everything after that
    follows the official VCRBench ``process.py`` cascade, including its habit of
    filling the prediction with zeros when no permutation of ``1..num_steps``
    can be recovered.

    Args:
        response: Raw model output for one example.
        num_steps: Number of clips in the ground-truth order.

    Returns:
        A list of ``num_steps`` clip numbers, or ``num_steps`` zeros on failure.
    """
    response = strip_reasoning_tags(response or "", _REASONING_TAG_PAIRS)

    wrapped = _wrapped_answer(response)
    if wrapped is not None:
        predicted_order = fetch_predicted_order(wrapped)
        if len(predicted_order) == num_steps and is_consecutive_in_range(predicted_order, 1, num_steps):
            return predicted_order

    for phrase in ANSWER_PHRASES:
        cleaned = response.replace("*", "")
        candidates = extract_following_text(cleaned, phrase=phrase)
        if len(candidates) == 0:
            continue
        for candidate in candidates:
            predicted_order = fetch_predicted_order(candidate)
            if len(predicted_order) == num_steps and is_consecutive_in_range(predicted_order, 1, num_steps):
                return predicted_order
        # The official implementation stops after the first phrase that matches a sentence.
        break

    return [0] * num_steps


def vcrbench_process_results(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Parse a prediction and emit the shared VCRBench metric record."""
    prediction = results[0] if results else ""
    target_order = vcrbench_target_order(doc)
    predicted_order = extract_predicted_order(prediction, len(target_order))
    if predicted_order == [0] * len(target_order):
        eval_logger.debug(f"VCRBench qid={doc['qid']}: no valid clip order found in the response.")
    record = {
        "qid": doc["qid"],
        "goal": doc["goal"],
        "num_steps": len(target_order),
        "prediction": prediction,
        "predicted_order": predicted_order,
        "answer": target_order,
        "score": float(predicted_order == target_order),
        "step_scores": compare_lists(target_order, predicted_order),
    }
    return {
        "vcrbench_accuracy": record,
        "vcrbench_step_accuracy": record,
        "vcrbench_weighted_accuracy": record,
    }


def vcrbench_aggregate_accuracy(results: Sequence[MetricRecord]) -> float:
    """Aggregate the official ``avg_accuracy``: exact match of the whole clip order."""
    if not results:
        return 0.0
    accuracy = 100.0 * sum(result["score"] for result in results) / len(results)
    step_scores = defaultdict(list)
    for result in results:
        step_scores[result["num_steps"]].append(result["score"])
    for num_steps, scores in sorted(step_scores.items()):
        eval_logger.info(f"VCRBench/{num_steps}-step accuracy: {100.0 * sum(scores) / len(scores):.2f}")
    return accuracy


def vcrbench_aggregate_step_accuracy(results: Sequence[MetricRecord]) -> float:
    """Aggregate the official ``avg_step_accuracy``: position-wise match over all clips."""
    step_scores = [score for result in results for score in result["step_scores"]]
    if not step_scores:
        return 0.0
    return 100.0 * sum(step_scores) / len(step_scores)


def vcrbench_aggregate_weighted_accuracy(results: Sequence[MetricRecord]) -> float:
    """Aggregate the official ``weighted_avg_accuracy``: unweighted mean over the goal classes."""
    if not results:
        return 0.0
    goal_scores = defaultdict(list)
    for result in results:
        goal_scores[result["goal"]].append(result["score"])
    per_goal = {}
    for goal, scores in sorted(goal_scores.items()):
        per_goal[goal] = 100.0 * sum(scores) / len(scores)
        eval_logger.info(f"VCRBench/{goal}: {per_goal[goal]:.2f}")
    return sum(per_goal.values()) / len(per_goal)
