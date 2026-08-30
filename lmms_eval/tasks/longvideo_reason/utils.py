import os
import re
from typing import Any, Mapping, Sequence

from loguru import logger as eval_logger

from lmms_eval.api.reasoning import strip_reasoning_tags
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

Document = Mapping[str, Any]
TaskKwargs = Mapping[str, Any]
MetricRecord = dict[str, Any]

CACHE_DIR_NAME = "longvideo_reason"
VIDEO_ENV_VARS = ("LONGVIDEO_REASON_VIDEO_DIR", "LONGVIDEO_REASON_ROOT")
REASONING_TAG_PAIRS = [["<think>", "</think>"], ["<thinking>", "</thinking>"]]

# The four reasoning perspectives the paper reports. `problem_type` carries
# exactly these four strings on all 1,000 test rows.
PROBLEM_TYPES = ("temporal", "goal", "spatial", "plot")

# The official prompt, copied verbatim from longvideo-reason/eval.py
# (QUESTION_TEMPLATE_VIDEO) in NVlabs/Long-RL, including its typo ("then he
# solves it") and the leading space before "Question:". The options are part of
# `problem`, so the template needs no option block of its own.
QUESTION_TEMPLATE_VIDEO = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

# Official extraction, ported from accuracy_reward()/format_reward().
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ANSWER_TAG_PHRASE = re.compile(r"<answer>Therefore the final answer is: (.*?)</answer>", re.DOTALL)
_FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_BOXED_ANSWER = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")

# Option lines look like "A. <text>", one per line. Used to read back the
# letters an example actually offers and to flag the malformed rows.
_OPTION_LETTER = re.compile(r"^\s*([A-Z])\.\s", re.M)

EXPECTED_OPTION_COUNT = 4


def _choices(doc: Document) -> list[str]:
    """Return the option letters this example actually offers.

    Read back from the question text rather than assumed to be A-D: five rows
    of the test split do not carry four options (see `_is_wellformed`).
    """
    return sorted(set(_OPTION_LETTER.findall(str(doc.get("problem", "")))))


def _is_wellformed(doc: Document) -> bool:
    """True when the example offers exactly four lettered options.

    FIVE of the 1,000 test rows (problem_id 147, 379, 743, 825 and 857) do not.
    The upstream generator leaked its own deliberation into the option block:
    three rows carry a paragraph of model scratchpad where an option should be
    ("A. But this requires the test-taker to weigh..."), and two rows (379,
    857) carry no options at all and cannot be answered as multiple choice by
    any model. They are NOT dropped -- upstream scores all 1,000 and changing
    the denominator would make this task incomparable with the paper -- but
    `longvideo_reason_wellformed_accuracy` reports the clean 995 beside the
    headline so the contamination is visible rather than absorbed.
    """
    return len(_choices(doc)) == EXPECTED_OPTION_COUNT


# The archives do NOT lay out what the annotations reference. Every `videos`
# field reads "longvila_videos/<stem>.<ext>" and the upstream README documents
# that path, but each of the ten shards extracts into its own flat
# "longvideo_eval_subset<N>/" directory instead. Searching those directories as
# well means a correct, complete download resolves without the user first having
# to reorganise 195 GB by hand.
VIDEO_SUBDIRS = ("longvila_videos", "videos", *(f"longvideo_eval_subset{index}" for index in range(10)))


def longvideo_reason_doc_to_visual(doc: Document) -> list[str]:
    """Resolve the local video for one LongVideo-Reason example."""
    reference = str(doc["videos"])
    video_path = resolve_media_reference(
        reference,
        media_type="video",
        cache_dir=CACHE_DIR_NAME,
        env_vars=VIDEO_ENV_VARS,
        extra_subdirs=VIDEO_SUBDIRS,
    )
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"LongVideo-Reason video not found: {reference}. The videos live in a SEPARATE "
            "repository from the annotations: download the ten longvideo_eval_subset*.tar.gz "
            "shards of LongVideo-Reason/longvideo_eval_videos (195 GB), extract them all into "
            "one directory, and set LONGVIDEO_REASON_VIDEO_DIR to it. Note that the shards "
            "extract into per-shard 'longvideo_eval_subset<N>/' directories rather than into "
            "the 'longvila_videos/' directory the upstream README describes; both layouts are "
            "searched."
        )
    return [video_path]


def longvideo_reason_doc_to_text(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Format the official prompt. Options are already inside `problem`."""
    kwargs = lmms_eval_specific_kwargs or {}
    prompt = QUESTION_TEMPLATE_VIDEO.format(question=str(doc["problem"]).rstrip())
    return f"{kwargs.get('pre_prompt', '')}{prompt}{kwargs.get('post_prompt', '')}"


def longvideo_reason_doc_to_target(doc: Document) -> str:
    """Return the gold answer as it ships, i.e. still wrapped in <answer> tags."""
    return str(doc["answer"])


def _unwrap_answer(text: str) -> str:
    """Strip one <answer> wrapper if present, else return the text unchanged.

    The gold field ships as "<answer>B</answer>"; the official scorer unwraps it
    with this same regex before comparing.
    """
    match = _ANSWER_TAG.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def extract_official_answer(response: str) -> str:
    """Reproduce the official student-answer extraction, byte for byte.

    Ported from accuracy_reward() in longvideo-reason/eval.py. The phrase branch
    comes first there, so a completion that says "Therefore the final answer is:"
    yields the text AFTER the phrase, not the whole span. A completion with no
    <answer> tag falls back to the entire stripped completion, which is why the
    official metric is close to zero for any model that narrates.
    """
    content = response or ""
    if "Therefore the final answer is:" in content:
        match = _ANSWER_TAG_PHRASE.search(content)
    else:
        match = _ANSWER_TAG.search(content)
    return match.group(1).strip() if match else content.strip()


def extract_longvideo_reason_answer(response: str, choices: Sequence[str]) -> str:
    """Extract one offered option letter, preferring the official format.

    Layered, most faithful first, and every layer is ADDITIVE: a completion the
    official extractor already resolves to an offered letter is returned
    unchanged, so this can only recover answers the official parser drops (a
    trailing period, a \\boxed{} wrapper, a bare "The answer is C"). It can
    never turn an official hit into a miss. That property is what makes
    `overall_accuracy` >= `strict_accuracy` by construction.
    """
    official = extract_official_answer(response)
    normalized = re.sub(r"[^A-Za-z]", "", official).upper()
    if len(normalized) == 1 and normalized in choices:
        return normalized

    text = strip_reasoning_tags(response or "", REASONING_TAG_PAIRS)
    boxed = _BOXED_ANSWER.findall(text)
    if boxed and boxed[-1].upper() in choices:
        return boxed[-1].upper()
    return extract_mcq_answer(text, choices=list(choices))


def longvideo_reason_process_results(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Parse a prediction and emit the overall, per-perspective and format records."""
    problem_type = str(doc["problem_type"])
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"Unknown LongVideo-Reason problem_type: {problem_type!r}")

    prediction = str(results[0]) if results else ""
    choices = _choices(doc)
    gold_raw = str(doc["answer"])
    answer = re.sub(r"[^A-Za-z]", "", _unwrap_answer(gold_raw)).upper()

    parsed_prediction = extract_longvideo_reason_answer(prediction, choices)
    # Official score: strict string equality between the two unwrapped spans, no
    # normalisation at all. This is the number the paper reports.
    strict_score = float(extract_official_answer(prediction) == _unwrap_answer(gold_raw))

    record = {
        "problem_id": int(doc["problem_id"]),
        "video": str(doc["videos"]),
        "problem_type": problem_type,
        "wellformed": _is_wellformed(doc),
        "prediction": prediction,
        "pred_answer": parsed_prediction,
        "answer": answer,
        "score": float(parsed_prediction == answer),
        "strict_score": strict_score,
        "format_score": float(bool(_FORMAT_PATTERN.match(prediction))),
    }
    return {
        "longvideo_reason_overall_accuracy": record,
        "longvideo_reason_strict_accuracy": record,
        "longvideo_reason_wellformed_accuracy": record,
        "longvideo_reason_format_accuracy": record,
        f"longvideo_reason_{problem_type}_accuracy": record,
    }


def _aggregate(results: Sequence[MetricRecord], key: str = "score", problem_type: str | None = None, wellformed_only: bool = False) -> float:
    """Question-weighted accuracy as a percentage over the selected records."""
    selected = [result for result in results if (problem_type is None or result["problem_type"] == problem_type) and (not wellformed_only or result["wellformed"])]
    if not selected:
        return 0.0
    correct = sum(result[key] for result in selected)
    accuracy = 100.0 * correct / len(selected)
    label = problem_type or ("wellformed" if wellformed_only else key)
    eval_logger.info(f"LongVideo-Reason {label}: {accuracy:.2f}% ({int(correct)}/{len(selected)})")
    return accuracy


def longvideo_reason_aggregate_overall(results: Sequence[MetricRecord]) -> float:
    """Headline accuracy over all 1,000 rows, with the robust extractor.

    The answer key of this benchmark is NOT uniform: B=441, C=299, A=153,
    D=107. A model that always answers B scores 44.1%. Read this number
    against 44.1%, not against 25%.
    """
    return _aggregate(results)


def longvideo_reason_aggregate_strict(results: Sequence[MetricRecord]) -> float:
    """Accuracy under the official byte-exact comparison, for paper parity."""
    return _aggregate(results, key="strict_score")


def longvideo_reason_aggregate_wellformed(results: Sequence[MetricRecord]) -> float:
    """Accuracy over the 995 rows whose option block is intact."""
    return _aggregate(results, wellformed_only=True)


def longvideo_reason_aggregate_format(results: Sequence[MetricRecord]) -> float:
    """Share of completions matching <think>...</think><answer>...</answer>."""
    return _aggregate(results, key="format_score")


def longvideo_reason_aggregate_temporal(results: Sequence[MetricRecord]) -> float:
    return _aggregate(results, problem_type="temporal")


def longvideo_reason_aggregate_goal(results: Sequence[MetricRecord]) -> float:
    return _aggregate(results, problem_type="goal")


def longvideo_reason_aggregate_spatial(results: Sequence[MetricRecord]) -> float:
    return _aggregate(results, problem_type="spatial")


def longvideo_reason_aggregate_plot(results: Sequence[MetricRecord]) -> float:
    return _aggregate(results, problem_type="plot")
