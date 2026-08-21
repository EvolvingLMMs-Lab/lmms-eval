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

CACHE_DIR_NAME = "mmr_v"
VIDEO_ENV_VARS = ("MMR_V_VIDEO_DIR", "MMR_V_ROOT")
REASONING_TAG_PAIRS = [["<think>", "</think>"], ["<thinking>", "</thinking>"]]

# The 10 abilityType_L2 labels and the 6 videoType labels of the test split.
ABILITY_TYPES = (
    "Theme Understanding",
    "Metaphor Understanding",
    "Emotion Recognition",
    "Implicit Symbol",
    "Causal Reasoning",
    "Counterintuitive Reasoning",
    "Video Type and Intent",
    "Sequential Structure Reasoning",
    "Cross-modal Creative Transfer",
    "Comment Matching",
)
VIDEO_TYPES = ("Animation", "movie", "TV", "Life", "Art", "Philosophy")

# Options ship with their own "(A) " prefix and a few items skip a letter, so the
# valid letters are read back from the option strings instead of being rebuilt
# from the option count.
_OPTION_LETTER = re.compile(r"^\s*\(([A-Za-z])\)")
# Official answer formats, tried before the shared multiple-choice extractor.
_BRACKET_ANSWER = re.compile(r"\[\[\s*([A-Za-z])\s*\]\]")
_BOXED_ANSWER = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _options(doc: Document) -> list[str]:
    """Return the non-empty option strings of one example, prefixes included."""
    return [str(option).strip() for option in doc["options"] if str(option).strip()]


def _choices(doc: Document) -> list[str]:
    """Return the option letters this example actually offers."""
    letters = []
    for option in _options(doc):
        match = _OPTION_LETTER.match(option)
        if match:
            letters.append(match.group(1).upper())
    return letters


def mmr_v_doc_to_visual(doc: Document) -> list[str]:
    """Resolve the local video for one MMR-V example."""
    video_path = resolve_media_reference(
        str(doc["video"]),
        media_type="video",
        cache_dir=CACHE_DIR_NAME,
        env_vars=VIDEO_ENV_VARS,
        extra_subdirs=("videos", "videos_extracted"),
    )
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"MMR-V video not found: {doc['video']}. Concatenate the videos.tar.part.* archives of JokerJan/MMR-VBench " "and set MMR_V_VIDEO_DIR to the extracted video directory if needed.")
    return [video_path]


def _format_question(doc: Document) -> str:
    return f"{doc['question']}\nOptions:\n" + "\n".join(_options(doc))


def mmr_v_doc_to_text(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Format the multiple-choice prompt; the answer format comes from post_prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    instruction = "Please select the best answer to the following multiple-choice question based on the video. " "Only one option is the most accurate answer in relation to the question and the video."
    return f"{kwargs.get('pre_prompt', '')}{instruction}\n{_format_question(doc)}{kwargs.get('post_prompt', '')}"


def extract_mmr_v_answer(response: str, choices: Sequence[str]) -> str:
    """Extract one offered option letter, preferring the official "[[X]]" format."""
    text = strip_reasoning_tags(response or "", REASONING_TAG_PAIRS)
    for pattern in (_BRACKET_ANSWER, _BOXED_ANSWER):
        matches = pattern.findall(text)
        if matches and matches[-1].upper() in choices:
            return matches[-1].upper()
    return extract_mcq_answer(text, choices=list(choices))


def mmr_v_process_results(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Parse a prediction and emit the overall, ability and video-type records."""
    ability_type = str(doc["abilityType_L2"])
    video_type = str(doc["videoType"])
    if ability_type not in ABILITY_TYPES:
        raise ValueError(f"Unknown MMR-V abilityType_L2: {ability_type!r}")
    if video_type not in VIDEO_TYPES:
        raise ValueError(f"Unknown MMR-V videoType: {video_type!r}")

    prediction = results[0] if results else ""
    parsed_prediction = extract_mmr_v_answer(str(prediction), _choices(doc))
    answer = re.sub(r"[^A-Za-z]", "", str(doc["correctAnswer"])).upper()
    record = {
        "question_idx": int(doc["question_idx"]),
        "video": str(doc["video"]),
        "ability_type": ability_type,
        "ability_type_l3": str(doc["abilityType_L3"]),
        "video_type": video_type,
        "prediction": prediction,
        "pred_answer": parsed_prediction,
        "answer": answer,
        "score": float(parsed_prediction == answer),
    }
    return {
        "mmr_v_overall_accuracy": record,
        f"mmr_v_{_slug(ability_type)}_accuracy": record,
        f"mmr_v_{_slug(video_type)}_video_accuracy": record,
    }


def _aggregate_accuracy(results: Sequence[MetricRecord], ability_type: str | None = None, video_type: str | None = None) -> float:
    """Compute question-weighted accuracy as a percentage over the selected records."""
    selected = [result for result in results if (ability_type is None or result["ability_type"] == ability_type) and (video_type is None or result["video_type"] == video_type)]
    if not selected:
        return 0.0
    correct = sum(result["score"] for result in selected)
    accuracy = 100.0 * correct / len(selected)
    label = ability_type or video_type or "overall"
    eval_logger.info(f"MMR-V {label} accuracy: {accuracy:.2f}% ({int(correct)}/{len(selected)})")
    return accuracy


def mmr_v_aggregate_overall(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results)


def mmr_v_aggregate_theme_understanding(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Theme Understanding")


def mmr_v_aggregate_metaphor_understanding(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Metaphor Understanding")


def mmr_v_aggregate_emotion_recognition(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Emotion Recognition")


def mmr_v_aggregate_implicit_symbol(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Implicit Symbol")


def mmr_v_aggregate_causal_reasoning(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Causal Reasoning")


def mmr_v_aggregate_counterintuitive_reasoning(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Counterintuitive Reasoning")


def mmr_v_aggregate_video_type_and_intent(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Video Type and Intent")


def mmr_v_aggregate_sequential_structure_reasoning(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Sequential Structure Reasoning")


def mmr_v_aggregate_cross_modal_creative_transfer(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Cross-modal Creative Transfer")


def mmr_v_aggregate_comment_matching(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, ability_type="Comment Matching")


def mmr_v_aggregate_animation_video(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, video_type="Animation")


def mmr_v_aggregate_movie_video(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, video_type="movie")


def mmr_v_aggregate_tv_video(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, video_type="TV")


def mmr_v_aggregate_life_video(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, video_type="Life")


def mmr_v_aggregate_art_video(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, video_type="Art")


def mmr_v_aggregate_philosophy_video(results: Sequence[MetricRecord]) -> float:
    return _aggregate_accuracy(results, video_type="Philosophy")
