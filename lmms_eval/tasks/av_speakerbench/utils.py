import ast
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from loguru import logger as eval_logger

DEFAULT_PRE_PROMPT = ""
DEFAULT_POST_PROMPT = "The best answer is:"

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface")
CACHE_ROOT = os.path.expanduser(os.getenv("AV_SPEAKERBENCH_CACHE", os.path.join(HF_HOME, "AV-SpeakerBench")))


def _parse_choices(raw_choices: Union[str, List, Tuple]) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Args:
        raw_choices: A Python list literal or list/tuple from the CSV, e.g., ["A. foo", "B. bar"].

    Returns:
        normalized_choices: Choices as strings (unchanged aside from stripping).
        labels: Option labels, typically ["A", "B", ...].
        index2ans: Mapping from label to the answer text (prefix removed).
    """
    if isinstance(raw_choices, str):
        choices = ast.literal_eval(raw_choices)
    elif isinstance(raw_choices, (list, tuple)):
        choices = list(raw_choices)
    else:
        choices = [str(raw_choices)]

    normalized_choices = [str(c).strip() for c in choices]
    labels = []
    index2ans = {}
    for i, text in enumerate(normalized_choices):
        label = text[0].upper() if text else chr(ord("A") + i)
        body = text[2:].strip() if len(text) >= 2 and text[1] in [".", ")", ":"] else text
        labels.append(label)
        index2ans[label] = body
    return normalized_choices, labels, index2ans


def av_speakerbench_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter (A/B/C/D) from the doc."""
    return str(doc["answer"]).strip().upper()


def _build_prompt(doc: Dict[str, Any], modality_instruction: str, lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Build the textual prompt: modality instruction + question + choices + post_prompt.
    """
    pre_prompt = DEFAULT_PRE_PROMPT
    post_prompt = DEFAULT_POST_PROMPT
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)

    choices, _, _ = _parse_choices(doc["choices"])
    prompt_parts = [
        pre_prompt,
        modality_instruction,
        doc["question"],
        "\n",
        "\n".join(choices),
        "\n",
        post_prompt,
    ]
    return "".join(prompt_parts).strip()


def av_speakerbench_doc_to_text_av(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Prompt for audiovisual mode."""
    modality_instruction = "Select the best answer to the following multiple-choice question based on the audiovisual clip. " "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    return _build_prompt(doc, modality_instruction, lmms_eval_specific_kwargs)


def av_speakerbench_doc_to_text_audio(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Prompt for audio-only mode."""
    modality_instruction = "Select the best answer to the following multiple-choice question based on the audio clip. " "Focus on the audio and respond with only the letter (A, B, C, or D).\n"
    return _build_prompt(doc, modality_instruction, lmms_eval_specific_kwargs)


def av_speakerbench_doc_to_text_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Prompt for visual-only mode."""
    modality_instruction = "Select the best answer to the following multiple-choice question based on the silent visual clip. " "Rely on the visuals only and respond with the letter (A, B, C, or D).\n"
    return _build_prompt(doc, modality_instruction, lmms_eval_specific_kwargs)


def av_speakerbench_doc_to_audiovisual(doc: Dict[str, Any]) -> List[str]:
    """Return the audiovisual clip path, joined with CACHE_ROOT."""
    path = doc.get("audio_visual_path") or doc.get("audiovisual_path") or doc.get("video_path")
    if not path:
        return []
    path = os.path.join(CACHE_ROOT, path)
    path = os.path.expanduser(path)
    return [path]


def av_speakerbench_doc_to_audio(doc: Dict[str, Any]) -> List[Union[Dict[str, Any], str]]:
    """
    Return audio as a dict with waveform and sampling_rate for unified audio handling.
    Falls back to the absolute path if loading fails.
    """
    path = doc.get("audio_path")
    if not path:
        return []
    abs_path = os.path.expanduser(os.path.join(CACHE_ROOT, path))
    try:
        audio, sr = librosa.load(abs_path, sr=None, mono=True)
        return [{"array": audio.astype(np.float32), "sampling_rate": sr}]
    except Exception as e:
        eval_logger.warning(f"Failed to load audio {abs_path}: {e}")
        return [abs_path]


def av_speakerbench_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """Return the visual-only clip path, joined with CACHE_ROOT."""
    path = doc.get("visual_path") or doc.get("video_path")
    if not path:
        return []
    path = os.path.join(CACHE_ROOT, path)
    path = os.path.expanduser(path)
    return [path]


def parse_multi_choice_response(response: Optional[str], all_choices: List[str]) -> str:
    """
    Parse the model text into a choice label.

    Args:
        response: Raw model string.
        all_choices: Valid labels, e.g., ["A","B","C","D"].

    Returns:
        The parsed label (uppercased). Falls back to the first label if nothing matches.
    """
    response = response or ""
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "Based",
        "Correct answer",
        "\u261e",
        "<|im_end|>",
    ]
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")

    response = response.strip()
    response = re.sub(r"[.,:!\"'`;\\/?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]", " ", response)
    tokens = response.split()

    for token in tokens:
        if token in all_choices or token.upper() in all_choices:
            return token.upper()

    # Fallback: pick the first valid choice to avoid empty return
    return all_choices[0]


def av_speakerbench_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compare model prediction with ground truth.

    Args:
        doc: Single dataset example.
        results: List of model outputs for the example.

    Returns:
        Dict keyed by metric name with per-sample fields used in aggregation.
    """
    pred = results[0] if results else ""
    _, labels, _ = _parse_choices(doc["choices"])
    answer = parse_multi_choice_response(pred, labels)
    gt_answer = av_speakerbench_doc_to_target(doc)
    score = 1.0 if answer == gt_answer else 0.0

    return {
        "av_speakerbench_score": {
            "question_id": doc.get("question_id", doc.get("id", "")),
            "task_id": doc.get("task_id", "unknown"),
            "score": score,
        }
    }


def av_speakerbench_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate per-sample scores into per-task_id accuracy and overall accuracy.

    Args:
        results: List of dicts produced by av_speakerbench_process_results for the metric.

    Returns:
        Overall accuracy (%).
    """
    task_score_map = defaultdict(list)
    total_score = 0.0

    for result in results:
        task_id = result.get("task_id", "unknown")
        score = float(result.get("score", 0.0))
        task_score_map[task_id].append(score)
        total_score += score

    overall = (total_score / len(results) * 100.0) if results else 0.0

    for task_id, scores in sorted(task_score_map.items()):
        task_avg = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"AV-SpeakerBench ({task_id}): {task_avg:.2f}")

    eval_logger.info(f"Overall AV-SpeakerBench: {overall:.2f}")
    return overall
