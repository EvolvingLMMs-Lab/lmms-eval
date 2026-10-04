import os
import re
import unicodedata
from pathlib import Path
from typing import Dict

import yaml
from loguru import logger as eval_logger

from lmms_eval.utils import resolve_cache_dir

with open(Path(__file__).parent / "timeblind.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
cache_dir = resolve_cache_dir(cache_name, base_dir=hf_home)

SUFFIX_FOR_VQA = {"yes_no": "Please output Yes or No.", "multiple_choice": "Please output A or B."}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer(output_string: str, task_type: str = "yes_no") -> int:
    """
    Extract answer from model output.
    Returns: 1 (yes/A), 0 (no/B), -1 (invalid)
    """
    if not output_string or not str(output_string).strip():
        return -1

    if task_type not in ("yes_no", "multiple_choice"):
        raise ValueError("task_type must be 'yes_no' or 'multiple_choice'")

    text = _normalize(output_string)

    if task_type == "yes_no":
        patterns = [
            r"(?i)(?:final(?: answer)?|answer|prediction)\s*[:：]\s*(yes|no)\b",
            r"(?i)\b(yes|no)\b(?=[\s\.\,\!\?\)]|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return 1 if m.group(1).lower() == "yes" else 0
        return -1

    else:  # multiple_choice
        patterns = [
            r"(?i)(?:final(?: answer)?|answer|prediction)\s*[:：]\s*([AB])\b",
            r"(?i)(?:option|choice)\s*[:：]?\s*([AB])\b",
            r"(?i)[\(\[\{]\s*([AB])\s*[\)\]\}]",
            r"(?i)\b([AB])\s*[\.\)]\b",
            r"(?i)(?<![A-Za-z0-9/])([AB])(?![A-Za-z0-9/])",
        ]
        for i, pat in enumerate(patterns):
            if i < len(patterns) - 1:
                m = re.search(pat, text)
                if m:
                    return 1 if m.group(1).upper() == "A" else 0
            else:
                answer_keyword = re.search(r"(?i)\b(final(?: answer)?|answer|prediction)\b", text)
                if answer_keyword:
                    before = text[: answer_keyword.start()]
                    m_before = re.search(pat, before)
                    if m_before:
                        return 1 if m_before.group(1).upper() == "A" else 0

                    after = text[answer_keyword.end() :]
                    m_after = re.search(pat, after)
                    if m_after:
                        return 1 if m_after.group(1).upper() == "A" else 0

                m = re.search(pat, text)
                if m:
                    return 1 if m.group(1).upper() == "A" else 0
        return -1


def get_scores(scores):
    """
    Calculate Q_Acc, V_Acc, Acc, I_Acc from answer results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and videos:
        - "q0_i0" means question_0 on video_0
        - "q0_i1" means question_0 on video_1
        - "q1_i0" means question_1 on video_0
        - "q1_i1" means question_1 on video_1

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'Q_Acc': Average question acc
            - 'V_Acc': Average video acc
            - 'Acc': Average binary VQA acc
            - 'I_Acc': Average instance (group) acc
    """
    Q_Acc = V_Acc = Acc = I_Acc = 0.0
    num_samples = len(scores)
    if num_samples == 0:
        return {"Q_Acc": 0.0, "V_Acc": 0.0, "Acc": 0.0, "I_Acc": 0.0}

    def calc_video_score(r):
        score = 0
        if isinstance(r, dict):
            if r["q0_i0"] == 1.0 and r["q1_i0"] == 0.0:
                score += 1
            if r["q1_i1"] == 1.0 and r["q0_i1"] == 0.0:
                score += 1
        else:
            if r[0] == 1.0 and r[2] == 0.0:
                score += 1
            if r[3] == 1.0 and r[1] == 0.0:
                score += 1
        return score

    def calc_question_score(r):
        score = 0
        if isinstance(r, dict):
            if r["q0_i0"] == 1.0 and r["q0_i1"] == 0.0:
                score += 1
            if r["q1_i1"] == 1.0 and r["q1_i0"] == 0.0:
                score += 1
        else:
            if r[0] == 1.0 and r[1] == 0.0:
                score += 1
            if r[3] == 1.0 and r[2] == 0.0:
                score += 1
        return score

    def calc_binary_score(r):
        if isinstance(r, dict):
            return sum(
                [
                    r["q0_i0"] == 1.0,
                    r["q0_i1"] == 0.0,
                    r["q1_i0"] == 0.0,
                    r["q1_i1"] == 1.0,
                ]
            )
        return sum([r[0] == 1.0, r[1] == 0.0, r[2] == 0.0, r[3] == 1.0])

    def calc_instance_score(r):
        return 1 if calc_question_score(r) == 2 and calc_video_score(r) == 2 else 0

    results = scores.values() if isinstance(scores, dict) else scores
    for r in results:
        Q_Acc += calc_question_score(r)
        V_Acc += calc_video_score(r)
        Acc += calc_binary_score(r)
        I_Acc += calc_instance_score(r)

    return {"Q_Acc": Q_Acc / (num_samples * 2), "V_Acc": V_Acc / (num_samples * 2), "Acc": Acc / (num_samples * 4), "I_Acc": I_Acc / num_samples}


def timeblind_doc_to_visual(doc):
    video_rel = doc["video_path"]
    if video_rel.startswith("TimeBlind/"):
        video_rel = video_rel[len("TimeBlind/") :]
    video_path = os.path.join(cache_dir, video_rel)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video path: {video_path} does not exist, please check")
    return [video_path]


def timeblind_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    question_type = doc["type"]
    if question_type in SUFFIX_FOR_VQA:
        question = question + " " + SUFFIX_FOR_VQA[question_type]
    return question


def timeblind_process_results(doc, results):
    """
    Args:
        doc: an instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric payload
    """
    pred = results[0] if results else ""
    score = extract_answer(pred, task_type=doc["type"])
    payload = {"index": doc["index"], "score": score}
    return {
        "timeblind_I_Acc": payload,
        "timeblind_Q_Acc": payload,
        "timeblind_V_Acc": payload,
        "timeblind_Acc": payload,
    }


def _aggregate_results(results) -> Dict[str, float]:
    """Group per-question results into instances (4 questions each) and score them.

    Each instance covers indices [4k, 4k+3] with roles (q0_i0, q0_i1, q1_i0, q1_i1).
    Instances with missing predictions are treated as all-invalid (-1), matching
    the official TimeBlind evaluation.
    """
    groups: Dict[int, Dict[int, float]] = {}
    for r in results:
        idx = r["index"]
        groups.setdefault(idx // 4, {})[idx % 4] = float(r["score"])

    answers: Dict[str, Dict[str, float]] = {}
    for sample_id, group in groups.items():
        if len(group) != 4:
            eval_logger.warning(f"TimeBlind instance {sample_id} is incomplete ({len(group)}/4 predictions); scoring it as invalid.")
            group = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}
        answers[str(sample_id)] = {"q0_i0": group[0], "q0_i1": group[1], "q1_i0": group[2], "q1_i1": group[3]}

    return get_scores(answers)


def timeblind_aggregate_results_I_Acc(results):
    return _aggregate_results(results)["I_Acc"]


def timeblind_aggregate_results_Q_Acc(results):
    return _aggregate_results(results)["Q_Acc"]


def timeblind_aggregate_results_V_Acc(results):
    return _aggregate_results(results)["V_Acc"]


def timeblind_aggregate_results_Acc(results):
    return _aggregate_results(results)["Acc"]
