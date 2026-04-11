"""Utility functions for PhysReason benchmark evaluation.

Handles data preparation (zip download + JSONL generation), prompt
construction, image extraction, and scoring for open-ended physics
problem solving.

Dataset: https://huggingface.co/datasets/zhibei1204/PhysReason
Paper:   https://arxiv.org/abs/2502.12054
"""

import json
import os
import re
import zipfile

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

# ---------------------------------------------------------------------------
# Data preparation: download zip from HF, extract, build JSONL for datasets
# ---------------------------------------------------------------------------

_HF_REPO = "zhibei1204/PhysReason"
_ZIP_NAMES = {
    "full": "PhysReason-full.zip",
    "mini": "PhysReason-mini.zip",
}
_ZIP_ROOTS = {
    "full": "PhysReason_full",
    "mini": "PhysReason-mini",
}
_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_TASK_DIR, "physreason_data")

# Cache directory for extracted zip contents (images live here)
_CACHE_BASE = os.path.join(
    os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")),
    "physreason",
)


def _ensure_data_prepared(config_name: str) -> None:
    """Download the PhysReason zip, extract it, and build a JSONL file.

    The JSONL is placed at ``physreason_data/<config_name>/data.jsonl`` so
    that ``datasets.load_dataset`` auto-discovers it.  Images are extracted
    to ``$HF_HOME/physreason/<config_name>/`` and referenced by path in the
    ``image_path`` field.
    """
    split_dir = os.path.join(_DATA_DIR, config_name, "test")
    jsonl_path = os.path.join(split_dir, "data.jsonl")

    if os.path.exists(jsonl_path):
        return  # already prepared

    eval_logger.info(f"[physreason] Preparing {config_name} data ...")

    from huggingface_hub import hf_hub_download

    zip_file = hf_hub_download(
        repo_id=_HF_REPO,
        repo_type="dataset",
        filename=_ZIP_NAMES[config_name],
    )

    extract_dir = os.path.join(_CACHE_BASE, config_name)
    os.makedirs(extract_dir, exist_ok=True)

    zip_root = _ZIP_ROOTS[config_name]
    marker = os.path.join(extract_dir, ".extracted")
    if not os.path.exists(marker):
        eval_logger.info(f"[physreason] Extracting {zip_file} -> {extract_dir}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(extract_dir)
        with open(marker, "w") as f:
            f.write("done\n")

    problems_root = os.path.join(extract_dir, zip_root)
    if not os.path.isdir(problems_root):
        problems_root = extract_dir

    rows = []
    for problem_dir_name in sorted(os.listdir(problems_root)):
        problem_path = os.path.join(problems_root, problem_dir_name)
        if not os.path.isdir(problem_path):
            continue
        json_path = os.path.join(problem_path, "problem.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qs = data.get("question_structure", {})
        context = qs.get("context", "")

        sub_questions = []
        i = 1
        while f"sub_question_{i}" in qs:
            sub_questions.append(qs[f"sub_question_{i}"])
            i += 1

        answers = data.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]

        difficulty = data.get("difficulty", "unknown")

        image_list = data.get("question_image_list", [])
        image_rel = image_list[0] if image_list else ""
        # Absolute path to image within the extraction cache
        if image_rel:
            image_abs = os.path.join(problem_path, image_rel)
        else:
            image_abs = ""

        image_caption = data.get("image_captions", "")
        if isinstance(image_caption, list):
            image_caption = " ".join(image_caption)

        explanation_steps = data.get("explanation_steps", {})
        num_steps = sum(len(sq_steps) for sq_steps in explanation_steps.values() if isinstance(sq_steps, dict))

        rows.append(
            {
                "problem_id": problem_dir_name,
                "context": context,
                "sub_questions": sub_questions,
                "answers": answers,
                "difficulty": difficulty,
                "image_path": image_abs,
                "image_caption": image_caption or "",
                "num_sub_questions": len(sub_questions),
                "num_steps": num_steps,
            }
        )

    os.makedirs(split_dir, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_logger.info(f"[physreason] Wrote {len(rows)} rows to {jsonl_path}")


# Prepare both configs at import time so the JSONL files exist before
# datasets.load_dataset is called.
for _cfg in ("full", "mini"):
    try:
        _ensure_data_prepared(_cfg)
    except Exception:
        # Don't crash on import if only one config is needed and the
        # other can't be downloaded (e.g. offline).  The actual load
        # will fail later with a clear error.
        pass


# ---------------------------------------------------------------------------
# doc_to_visual / doc_to_text / scoring
# ---------------------------------------------------------------------------


def physreason_doc_to_visual(doc):
    """Load the problem image from the cached extraction directory."""
    image_path = doc.get("image_path", "")
    if not image_path or not os.path.exists(image_path):
        return []
    try:
        img = Image.open(image_path).convert("RGB")
        return [img]
    except Exception:
        eval_logger.warning(f"[physreason] Failed to load image: {image_path}")
        return []


def physreason_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build the prompt from context + sub-questions.

    Formats the physics problem with all sub-questions numbered,
    and asks for step-by-step reasoning with clearly labeled answers.
    """
    context = doc.get("context", "")
    sub_questions = doc.get("sub_questions", [])

    prompt_parts = []
    prompt_parts.append(context.strip())

    if sub_questions:
        prompt_parts.append("")
        for i, sq in enumerate(sub_questions, 1):
            prompt_parts.append(f"({i}) {sq.strip()}")

    prompt_parts.append("")
    prompt_parts.append("Solve each sub-question step by step. " "For each sub-question, show your reasoning and then give the final answer. " "Format each final answer as: Answer (N): <your answer>")

    return "\n".join(prompt_parts)


def _normalize_answer(text):
    """Normalize a LaTeX/math answer string for comparison."""
    s = text.strip()
    s = s.strip("$")
    s = re.sub(r"\\(?:text|mathrm|mathsf|mathit)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[,;:!\s]", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_answers_from_response(response, num_expected):
    """Try to extract numbered answers from model response."""
    answers = []

    pattern = r"Answer\s*\(?(\d+)\)?[:\s]+(.+?)(?=Answer\s*\(?\d|$)"
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

    if matches:
        matches.sort(key=lambda x: int(x[0]))
        for _, ans in matches:
            ans_clean = ans.strip().split("\n")[0].strip()
            ans_clean = ans_clean.rstrip(".")
            answers.append(ans_clean)

    if len(answers) < num_expected:
        boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
        if len(boxed) >= num_expected:
            answers = boxed[:num_expected]

    return answers


def physreason_process_results(doc, results):
    """Process model output and compare against ground truth answers."""
    prediction = results[0].strip() if results else ""
    answers_gt = doc.get("answers", [])
    num_sq = len(answers_gt)
    difficulty = doc.get("difficulty", "unknown")

    extracted = _extract_answers_from_response(prediction, num_sq)

    correct = 0
    for i, gt in enumerate(answers_gt):
        gt_norm = _normalize_answer(gt)
        if i < len(extracted):
            pred_norm = _normalize_answer(extracted[i])
            if gt_norm == pred_norm:
                correct += 1

    accuracy = correct / max(num_sq, 1)

    eval_result = {
        "problem_id": doc.get("problem_id", ""),
        "difficulty": difficulty,
        "num_sub_questions": num_sq,
        "correct": correct,
        "accuracy": accuracy,
    }

    return {"physreason_accuracy": eval_result}


def physreason_aggregate_results(results):
    """Aggregate per-problem accuracy into overall score."""
    if not results:
        eval_logger.warning("Empty results list for PhysReason. Returning 0.0")
        return 0.0

    accuracies = [r["accuracy"] for r in results]
    overall = float(np.mean(accuracies))

    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = []
        by_difficulty[d].append(r["accuracy"])

    for d in sorted(by_difficulty.keys()):
        acc = float(np.mean(by_difficulty[d]))
        count = len(by_difficulty[d])
        eval_logger.info(f"PhysReason [{d}]: {acc:.4f} ({count} problems)")

    eval_logger.info(f"PhysReason [overall]: {overall:.4f} ({len(results)} problems)")
    return overall
