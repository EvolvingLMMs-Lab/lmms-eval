"""ContPhy benchmark utilities for lmms-eval.

ContPhy: Continuum Physical Concept Learning and Reasoning from Videos (ICML 2024)
https://arxiv.org/abs/2402.06119

Handles multiple-choice QA over physics simulation videos covering four
scenarios: fluid dynamics, rope/pulley systems, cloth manipulation, and
soft-body ball interactions.
"""

import os
import re
from collections import defaultdict

import datasets
from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Scenario categories for per-scenario reporting
# ---------------------------------------------------------------------------
SCENARIOS = ["fluid", "rope", "cloth", "ball"]

QUESTION_TYPES = ["property", "predictive", "counterfactual", "goal_driven"]

# Option letters
OPTION_LETTERS = "ABCDEFGHIJ"


# ---------------------------------------------------------------------------
# Data directory resolution
# ---------------------------------------------------------------------------
def _get_data_dir() -> str:
    """Resolve the ContPhy data directory containing extracted video files.

    Checks in order:
      1. CONTPHY_DATA_DIR env var
      2. $HF_HOME/contphy/contphy_data/
    """
    explicit = os.getenv("CONTPHY_DATA_DIR", "").strip()
    if explicit:
        return os.path.expanduser(explicit)

    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
    return os.path.join(hf_home, "contphy", "contphy_data")


# ---------------------------------------------------------------------------
# process_docs: format options with letter prefixes
# ---------------------------------------------------------------------------
def contphy_process_docs(dataset):
    """Add letter-prefixed option strings and answer letter to each doc."""
    processed = []
    for doc in dataset:
        doc = dict(doc)
        options = doc.get("options", [])
        answer_text = doc.get("answer", "")

        # Build lettered options: "A. Yes", "B. No", ...
        lettered = []
        answer_letter = ""
        for i, opt in enumerate(options):
            letter = OPTION_LETTERS[i]
            lettered.append(f"{letter}. {opt}")
            if opt == answer_text:
                answer_letter = letter

        doc["options_str"] = "\n".join(lettered)
        doc["answer_letter"] = answer_letter
        doc["answer"] = answer_letter  # override for doc_to_target
        processed.append(doc)
    return datasets.Dataset.from_list(processed)


# ---------------------------------------------------------------------------
# doc_to_visual: locate the video file
# ---------------------------------------------------------------------------
def contphy_doc_to_visual(doc):
    """Return the video file path for this question."""
    video_id = doc.get("video_id", "")
    if not video_id:
        eval_logger.warning("ContPhy: no video_id in doc")
        return []

    data_dir = _get_data_dir()
    video_path = os.path.join(data_dir, video_id, "output_Full.mp4")

    if os.path.exists(video_path):
        return [video_path]

    # Try without subdirectory nesting
    alt_path = os.path.join(data_dir, f"{video_id}.mp4")
    if os.path.exists(alt_path):
        return [alt_path]

    eval_logger.warning(
        "ContPhy: video not found for {} (tried {})",
        video_id,
        video_path,
    )
    return []


# ---------------------------------------------------------------------------
# doc_to_text: build the MC prompt
# ---------------------------------------------------------------------------
def contphy_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format the question with lettered options."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    options_str = doc.get("options_str", "")

    return f"{pre_prompt}{question}\n{options_str}{post_prompt}"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def _extract_letter(text: str, num_options: int = 5) -> str:
    """Extract a single option letter from model response."""
    text = text.strip()

    # Common answer prefixes to strip
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "Best answer:",
        "Answer:",
    ]
    text_lower = text.lower()
    for prefix in answer_prefixes:
        if text_lower.startswith(prefix.lower()):
            text = text[len(prefix) :].strip(" :.-")
            break

    valid_letters = OPTION_LETTERS[:num_options]

    # Try to find a letter at the very start
    if text and text[0].upper() in valid_letters:
        # Make sure it's not part of a word (e.g. "A" vs "And")
        if len(text) == 1 or not text[1].isalpha():
            return text[0].upper()

    # Look for standalone letter (word boundary) or "letter." / "letter)"
    pattern = rf"\b([{valid_letters}])(?:\b|[.\)\]:,])"
    matches = re.findall(pattern, text.upper())
    if matches:
        return matches[0]

    # Fallback: any letter in the response
    fallback = re.findall(rf"[{valid_letters}]", text.upper())
    if fallback:
        return fallback[0]

    return ""


# ---------------------------------------------------------------------------
# process_results
# ---------------------------------------------------------------------------
def contphy_process_results(doc, results):
    """Compare model prediction against ground truth."""
    pred = results[0] if results else ""
    num_options = len(doc.get("options", []))
    pred_letter = _extract_letter(pred, num_options)

    gt_letter = doc.get("answer_letter", doc.get("answer", ""))

    return {
        "contphy_accuracy": {
            "video_id": doc.get("video_id", ""),
            "scenario": doc.get("scenario", "unknown"),
            "question_type": doc.get("question_type", "unknown"),
            "question_class": doc.get("question_class", "unknown"),
            "pred_answer": pred_letter,
            "answer": gt_letter,
        }
    }


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------
def contphy_aggregate_results(results):
    """Compute per-scenario, per-type, and overall accuracy."""
    # Per-scenario stats
    scenario_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    # Per-question-type stats
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    # Per scenario+type combo
    combo_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total = 0

    for result in results:
        scenario = result.get("scenario", "unknown")
        q_type = result.get("question_type", "unknown")
        correct = result["pred_answer"] == result["answer"]

        scenario_stats[scenario]["total"] += 1
        type_stats[q_type]["total"] += 1
        combo_stats[f"{scenario}/{q_type}"]["total"] += 1
        total += 1

        if correct:
            scenario_stats[scenario]["correct"] += 1
            type_stats[q_type]["correct"] += 1
            combo_stats[f"{scenario}/{q_type}"]["correct"] += 1
            total_correct += 1

    # Log per-scenario results
    for scenario in SCENARIOS:
        stats = scenario_stats.get(scenario, {"correct": 0, "total": 0})
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info(
                "ContPhy [{}]: {:.1f}% ({}/{})",
                scenario,
                acc,
                stats["correct"],
                stats["total"],
            )

    # Log per-type results
    for q_type in sorted(type_stats):
        stats = type_stats[q_type]
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info(
                "ContPhy [{}]: {:.1f}% ({}/{})",
                q_type,
                acc,
                stats["correct"],
                stats["total"],
            )

    # Log combo results
    for combo in sorted(combo_stats):
        stats = combo_stats[combo]
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info(
                "ContPhy [{}]: {:.1f}% ({}/{})",
                combo,
                acc,
                stats["correct"],
                stats["total"],
            )

    if total == 0:
        return 0.0

    overall = 100 * total_correct / total
    eval_logger.info(
        "ContPhy overall: {:.1f}% ({}/{})",
        overall,
        total_correct,
        total,
    )
    return overall
