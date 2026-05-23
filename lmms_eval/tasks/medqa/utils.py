from typing import Any, Dict, List

from lmms_eval.tasks.medevalkit.eval_utils import (
    judge_multi_choice,
    strip_thinking,
)

MEDQA_PROMPT = (
    "Answer the following multiple choice question. " "There is only one correct answer. " "The last line of your response should be in the format " "'Answer: $LETTER' (without quotes), " "where LETTER is one of A, B, C, D, or E."
)


def medqa_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> str:
    question = doc.get("question", "").strip()

    # Normalize options into A..E style lines
    options = doc.get("options")
    if isinstance(options, dict):
        ordered_keys = [k for k in ["A", "B", "C", "D", "E"] if k in options]
        options_block = "\n".join(f"{k}. {str(options[k]).strip()}" for k in ordered_keys)
    elif isinstance(options, list):
        letters = ["A", "B", "C", "D", "E"]
        options_block = "\n".join(f"{letters[i]}. {str(opt).strip()}" for i, opt in enumerate(options))
    else:
        options_block = str(options) if options is not None else ""

    return f"{MEDQA_PROMPT}\nQuestion: {question}\n{options_block}\n"


def medqa_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter."""
    # Prefer explicit answer letter field when present
    if "answer_idx" in doc and isinstance(doc["answer_idx"], str) and len(doc["answer_idx"]) == 1:
        return doc["answer_idx"].strip()

    # Some variants store the letter in "answer" directly
    ans = doc.get("answer")
    if isinstance(ans, str) and len(ans.strip()) == 1 and ans.strip().upper() in ["A", "B", "C", "D", "E"]:
        return ans.strip().upper()

    # If answer is provided as text, try to map back to a letter via options
    options = doc.get("options")
    if isinstance(options, dict) and isinstance(ans, str):
        for k, v in options.items():
            if isinstance(v, str) and v.strip() == ans.strip():
                return k

    # Fallback: unknown -> choose a dummy; evaluation will mark as incorrect
    return "A"


def medqa_doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    # Detect how many choices are present and return corresponding letters
    if isinstance(doc.get("options"), dict):
        present = [k for k in ["A", "B", "C", "D", "E"] if k in doc["options"]]
        if present:
            return present
    if isinstance(doc.get("options"), list):
        n = min(len(doc["options"]), 5)
        return ["A", "B", "C", "D", "E"][:n]
    # Default to 5-way if uncertain
    return ["A", "B", "C", "D", "E"]


def _get_choice_texts(doc: Dict[str, Any]) -> List[str]:
    """Return the option texts in A..E order."""
    options = doc.get("options")
    if isinstance(options, dict):
        return [str(options.get(k, "")).strip() for k in ["A", "B", "C", "D", "E"] if k in options]
    if isinstance(options, list):
        return [str(opt).strip() for opt in options]
    return []


def medqa_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    """Parse model output and compute accuracy against the gold letter."""
    response = strip_thinking(result[0]).strip()
    choice_texts = _get_choice_texts(doc)
    gt_ans = medqa_doc_to_target(doc)
    score = float(judge_multi_choice(choice_texts, gt_ans, response))
    return {"accuracy": score}
