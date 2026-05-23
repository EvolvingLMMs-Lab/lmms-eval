from typing import Any, Dict, List

from lmms_eval.tasks.medevalkit.eval_utils import (
    judge_multi_choice,
    strip_thinking,
)

PUBMEDQA_PROMPT = (
    "Answer the following multiple choice question about the given "
    "biomedical research context. There is only one correct answer. "
    "The last line of your response should be in the format "
    "'Answer: $LETTER' (without quotes), where LETTER is one of A, B, or C."
)

ANSWER_LETTERS = ["A", "B", "C"]


def pubmedqa_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> str:
    """Format PubMedQA sample into a prompt with context, question, and options."""
    data = doc["data"]
    context = "\n".join(data["Context"])
    question = data["Question"]
    options = data["Options"]

    options_block = "\n".join(f"{letter}. {options[letter]}" for letter in ANSWER_LETTERS)

    return f"{PUBMEDQA_PROMPT}\n\n" f"Context: {context}\n\n" f"Question: {question}\n" f"{options_block}\n"


def pubmedqa_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter."""
    return doc["data"]["Correct Option"]


def pubmedqa_doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    """Return the list of choice letters."""
    return ANSWER_LETTERS


def pubmedqa_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    """Parse model output and compute accuracy against the gold letter."""
    response = strip_thinking(result[0]).strip()
    options = doc["data"]["Options"]
    choice_texts = [str(options[letter]) for letter in ANSWER_LETTERS]
    gt_ans = pubmedqa_doc_to_target(doc)
    score = float(judge_multi_choice(choice_texts, gt_ans, response))
    return {"accuracy": score}
