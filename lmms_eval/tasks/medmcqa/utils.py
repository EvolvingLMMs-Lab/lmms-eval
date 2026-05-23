from typing import Any, Dict, List

from lmms_eval.tasks.medevalkit.eval_utils import (
    judge_multi_choice,
    strip_thinking,
)

MEDMCQA_PROMPT = (
    "Answer the following multiple choice medical question. " "There is only one correct answer. " "The last line of your response should be in the format " "'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, or D."
)

ANSWER_LETTERS = ["A", "B", "C", "D"]
OPTION_KEYS = ["opa", "opb", "opc", "opd"]


def medmcqa_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> str:
    """Format MedMCQA sample into a prompt with question and options."""
    question = doc["question"]
    options_block = "\n".join(f"{letter}. {doc[key]}" for letter, key in zip(ANSWER_LETTERS, OPTION_KEYS))
    return f"{MEDMCQA_PROMPT}\nQuestion: {question}\n{options_block}\n"


def medmcqa_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter."""
    return ANSWER_LETTERS[doc["cop"]]


def medmcqa_doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    """Return the list of choice letters."""
    return ANSWER_LETTERS


def medmcqa_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    """Parse model output and compute accuracy against the gold letter."""
    response = strip_thinking(result[0]).strip()
    choice_texts = [str(doc[key]) for key in OPTION_KEYS]
    gt_ans = medmcqa_doc_to_target(doc)
    score = float(judge_multi_choice(choice_texts, gt_ans, response))
    return {"accuracy": score}
