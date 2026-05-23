from typing import Any, Dict, List

from lmms_eval.tasks.medevalkit.eval_utils import agg_mean  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import no_image_doc_to_visual  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import (
    judge_open,
    judge_yesno,
    parse_response,
)


def vqa_rad_doc_to_visual(doc: Dict[str, Any]):
    return [doc["image"].convert("RGB")]


def vqa_rad_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] = None,
) -> str:
    question = doc["question"].strip()
    answer = doc["answer"].lower().strip()
    if answer in ("yes", "no"):
        return question + "\nPlease answer 'yes' or 'no' (no extra output)."
    else:
        return question + "\nPlease answer the question concisely."


def vqa_rad_doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["answer"].lower().strip()


def vqa_rad_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Any]:
    raw_response = result[0] if result else ""
    response = parse_response(raw_response).lower().strip()
    answer = doc["answer"].lower().strip()
    is_close = answer in ("yes", "no")

    if is_close:
        correct = float(judge_yesno(answer, response))
        return {
            "close_accuracy": correct,
            "open_em": None,
            "bleu1": None,
            "bleu2": None,
            "bleu3": None,
            "bleu4": None,
            "rouge1": None,
            "rouge2": None,
            "rougel": None,
            "f1": None,
        }
    else:
        m = judge_open(answer, response)
        return {
            "close_accuracy": None,
            "open_em": m["em"],
            "bleu1": m["bleu1"],
            "bleu2": m["bleu2"],
            "bleu3": m["bleu3"],
            "bleu4": m["bleu4"],
            "rouge1": m["rouge1"],
            "rouge2": m["rouge2"],
            "rougel": m["rougel"],
            "f1": m["f1"],
        }
