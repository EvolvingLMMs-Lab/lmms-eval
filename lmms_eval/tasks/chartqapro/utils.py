"""ChartQAPro task utilities."""

import ast
import io
import re
from typing import Any, List, Optional

from anls import anls_score
from PIL import Image


def _load_image(doc):
    img = doc["image"]
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    return img.convert("RGB")


def chartqapro_doc_to_visual(doc):
    return [_load_image(doc)]


def _format_conversation(questions: List[str], answers: List[str]) -> str:
    lines: List[str] = []
    n = len(questions)
    for i in range(n - 1):
        lines.append(f"Q{i + 1}: {questions[i].strip()}")
        if i < len(answers):
            lines.append(f"A{i + 1}: {str(answers[i]).strip()}")
    lines.append(f"Q{n}: {questions[-1].strip()}")
    return "\n".join(lines)


def chartqapro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    questions = doc["Question"] if isinstance(doc["Question"], list) else [doc["Question"]]
    answers = doc.get("Answer") or []
    if not isinstance(answers, list):
        answers = [answers]
    qt = doc.get("Question Type", "")

    if qt == "Conversational" and len(questions) > 1:
        question = _format_conversation(questions, answers)
    else:
        question = questions[-1].strip()

    paragraph = (doc.get("Paragraph") or "").strip()
    if paragraph:
        question = f"Context: {paragraph}\n\n{question}"

    return f"{pre_prompt}{question}{post_prompt}"


def chartqapro_doc_to_target(doc):
    answers = doc["Answer"] if isinstance(doc["Answer"], list) else [doc["Answer"]]
    return str(answers[-1])


def _fix_list_format(item: str) -> Any:
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def _parse_to_list(text: str) -> Optional[List[str]]:
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text.strip().strip("%"))
    except (ValueError, AttributeError):
        return None


def _evaluate_single_answer(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
) -> float:
    t = target.strip().strip("%").strip()
    p = prediction.strip().strip("%").strip()
    t_f = _to_float(t)
    p_f = _to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0
    return float(anls_score(prediction=p.lower(), gold_labels=[t.lower()], threshold=0.5))


def _relaxed_correctness_chartqapro(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
    year_flags: Optional[List[str]] = None,
    always_use_exact_match: bool = False,
) -> float:
    fixed_t = _fix_list_format(target)
    t_list = _parse_to_list(str(fixed_t)) or [str(target)]
    p_list = _parse_to_list(str(prediction)) or [str(prediction)]
    n = len(t_list)
    if year_flags is None:
        year_flags = ["NO"] * max(n, len(p_list))
    elif len(year_flags) < n:
        year_flags = list(year_flags) * n

    scores: List[float] = []
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue
        t_item, p_item = t_list[idx], p_list[idx]
        flag = year_flags[idx] if idx < len(year_flags) else "NO"
        flag_cond = isinstance(flag, str) and flag.upper() == "YES"
        if flag_cond or always_use_exact_match:
            try:
                scores.append(1.0 if t_item.strip().lower() == p_item.strip().lower() else 0.0)
            except (AttributeError, ValueError):
                scores.append(0.0)
        else:
            scores.append(_evaluate_single_answer(t_item, p_item, max_relative_change))
    return sum(scores) / len(scores) if scores else 0.0


_SPLIT_TO_METRIC = {
    "Factoid": "relaxed_factoid",
    "Conversational": "relaxed_conversational",
    "Hypothetical": "relaxed_hypothetical",
    "Fact Checking": "relaxed_fact_checking",
    "Multi Choice": "relaxed_multi_choice",
}


def chartqapro_process_results(doc, results):
    pred = (results[0] or "").strip().strip(".").strip("\n")
    answers = doc["Answer"] if isinstance(doc["Answer"], list) else [doc["Answer"]]
    gt = str(answers[-1]).strip().strip(".").strip("\n")

    qt = doc.get("Question Type", "")
    year_flags = doc.get("Year") or []
    if not isinstance(year_flags, list):
        year_flags = [year_flags]
    if qt == "Conversational" and year_flags:
        year_flags = year_flags[-1:]

    always_use_exact_match = qt in ("Fact Checking", "Multi Choice")
    score = _relaxed_correctness_chartqapro(
        gt,
        pred,
        year_flags=year_flags,
        always_use_exact_match=always_use_exact_match,
    )

    out = {"relaxed_overall": score}
    metric = _SPLIT_TO_METRIC.get(qt)
    if metric is not None:
        out[metric] = score
    return out
