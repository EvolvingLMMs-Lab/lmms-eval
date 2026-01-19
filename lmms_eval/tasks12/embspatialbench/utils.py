"""
EmbSpatialBench (OmniSpatial) Task Utilities - Aligned with EASI
"""
import ast
import json
import re
import string
from typing import Dict, List


ZW_RE = re.compile(r'[\u200b\u200c\u200d\ufeff]')


def embspatialbench_doc_to_visual(doc: Dict) -> List[str]:
    """Get visual inputs"""
    image_path = doc.get("image_path", "")
    if isinstance(image_path, list):
        return image_path
    return [image_path] if image_path else []


def embspatialbench_doc_to_text(doc: Dict) -> str:
    """Build prompt following EASI EmbSpatialBench (line 27-57)"""
    question = doc.get("question", "")
    options = doc.get("options", [])

    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = ast.literal_eval(options)

    UpperLetters = list(string.ascii_uppercase)
    option_text = "\n".join(
        f"{UpperLetters[i]}: {options[i]}"
        for i in range(len(options))
    )

    # EmbSpatial uses SiteBench prompt format (line 52-57)
    prompt = ''
    prompt += "Question: " + question + "\n"
    prompt += "Options:\n" + option_text + "\n"
    post_prompt = "Give me the answer letter directly. The best answer is:"
    prompt += post_prompt

    return prompt


def embspatialbench_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process results following EASI compute_mcq_score"""
    pred_raw = results[0] if results else ""
    gt_raw = str(doc.get("answer", "")).strip()

    pred = can_match_option(pred_raw)
    gt = can_match_option(gt_raw)

    acc = exact_match(pred, gt)
    return {"accuracy": acc}


def exact_match(pred, target) -> float:
    """EASI exact_match"""
    pred = str(pred).strip().lower()
    target = str(target).strip().lower()
    return 1.0 if pred == target else 0.0


def can_match_option(answer_text: str):
    """Extract option letter"""
    if not isinstance(answer_text, str):
        return False

    text = ZW_RE.sub('', answer_text.strip())
    letters = 'ABCDEFGHIJ'

    # <answer>X</answer>
    m = re.search(
        r'<\s*answer\b[^>]*>\s*([A-Ja-j])(?:\s*[\.．:：\)\]】、])?.*?<\s*/\s*answer\s*>',
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group(1).upper()

    # Phrase-style
    m = re.search(
        r'(?i)(?:final\s*answer|the\s*answer\s*is|answer(?:\s*is)?|best\s*answer)\s*[:：>＝=]?\s*'
        r'[\(\[\{（【]?\s*([' + letters + r'])\s*[\)\]\}）】]?',
        text
    )
    if m:
        return m.group(1).upper()

    # Single letter
    m = re.match(r'^\s*([A-J])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Standalone
    m = re.search(r'(?<![A-Za-z])\s*([A-J])\s*(?![A-Za-z])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return False
