# VideoMME Video Only ABCD Setting Utils
# Only provide video, no question or options - model guesses A/B/C/D

import re
from typing import Any, Dict, Optional

from lmms_eval.tasks.videomme.utils import videomme_doc_to_visual

_DEFAULT_SYSTEM_PROMPT = "You will receive ONLY a video. There is no question. " "Guess which option letter (A, B, C, or D) is correct. " "Output exactly one capital letter: A, B, C, or D."


def doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None) -> str:
    """Return empty text (no question provided)."""
    return ""


def doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None):
    """System-only instruction + user contains video only."""
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}

    # Get system prompt
    system_prompt = lmms_eval_specific_kwargs.get("system_prompt") or (lmms_eval_specific_kwargs.get("default", {}) or {}).get("system_prompt") or _DEFAULT_SYSTEM_PROMPT
    user_text = lmms_eval_specific_kwargs.get("user_text") or (lmms_eval_specific_kwargs.get("default", {}) or {}).get("user_text") or ""

    videos = videomme_doc_to_visual(doc)

    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}],
    }

    user_content = [{"type": "video", "url": v} for v in videos]
    user_content.append({"type": "text", "text": user_text})
    user_msg = {"role": "user", "content": user_content}

    return [system_msg, user_msg]


def _extract_abcd(s: str) -> str:
    """Extract A/B/C/D from string."""
    if not s:
        return ""
    s = s.strip().upper()

    # Prefer standalone token
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)

    # Fallback to first occurrence
    m = re.search(r"[ABCD]", s)
    return m.group(0) if m else ""


def process_results(doc: Dict[str, Any], results):
    """Process results for video-only format."""
    gt = doc.get("answer", "").upper()

    if not results:
        return {"acc_score": 0.0, "format_score": 0.0}

    acc = 0.0
    fmt = 0.0
    for pred in results:
        pred_letter = _extract_abcd(pred or "")
        if pred_letter:
            fmt += 1.0
        if pred_letter == gt:
            acc += 1.0

    n = len(results)
    return {"acc_score": acc / n, "format_score": fmt / n}
