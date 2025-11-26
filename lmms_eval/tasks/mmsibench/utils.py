"""
MMSIBench Task Utilities - Fully Aligned with EASI
Multi-modal Spatial Intelligence Benchmark
"""
import re
import string
from typing import Dict, List
import pandas as pd


ZW_RE = re.compile(r'[\u200b\u200c\u200d\ufeff]')
TAGGED_ANSWER_BLOCK = re.compile(
    r'<\s*answer\b[^>]*>\s*([A-Ja-j])(?:\s*[\.．:：\)\]】、])?.*?<\s*/\s*answer\s*>',
    flags=re.IGNORECASE | re.DOTALL
)


def mmsibench_doc_to_visual(doc: Dict) -> List[str]:
    """Get visual inputs"""
    image_path = doc.get("image_path", "")
    if isinstance(image_path, list):
        return image_path
    return [image_path] if image_path else []


def mmsibench_doc_to_text(doc: Dict) -> str:
    """Build prompt following EASI MMSIBench (line 39-74)"""
    question = doc.get("question", "")

    # Build options from columns A, B, C, D, etc. (line 49-53)
    options = {
        cand: doc[cand]
        for cand in string.ascii_uppercase
        if cand in doc and not pd.isna(doc.get(cand))
    }

    # Prompt format aligned with mmsi code base (line 56-58)
    options_prompt = 'Options: '
    for key, item in options.items():
        options_prompt += f'{key}: {item}, '

    # Handle hint (line 59)
    hint = doc.get('hint') if ('hint' in doc and not pd.isna(doc.get('hint'))) else None

    # Build prompt (line 61-66)
    prompt = ''
    if hint is not None:
        prompt += f'Hint: {hint}\n'
    prompt += f'{question}\n'
    if len(options):
        prompt += options_prompt

    # MMSI Direct format (line 69-72)
    post_prompt = (
        "Answer with the option's letter from the given choices directly. "
        "Enclose the option's letter within ``."
    )

    prompt = f'{prompt}\n{post_prompt}'
    return prompt


def mmsibench_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process results following EASI compute_mcq_score (line 72-87)"""
    pred_raw = results[0] if results else ""
    gt_raw = str(doc.get("answer", "")).strip()

    # Extract both pred and gt using can_match_option (line 78-79)
    pred = can_match_option(pred_raw)
    gt = can_match_option(gt_raw)

    # Use exact_match (line 82)
    acc = exact_match(pred, gt)
    return {"accuracy": acc}


def exact_match(pred, target) -> float:
    """EASI exact_match from cal_scores.py (line 13-16)"""
    pred = str(pred).strip().lower()
    target = str(target).strip().lower()
    return 1.0 if pred == target else 0.0


def can_match_option(
    answer_text: str,
    choices=None,
    tail_lines: int = 6,
    tail_window: int = 800
):
    """
    Full EASI matching_func.py can_match_option implementation (line 20-148)
    Extract single-choice option letter from model output.
    """
    # 1) Dynamic letter set (line 47-56)
    if not isinstance(answer_text, str):
        return False
    text = ZW_RE.sub('', answer_text.strip())

    if choices:
        letters_sorted = ''.join(sorted({str(c).strip().upper()[:1] for c in choices if str(c)}))
        letters = ''.join([ch for ch in 'ABCDEFGHIJ' if ch in letters_sorted]) or 'ABCDEF'
    else:
        letters = 'ABCDEF'

    # 2) Block-level <answer>...</answer> (line 58-61)
    m_block = TAGGED_ANSWER_BLOCK.search(text)
    if m_block:
        return m_block.group(1).upper()

    # 3) Tail anchors: before </answer> / </think> (line 63-76)
    tail_block = text[-tail_window:]
    PAT_ANS = re.compile(
        r'(?<![A-Za-z])[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?\s*</\s*answer\s*>' % letters,
        re.IGNORECASE
    )
    PAT_THINK = re.compile(
        r'(?<![A-Za-z])[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?\s*</\s*think\s*>' % letters,
        re.IGNORECASE
    )
    for pat in (PAT_ANS, PAT_THINK):
        m = pat.search(tail_block)
        if m:
            return m.group(1).upper()

    # Helpers for steps 4 & 5 (line 78-86)
    _PUNC_TIGHT = r"\.,:;!?\)\]】》」』，。；、：）】》」』"
    OPTION_LINE_PREFIX = re.compile(r'^(?:[*_>\-\s]*)(?:option|选项)\s+[A-J]\s*[:：]', re.IGNORECASE)
    MD_SINGLE = re.compile(r'^\s*[*_`>（）\[\]【】\(\)]*\s*([A-Fa-f])\s*[*_`（）\[\]【】\(\)]*\s*$')
    LINE_START_LABELED = re.compile(r'^\s*([A-F])\s*[\.．:：\)\]】、-]\s+', re.IGNORECASE)
    TOKEN_INLINE = re.compile(
        r'(?<![A-Za-z])[*_`（\[\{\(]*\s*([A-F])\s*[*_`）\]\}\)]*(?=$|[\s%s])' % _PUNC_TIGHT
    )

    def _pick_from_lines(lines):
        """Helper function (line 88-103)"""
        for line in reversed([ln.strip() for ln in lines if ln.strip()]):
            if OPTION_LINE_PREFIX.search(line):
                continue
            m = MD_SINGLE.fullmatch(line)
            if m:
                return m.group(1).upper()
            m = LINE_START_LABELED.match(line)
            if m:
                return m.group(1).upper()
            tokens = [t.upper() for t in TOKEN_INLINE.findall(line)]
            if tokens:
                uniq = sorted(set(tokens))
                if len(uniq) == 1:
                    return uniq[0]
        return None

    # 4) Last-lines after think-tail (line 105-115)
    if re.search(r'</\s*think\s*>', text, re.IGNORECASE):
        tail_segment = text[list(re.finditer(r'</\s*think\s*>', text, re.IGNORECASE))[-1].end():].strip()
    elif re.search(r'<\s*think\s*>', text, re.IGNORECASE):
        tail_segment = text[list(re.finditer(r'<\s*think\s*>', text, re.IGNORECASE))[-1].end():].strip()
    else:
        tail_segment = text

    pick = _pick_from_lines(tail_segment.splitlines()[-tail_lines:])
    if pick:
        return pick

    # 5) Last-lines in global tail window (line 117-120)
    pick = _pick_from_lines(text[-tail_window:].splitlines()[-tail_lines:])
    if pick:
        return pick

    # 6) Phrase-style conclusion (line 122-130)
    PHRASE_AFTER = re.compile(
        r'(?i)(?:final\s*answer|the\s*answer\s*is|answer(?:\s*is)?|correct\s*answer|'
        r'答案|最终答案|结论|所以|因此|我选(?:择)?|选择|选)\s*[:：>＝=]?\s*'
        r'[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?(?:\b|[.)、。])' % letters
    )
    m = PHRASE_AFTER.search(text)
    if m:
        return m.group(1).upper()

    # 7) Global fallback (line 132-146)
    cleaned_lines = []
    for ln in text.splitlines():
        if OPTION_LINE_PREFIX.search(ln):
            continue
        cleaned_lines.append(ln)
    cleaned = "\n".join(cleaned_lines)

    TOKEN_UPPER_GLOBAL = re.compile(
        r'(?<![A-Za-z])[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?(?![A-Za-z])' % letters
    )
    tokens = TOKEN_UPPER_GLOBAL.findall(cleaned)
    uniq = sorted(set(tokens))
    if len(uniq) == 1:
        return uniq[0]

    return False
