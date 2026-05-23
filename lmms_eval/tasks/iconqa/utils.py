import logging
import re

logger = logging.getLogger("lmms-eval")

# ICON-QA mixes three subtypes under one task:
#   choose_img   (54%): prompt asks for option letter ("A. The first image. / B. ..."),
#                       gold answer is a filename ("choice_1.png")
#   choose_txt   (29%): prompt asks for option letter, gold answer is the choice text ("7")
#   fill_in_blank(17%): prompt asks for a single word/phrase, gold answer is free text
#
# Scoring contract:
#   choose_img / choose_txt -> extract first option letter the model emitted, map it to
#                              the corresponding choice (filename or text), exact-match
#                              that against gold answer.
#   fill_in_blank           -> ANLS (normalized Levenshtein) with the standard 0.5
#                              threshold used by DocVQA-family ANLS implementations.

ANLS_THRESHOLD = 0.5


def options_to_str(options_prompt):
    option_prompt_str = ""
    for i, option in enumerate(options_prompt):
        option_choice = chr(ord("A") + i)
        option_prompt_str += f"{option_choice}. {option}\n"

    option_prompt_str = option_prompt_str.rstrip("\n")
    return option_prompt_str


def doc_to_visual(doc):
    image_list = []
    if "query_image" in doc:
        image_list.append(doc["query_image"].convert("RGB"))
    for i in range(5):
        id = f"choice_image_{i}"
        if id in doc and doc[id] is not None:
            image_list.append(doc[id].convert("RGB"))
    assert len(image_list) < 6, "Maximum 5 images allowed for ICON-QA"
    return image_list


def doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    ques_type = doc["ques_type"]
    options_prompt = []

    if ques_type == "choose_img":
        options_prompt.append("The first image.")
        options_prompt.append("The second image.")

        options_str = options_to_str(options_prompt)
        full_prompt = f"{lmms_eval_specific_kwargs['pre_prompt']}{lmms_eval_specific_kwargs['statement']}{lmms_eval_specific_kwargs['options_statement'].format(question=question, options=options_str)}"

    elif ques_type == "choose_txt":
        choices = doc["choices"].split(",")
        for i, choice in enumerate(choices):
            options_prompt.append(f"{choice}")

        options_str = options_to_str(options_prompt)
        full_prompt = f"{lmms_eval_specific_kwargs['pre_prompt']}{lmms_eval_specific_kwargs['statement']}{lmms_eval_specific_kwargs['options_statement'].format(question=question, options=options_str)}"

    elif ques_type == "fill_in_blank":
        full_prompt = f"{lmms_eval_specific_kwargs['pre_prompt']}{lmms_eval_specific_kwargs['statement']}{lmms_eval_specific_kwargs['freeform_statement'].format(question=question)}"

    return full_prompt


_LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def _extract_letter(text, n_choices):
    """Return the first option letter (A..A+n_choices-1) present in ``text``, or None."""
    text = str(text).strip()
    if not text:
        return None
    valid = set("ABCDE"[:n_choices])
    # Whole-string single letter
    if len(text) == 1 and text.upper() in valid:
        return text.upper()
    # Bounded letter token (matches "A", "A.", "(A)", "Option A", etc.)
    for m in _LETTER_RE.finditer(text):
        letter = m.group(1).upper()
        if letter in valid:
            return letter
    return None


def _anls_score(pred, gold, threshold=ANLS_THRESHOLD):
    """Normalized Levenshtein similarity, thresholded at ``threshold``."""
    pred = str(pred).strip().lower()
    gold = str(gold).strip().lower()
    if not gold:
        return 0.0
    if pred == gold:
        return 1.0
    try:
        from rapidfuzz.distance import Levenshtein

        dist = Levenshtein.distance(pred, gold)
    except ImportError:
        # Pure-Python fallback — O(len(pred)*len(gold)) but accurate.
        m, n = len(pred), len(gold)
        if m == 0:
            return 0.0
        prev = list(range(n + 1))
        for i, c1 in enumerate(pred, 1):
            cur = [i] + [0] * n
            for j, c2 in enumerate(gold, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c1 != c2))
            prev = cur
        dist = prev[n]
    max_len = max(len(pred), len(gold))
    sim = 1.0 - dist / max_len if max_len > 0 else 0.0
    return sim if sim >= threshold else 0.0


def _score_one(doc, pred_raw):
    ques_type = doc["ques_type"]
    gold = str(doc["answer"]).strip()
    pred_raw = str(pred_raw).strip()

    if ques_type in ("choose_img", "choose_txt"):
        choices = doc["choices"].split(",")
        letter = _extract_letter(pred_raw, len(choices))
        if letter is not None:
            pred_norm = choices[ord(letter) - ord("A")]
        else:
            # Model didn't emit an isolated letter; treat the raw response as
            # the predicted choice and let exact-match decide.
            pred_norm = pred_raw
        return 1.0 if pred_norm.strip().lower() == gold.lower() else 0.0

    # fill_in_blank
    return _anls_score(pred_raw, gold)


def iconqa_process_results(doc, results):
    pred = results[0] if results else ""
    score = _score_one(doc, pred)
    return {"anls": {"score": score, "ques_type": doc["ques_type"]}}


def iconqa_aggregate_anls(items, args=None):
    """Mean ANLS across all items; logs per-subtype mean for visibility."""
    scores = []
    by_type = {"choose_img": [], "choose_txt": [], "fill_in_blank": []}
    for item in items:
        s = float(item.get("score", 0.0))
        qt = item.get("ques_type", "")
        scores.append(s)
        if qt in by_type:
            by_type[qt].append(s)

    for qt, ss in by_type.items():
        if ss:
            logger.info(f"iconqa[{qt}]: {sum(ss) / len(ss):.4f} (n={len(ss)})")

    return sum(scores) / max(len(scores), 1)


# Backwards-compat alias: the previous module exported `test_process_results`
# (commented out in the yaml). Keep the name so a future revert / older config
# still resolves, but route it to the new logic.
test_process_results = iconqa_process_results
