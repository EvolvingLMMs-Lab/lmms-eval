NUMBER_WORD_TO_NUMERAL = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _normalize_count_answer(answer) -> str:
    normalized = str(answer).strip().lower()
    return NUMBER_WORD_TO_NUMERAL.get(normalized, normalized)


def countbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def countbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def countbench_doc_to_target(doc):
    return _normalize_count_answer(doc["number"])


def countbench_process_results(doc, results):
    prediction = _normalize_count_answer(results[0])
    target = countbench_doc_to_target(doc)
    return {"acc": float(prediction == target)}
