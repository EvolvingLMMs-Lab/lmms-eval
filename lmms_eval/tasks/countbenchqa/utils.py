from lmms_eval.tasks.countqa.utils import parse_count


def countbenchqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def countbenchqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    return f"{pre_prompt}{question}{post_prompt}"


def countbenchqa_process_results(doc, results):
    # Compare the count as a number, so answers like "3." or "There are 3 dogs." still match.
    pred = parse_count(results[0] if results else "")
    score = 1.0 if pred == int(doc["number"]) else 0.0
    return {"acc": score, "exact_match": score}
