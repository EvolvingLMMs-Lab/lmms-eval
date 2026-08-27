def pixmo_count_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def pixmo_count_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].replace("<image>", "").strip()
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    return f"{pre_prompt}{question}{post_prompt}"


def pixmo_count_process_results(doc, results):
    pred = results[0].strip()
    ground_truth = doc["answer"].strip()

    import re

    match = re.search(r"\d+", ground_truth)
    if match:
        ground_truth = match.group()

    match = re.search(r"\d+", pred)
    if match:
        pred_num = match.group()
    else:
        pred_num = pred

    score = 1.0 if pred_num == ground_truth else 0.0
    return {"acc": score, "exact_match": score}
