def countbenchqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def countbenchqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    if doc.get("text"):
        question = f"{doc['text']}\n\n{question}"
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    return f"{pre_prompt}{question}{post_prompt}"


def countbenchqa_process_results(doc, results):
    pred = results[0].strip()
    ground_truth = str(doc["number"])
    score = 1.0 if pred == ground_truth else 0.0
    return {"acc": score, "exact_match": score}
