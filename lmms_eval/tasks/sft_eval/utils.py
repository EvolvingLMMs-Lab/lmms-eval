from rapidfuzz.distance import Levenshtein


def _normalize_text(text):
    import string

    text = "".join(filter(lambda x: x in (string.digits + string.ascii_letters), text))


def process_result_ocr_rec(doc, result):
    return {"edit_distance": {"gts": [gt.replace("\n", "") for gt in doc["answer"]], "predictions": result}, "edit_acc": {"gts": [gt.replace("\n", "") for gt in doc["answer"]], "predictions": result}}


def process_result_rest(doc, result):
    return {"fuzzy_match": {"gts": [gt.replace("\n", "") for gt in doc["answer"]], "predictions": result, "question_id": doc["question_id"]}}


def run_editdistance(gts, predictions, ignore_space=True, is_filter=False):
    eps = 1e-6
    correct_num = 0
    all_num = 0
    norm_edit_dis = 0.0
    edit_norm_score_list = list()
    for idx in range(len(predictions)):
        target, pred = gts[idx][0], predictions[idx]
        if ignore_space:
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
        if is_filter:
            pred = _normalize_text(pred)
            target = _normalize_text(target)

        ned = Levenshtein.normalized_distance(pred, target)
        norm_edit_dis += ned
        edit_norm_score_list.append(1 - ned)
        if pred == target:
            correct_num += 1
        all_num += 1
    # import pdb; pdb.set_trace()
    metric = {"acc": correct_num / (all_num + eps), "norm_edit_dis": 1 - norm_edit_dis / (all_num + eps)}
    return metric, edit_norm_score_list


def fuzzy_match_multi_answers(results, gt_dict):
    acc = []
    for result in results:
        question_id = result["question_id"]
        try:
            gt_ans = gt_dict[question_id]
        except:
            import pdb

            pdb.set_trace()
        pred = result["text"]
        for gt in gt_ans:
            vqa_acc = 1
            if not (
                (gt == "是" and gt in pred and "不是" not in pred)
                or (gt == "对" and gt in pred and "不对" not in pred)
                or (gt == "相同" and gt in pred and "不相同" not in pred)
                or (gt == "有" and gt in pred and "没有" not in pred)
                or (gt == "在" and gt in pred and "不在" not in pred)
                or (gt == "一样" and gt in pred and "不一样" not in pred)
                or (gt not in ["是", "在", "对", "有", "一样", "相同"] and gt.lower() in pred.lower())
            ):
                vqa_acc = 0
            if vqa_acc == 1:
                break
        acc.append(vqa_acc)
    accuracy = sum(acc) / len(acc) * 100
    return {"Acc": accuracy}


def sft_eval_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def sft_eval_doc_to_text(doc):
    return doc["question"]


def sft_eval_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")
    question = f"{pre_prompt}{doc['question']}{post_prompt}"
    return question


def sft_eval_edit_dist_acc_agg(results):
    predictions = [result["predictions"][0] for result in results]
    gts = [result["gts"] for result in results]
    acc, _ = run_editdistance(gts, predictions)
    return acc["acc"]


def sft_eval_edit_dist_agg(results):
    predictions = [result["predictions"][0] for result in results]
    gts = [result["gts"] for result in results]
    acc, _ = run_editdistance(gts, predictions)
    return acc["norm_edit_dis"]


def sft_eval_acc_agg(results):
    gts_dict = {result["question_id"]: result["gts"] for result in results}
    predictions = [{"question_id": result["question_id"], "text": result["predictions"][0]} for result in results]
    acc = fuzzy_match_multi_answers(predictions, gts_dict)
    return acc["Acc"]
