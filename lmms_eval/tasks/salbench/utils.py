# Utils for processing p3o3 dataset


def p3o3_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    if not doc.get("image", None):
        return None
    if isinstance(doc["image"], list):
        return [img.convert("RGB") for img in doc["image"]]
    return [doc["image"].convert("RGB")]


def p3o3_doc_to_text(doc, prompt_kwargs=None):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    if prompt_kwargs:
        pre, post = prompt_kwargs["pre_prompt"], prompt_kwargs["post_prompt"]
        return f"{pre}\n{question}\n{post}"
    return f"{question}\nAnswer the question using a single word or phrase."


def p3_process_results(doc, results):
    pred = {x.strip() for x in results[0].lower().strip("[").strip("]").split(",")}
    gt_ans = {x.strip() for x in doc["answer"].lower().split(",")}

    exact_match = int(pred == gt_ans)

    # Per sample
    matches = pred.intersection(gt_ans)
    # how many retrieved categories are relevant
    precision = len(matches) / (len(pred) + 1e-8)
    # how many applicable categories are retrieved
    recall = len(matches) / len(gt_ans)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    categories = [
        "orientation",
        "color",
        "size",
    ]
    cat_preds = {}
    for cat in categories:
        cat_preds[cat] = {}
        cat_preds[cat]["true_pos"] = int(cat in pred and cat in gt_ans)
        cat_preds[cat]["false_pos"] = int(cat in pred and cat not in gt_ans)
        cat_preds[cat]["true_neg"] = int(cat not in pred and cat not in gt_ans)
        cat_preds[cat]["false_neg"] = int(cat not in pred and cat in gt_ans)

    return {
        "exact_match": {"score": exact_match},
        "sample_precision": {"score": precision},
        "sample_recall": {"score": recall},
        "sample_f1": {"score": f1},
        "all_cat_precision": {"id": doc["image_id"], "pred": cat_preds},
        "all_cat_recall": {"id": doc["image_id"], "pred": cat_preds},
        "all_cat_f1": {"id": doc["image_id"], "pred": cat_preds},
        "orientation_precision": {"pred": cat_preds["orientation"]},
        "orientation_recall": {"pred": cat_preds["orientation"]},
        "orientation_f1": {"pred": cat_preds["orientation"]},
        "color_precision": {"pred": cat_preds["color"]},
        "color_recall": {"pred": cat_preds["color"]},
        "color_f1": {"pred": cat_preds["color"]},
        "size_precision": {"pred": cat_preds["size"]},
        "size_recall": {"pred": cat_preds["size"]},
        "size_f1": {"pred": cat_preds["size"]},
    }


def o3_process_results(doc, results):
    pred = {x.strip() for x in results[0].lower().strip("[").strip("]").split(",")}
    gt_ans = {x.strip() for x in doc["answer"].lower().split(",")}

    exact_match = int(pred == gt_ans)

    # Per sample
    matches = pred.intersection(gt_ans)
    # how many retrieved categories are relevant
    precision = len(matches) / (len(pred) + 1e-8)
    # how many applicable categories are retrieved
    recall = len(matches) / len(gt_ans)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    categories = [
        "orientation",
        "color",
        "focus",
        "shape",
        "size",
        "location",
        "pattern",
    ]
    cat_preds = {}
    for cat in categories:
        cat_preds[cat] = {}
        cat_preds[cat]["true_pos"] = int(cat in pred and cat in gt_ans)
        cat_preds[cat]["false_pos"] = int(cat in pred and cat not in gt_ans)
        cat_preds[cat]["true_neg"] = int(cat not in pred and cat not in gt_ans)
        cat_preds[cat]["false_neg"] = int(cat not in pred and cat in gt_ans)

    return {
        "exact_match": {"score": exact_match},
        "sample_precision": {"score": precision},
        "sample_recall": {"score": recall},
        "sample_f1": {"score": f1},
        "all_cat_precision": {"id": doc["image_id"], "pred": cat_preds},
        "all_cat_recall": {"id": doc["image_id"], "pred": cat_preds},
        "all_cat_f1": {"id": doc["image_id"], "pred": cat_preds},
        "orientation_precision": {"pred": cat_preds["orientation"]},
        "orientation_recall": {"pred": cat_preds["orientation"]},
        "orientation_f1": {"pred": cat_preds["orientation"]},
        "color_precision": {"pred": cat_preds["color"]},
        "color_recall": {"pred": cat_preds["color"]},
        "color_f1": {"pred": cat_preds["color"]},
        "focus_precision": {"pred": cat_preds["focus"]},
        "focus_recall": {"pred": cat_preds["focus"]},
        "focus_f1": {"pred": cat_preds["focus"]},
        "shape_precision": {"pred": cat_preds["shape"]},
        "shape_recall": {"pred": cat_preds["shape"]},
        "shape_f1": {"pred": cat_preds["shape"]},
        "size_precision": {"pred": cat_preds["size"]},
        "size_recall": {"pred": cat_preds["size"]},
        "size_f1": {"pred": cat_preds["size"]},
        "location_precision": {"pred": cat_preds["location"]},
        "location_recall": {"pred": cat_preds["location"]},
        "location_f1": {"pred": cat_preds["location"]},
        "pattern_precision": {"pred": cat_preds["pattern"]},
        "pattern_recall": {"pred": cat_preds["pattern"]},
        "pattern_f1": {"pred": cat_preds["pattern"]},
    }


def process_results_multiple_choices(doc, results):
    # Choices are 1/2/3/4 etc
    assert len(results) == 1, "Not support batch size > 1"
    pred = results[0].lower().strip(",").strip(")").strip(",")
    # HACK: Get first character only, because some model just like talking
    pred = pred[0]
    # pred = {x.strip() for x in result.lower().strip(".").strip(")").strip(",")}
    gt_ans = doc["answer"].lower()

    acc = int(pred == gt_ans)
    return {
            "acc": {"score": acc},
    }


def aggregate_per_sample_score(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def _aggregate_per_category(results):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for result in results:
        true_pos += result["pred"]["true_pos"]
        true_neg += result["pred"]["true_neg"]
        false_pos += result["pred"]["false_pos"]
        false_neg += result["pred"]["false_neg"]

    precision = true_pos / (true_pos + false_pos + 1e-8)
    recall = true_pos / (true_pos + false_neg + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1


def aggregate_per_category_precision(results):
    precision, recall, f1 = _aggregate_per_category(results)
    return precision


def aggregate_per_category_recall(results):
    precision, recall, f1 = _aggregate_per_category(results)
    return recall


def aggregate_per_category_f1(results):
    precision, recall, f1 = _aggregate_per_category(results)
    return f1


def _aggregate_all_category(results, categories):
    precisions = {}
    recalls = {}
    f1s = {}
    for cat in categories:
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for result in results:
            true_pos += result["pred"][cat]["true_pos"]
            true_neg += result["pred"][cat]["true_neg"]
            false_pos += result["pred"][cat]["false_pos"]
            false_neg += result["pred"][cat]["false_neg"]

        precisions[cat] = true_pos / (true_pos + false_pos + 1e-8)
        recalls[cat] = true_pos / (true_pos + false_neg + 1e-8)
        f1s[cat] = (
            (2 * precisions[cat] * recalls[cat])
            / (precisions[cat] + recalls[cat] + 1e-8)
        )

    agg_precision = sum(list(precisions.values())) / len(categories)
    agg_recall = sum(list(recalls.values())) / len(categories)
    agg_f1 = sum(list(f1s.values())) / len(categories)
    return agg_precision, agg_recall, agg_f1


def p3_aggregate_all_category_precision(results):
    categories = [
        "orientation",
        "color",
        "size",
    ]
    precision, recall, f1 = _aggregate_all_category(results, categories)
    return precision


def p3_aggregate_all_category_recall(results):
    categories = [
        "orientation",
        "color",
        "size",
    ]
    precision, recall, f1 = _aggregate_all_category(results, categories)
    return recall


def p3_aggregate_all_category_f1(results):
    categories = [
        "orientation",
        "color",
        "size",
    ]
    precision, recall, f1 = _aggregate_all_category(results, categories)
    return f1


def o3_aggregate_all_category_precision(results):
    categories = [
        "orientation",
        "color",
        "focus",
        "shape",
        "size",
        "location",
        "pattern",
    ]
    precision, recall, f1 = _aggregate_all_category(results, categories)
    return precision


def o3_aggregate_all_category_recall(results):
    categories = [
        "orientation",
        "color",
        "focus",
        "shape",
        "size",
        "location",
        "pattern",
    ]
    precision, recall, f1 = _aggregate_all_category(results, categories)
    return recall


def o3_aggregate_all_category_f1(results):
    categories = [
        "orientation",
        "color",
        "focus",
        "shape",
        "size",
        "location",
        "pattern",
    ]
    precision, recall, f1 = _aggregate_all_category(results, categories)
    return f1
