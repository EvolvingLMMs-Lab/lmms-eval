import os
import json
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

TEXTCAPS_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]


def textcaps_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textcaps_doc_to_text(doc):
    question = doc["question"]
    return f"{question}\nAnswer the question with a short phrase."


def textcaps_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""

    data_dict = {"answer": doc["caption_str"], "pred": pred, "image_id": doc["image_id"]}

    return {f"textcaps_{metric}": data_dict for metric in TEXTCAPS_METRICS}


def textcaps_aggregation_result(results, metric):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr"), (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": result["image_id"], "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": result["image_id"], "caption": a, "id": idx})
            idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    textcaps_result = coco.loadRes(stored_results)
    textcaps_eval = COCOEvalCap(coco, textcaps_result)

    imgIds = textcaps_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = textcaps_eval.coco.imgToAnns[imgId]
        res[imgId] = textcaps_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    os.makedirs("./submissions", exist_ok=True)
    if not os.path.exists("./submissions/textcaps_captions_val2014_alg_results.json"):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open("./submissions/textcaps_captions_val2014_alg_results.json", "w") as f:
            json.dump(stored_results, f, indent=4)

    return score


def textcaps_bleu4(results):
    return textcaps_aggregation_result(results, "Bleu_4")


def textcaps_bleu3(results):
    return textcaps_aggregation_result(results, "Bleu_3")


def textcaps_bleu2(results):
    return textcaps_aggregation_result(results, "Bleu_2")


def textcaps_bleu1(results):
    return textcaps_aggregation_result(results, "Bleu_1")


def textcaps_meteor(results):
    return textcaps_aggregation_result(results, "METEOR")


def textcaps_rougel(results):
    return textcaps_aggregation_result(results, "ROUGE_L")


def textcaps_cider(results):
    return textcaps_aggregation_result(results, "CIDEr")


def textcaps_spice(results):
    return textcaps_aggregation_result(results, "SPICE")


def textcaps_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case textcaps_passthrough), value: metric value
    """
    return {"textcaps_passthrough": {"pred": result, "image_id": doc["image_id"]}}


def textcaps_test_aggregation_result(results):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": result["image_id"], "caption": result["pred"]})

    os.makedirs("./submissions", exist_ok=True)
    if not os.path.exists("./submissions/captions_test2014_alg_results.json"):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open("./submissions/captions_test2014_alg_results.json", "w") as f:
            json.dump(stored_results, f, indent=4)

    eval_logger.info("Your test result has been stored. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
    return -1
