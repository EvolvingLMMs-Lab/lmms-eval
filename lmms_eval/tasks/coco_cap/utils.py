import os
import json
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

COCO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]


def coco_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def coco_doc_to_text(doc):
    return f"Provide a one-sentence caption for the provided image."


def coco_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    question_id = doc["question_id"]
    # The question id in our dataset is the image file itself
    image_id = int(question_id.split("_")[-1].split(".")[0])
    id = doc["id"]

    data_dict = {"answer": doc["answer"], "pred": pred, "image_id": image_id, "id": id}

    return {f"coco_{metric}": data_dict for metric in COCO_METRICS}


def coco_aggregation_result(results, metric, args):
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
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": int(result["image_id"]), "caption": a, "id": idx})
            idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

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

    path = generate_submission_file("coco_captions_val2014_alg_results.json", args)
    if not os.path.exists(path):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open(path, "w") as f:
            json.dump(stored_results, f, indent=4)

    return score


def coco_bleu4(results, args):
    return coco_aggregation_result(results, "Bleu_4", args)


def coco_bleu3(results, args):
    return coco_aggregation_result(results, "Bleu_3", args)


def coco_bleu2(results, args):
    return coco_aggregation_result(results, "Bleu_2", args)


def coco_bleu1(results, args):
    return coco_aggregation_result(results, "Bleu_1", args)


def coco_meteor(results, args):
    return coco_aggregation_result(results, "METEOR", args)


def coco_rougel(results, args):
    return coco_aggregation_result(results, "ROUGE_L", args)


def coco_cider(results, args):
    return coco_aggregation_result(results, "CIDEr", args)


def coco_spice(results, args):
    return coco_aggregation_result(results, "SPICE", args)


def coco_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_passthrough), value: metric value
    """
    question_id = doc["question_id"]
    # The question id in our dataset is the image file itself
    image_id = int(question_id.split("_")[-1].split(".")[0])
    return {"coco_passthrough": {"pred": result, "image_id": image_id}}


def coco_test_aggregation_result(results, args):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})

    path = generate_submission_file("coco_captions_test2014_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored in to {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
