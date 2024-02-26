import os
import json
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
import datetime

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

FLICKR_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]


def flickr_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def flickr_doc_to_text(doc):
    # question = "Please carefully observe the image and come up with a caption for the image"
    return f"Provide a one-sentence caption for the provided image."


def flickr_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    image_id = int(doc["img_id"])

    data_dict = {"answer": doc["caption"], "pred": pred, "image_id": image_id}

    return {f"flickr_{metric}": data_dict for metric in FLICKR_METRICS}


def flickr_aggregation_result(results, metric):
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
        dataset["images"].append({"id": int(result["image_id"])})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    flickr_result = coco.loadRes(stored_results)
    flickr_eval = COCOEvalCap(coco, flickr_result)

    imgIds = flickr_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = flickr_eval.coco.imgToAnns[imgId]
        res[imgId] = flickr_eval.cocoRes.imgToAnns[imgId]

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
    if not os.path.exists("./submissions/flickr30k_captions_val2014_alg_results.json"):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open("./submissions/flickr30k_captions_val2014_alg_results.json", "w") as f:
            json.dump(stored_results, f, indent=4)

    return score


def flickr_bleu4(results):
    return flickr_aggregation_result(results, "Bleu_4")


def flickr_bleu3(results):
    return flickr_aggregation_result(results, "Bleu_3")


def flickr_bleu2(results):
    return flickr_aggregation_result(results, "Bleu_2")


def flickr_bleu1(results):
    return flickr_aggregation_result(results, "Bleu_1")


def flickr_meteor(results):
    return flickr_aggregation_result(results, "METEOR")


def flickr_rougel(results):
    return flickr_aggregation_result(results, "ROUGE_L")


def flickr_cider(results):
    return flickr_aggregation_result(results, "CIDEr")


def flickr_spice(results):
    return flickr_aggregation_result(results, "SPICE")


def flickr_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case flickr_passthrough), value: metric value
    """
    # The question id in our dataset is the image file itself
    image_id = doc["img_id"]
    return {"flickr_passthrough": {"pred": result, "image_id": image_id}}
