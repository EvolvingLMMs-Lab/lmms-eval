import os
import json
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

NOCAPS_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]


def nocaps_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def nocaps_doc_to_text(doc, model_specific_prompt_kwargs=None):
    # question = "Please carefully observe the image and come up with a caption for the image"
    return model_specific_prompt_kwargs["prompt"]


def nocaps_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0]
    # The question id in our dataset is the image file itself
    image_id = doc["image_id"]

    data_dict = {"answer": doc["annotations_captions"], "pred": pred, "image_id": image_id}

    return {f"nocaps_{metric}": data_dict for metric in NOCAPS_METRICS}


def nocaps_aggregation_result(results, metric, args=None):
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

    nocaps_result = coco.loadRes(stored_results)
    nocaps_eval = COCOEvalCap(coco, nocaps_result)

    imgIds = nocaps_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = nocaps_eval.coco.imgToAnns[imgId]
        res[imgId] = nocaps_eval.cocoRes.imgToAnns[imgId]

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

    path = generate_submission_file(f"nocaps_val_{metric}_scores.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)
    eval_logger.info(f"Your result has been saved to {path}.")

    return score


def nocaps_bleu4(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_4", args)


def nocaps_bleu3(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_3", args)


def nocaps_bleu2(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_2", args)


def nocaps_bleu1(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_1", args)


def nocaps_meteor(results, args=None):
    return nocaps_aggregation_result(results, "METEOR", args)


def nocaps_rougel(results, args=None):
    return nocaps_aggregation_result(results, "ROUGE_L", args)


def nocaps_cider(results, args=None):
    return nocaps_aggregation_result(results, "CIDEr", args)


def nocaps_spice(results, args=None):
    return nocaps_aggregation_result(results, "SPICE", args)


def nocaps_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case nocaps_passthrough), value: metric value
    """
    return {"nocaps_passthrough": {"pred": result[0], "image_id": doc["image_id"]}}


def nocaps_test_aggregation_result(results, args=None):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})

    path = generate_submission_file("nocaps_captions_nocaps_test_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored in {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
