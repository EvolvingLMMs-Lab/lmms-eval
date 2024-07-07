import collections
import os
import json
from capture_metric.capture import CAPTURE
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
import io
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

detailcaps_METRICS = ["CAPTURE", "Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]


def detailcaps_doc_to_visual(doc):
    return [Image.open(io.BytesIO(doc["binary"])).convert("RGB")]


def detailcaps_doc_to_text(doc, model_specific_prompt_kwargs=None):
    # question = "Please carefully observe the image and come up with a caption for the image"
    return model_specific_prompt_kwargs["prompt"]

def detailcaps_doc_to_target(doc):
    references = [
        doc['GT_Caption_GPT4O'],
        doc['GT_Caption_GPT4V'],
        doc['GT_Caption_Gemini15Pro'],
    ]
    return references


def detailcaps_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """

    pred = result[0]
    # The question id in our dataset is the image file itself
    image_id = doc["image"]

    data_dict = {"answer": detailcaps_doc_to_target(doc), "pred": pred, "image_id": image_id}

    return {f"detailcaps_{metric}": data_dict for metric in detailcaps_METRICS}


def check_if_context_is_set(expected_context='spawn'):
    # 获取默认上下文的名称
    default_context_name = mp.get_context().get_start_method()
    
    # 检查当前上下文是否与预期的上下文相匹配
    is_set_to_expected = default_context_name == expected_context
    
    return is_set_to_expected


def detailcaps_aggregation_result(results, metric, args=None):

    scorers = [
        (Bleu(4), "Bleu_1"), 
        (Bleu(4), "Bleu_2"), 
        (Bleu(4), "Bleu_3"), 
        (Bleu(4), "Bleu_4"), 
        (Meteor(), "METEOR"), 
        (Rouge(), "ROUGE_L"), 
        (Cider(), "CIDEr"),
        (CAPTURE(), "CAPTURE")
    ]
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

    detailcaps_result = coco.loadRes(stored_results)
    detailcaps_eval = COCOEvalCap(coco, detailcaps_result)

    imgIds = detailcaps_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = detailcaps_eval.coco.imgToAnns[imgId]
        res[imgId] = detailcaps_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()

    if metric == 'CAPTURE':
        reorg_gts, reorg_res = collections.defaultdict(list), collections.defaultdict(list)
        for _, samples in gts.items():
            for sample in samples:
                reorg_gts[sample['image_id']].append(sample['caption'])
        for _, samples in res.items():
            for sample in samples:
                reorg_res[sample['image_id']].append(sample['caption'])
        gts, res = reorg_gts, reorg_res
    else:
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    # if int(os.environ.get("RANK", 0)) == 0:        
    #     from IPython import embed; embed()
    # else:
    #     import time; time.sleep(1200)

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    path = generate_submission_file(f"detailcaps_val_{metric}_scores.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)
    eval_logger.info(f"Your result has been saved to {path}.")

    return score


def detailcaps_bleu4(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_4", args)


def detailcaps_bleu3(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_3", args)


def detailcaps_bleu2(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_2", args)


def detailcaps_bleu1(results, args=None):
    return detailcaps_aggregation_result(results, "Bleu_1", args)


def detailcaps_meteor(results, args=None):
    return detailcaps_aggregation_result(results, "METEOR", args)


def detailcaps_rougel(results, args=None):
    return detailcaps_aggregation_result(results, "ROUGE_L", args)


def detailcaps_cider(results, args=None):
    return detailcaps_aggregation_result(results, "CIDEr", args)


def detailcaps_spice(results, args=None):
    return detailcaps_aggregation_result(results, "SPICE", args)


def detailcaps_capture(results, args=None):
    return detailcaps_aggregation_result(results, "CAPTURE", args)


def detailcaps_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case detailcaps_passthrough), value: metric value
    """
    return {"detailcaps_passthrough": {"pred": result[0], "image_id": doc["image_id"]}}


def detailcaps_test_aggregation_result(results, args=None):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})

    path = generate_submission_file("detailcaps_captions_detailcaps_test_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored in {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
