from PIL import ImageDraw
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

# COCO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]
COCO_METRICS = ["CIDEr"]

import logging

eval_logger = logging.getLogger("lmms-eval")


def screenspot_bbox_doc_to_visual(doc):
    bbox = doc["bbox"]
    image = doc["image"].convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox_xy = [bbox[0], bbox[1], bbox[2], bbox[3]]
    draw.rectangle(bbox_xy, outline="red", width=3)
    return [image.convert("RGB")]


def screenspot_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    ann_id = doc["file_name"]
    data_dict = {"instruction": doc["instruction"], "pred": pred, "ann_id": ann_id, "data_type": doc["data_type"], "data_source": doc["data_source"]}
    return {f"screenspot_{metric}": data_dict for metric in COCO_METRICS}


def screenspot_doc_to_text(doc):
    return f"Direct a user to interact with the highlighted region [{doc['bbox'][0]:.2f}, {doc['bbox'][1]:.2f}, {doc['bbox'][2]:.2f}, {doc['bbox'][3]:.2f}]."


def screenspot_aggregation_result(results, metric):
    # scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr"), (Spice(), "SPICE")]
    scorers = [(Cider(), "CIDEr")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    ann_id = 0
    for result in results:
        stored_results.append({"image_id": idx, "caption": result["pred"]})
        # for s in result["answer"]:
        dataset["annotations"].append({"image_id": idx, "caption": result["instruction"], "id": ann_id})
        ann_id += 1

        dataset["images"].append({"id": idx})
        idx += 1

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
    # coco_eval.setEval(score, metric)

    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    return score


def screenspot_bleu4(results):
    return screenspot_aggregation_result(results, "Bleu_4")


def screenspot_bleu3(results):
    return screenspot_aggregation_result(results, "Bleu_3")


def screenspot_bleu2(results):
    return screenspot_aggregation_result(results, "Bleu_2")


def screenspot_bleu1(results):
    return screenspot_aggregation_result(results, "Bleu_1")


def screenspot_meteor(results):
    return screenspot_aggregation_result(results, "METEOR")


def screenspot_rougel(results):
    return screenspot_aggregation_result(results, "ROUGE_L")


def screenspot_cider(results):
    return screenspot_aggregation_result(results, "CIDEr")


def screenspot_spice(results):
    return screenspot_aggregation_result(results, "SPICE")
