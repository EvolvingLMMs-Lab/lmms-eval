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



def olympiadbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def olympiadbench_doc_to_text(doc):
    # question = "Please carefully observe the image and come up with a caption for the image"
    return f"Provide a one-sentence caption for the provided image."


def olympiadbench_process_result(doc, result):
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


def olympiadbench_aggregation_result(results, metric, args):
    pass