import os
import json
import datetime
from lmms_eval.tasks.olympiadbench.olympiadbench_evals import OlympiadBenchEvaluator
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import logging
eval_logger = logging.getLogger("lmms-eval")
dir_name = os.path.dirname(os.path.abspath(__file__))

olympiadbench_evaluator = OlympiadBenchEvaluator()

def olympiadbench_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["images"]]

def olympiadbench_doc_to_text(doc):
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"]
    }
    pass

def olympiadbench_process_results(doc, result):
    pass

def olympiadbench_aggregation_results(results, metric, args):
    pass

def auto_scoring(results, metric, args):
    pass