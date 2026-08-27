from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)
from lmms_eval.tasks.seedbench.utils import seed_doc_to_text as _seed_doc_to_text
from lmms_eval.tasks.seedbench.utils import seed_doc_to_visual


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _seed_doc_to_text(doc)


seed_reasoning_doc_to_messages = make_reasoning_doc_to_messages(seed_doc_to_visual, seed_doc_to_text)

# Factory handles scoring; we just expand to per-data_type metric keys
_base_process = make_reasoning_process_results("seedbench", seed_doc_to_text)


def seed_reasoning_process_results(doc, results):
    base = _base_process(doc, results)
    data_type = doc["data_type"]
    return {
        f"seed_{data_type}_acc_score": base["acc_score"],
        f"seed_{data_type}_format_score": base["format_score"],
        "seed_all_acc_score": base["acc_score"],
        "seed_all_format_score": base["format_score"],
    }
