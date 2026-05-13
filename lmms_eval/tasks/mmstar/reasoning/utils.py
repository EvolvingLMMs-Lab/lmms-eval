from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)
from lmms_eval.tasks.mmstar.utils import mmstar_doc_to_text as _mmstar_doc_to_text
from lmms_eval.tasks.mmstar.utils import mmstar_doc_to_visual


def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _mmstar_doc_to_text(doc, lmms_eval_specific_kwargs or {})


mmstar_reasoning_doc_to_messages = make_reasoning_doc_to_messages(mmstar_doc_to_visual, mmstar_doc_to_text)
mmstar_reasoning_process_results = make_reasoning_process_results("mmstar", mmstar_doc_to_text)
