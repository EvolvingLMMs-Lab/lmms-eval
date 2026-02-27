from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)
from lmms_eval.tasks.countbenchqa.utils import (
    countbenchqa_doc_to_text,
    countbenchqa_doc_to_visual,
)

countbenchqa_reasoning_doc_to_messages = make_reasoning_doc_to_messages(countbenchqa_doc_to_visual, countbenchqa_doc_to_text)
countbenchqa_reasoning_process_results = make_reasoning_process_results("countbenchqa", countbenchqa_doc_to_text, gt_key="number")
