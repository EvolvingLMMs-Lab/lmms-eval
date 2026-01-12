# VideoMMMU No Visual Setting Utils
from lmms_eval.tasks.videommmu.utils import (
    videommmu_aggregate_results,
    videommmu_doc_to_answer,
    videommmu_doc_to_text_adaptation,
    videommmu_doc_to_text_perception_comprehension,
    videommmu_process_results,
)


def videommmu_doc_to_visual_empty(doc):
    """Return empty visual for no_visual setting."""
    return []
