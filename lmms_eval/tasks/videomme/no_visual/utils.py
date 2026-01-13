# VideoMME No Visual Setting Utils
# Import base functions from parent and add no_visual specific functions

from lmms_eval.tasks.videomme.utils import (
    videomme_aggregate_results,
    videomme_doc_to_text,
    videomme_process_results,
)


def videomme_doc_to_visual_empty(doc):
    """Return empty visual for no_visual setting."""
    return []
