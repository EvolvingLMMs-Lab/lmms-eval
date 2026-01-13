# LongVT No Visual Setting Utils
from lmms_eval.tasks.longvt.utils import (
    longvt_doc_to_text,
    longvt_process_results,
)


def longvt_doc_to_visual_empty(doc):
    """Return empty visual for no_visual setting."""
    return []
