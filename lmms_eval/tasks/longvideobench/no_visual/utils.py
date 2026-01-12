# LongVideoBench No Visual Setting Utils
from lmms_eval.tasks.longvideobench.utils import (
    longvideobench_aggregate_results,
    longvideobench_doc_to_text,
    longvideobench_process_results,
)


def longvideobench_doc_to_visual_empty(doc):
    """Return empty visual for no_visual setting."""
    return []
