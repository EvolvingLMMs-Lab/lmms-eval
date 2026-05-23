from collections import defaultdict
from typing import Any, Dict, List

from loguru import logger
from PIL import Image

from lmms_eval.tasks.medevalkit.eval_utils import no_image_doc_to_visual  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import (
    judge_multi_choice,
    strip_thinking,
)
from lmms_eval.tasks.path_mmu.path_mmu_dataset import build_dataset

CHOICE_LETTERS = ["A", "B", "C", "D"]

# Auto-build the PathMMU dataset cache on import.
build_dataset("path_mmu_dataset_cache")


def path_mmu_doc_to_visual(doc: Dict[str, Any]):
    """Load the pathology image for a PathMMU sample."""
    img_path = doc["image_path"]
    return [Image.open(img_path).convert("RGB")]


def path_mmu_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] = None,
) -> str:
    """Format a PathMMU sample into a multiple-choice prompt."""
    question = doc["question"].strip()
    options = doc["options"]
    choices = []
    for letter, opt in zip(CHOICE_LETTERS, options):
        choices.append(f"{letter}. {opt}")
    options_block = "\n".join(choices)
    return f"Question: {question}\n" f"Options:\n{options_block}\n" "Answer with the option's letter from the given choices directly."


def path_mmu_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter."""
    return doc["answer"].strip().upper()


def path_mmu_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Any]:
    """Parse model output and compute per-category accuracy."""
    response = strip_thinking(result[0]).strip() if result else ""
    options = doc["options"]
    gt_letter = doc["answer"].strip().upper()
    correct = float(judge_multi_choice(options, gt_letter, response))

    return {
        "accuracy": {
            "category": doc.get("category", "unknown"),
            "subcategory": doc.get("subcategory", doc.get("category", "unknown")),
            "correct": correct,
        },
    }


def path_mmu_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """Aggregate accuracy overall, per category, and per subcategory."""
    cat_scores = defaultdict(list)
    subcat_scores = defaultdict(list)

    for r in results:
        cat_scores[r["category"]].append(r["correct"])
        subcat_scores[r["subcategory"]].append(r["correct"])

    def _acc(scores: list) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    # Print results
    all_scores = [r["correct"] for r in results]
    overall = _acc(all_scores)

    lines = []
    lines.append(f"{'':2s}{'Category':30s} {'N':>6s} {'Acc':>8s}")
    lines.append("-" * 50)

    for cat in ["PubMed", "EduContent", "PathCLS", "Atlas", "SocialPath"]:
        if cat not in cat_scores:
            continue
        acc = _acc(cat_scores[cat])
        lines.append(f"  {cat:30s} {len(cat_scores[cat]):>6d} {acc:>8.4f}")

        # Print subcategories for PathCLS
        if cat == "PathCLS":
            for sub in sorted(subcat_scores):
                if sub == cat or sub in cat_scores:
                    continue
                acc_sub = _acc(subcat_scores[sub])
                lines.append(f"    {sub:28s} {len(subcat_scores[sub]):>6d} {acc_sub:>8.4f}")

    lines.append("-" * 50)
    lines.append(f"  {'Overall':30s} {len(all_scores):>6d} {overall:>8.4f}")

    logger.info("PathMMU results:\n" + "\n".join(lines))
    return overall
