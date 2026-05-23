import io
import zipfile
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.tasks.medevalkit.eval_utils import agg_mean  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import no_image_doc_to_visual  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import judge_multi_choice

# ---------------------------------------------------------------------------
# Dataset images: read directly from images_2.zip (no extraction, no inode
# churn, no extract-race). One ZipFile handle per process.
# ---------------------------------------------------------------------------

_ARCHIVE: Optional[zipfile.ZipFile] = None

CHOICE_LETTERS = ["A", "B", "C", "D"]


def _get_archive() -> zipfile.ZipFile:
    global _ARCHIVE
    if _ARCHIVE is None:
        zip_path = hf_hub_download(
            repo_id="RadGenome/PMC-VQA",
            filename="images_2.zip",
            repo_type="dataset",
        )
        _ARCHIVE = zipfile.ZipFile(zip_path, "r")
    return _ARCHIVE


# ---------------------------------------------------------------------------
# lmms-eval interface
# ---------------------------------------------------------------------------


def pmc_vqa_doc_to_visual(doc: Dict[str, Any]):
    archive = _get_archive()
    with archive.open(f"figures/{doc['Figure_path']}") as fp:
        return [Image.open(io.BytesIO(fp.read())).convert("RGB")]


def pmc_vqa_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] = None,
) -> str:
    question = doc["Question"].strip()
    choices = []
    for letter in CHOICE_LETTERS:
        choice_text = doc[f"Choice {letter}"].strip()
        # Choices may already include the letter prefix like " A:..."
        if not choice_text.startswith(letter):
            choice_text = f"{letter}. {choice_text}"
        choices.append(choice_text)
    options = "\n".join(choices)
    return f"Question: {question}\n" f"Options:\n{options}\n" "Answer with the option's letter from the given choices directly."


def pmc_vqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["Answer"].strip().upper()


def pmc_vqa_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    raw_response = result[0] if result else ""
    choices = [doc[f"Choice {letter}"].strip() for letter in CHOICE_LETTERS]
    answer = doc["Answer"].strip()
    correct = judge_multi_choice(choices, answer, raw_response)
    return {"accuracy": float(correct)}
