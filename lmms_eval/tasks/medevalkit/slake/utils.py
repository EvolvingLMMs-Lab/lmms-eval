import io
import os
import zipfile
from typing import Any, Dict, List, Optional

from huggingface_hub import snapshot_download
from PIL import Image

from lmms_eval.tasks.medevalkit.eval_utils import agg_mean  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import no_image_doc_to_visual  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import (
    judge_close_end,
    judge_open,
    judge_yesno,
    parse_response,
)

# ---------------------------------------------------------------------------
# Dataset images: read directly from imgs.zip (no extraction, no inode churn,
# no extract-race). One ZipFile handle per process.
# ---------------------------------------------------------------------------

_ARCHIVE: Optional[zipfile.ZipFile] = None


def _get_archive() -> zipfile.ZipFile:
    global _ARCHIVE
    if _ARCHIVE is None:
        cache_root = snapshot_download(repo_id="BoKelvin/SLAKE", repo_type="dataset")
        _ARCHIVE = zipfile.ZipFile(os.path.join(cache_root, "imgs.zip"), "r")
    return _ARCHIVE


# ---------------------------------------------------------------------------
# Dataset filtering
# ---------------------------------------------------------------------------


def slake_filter_english(dataset):
    """Keep only English questions."""
    return dataset.filter(lambda doc: doc["q_lang"] == "en")


# ---------------------------------------------------------------------------
# lmms-eval interface
# ---------------------------------------------------------------------------


def slake_doc_to_visual(doc: Dict[str, Any]):
    archive = _get_archive()
    with archive.open(f"imgs/{doc['img_name']}") as fp:
        return [Image.open(io.BytesIO(fp.read())).convert("RGB")]


def slake_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] = None,
) -> str:
    question = doc["question"].strip()
    answer_type = doc["answer_type"]
    answer = str(doc["answer"]).lower().strip()
    if answer_type == "CLOSED" and answer in ("yes", "no"):
        return question + "\nPlease answer 'yes' or 'no' (no extra output)."
    elif answer_type == "CLOSED":
        return question + "\nAnswer the question using a single word or phrase."
    else:
        return question + "\nPlease answer the question concisely."


def slake_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["answer"]).lower().strip()


def slake_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Any]:
    raw_response = result[0] if result else ""
    response = parse_response(raw_response).lower().strip()
    answer = str(doc["answer"]).lower().strip()
    answer_type = doc["answer_type"]

    if answer_type == "CLOSED":
        if answer in ("yes", "no"):
            correct = float(judge_yesno(answer, response))
        else:
            correct = float(judge_close_end(answer, response))
        return {
            "close_accuracy": correct,
            "open_em": None,
            "bleu1": None,
            "bleu4": None,
            "rouge1": None,
            "rougel": None,
            "f1": None,
        }
    else:
        m = judge_open(answer, response)
        return {
            "close_accuracy": None,
            "open_em": m["em"],
            "bleu1": m["bleu1"],
            "bleu4": m["bleu4"],
            "rouge1": m["rouge1"],
            "rougel": m["rougel"],
            "f1": m["f1"],
        }
