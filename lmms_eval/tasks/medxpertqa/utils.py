import io
import os
import re
import zipfile
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.tasks.medevalkit.eval_utils import agg_mean  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import no_image_doc_to_visual  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import judge_multi_choice, strip_thinking

_MEDXPERT_REPO = "TsinghuaC3I/MedXpertQA"
_ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _ordered_options(doc: Dict[str, Any]) -> List[str]:
    """Option texts ordered by their letter key (A, B, C, ...)."""
    opts = doc["options"]
    return [opts[k] for k in sorted(opts.keys()) if opts[k] is not None]


def _format_question(doc: Dict[str, Any]) -> str:
    question = doc["question"].strip()
    # MedXpertQA already embeds the options inline ("Answer Choices: (A) ...").
    # Only append them when missing, to avoid duplicating the choices.
    if "Answer Choices" not in question and not re.search(r"\(\s*A\s*\)", question):
        opts = doc["options"]
        lines = [f"{k}. {opts[k].strip()}" for k in sorted(opts.keys()) if opts[k] is not None]
        question = question + "\nAnswer Choices:\n" + "\n".join(lines)
    return f"{question}\n{_ANSWER_INSTRUCTION}"


def medxpertqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] = None) -> str:
    return _format_question(doc)


def medxpertqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["label"].strip().upper()


def medxpertqa_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    raw_response = result[0] if result else ""
    response = strip_thinking(raw_response)
    choices = _ordered_options(doc)
    answer = doc["label"].strip()
    correct = judge_multi_choice(choices, answer, response)
    return {"accuracy": float(correct)}


# ---------------------------------------------------------------------------
# MM image loading: read directly from images.zip (mirrors pmc_vqa / slake).
# One ZipFile handle per process; basename index makes it robust to whatever
# internal directory prefix the archive uses.
# ---------------------------------------------------------------------------

_ARCHIVE: Optional[zipfile.ZipFile] = None
_NAME_INDEX: Optional[Dict[str, str]] = None


def _get_archive() -> zipfile.ZipFile:
    global _ARCHIVE
    if _ARCHIVE is None:
        zip_path = hf_hub_download(repo_id=_MEDXPERT_REPO, filename="images.zip", repo_type="dataset")
        _ARCHIVE = zipfile.ZipFile(zip_path, "r")
    return _ARCHIVE


def _resolve(name: str) -> str:
    global _NAME_INDEX
    if _NAME_INDEX is None:
        _NAME_INDEX = {os.path.basename(n): n for n in _get_archive().namelist() if not n.endswith("/")}
    return _NAME_INDEX.get(os.path.basename(name), name)


def medxpertqa_mm_doc_to_visual(doc: Dict[str, Any]):
    archive = _get_archive()
    images = []
    for fn in doc["images"]:
        with archive.open(_resolve(fn)) as fp:
            images.append(Image.open(io.BytesIO(fp.read())).convert("RGB"))
    return images
