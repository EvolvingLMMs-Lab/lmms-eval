"""Pinned TAT-DQA test data, ordered pages and a generative JSON adaptation.

The authors publish answer/scale predictions, not a universal VLM prompt.
EM/F1/scale match the author-linked Doc2SoarGraph scorer; parse_success measures
this adapter's JSON contract and is not an official benchmark score.
"""

import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.tasks.tat_dqa._scorer import score_answer

DATASET_REPO = "next-tat/TAT-DQA"
DATASET_REVISION = "bc46208b22f0bfe47152c98fd0938d238ce868c2"
SCALES = ("", "thousand", "million", "billion", "percent")
RENDER_DPI = 144
ANSWER_TYPES = ("span", "multi-span", "arithmetic", "count")
PROMPT = (
    "Answer the question using the provided document pages.\n"
    'Return only a JSON object with keys "answer" and "scale".\n'
    '"answer" must be a string, a number, or a list of strings for multiple spans.\n'
    '"scale" must be one of "", "thousand", "million", "billion", or "percent".\n'
    "For a numerical answer, give the value in the chosen scale; for example,\n"
    'use {"answer": 12.5, "scale": "percent"} for 12.5%.\n'
    "Do not include an explanation or any other keys.\n\n"
    "Question: "
)
_FEATURES = datasets.Features(
    {
        "question_id": datasets.Value("string"),
        "doc_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answer_json": datasets.Value("string"),
        "answer_type": datasets.Value("string"),
        "scale": datasets.Value("string"),
        "page_names": datasets.Sequence(datasets.Value("string")),
        "images": datasets.Sequence(datasets.Image()),
    },
)


def _render_pdf(pdf_bytes: bytes, expected_pages: int) -> list[dict[str, Any]]:
    """Render full PDF pages at 144 DPI; the release PNGs are 224px thumbnails."""
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("TAT-DQA requires Poppler's pdftoppm to render the original PDFs")
    with tempfile.TemporaryDirectory(prefix="tat_dqa_") as temporary:
        root = Path(temporary)
        pdf = root / "document.pdf"
        pdf.write_bytes(pdf_bytes)
        subprocess.run(
            ["pdftoppm", "-r", str(RENDER_DPI), "-png", str(pdf), str(root / "page")],
            check=True,
            capture_output=True,
            timeout=120,
        )
        pages = sorted(root.glob("page-*.png"), key=lambda path: int(path.stem.rsplit("-", 1)[1]))
        if len(pages) != expected_pages:
            raise ValueError("TAT-DQA PDF page count disagrees with document metadata")
        return [{"bytes": page.read_bytes(), "path": None} for page in pages]


def _document_pages(archive: zipfile.ZipFile, doc_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    if not re.fullmatch(r"[0-9a-f]{32}", doc_id):
        raise ValueError("TAT-DQA has an invalid document ID")
    try:
        metadata = json.loads(archive.read(f"test/{doc_id}.json"))
        pages = metadata["pages"]
        if not isinstance(pages, list) or not pages:
            raise ValueError("TAT-DQA document metadata has no pages")
        pdf = archive.read(f"test/{doc_id}.pdf")
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"TAT-DQA is missing document metadata or PDF for {doc_id}") from error
    names = [f"test/{doc_id}.pdf#page={index}" for index in range(1, len(pages) + 1)]
    return names, _render_pdf(pdf, len(pages))


def build_dataset(records: list[dict[str, Any]], archive_path: str | Path) -> datasets.Dataset:
    """Expand raw documents into questions with embedded, ordered page bytes."""
    rows, seen_questions, seen_documents = [], set(), set()
    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            doc_id = record["doc"]["uid"]
            if doc_id in seen_documents:
                raise ValueError("TAT-DQA contains a duplicate document ID")
            seen_documents.add(doc_id)
            page_names, images = _document_pages(archive, doc_id)
            for question in record["questions"]:
                question_id = question["uid"]
                if not isinstance(question_id, str) or not question_id or question_id in seen_questions:
                    raise ValueError("TAT-DQA contains a missing or duplicate question ID")
                seen_questions.add(question_id)
                if question["answer_type"] not in ANSWER_TYPES or question["scale"] not in SCALES:
                    raise ValueError("TAT-DQA contains an unsupported answer type or scale")
                if not isinstance(question["question"], str) or not question["question"].strip():
                    raise ValueError("TAT-DQA contains an empty question")
                answer = question["answer"]
                if question["answer_type"] in {"span", "multi-span"}:
                    if not isinstance(answer, list) or not answer or not all(isinstance(item, str) for item in answer):
                        raise ValueError("TAT-DQA span gold must be a nonempty list of strings")
                elif not _answer_value_valid(answer) or isinstance(answer, list) or answer == "":
                    raise ValueError("TAT-DQA numerical gold must be a nonempty scalar")
                answer_json = json.dumps(answer, ensure_ascii=False, allow_nan=False)
                rows.append(
                    {
                        "question_id": question_id,
                        "doc_id": doc_id,
                        "question": question["question"],
                        "answer_json": answer_json,
                        "answer_type": question["answer_type"],
                        "scale": question["scale"],
                        "page_names": page_names,
                        "images": images,
                    },
                )
    if not rows:
        raise ValueError("TAT-DQA contains no questions")
    return datasets.Dataset.from_list(rows, features=_FEATURES)


def tat_dqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Parse the pinned raw JSON, or reuse the same portable prepared schema."""
    if "text" not in dataset.column_names:
        if set(dataset.column_names) != set(_FEATURES):
            raise ValueError("TAT-DQA prepared dataset has an unexpected schema")
        return dataset
    if len(dataset) != 1:
        raise ValueError("TAT-DQA requires one raw JSON document; use sample_by: document")
    records = json.loads(dataset[0]["text"])
    if not isinstance(records, list):
        raise ValueError("TAT-DQA annotations must be a list of documents")
    archive_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="tatdqa_docs_test.zip",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return build_dataset(records, archive_path)


def tat_dqa_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Return every page in its original document order, converted to RGB."""
    visuals = []
    for page in doc["images"]:
        if isinstance(page, dict):
            source = io.BytesIO(page["bytes"]) if page.get("bytes") is not None else page["path"]
            with Image.open(source) as opened:
                visuals.append(opened.convert("RGB"))
        else:
            visuals.append(page.convert("RGB"))
    if not visuals or len(visuals) != len(doc["page_names"]):
        raise ValueError("TAT-DQA page inventory is incomplete")
    return visuals


def tat_dqa_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    """Return the fixed adaptation prompt without any gold fields."""
    return PROMPT + doc["question"]


def tat_dqa_doc_to_messages(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Use one user turn with ordered pages followed by the shared prompt."""
    content = [{"type": "image", "url": page} for page in tat_dqa_doc_to_visual(doc)]
    content.append({"type": "text", "text": tat_dqa_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [{"role": "user", "content": content}]


def tat_dqa_doc_to_target(doc: dict[str, Any]) -> str:
    """Serialize the reference answer without losing its original value type."""
    return json.dumps({"answer": json.loads(doc["answer_json"]), "scale": doc["scale"]}, ensure_ascii=False)


def _answer_value_valid(value: Any) -> bool:
    if isinstance(value, list):
        return all(isinstance(item, str) for item in value)
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return False


def parse_prediction(response: str) -> tuple[Any, str | None, bool]:
    """Read one JSON answer; malformed responses never disappear from scoring."""
    if not isinstance(response, str):
        return None, None, False
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict) or set(payload) != {"answer", "scale"}:
            return None, None, False
        scale = payload["scale"]
        if isinstance(scale, str) and scale in SCALES and _answer_value_valid(payload["answer"]):
            return payload["answer"], scale, True
    except (ValueError, TypeError, OverflowError, RecursionError):
        pass
    return None, None, False


def tat_dqa_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score parsed predictions and report the adapter's parsing success separately."""
    answer, scale, valid = parse_prediction(results[0] if results else "")
    gold = {"answer": json.loads(doc["answer_json"]), "answer_type": doc["answer_type"], "scale": doc["scale"]}
    exact, f1, scale_accuracy = score_answer(gold, answer, scale)
    return {
        "tat_dqa_em": exact,
        "tat_dqa_f1": f1,
        "tat_dqa_scale_accuracy": scale_accuracy,
        "tat_dqa_parse_success": float(valid),
    }
