"""olmOCR-Bench page loading, plain-response scoring and category aggregation."""

import io
import json
import math
import shutil
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.utils import eval_logger

DATASET_REPO_ID = "allenai/olmOCR-bench"
DATASET_REVISION = "54a96a6fb6a2bd3b297e59869491db4d3625b711"
TARGET_LONGEST_IMAGE_DIM = 2048

SOURCE_CATEGORIES: tuple[str, ...] = (
    "arxiv_math",
    "old_scans_math",
    "table_tests",
    "old_scans",
    "headers_footers",
    "multi_column",
    "long_tiny_text",
)
ALL_CATEGORIES: tuple[str, ...] = (*SOURCE_CATEGORIES, "baseline")
ALL_METRICS: tuple[str, ...] = (
    "olmocr_bench_overall",
    *(f"olmocr_bench_{category}" for category in ALL_CATEGORIES),
)

PROMPT_FULL_NO_DOCUMENT_ANCHORING = (
    "Below is the image of one page of a PDF document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "Turn equations into a LaTeX representation, and tables into markdown format. "
    "Remove the headers and footers, but keep references and footnotes.\n"
    "Read any natural handwriting.\n"
    "This is likely one page out of several in the document, so be sure to preserve any sentences "
    "that come from the previous page, or continue onto the next page, exactly as they are.\n"
    "If there is no text at all that you think you should read, you can output null.\n"
    "Do not hallucinate."
)

PROMPT_BASIC = (
    "Please provide a natural, plain text representation of the document, formatted in Markdown. "
    "Skip any headers and footers. For ALL mathematical expressions, use LaTeX notation with "
    r"\( and \) for inline equations and \[ and \] for display equations. "
    "Convert any tables into Markdown format."
)

PROMPT_YAML = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

PROMPT_TEMPLATES: dict[str, str] = {
    "full_no_document_anchoring": PROMPT_FULL_NO_DOCUMENT_ANCHORING,
    "basic": PROMPT_BASIC,
    "yaml": PROMPT_YAML,
}


def clean_model_prediction(raw_prediction: str | None) -> str:
    """Preserve the official plain Markdown boundary, including whitespace and null."""
    return "" if raw_prediction is None else raw_prediction


@lru_cache(maxsize=8192)
def _official_test(test_json: str):
    from lmms_eval.tasks.olmocr_bench import _scorer

    definitions = {
        "present": _scorer.TextPresenceTest,
        "absent": _scorer.TextPresenceTest,
        "order": _scorer.TextOrderTest,
        "table": _scorer.TableTest,
        "math": _scorer.MathTest,
        "baseline": _scorer.BaselineTest,
    }
    spec = json.loads(test_json)
    test_type = spec.get("type")
    if test_type not in definitions:
        raise ValueError(f"olmOCR-Bench: unsupported test type {test_type!r}")
    return definitions[test_type](**spec)


def run_single_pdf_test(test: dict[str, Any], md_content: str) -> bool:
    """Apply the pinned official test; environment failures remain actionable errors."""
    spec = {key: value for key, value in test.items() if key != "group" and value is not None}
    # The compatibility helpers also accept standalone checks without dataset identity.
    spec = {"id": "page-check", "pdf": "page.pdf", "page": 1, **spec}
    check = _official_test(json.dumps(spec, sort_keys=True, ensure_ascii=False))
    try:
        return bool(check.run(md_content)[0])
    except (ValueError, TypeError, IndexError, OverflowError, RecursionError):
        # benchmark.py counts failed evaluations in the denominator.
        return False


def run_math_test(test: dict[str, Any], content: str) -> bool:
    return run_single_pdf_test({**test, "type": "math"}, content)


def run_table_test(test: dict[str, Any], content: str) -> bool:
    return run_single_pdf_test({**test, "type": "table"}, content)


def run_text_presence_test(test: dict[str, Any], content: str) -> bool:
    return run_single_pdf_test(test, content)


def run_text_order_test(test: dict[str, Any], content: str) -> bool:
    return run_single_pdf_test({**test, "type": "order"}, content)


def run_baseline_test(test: dict[str, Any], content: str) -> bool:
    return run_single_pdf_test({**test, "type": "baseline"}, content)


def normalize_text(content: str | None) -> str | None:
    from lmms_eval.tasks.olmocr_bench._scorer import normalize_text as normalize

    return normalize(content)


def parse_html_tables(content: str):
    from lmms_eval.tasks.olmocr_bench._tables import parse_html_tables as parse

    return parse(content)


def parse_markdown_tables(content: str):
    from lmms_eval.tasks.olmocr_bench._tables import parse_markdown_tables as parse

    return parse(content)


def _load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


@lru_cache(maxsize=16)
def _fetch_official_category_records(category: str) -> list[dict[str, Any]]:
    jsonl_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=f"bench_data/{category}.jsonl",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return _load_jsonl_records(jsonl_path)


def _group_tests_into_pages(
    records_by_category: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Group unit tests into 1 row per PDF page, injecting ``BaselineTest``s and interleaving."""
    pages_by_cat: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = {}
    for cat in SOURCE_CATEGORIES:
        pages_by_cat[cat] = defaultdict(list)
    extra_pages: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for category, records in records_by_category.items():
        for raw_test in records:
            pdf = str(raw_test["pdf"])
            page = int(raw_test.get("page", 1))
            test_entry = {k: v for k, v in raw_test.items() if v is not None}
            test_entry["group"] = category
            if category in pages_by_cat:
                pages_by_cat[category][(pdf, page)].append(test_entry)
            else:
                extra_pages[(pdf, page)].append(test_entry)

    category_page_lists: list[list[dict[str, Any]]] = []
    for category in SOURCE_CATEGORIES:
        cat_pages: list[dict[str, Any]] = []
        for (pdf, page), tests in sorted(pages_by_cat[category].items()):
            all_page_tests = list(tests)
            if not any(t.get("type") == "baseline" for t in all_page_tests):
                all_page_tests.append(
                    {
                        "id": f"{pdf}_baseline",
                        "pdf": pdf,
                        "page": page,
                        "type": "baseline",
                        "group": "baseline",
                    },
                )
            cat_pages.append(
                {
                    "doc_id": f"{category}/{pdf}:p{page}",
                    "pdf": pdf,
                    "page": page,
                    "category": category,
                    "num_tests": len(all_page_tests),
                    "tests_json": json.dumps(all_page_tests, ensure_ascii=False),
                },
            )
        if cat_pages:
            category_page_lists.append(cat_pages)

    for (pdf, page), tests in sorted(extra_pages.items()):
        all_page_tests = list(tests)
        cat = str(all_page_tests[0].get("group", "custom"))
        if not any(t.get("type") == "baseline" for t in all_page_tests):
            all_page_tests.append(
                {
                    "id": f"{pdf}_baseline",
                    "pdf": pdf,
                    "page": page,
                    "type": "baseline",
                    "group": "baseline",
                },
            )
        category_page_lists.append(
            [
                {
                    "doc_id": f"{cat}/{pdf}:p{page}",
                    "pdf": pdf,
                    "page": page,
                    "category": cat,
                    "num_tests": len(all_page_tests),
                    "tests_json": json.dumps(all_page_tests, ensure_ascii=False),
                },
            ],
        )

    interleaved: list[dict[str, Any]] = []
    max_len = max((len(lst) for lst in category_page_lists), default=0)
    for idx in range(max_len):
        for lst in category_page_lists:
            if idx < len(lst):
                interleaved.append(lst[idx])
    return interleaved


def _infer_category_from_pdf(pdf_relpath: str, explicit_category: Any = None) -> str:
    if explicit_category:
        return str(explicit_category)
    prefix = pdf_relpath.split("/", 1)[0]
    if prefix == "tables":
        return "table_tests"
    if prefix in SOURCE_CATEGORIES:
        return prefix
    return "arxiv_math"


def olmocr_bench_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Group unit-test rows into 1,403 page-level rows with deterministic category interleaving."""
    if "tests_json" in dataset.column_names:
        return dataset

    # Synthetic / unit-test dataset passed directly to process_docs
    if "_synthetic" in dataset.column_names or len(dataset) != 2927:
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset:
            cat = _infer_category_from_pdf(str(row["pdf"]), row.get("category"))
            by_cat[cat].append(dict(row))
        pages = _group_tests_into_pages(by_cat)
        return datasets.Dataset.from_list(pages)

    records_by_category = {cat: _fetch_official_category_records(cat) for cat in SOURCE_CATEGORIES}
    pages = _group_tests_into_pages(records_by_category)
    total_tests = sum(int(p["num_tests"]) for p in pages)
    eval_logger.info(
        f"olmOCR-Bench: grouped {total_tests} unit tests across {len(pages)} PDF pages.",
    )
    return datasets.Dataset.from_list(pages)


def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    """Official ``get_pdf_media_box_width_height`` from ``olmocr/data/renderpdf.py:9-36``."""
    command = [
        "pdfinfo",
        "-f",
        str(page_num),
        "-l",
        str(page_num),
        "-box",
        "-enc",
        "UTF-8",
        local_pdf_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")
    for line in result.stdout.splitlines():
        if "MediaBox" in line:
            media_box = [float(x) for x in line.split(":", 1)[1].strip().split()]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])
    raise ValueError("MediaBox not found in the PDF info.")


@lru_cache(maxsize=64)
def render_pdf_page_to_png_bytes(
    local_pdf_path: str,
    page_num: int = 1,
    target_longest_image_dim: int = TARGET_LONGEST_IMAGE_DIM,
) -> bytes:
    """Render a PDF page to PNG bytes at ``target_longest_image_dim`` (``renderpdf.py:39-60``)."""
    if shutil.which("pdfinfo") and shutil.which("pdftoppm"):
        width_pt, height_pt = get_pdf_media_box_width_height(local_pdf_path, page_num)
        longest_dim = max(width_pt, height_pt)
        dpi = target_longest_image_dim * 72.0 / longest_dim
        pdftoppm_result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                "-r",
                str(dpi),
                local_pdf_path,
            ],
            timeout=120,
            capture_output=True,
            check=False,
        )
        if pdftoppm_result.returncode != 0:
            err_msg = pdftoppm_result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"pdftoppm failed on {local_pdf_path} p{page_num}: {err_msg}")
        return pdftoppm_result.stdout

    message = "olmOCR-Bench requires Poppler ('pdfinfo' and 'pdftoppm'). "
    message += "Install poppler-utils on Linux or poppler with Homebrew on macOS."
    raise RuntimeError(message)


@lru_cache(maxsize=256)
def _resolve_hf_pdf_path(pdf_relpath: str) -> str:
    return hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=f"bench_data/pdfs/{pdf_relpath}",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )


def _decode_visual_item(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (bytes, bytearray)):
        with Image.open(io.BytesIO(item)) as img:
            return img.convert("RGB")
    if isinstance(item, dict):
        if item.get("bytes") is not None:
            with Image.open(io.BytesIO(bytes(item["bytes"]))) as img:
                return img.convert("RGB")
        if item.get("path"):
            with Image.open(item["path"]) as img:
                return img.convert("RGB")
    raise TypeError(f"Unsupported visual representation: {type(item)!r}")


def olmocr_bench_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Render or decode the single-page RGB image for an olmOCR-Bench document."""
    for key in ("image", "page_image"):
        if doc.get(key) is not None:
            return [_decode_visual_item(doc[key])]

    page_num = int(doc.get("page", 1))
    pdf_path = doc.get("pdf_path") or _resolve_hf_pdf_path(str(doc["pdf"]))
    png_bytes = render_pdf_page_to_png_bytes(
        str(pdf_path),
        page_num=page_num,
        target_longest_image_dim=TARGET_LONGEST_IMAGE_DIM,
    )
    with Image.open(io.BytesIO(png_bytes)) as img:
        return [img.convert("RGB")]


def olmocr_bench_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: dict[str, Any] | None = None,
) -> str:
    """Return the official olmOCR-Bench VLM prompt."""
    del doc
    kwargs = lmms_eval_specific_kwargs or {}
    template_name = str(kwargs.get("prompt_template", "full_no_document_anchoring"))
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(
            f"olmOCR-Bench: unknown prompt_template={template_name!r}; expected one of {sorted(PROMPT_TEMPLATES)}",
        )
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    post_prompt = str(kwargs.get("post_prompt", ""))
    return f"{pre_prompt}{PROMPT_TEMPLATES[template_name]}{post_prompt}"


# Scoring & rank-0 macro-aggregation


def _extract_doc_tests(doc: dict[str, Any]) -> list[dict[str, Any]]:
    raw = doc.get("tests_json") or doc.get("tests")
    if isinstance(raw, str):
        parsed = json.loads(raw)
    elif isinstance(raw, list):
        parsed = raw
    else:
        raise ValueError(f"olmOCR-Bench: missing 'tests_json' in doc={doc.get('pdf')!r}")
    if not parsed:
        raise ValueError(f"olmOCR-Bench: empty unit-test list for doc={doc.get('pdf')!r}")
    return list(parsed)


def olmocr_bench_process_results(
    doc: dict[str, Any],
    results: list[str],
) -> dict[str, Any]:
    """Run all unit tests for one PDF page and emit per-category pass/total counts."""
    raw_prediction = results[0] if results else ""
    cleaned_md = clean_model_prediction(raw_prediction)
    tests = _extract_doc_tests(doc)

    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    total_passed = 0
    for test in tests:
        group = str(test.get("group") or doc.get("category") or "baseline")
        passed = bool(run_single_pdf_test(test, cleaned_md))
        group_counts[group]["total"] += 1
        if passed:
            group_counts[group]["passed"] += 1
            total_passed += 1

    serializable_counts: dict[str, dict[str, int]] = {}
    for grp, counts in group_counts.items():
        serializable_counts[grp] = {
            "passed": int(counts["passed"]),
            "total": int(counts["total"]),
        }
    overall_record = {
        "pdf": str(doc.get("pdf", "")),
        "category": str(doc.get("category", "")),
        "passed": int(total_passed),
        "total": int(len(tests)),
        "group_counts": serializable_counts,
    }

    out: dict[str, Any] = {"olmocr_bench_overall": overall_record}
    for category in ALL_CATEGORIES:
        out[f"olmocr_bench_{category}"] = serializable_counts.get(category)
    return out


def _aggregate_category_counts(results: list[Any], category: str) -> float:
    passed_sum = 0
    total_sum = 0
    for item in results:
        if item is None:
            continue
        if isinstance(item, dict):
            if "group_counts" in item:
                grp = item["group_counts"].get(category)
                if grp:
                    passed_sum += int(grp["passed"])
                    total_sum += int(grp["total"])
            elif "passed" in item and "total" in item:
                passed_sum += int(item["passed"])
                total_sum += int(item["total"])
    return (passed_sum / total_sum) if total_sum > 0 else float("nan")


def olmocr_bench_aggregate_overall(results: list[dict[str, Any]]) -> float:
    """Unweighted mean (macro-average) of active category pass rates (``benchmark.py:380-389``)."""
    if not results:
        return float("nan")
    category_rates: list[float] = []
    for category in ALL_CATEGORIES:
        rate = _aggregate_category_counts(results, category)
        if not math.isnan(rate):
            category_rates.append(rate)
            eval_logger.info(f"olmOCR-Bench {category}: {rate * 100.0:.2f}%")
    if not category_rates:
        return float("nan")
    overall = sum(category_rates) / len(category_rates)
    eval_logger.info(f"olmOCR-Bench overall ({len(category_rates)} categories): {overall * 100.0:.2f}%")
    return overall


def olmocr_bench_aggregate_arxiv_math(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "arxiv_math")


def olmocr_bench_aggregate_old_scans_math(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "old_scans_math")


def olmocr_bench_aggregate_table_tests(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "table_tests")


def olmocr_bench_aggregate_old_scans(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "old_scans")


def olmocr_bench_aggregate_headers_footers(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "headers_footers")


def olmocr_bench_aggregate_multi_column(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "multi_column")


def olmocr_bench_aggregate_long_tiny_text(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "long_tiny_text")


def olmocr_bench_aggregate_baseline(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "baseline")
