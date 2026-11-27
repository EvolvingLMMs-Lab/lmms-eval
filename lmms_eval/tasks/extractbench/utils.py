"""Full-document ExtractBench adaptation using the pinned official evaluator."""

import importlib.metadata
import json
import re
import sys
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from huggingface_hub import hf_hub_download

DATASET_REPO = "llamaindex/ExtractBench"
DATASET_REVISION = "f9b910906cd7e73cc9eb5379a8d5d935c67b92c4"
SCORER_VERSION = "1.0.0"
SCHEMA_VERSION = "2.0.0"
RENDER_DPI = 150
_THINK_RE = re.compile(r"^\s*<think>.*?(?:</think>\s*|\Z)", re.DOTALL)


@lru_cache(maxsize=1)
def check_dependency():
    """Require the published scorer and schema versions used by this task."""
    message = "ExtractBench requires Python >=3.12 and the pinned scorer; install lmms_eval[extractbench]."
    if sys.version_info < (3, 12):
        raise RuntimeError(message)
    try:
        for package, expected in (("llama-extract-bench", SCORER_VERSION), ("parse-bench", SCHEMA_VERSION)):
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f"{message} Expected {package}=={expected}.")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(message) from error


def _decode(doc, key, default=None):
    value = doc.get(key, default)
    return json.loads(value) if isinstance(value, str) else value


def _pdf_name(doc):
    path = PurePosixPath(doc["pdf"])
    unsafe = path.is_absolute() or ".." in path.parts or "\\" in str(path)
    if unsafe or path.parts[:1] != ("docs",) or path.suffix.lower() != ".pdf":
        raise ValueError(f"ExtractBench has an unsafe PDF path: {doc['pdf']!r}")
    return str(path)


def case_from_doc(doc):
    """Mirror the official HF downloader and sidecar loader without writing files."""
    check_dependency()
    from extract_bench.test_cases.schema import ExtractTestCase

    rules = _decode(doc, "field_rules", {}) or {}
    if isinstance(rules, dict):
        converted = []
        for field_path, body in rules.items():
            if not isinstance(body, dict) or body.get("type") not in (None, "extract_field"):
                raise ValueError(f"Invalid ExtractBench field rule: {field_path!r}")
            converted.append({**body, "type": "extract_field", "field_path": field_path})
        rules = converted
    elif not isinstance(rules, list):
        raise ValueError("ExtractBench field_rules must be an object or list")
    return ExtractTestCase(
        test_id=doc["id"],
        group=doc["category"],
        file_path=Path(_pdf_name(doc)),
        tags=doc.get("tags") or [],
        schema=_decode(doc, "data_schema"),
        expected_output=_decode(doc, "expected_output"),
        test_rules=rules,
        _eval_row_identity=_decode(doc, "repeated_structure", {}) or None,
    )


def extractbench_process_docs(dataset):
    """Validate complete rows before inference without filtering or reordering them."""
    seen = set()
    for doc in dataset:
        if doc["id"] in seen:
            raise ValueError(f"Duplicate ExtractBench document ID: {doc['id']}")
        seen.add(doc["id"])
        if doc["category"] not in {"short", "medium", "long"} or not doc["id"].startswith(doc["category"] + "/"):
            raise ValueError(f"Invalid ExtractBench split/ID: {doc['id']}")
        case = case_from_doc(doc)
        if case.expected_output is None:
            raise ValueError(f"ExtractBench document has no expected output: {doc['id']}")
    return dataset


def extractbench_doc_to_visual(doc):
    """Render every PDF page, in source order, with the official Poppler path."""
    filename = _pdf_name(doc)
    try:
        from pdf2image import convert_from_path
    except ImportError as error:
        raise RuntimeError("Install the extractbench extra and Poppler to render ExtractBench PDFs") from error
    pdf = hf_hub_download(DATASET_REPO, filename, repo_type="dataset", revision=DATASET_REVISION)
    pages = convert_from_path(pdf, dpi=RENDER_DPI)
    if not pages:
        raise ValueError(f"ExtractBench PDF has no pages: {doc['id']}")
    return [page.convert("RGB") for page in pages]


def _prompts(doc):
    check_dependency()
    from extract_bench.inference.providers.extract.direct_model_utils import (
        DEFAULT_SYSTEM_PROMPT,
        DEFAULT_USER_INSTRUCTION,
        add_additional_properties_false,
        promote_repeated_structure,
    )

    schema = add_additional_properties_false(promote_repeated_structure(_decode(doc, "data_schema")))
    prompt = (
        "Extract every field from the attached document page image(s) according to the JSON schema below.\n\n"
        "JSON schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
        f"{DEFAULT_USER_INSTRUCTION}"
    )
    return DEFAULT_SYSTEM_PROMPT, prompt


def extractbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Flatten the two prompts for simple models; chat models preserve their roles."""
    return "\n\n".join(_prompts(doc))


def extractbench_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    system, prompt = _prompts(doc)
    content = [{"type": "image", "url": page} for page in extractbench_doc_to_visual(doc)]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


def extractbench_doc_to_target(doc):
    return json.dumps(_decode(doc, "expected_output"), ensure_ascii=False)


def parse_prediction(response):
    """Use official leading-think removal, truncated-JSON salvage and depth pruning."""
    check_dependency()
    from extract_bench.inference.providers.extract.vllm_extract import prune_excess_depth, salvage_truncated_json

    if not isinstance(response, str):
        raise TypeError("ExtractBench expects a captured text response, not a transport-failure sentinel")
    if response.lstrip().startswith(("[LMMS_EVAL_REQUEST_FAILED ", "[LMMS_EVAL_BUDGET_EXCEEDED]")):
        raise RuntimeError("ExtractBench received a backend failure sentinel; inspect the request logs")
    content = _THINK_RE.sub("", response, count=1)
    salvaged = False
    try:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = salvage_truncated_json(content)
            if data is None:
                return None, False, False
            salvaged = True
        data, _ = prune_excess_depth(data)
    except (ValueError, RecursionError):
        # Oversized integers or excessive nesting are model parse failures.
        return None, False, False
    return data, True, salvaged


@lru_cache(maxsize=1)
def _evaluator():
    check_dependency()
    from extract_bench.evaluation.evaluators.extract import ExtractEvaluator

    return ExtractEvaluator()


def extractbench_process_results(doc, results):
    """Score one response, retaining every document in every metric's denominator."""
    if len(results) != 1:
        raise ValueError("ExtractBench expects exactly one response per document")
    case = case_from_doc(doc)
    data, parsed, salvaged = parse_prediction(results[0])
    value_f1 = 0.0
    if parsed:
        from extract_bench.schemas.extract_output import ExtractOutput
        from extract_bench.schemas.product import ProductType

        # This is the official direct-provider normalization for falsy JSON roots.
        normalized = data or {}
        valid_rows = isinstance(normalized, list) and all(isinstance(item, dict) for item in normalized)
        if isinstance(normalized, dict) or valid_rows:
            output = ExtractOutput(
                task_type="extract",
                example_id=doc["id"],
                pipeline_name="extractbench_full_document",
                extracted_data=normalized,
                field_citations=[],
            )
            scoring_input = SimpleNamespace(product_type=ProductType.EXTRACT, output=output)
            bundle = _evaluator().compute_metrics(scoring_input, case)
            scores = {metric.metric_name: metric.value for metric in bundle.metrics}
            value_f1 = float(scores.get("extract_unified_value_f1", 0.0))
    return {
        "extractbench_value_f1": value_f1,
        "extractbench_perfect": float(value_f1 == 1.0),
        "extractbench_parse_success": float(parsed),
        "extractbench_salvaged": float(salvaged),
    }
