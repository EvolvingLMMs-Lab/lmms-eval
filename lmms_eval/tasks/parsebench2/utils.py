"""ParseBench-2 inputs and pinned official scoring; no provider inference here."""

import copy
import json
import math
from functools import lru_cache
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any

DATASET = "llamaindex/ParseBench-2"
DATA_REVISION = "a12c9785cd8cd015b5284e0ad0d74722091762f2"
SCORER_VERSION = "2.0.1"
RENDERER_VERSION = "4.30.0"
SUBSETS = (
    "text",
    "headings",
    "strikeout",
    "underline",
    "tables",
    "complex_table_headers",
    "financial_tables",
    "insurance_filing_tables",
    "dense_tables",
    "charts",
    "layout",
    "forms",
    "fillable_forms",
)
DIMENSIONS = ("text", "formatting", "tables", "charts", "layout", "forms")
PROMPT = """Transcribe every supplied document page in order. Return a JSON object with "markdown" (the full document) and "layout_pages" (one entry per page).
Preserve all text, headings, bold, italic, strikethrough (<s>), underline (<u>), lists, formulas, headers and footers. Use HTML tables with correct row/column spans. Represent charts as tables of their data. Preserve form labels, values, choices and checkbox states.
Each layout_pages entry has "page_number" (1-based), "width": 1, "height": 1, and "items" in reading order. Each item has "type" (text, heading, table or picture), "md" (its Markdown/HTML text), and "bbox": {"x": left, "y": top, "w": width, "h": height}, with coordinates normalized to [0,1] relative to that page. Use separate items for separate regions. Do not invent content. Return only the JSON object, without code fences."""


@lru_cache(maxsize=1)
def require_scorer() -> None:
    """Fail early on unsupported Python/dependencies, without importing at discovery."""
    import sys
    from importlib.metadata import PackageNotFoundError

    if sys.version_info < (3, 12):
        raise RuntimeError("ParseBench-2 requires Python >=3.12; install lmms_eval[parsebench2].")
    try:
        installed = version("parse-bench")
    except PackageNotFoundError as exc:
        raise RuntimeError("Install lmms_eval[parsebench2] to score ParseBench-2.") from exc
    if installed != SCORER_VERSION:
        raise RuntimeError(f"ParseBench-2 requires parse-bench=={SCORER_VERSION}, found {installed}.")


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError(f"Invalid benchmark path: {value!r}")
    return str(path)


def fetch(filename: str) -> str:
    """Resolve only files of the immutable public benchmark snapshot."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET, _safe_path(filename), repo_type="dataset", revision=DATA_REVISION)


def input_record(row: dict[str, Any], subset: str) -> dict[str, str]:
    """Whitelist inference fields; annotations never enter the harness dataset."""
    example_id = _safe_path(row["id"])
    if "/" not in example_id:
        example_id = f"{subset}/{example_id}"
    document = _safe_path(row["document"])
    if example_id.split("/")[0] != subset or len(PurePosixPath(example_id).parts) != 2:
        raise ValueError(f"Unexpected example ID: {example_id}")
    if str(PurePosixPath(document).with_suffix("")) != example_id:
        raise ValueError(f"Document and example ID disagree: {document}, {example_id}")
    return {"example_id": example_id, "subset": subset, "document": document}


def load_inputs() -> list[dict[str, str]]:
    """Interleave all 13 manifests so small limits cover multiple dimensions."""
    groups = []
    seen = set()
    for subset in SUBSETS:
        with open(fetch(f"{subset}.jsonl"), encoding="utf-8") as handle:
            rows = [input_record(json.loads(line), subset) for line in handle if line.strip()]
        for row in rows:
            if row["example_id"] in seen:
                raise ValueError(f"Duplicate example ID: {row['example_id']}")
            seen.add(row["example_id"])
        if not rows:
            raise ValueError(f"Empty benchmark subset: {subset}")
        groups.append(rows)
    return [row for batch in zip_longest(*groups) for row in batch if row is not None]


def doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Fixed, annotation-independent prompt (integration prompt v1)."""
    return PROMPT


def doc_to_visual(doc: dict[str, Any]) -> list[Any]:
    """Render every PDF page at 144 DPI or decode an original RGB image."""
    from loguru import logger
    from PIL import Image

    path = fetch(doc["document"])
    if Path(path).suffix.lower() != ".pdf":
        with Image.open(path) as image:
            images = [image.convert("RGB")]
        logger.debug("ParseBench-2 media {}: decoded RGB image {}", doc["example_id"], images[0].size)
        return images
    if version("pypdfium2") != RENDERER_VERSION:
        raise RuntimeError(f"ParseBench-2 rendering requires pypdfium2=={RENDERER_VERSION}.")
    import pypdfium2 as pdfium

    images = []
    pdf = pdfium.PdfDocument(path)
    try:
        for page in pdf:
            try:
                bitmap = page.render(scale=2)
                try:
                    images.append(bitmap.to_pil().convert("RGB"))
                finally:
                    bitmap.close()
            finally:
                page.close()
    finally:
        pdf.close()
    if not images:
        raise ValueError(f"Empty PDF: {doc['example_id']}")
    logger.debug("ParseBench-2 media {}: rendered {} RGB PDF page(s), sizes={}, PDFium {}, scale=2", doc["example_id"], len(images), [image.size for image in images], RENDERER_VERSION)
    return images


@lru_cache(maxsize=32)
def load_case(document: str) -> Any:
    """Load the official sidecar after generation, including local sidecar resolution."""
    from parse_bench.test_cases.loader import load_test_case

    source = Path(fetch(document))
    sidecar = Path(fetch(str(PurePosixPath(document).with_suffix(".test.json"))))
    case = load_test_case(source, sidecar, product_type_hint="PARSE")
    if case is None:
        raise ValueError(f"Missing reference test case: {document}")
    # HF snapshot paths can resolve to content-addressed blobs. The official
    # loader derives identity from filesystem names; the manifest is authoritative.
    identity = str(PurePosixPath(document).with_suffix(""))
    return case.model_copy(update={"test_id": identity, "group": PurePosixPath(document).parent.name})


def _native_result(doc: dict[str, Any], payload: dict[str, Any]) -> Any:
    from parse_bench.schemas.pipeline_io import InferenceResult

    payload = copy.deepcopy(payload)
    # A region bbox is a single attributed segment in our v1 response contract.
    # The official adapter deliberately ignores bbox-only attribution otherwise.
    for page in payload.get("layout_pages", []):
        if not isinstance(page, dict):
            raise ValueError("Invalid layout page")
        if page.get("width") != 1 or page.get("height") != 1:
            raise ValueError("Native layout pages require normalized width=height=1")
        for item in page.get("items", []):
            if "bbox" in item and "layout_segments" not in item:
                box = item["bbox"]
                if not isinstance(box, dict) or any(not isinstance(box.get(k), (int, float)) or not math.isfinite(box[k]) or not 0 <= box[k] <= 1 for k in ("x", "y", "w", "h")):
                    raise ValueError("Native layout bbox must be finite normalized xywh")
                if box["x"] + box["w"] > 1.000001 or box["y"] + box["h"] > 1.000001:
                    raise ValueError("Native layout bbox exceeds the page")
                item["layout_segments"] = [dict(box)]
        page.setdefault("md", "\n\n".join(item.get("md", "") for item in page.get("items", [])))
    # The pipeline is deliberately unique: official shape matching selects the
    # normalized ParseOutput adapter, without impersonating a provider/model.
    return InferenceResult.model_validate(
        {
            "request": {"example_id": doc["example_id"], "source_file_path": doc["document"], "product_type": "parse"},
            "pipeline_name": "lmms_eval_parsebench2_v1",
            "product_type": "parse",
            "raw_output": {},
            "output": {**payload, "task_type": "parse", "example_id": doc["example_id"], "pipeline_name": "lmms_eval_parsebench2_v1"},
            # Timing is unavailable at the task boundary, not an inference measurement.
            "started_at": "1970-01-01T00:00:00Z",
            "completed_at": "1970-01-01T00:00:00Z",
            "latency_in_ms": 0,
        }
    )


def decode_response(doc: dict[str, Any], response: str | None) -> tuple[Any, bool, str]:
    """Accept native JSON, plain Markdown, or a full official saved inference record.

    Missing/invalid responses become empty predictions, never dropped examples.
    Replay identity errors remain fatal. Full records retain raw provider fields.
    """
    from parse_bench.schemas.parse_output import ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceResult
    from pydantic import ValidationError

    if not response or not response.strip():
        return _native_result(doc, {"markdown": ""}), False, "missing"
    stripped = response.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    if not stripped.startswith("{"):
        return _native_result(doc, {"markdown": response}), False, "markdown_only"
    try:
        payload = json.loads(stripped)
        if "request" in payload and "output" in payload:
            record = InferenceResult.model_validate(payload)
            if record.request.example_id != doc["example_id"] or record.output.example_id != doc["example_id"]:
                raise ValueError("ParseBench-2 replay example IDs disagree")
            if record.pipeline_name != record.output.pipeline_name or not isinstance(record.output, ParseOutput):
                raise ValueError("ParseBench-2 replay requires a consistent parse pipeline")
            return record, "layout_pages" in payload["output"], "official_record"
        if not isinstance(payload.get("markdown"), str):
            return _native_result(doc, {"markdown": ""}), False, "invalid"
        try:
            record = _native_result(doc, payload)
        except (ValueError, TypeError, AttributeError):
            return _native_result(doc, {"markdown": ""}), False, "invalid"
        return record, "layout_pages" in payload, "structured"
    except (json.JSONDecodeError, ValidationError):
        return _native_result(doc, {"markdown": ""}), False, "invalid"


def _require_official_metrics(example_id: str, metrics: list[Any]) -> None:
    if not metrics:
        raise RuntimeError(
            f"ParseBench-2 {example_id}: no official extended metrics from parse-bench=={SCORER_VERSION} "
            f"at dataset revision {DATA_REVISION}; reference/scorer eligibility needs review. "
            "Refusing to omit this example or emit an incomplete aggregate."
        )


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    """Compute official document metrics once; emit JSON-safe records for rank 0."""
    require_scorer()
    from parse_bench.evaluation.metrics.parse.next_metrics import compute_next_metrics
    from parse_bench.schemas.evaluation import EvaluationResult

    if len(results) > 1:
        raise ValueError("ParseBench-2 requires one filtered response per document")
    record, layout_supported, status = decode_response(doc, results[0] if results else None)
    case = load_case(doc["document"])
    if case.test_id != doc["example_id"]:
        raise ValueError(f"Reference identity mismatch: {case.test_id}, {doc['example_id']}")
    # Call the underlying API directly: evaluator errors must stop the run,
    # rather than disappearing from the official survivor denominator.
    metrics = compute_next_metrics(record, case)
    _require_official_metrics(doc["example_id"], metrics)
    evaluation = EvaluationResult(
        test_id=case.test_id,
        example_id=doc["example_id"],
        pipeline_name=record.pipeline_name,
        product_type="layout_detection" if doc["subset"] == "layout" else "parse",
        success=True,
        metrics=metrics,
    )
    item = {"subset": doc["subset"], "evaluation": evaluation.model_dump(mode="json"), "layout_supported": layout_supported, "response_status": status}
    return {
        **{f"parsebench2_{name}": item for name in ("overall", *DIMENSIONS)},
        "parsebench2_response_failure_rate": float(status in {"missing", "invalid"}),
        "parsebench2_layout_coverage": float(layout_supported) if doc["subset"] == "layout" else None,
    }


def combined_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Use official subset macro aggregation and six-dimension weighting."""
    require_scorer()
    from parse_bench.evaluation.next_score import combine_next_score
    from parse_bench.evaluation.runner import EvaluationRunner
    from parse_bench.schemas.evaluation import EvaluationResult

    grouped: dict[str, list[Any]] = {}
    seen = set()
    for item in items:
        result = EvaluationResult.model_validate(item["evaluation"])
        _require_official_metrics(result.example_id, result.metrics)
        if result.example_id in seen:
            raise ValueError(f"Duplicate scored example: {result.example_id}")
        seen.add(result.example_id)
        grouped.setdefault(item["subset"], []).append(result)
    runner = EvaluationRunner(output_dir=Path("."), next_only=True)
    subset_metrics = {subset: runner.aggregate_metrics(rows) for subset, rows in grouped.items()}
    # Capability is declared for the run by any valid response carrying the
    # explicit layout channel (including an empty list), on any subset. Once
    # declared, missing/invalid/Markdown-only individual pages keep their
    # official zero predictions in the layout denominator. With no declaration,
    # a Markdown-only run must not look like a complete six-dimension evaluation.
    if not any(item["layout_supported"] for item in items):
        subset_metrics.pop("layout", None)
    return combine_next_score(subset_metrics, {subset: len(rows) for subset, rows in grouped.items()})


def _aggregate(items: list[dict[str, Any]], dimension: str) -> float:
    report = combined_report(items)
    value = report["next_score"] if dimension == "overall" else report["dimensions"][dimension]["score"]
    return float(value) if value is not None else math.nan


def aggregate_overall(items: list[dict[str, Any]]) -> float:
    """Official overall; NaN when any dimension is missing."""
    return _aggregate(items, "overall")


def aggregate_text(items: list[dict[str, Any]]) -> float:
    """Official text dimension."""
    return _aggregate(items, "text")


def aggregate_formatting(items: list[dict[str, Any]]) -> float:
    """Official formatting dimension."""
    return _aggregate(items, "formatting")


def aggregate_tables(items: list[dict[str, Any]]) -> float:
    """Official tables dimension."""
    return _aggregate(items, "tables")


def aggregate_charts(items: list[dict[str, Any]]) -> float:
    """Official charts dimension."""
    return _aggregate(items, "charts")


def aggregate_layout(items: list[dict[str, Any]]) -> float:
    """Official layout dimension, only for explicitly structured predictions."""
    return _aggregate(items, "layout")


def aggregate_forms(items: list[dict[str, Any]]) -> float:
    """Official forms dimension."""
    return _aggregate(items, "forms")


def aggregate_layout_coverage(items: list[float | None]) -> float:
    """Fraction of evaluated layout documents with an explicit layout channel."""
    present = [item for item in items if item is not None]
    return sum(present) / len(present) if present else math.nan
