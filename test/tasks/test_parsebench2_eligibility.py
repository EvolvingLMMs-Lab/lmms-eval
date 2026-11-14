"""Reference/scorer eligibility must never silently remove an example."""

import os
from pathlib import Path

import pytest

from lmms_eval.tasks.parsebench2 import utils


@pytest.fixture
def scorer() -> None:
    pytest.importorskip("parse_bench")
    utils.require_scorer()


@pytest.mark.parametrize("rules", [None, [{"type": "is_code_block", "language": "fortran", "code": "PRINT *, 181"}]])
def test_ineligible_reference_stops_before_success_record(scorer: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rules: list | None) -> None:
    from parse_bench.test_cases.schema import ParseTestCase

    doc = {"example_id": "text/ineligible", "subset": "text", "document": "text/ineligible.pdf"}
    case = ParseTestCase(test_id=doc["example_id"], group="text", file_path=tmp_path / "ineligible.pdf", expected_markdown=None, test_rules=rules, metadata={"scoring": "next"})
    monkeypatch.setattr(utils, "load_case", lambda document: case)
    # The actual official extended scorer returns [] for both corpus shapes.
    with pytest.raises(RuntimeError, match="text/ineligible.*no official extended metrics"):
        utils.process_results(doc, ["A non-empty provider prediction"])


@pytest.mark.parametrize("success", [True, False])
def test_empty_cached_metric_record_cannot_be_aggregated(scorer: None, success: bool) -> None:
    from parse_bench.schemas.evaluation import EvaluationResult

    value = EvaluationResult(test_id="text/old-cache", example_id="text/old-cache", pipeline_name="saved", product_type="parse", success=success, metrics=[])
    item = {"subset": "text", "evaluation": value.model_dump(mode="json"), "layout_supported": False, "response_status": "official_record"}
    with pytest.raises(RuntimeError, match="text/old-cache.*no official extended metrics"):
        utils.combined_report([item])


def test_missing_prediction_with_valid_reference_still_scores_zero(scorer: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from parse_bench.test_cases.schema import ParseTestCase

    doc = {"example_id": "text/eligible", "subset": "text", "document": "text/eligible.pdf"}
    case = ParseTestCase(test_id=doc["example_id"], group="text", file_path=tmp_path / "eligible.pdf", expected_markdown="# Heading\n\nA **bold** sentence.")
    monkeypatch.setattr(utils, "load_case", lambda document: case)
    result = utils.process_results(doc, [None])
    assert result["parsebench2_response_failure_rate"] == 1
    assert result["parsebench2_text"]["evaluation"]["success"] is True
    assert utils.aggregate_text([result["parsebench2_text"]]) == 0


@pytest.mark.parametrize("stem", ["text_ocr__code", "text_sparse__anex"])
def test_real_pinned_ineligible_text_cases_fail_closed(scorer: None, stem: str) -> None:
    if os.environ.get("LMMS_PARSEBENCH2_INTEGRATION") != "1":
        pytest.skip("set LMMS_PARSEBENCH2_INTEGRATION=1 for public reference checks")
    doc = {"example_id": f"text/{stem}", "subset": "text", "document": f"text/{stem}.pdf"}
    with pytest.raises(RuntimeError, match=stem + ".*no official extended metrics"):
        utils.process_results(doc, ["non-empty prediction"])
