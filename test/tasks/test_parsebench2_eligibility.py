"""References the official extended scorer cannot score are handled as `parse_bench evaluation run --next_only` does."""

import math
import os
from pathlib import Path

import pytest

from lmms_eval.tasks.parsebench2 import utils


@pytest.fixture
def scorer() -> None:
    pytest.importorskip("parse_bench")
    utils.require_scorer()


def _scored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, response: str | None, **reference: object) -> dict:
    from parse_bench.test_cases.schema import ParseTestCase

    doc = {"example_id": f"text/{name}", "subset": "text", "document": f"text/{name}.pdf"}
    case = ParseTestCase(test_id=doc["example_id"], group="text", file_path=tmp_path / f"{name}.pdf", **reference)
    monkeypatch.setattr(utils, "load_case", lambda document: case)
    return utils.process_results(doc, [response])["parsebench2_text"]


def test_reference_without_rules_or_markdown_is_skipped(scorer: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    good = _scored(monkeypatch, tmp_path, "good", "# Heading\n\nA **bold** sentence.", expected_markdown="# Heading\n\nA **bold** sentence.")
    skipped = _scored(monkeypatch, tmp_path, "empty", "A non-empty prediction", expected_markdown=None, test_rules=None)

    assert skipped["evaluation"] is None
    assert utils.aggregate_text([good, skipped]) == utils.aggregate_text([good])
    assert utils.combined_report([good, skipped])["dimensions"]["text"]["slots"][0]["documents"] == 1


def test_reference_without_an_extended_metric_fails_and_counts_zero(scorer: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    good = _scored(monkeypatch, tmp_path, "good", "# Heading\n\nA **bold** sentence.", expected_markdown="# Heading\n\nA **bold** sentence.")
    rules = [{"type": "is_code_block", "language": "fortran", "code": "PRINT *, 181"}]
    failed = _scored(monkeypatch, tmp_path, "code", "A non-empty prediction", expected_markdown=None, test_rules=rules)

    assert failed["evaluation"]["success"] is False
    assert failed["evaluation"]["error"] == "No extended metric applies to this output"
    assert utils.aggregate_text([good, failed]) == pytest.approx(utils.aggregate_text([good]) / 2)


def test_missing_prediction_with_valid_reference_still_scores_zero(scorer: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _scored(monkeypatch, tmp_path, "eligible", None, expected_markdown="# Heading\n\nA **bold** sentence.")

    assert result["evaluation"]["success"] is True
    assert utils.aggregate_text([result]) == 0


@pytest.mark.parametrize(("stem", "skipped"), [("text_ocr__code", False), ("text_sparse__anex", True)])
def test_real_pinned_text_cases_follow_the_official_runner(scorer: None, stem: str, skipped: bool) -> None:
    if os.environ.get("LMMS_PARSEBENCH2_INTEGRATION") != "1":
        pytest.skip("set LMMS_PARSEBENCH2_INTEGRATION=1 for public reference checks")
    doc = {"example_id": f"text/{stem}", "subset": "text", "document": f"text/{stem}.pdf"}
    item = utils.process_results(doc, ["non-empty prediction"])["parsebench2_text"]

    if skipped:
        assert item["evaluation"] is None
    else:
        assert item["evaluation"]["success"] is False
        # The official aggregation pads failures only next to successful documents; alone, the slot is missing.
        assert math.isnan(utils.aggregate_text([item]))
