"""Offline contracts for the FERMAT-PINK task (judge calls are faked)."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

pink = pytest.importorskip("pink")

from pink.rubric import SCORE_COLUMNS  # noqa: E402

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "fermat_pink"


def _load_utils():
    # lmms-eval imports utils.py separately for every `!function`; mimic that.
    spec = importlib.util.spec_from_file_location("utils", TASK_DIR / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doc(sid="img_1_pert_1.1"):
    return {
        "new_custom_id": sid,
        "image": Image.new("RGB", (32, 16), "white"),
        "orig_q": "Compute 2 \\times 3.\r\n",
        "pert_a": "2 \\times 3 = \\textcolor{red}{\\frac{10}{2}}",
    }


def test_messages_and_target():
    utils = _load_utils()
    doc = _doc()
    messages = utils.fermat_doc_to_messages(doc)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert [c["type"] for c in messages[1]["content"]] == ["image", "text"]
    assert utils.fermat_doc_to_target(doc) == "2 \\times 3 = \\frac{10}{2}"


def test_scoring_judges_once_and_applies_penalty(monkeypatch, tmp_path):
    utils_a, utils_b = _load_utils(), _load_utils()
    oracle = tmp_path / "oracle.csv"
    oracle.write_text("sample_id,success," + ",".join(SCORE_COLUMNS) + "\n" + "s1,True,20,20,20,5,5\n" + "s2,True,10,10,10,10,10\n")
    calls = []

    def fake_grade(client, cfg, problem, solution, sample_id=""):
        calls.append(sample_id)
        scores = [20, 20, 20, 20, 20] if sample_id == "s1" else [10, 10, 10, 10, 10]
        return {"sample_id": sample_id, "success": True, **dict(zip(SCORE_COLUMNS, scores))}

    monkeypatch.setattr(pink.judge, "grade", fake_grade)
    monkeypatch.setattr(pink.judge, "make_client", lambda cfg: object())
    monkeypatch.setenv("PINK_JUDGE", "qwen3-next-80b")
    monkeypatch.setenv("PINK_ORACLE_GRADES", str(oracle))

    items = [utils_a.fermat_process_results({**_doc(sid), "new_custom_id": sid}, [f"**Answer:** {sid}"])["pink"] for sid in ("s1", "s2")]
    pink_score = utils_a.fermat_pink_aggregate(items)
    oc_rate = utils_b.fermat_over_correction_aggregate(list(items))

    # s1: components 4-5 over-corrected by +15 -> nullified: 60; s2 faithful: 50. Oracle: 70 + 50.
    assert pink_score == pytest.approx((60 + 50) / (70 + 50))
    assert oc_rate == pytest.approx(0.5)
    assert sorted(calls) == ["s1", "s2"]  # graded once, shared across aggregations


def test_empty_prediction_scores_zero(monkeypatch, tmp_path):
    utils = _load_utils()
    oracle = tmp_path / "oracle.csv"
    oracle.write_text("sample_id,success," + ",".join(SCORE_COLUMNS) + "\n" + "e1,True,10,10,10,10,10\n")
    monkeypatch.setattr(pink.judge, "make_client", lambda cfg: object())
    monkeypatch.setenv("PINK_JUDGE", "qwen3-next-80b")
    monkeypatch.setenv("PINK_ORACLE_GRADES", str(oracle))
    item = utils.fermat_process_results({**_doc("e1"), "new_custom_id": "e1"}, [""])["pink"]
    assert utils.fermat_pink_aggregate([item]) == 0.0
