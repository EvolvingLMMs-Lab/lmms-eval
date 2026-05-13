from pathlib import Path
from types import SimpleNamespace

import pytest

from lmms_eval.tasks.ocrbench_v2 import spotting_metric, utils, vqa_metric


def test_ocrbench_v2_chart_parsing_uses_ground_truth(monkeypatch):
    captured = {}

    class DummyTEDS:
        def evaluate(self, pred_html, gt_html):
            captured["pred_html"] = pred_html
            captured["gt_html"] = gt_html
            return 0.42

    monkeypatch.setattr(utils, "teds", DummyTEDS())
    monkeypatch.setattr(utils, "convert_str_to_multi_dict", lambda raw: {"source": raw})
    monkeypatch.setattr(utils, "dict_to_html", lambda data: f"html::{data['source']}")

    result = utils.ocrbench_v2_process_results(
        {
            "question": "Parse the chart into a structured representation.",
            "answers": ["ground-truth-chart"],
            "type": "chart parsing en",
        },
        ["predicted-chart"],
    )

    assert captured == {
        "pred_html": "html::predicted-chart",
        "gt_html": "html::ground-truth-chart",
    }
    assert result["ocrbench_v2_accuracy"]["score"] == pytest.approx(0.42)


def test_ocrbench_v2_aggregate_accuracy_is_stateless(tmp_path):
    args = SimpleNamespace(output_path=str(tmp_path))

    first_score = utils.ocrbench_v2_aggregate_accuracy(
        [{"question_type": "text recognition en", "score": 1.0}],
        args,
    )
    second_score = utils.ocrbench_v2_aggregate_accuracy(
        [{"question_type": "text recognition en", "score": 0.0}],
        args,
    )

    assert first_score == pytest.approx(0.0625)
    assert second_score == 0.0


def test_spotting_evaluation_uses_temp_workdir(monkeypatch):
    captured = {}
    module_dir = Path(spotting_metric.__file__).resolve().parent / "spotting_eval"

    def fake_main_evaluation(command, default_params, validate, evaluate):
        captured["command"] = command
        return {"method": {"hmean": 0.9}}

    monkeypatch.setattr(spotting_metric.rrc_evaluation_funcs, "main_evaluation", fake_main_evaluation)

    score = spotting_metric.spotting_evaluation(
        [[0, 0, 10, 10, "hello"]],
        {"bbox_list": [[0, 0, 10, 0, 10, 10, 0, 10]], "content": ["hello"]},
    )

    assert score == pytest.approx(0.9)
    assert module_dir not in Path(captured["command"]["g"]).resolve().parents
    assert module_dir not in Path(captured["command"]["s"]).resolve().parents
    assert not (module_dir / "submit").exists()
    assert not (module_dir / "gt").exists()


def test_cn_vqa_evaluation_supports_scalar_answers():
    assert vqa_metric.cn_vqa_evaluation("答案是北京", "北京") == 1


def test_counting_evaluation_supports_scalar_answers():
    assert vqa_metric.counting_evaluation("There are 3 objects.", "3", "exact match") == 1
