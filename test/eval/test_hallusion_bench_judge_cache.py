import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lmms_eval.tasks.hallusion_bench import evaluate_hb as hb
from lmms_eval.tasks.hallusion_bench import utils as hb_utils

ANSWERS = ["yes", "no", "yes", "no"]


class _Judge:
    """Stands in for the GPT judge: 'correct' iff the prediction starts with the same Yes/No as the reference."""

    def __init__(self):
        self.calls = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        prompt = json["messages"][0]["content"]
        reference = prompt.split("Reference answer: ")[1].split("\nPrediction answer:")[0]
        prediction = prompt.split("Prediction answer:")[1].split("\nOutput:")[0]
        verdict = "correct" if reference.split()[0].strip(",.").lower() == prediction.split()[0].strip(",.").lower() else "incorrect"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"choices": [{"message": {"content": verdict}}]})


class _ChatCompletionsHandler(BaseHTTPRequestHandler):
    """Local stand-in that speaks the chat-completions shape the GPT judge posts."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.payloads.append(payload)
        prompt = payload["messages"][0]["content"]
        reference = prompt.split("Reference answer: ")[1].split("\nPrediction answer:")[0]
        prediction = prompt.split("Prediction answer:")[1].split("\nOutput:")[0]
        same_answer = reference.split()[0].strip(",.").lower() == prediction.split()[0].strip(",.").lower()
        verdict = "correct" if same_answer else "incorrect"
        body = json.dumps({"choices": [{"message": {"content": verdict}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def judge(monkeypatch):
    judge = _Judge()
    monkeypatch.setattr(hb_utils.requests, "post", judge.post)
    return judge


def _results(predictions):
    return [
        {
            "category": "VD",
            "subcategory": "illusion",
            "visual_input": "1",
            "set_id": "0",
            "figure_id": str(i),
            "question_id": "0",
            "question": f"Is it {i}?",
            "gt_answer": "1" if answer == "yes" else "0",
            "gt_answer_details": f"{answer.capitalize()}, it is.",
            "model_prediction": prediction,
        }
        for i, (answer, prediction) in enumerate(zip(ANSWERS, predictions))
    ]


def _right():
    return [answer.capitalize() for answer in ANSWERS]


def _wrong():
    return ["No" if answer == "yes" else "Yes" for answer in ANSWERS]


def test_second_model_does_not_reuse_judgments_of_the_first_in_the_same_output_path(judge, tmp_path):
    args = SimpleNamespace(output_path=str(tmp_path))

    assert hb.hb_aggregation_result_aAcc(_results(_right()), args) == 100.0
    assert hb.hb_aggregation_result_aAcc(_results(_wrong()), args) == 0.0
    assert judge.calls == 2 * len(ANSWERS)


def test_metrics_of_one_run_share_the_judgments(judge, tmp_path):
    args = SimpleNamespace(output_path=str(tmp_path))
    results = _results(_right())

    assert hb.hb_aggregation_result_aAcc(results, args) == 100.0
    assert hb.hb_aggregation_result_qAcc(results, args) == 100.0
    assert hb.hb_aggregation_result_fAcc(results, args) == 100.0
    assert judge.calls == len(ANSWERS)


@pytest.mark.parametrize("args", [SimpleNamespace(output_path=None), None])
def test_aggregation_without_output_path(judge, tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)

    assert hb.hb_aggregation_result_aAcc(_results(_right()), args) == 100.0
    assert os.listdir(tmp_path / "gpt_response")


def test_offline_chat_completions_judge_is_keyed_per_prediction(tmp_path):
    """The real requests path, against a local chat-completions server, not GPT-4."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletionsHandler)
    server.payloads = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous_url = hb_utils.API_URL
    previous_headers = hb_utils.headers
    try:
        port = server.server_address[1]
        hb_utils.API_URL = f"http://127.0.0.1:{port}/v1/chat/completions"
        hb_utils.headers = {"Authorization": "Bearer offline-judge", "Content-Type": "application/json"}
        args = SimpleNamespace(output_path=str(tmp_path))

        assert hb.hb_aggregation_result_aAcc(_results(_right()), args) == 100.0
        assert hb.hb_aggregation_result_aAcc(_results(_wrong()), args) == 0.0
    finally:
        hb_utils.API_URL = previous_url
        hb_utils.headers = previous_headers
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert len(server.payloads) == 2 * len(ANSWERS)
    first = server.payloads[0]
    assert first["model"] == "gpt-4"
    assert first["max_tokens"] == 16
    assert first["messages"] == [{"role": "user", "content": first["messages"][0]["content"]}]
    assert "Prediction answer:Yes" in first["messages"][0]["content"]
    second_model = server.payloads[len(ANSWERS)]
    assert "Prediction answer:No" in second_model["messages"][0]["content"]

    cache_files = sorted((tmp_path / "gpt_response").glob("hallusion_output_vd_*.json"))
    assert len(cache_files) == 2
    assert all(path.name != "hallusion_output_vd_model.json" for path in cache_files)
    stored = [json.loads(path.read_text()) for path in cache_files]
    sequences = [[row["model_prediction"] for row in rows] for rows in stored]
    assert sorted(sequences) == sorted([_right(), _wrong()])
