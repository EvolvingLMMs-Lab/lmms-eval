import json
import re
import zipfile
from pathlib import Path

import pytest

from lmms_eval.tasks.full_duplex_bench import task as fdb_task
from lmms_eval.tasks.full_duplex_bench import utils as fdb

TASK_DIR = Path(fdb.__file__).parent
JUDGE_ENV = ("FDB_JUDGE_API_KEY", "FDB_JUDGE_BASE_URL", "FDB_JUDGE_MODEL", "DUPLEXEVAL_API_KEY", "DUPLEXEVAL_BASE_URL", "DUPLEXEVAL_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_ID")


class _Response:
    def __init__(self, status_code=200, content=None, text=""):
        self.status_code = status_code
        self._content = content
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture
def judge(monkeypatch):
    """Stub the judge endpoint; each reply is a Response or an exception."""
    for name in JUDGE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FDB_JUDGE_API_KEY", "test-key")
    monkeypatch.setattr(fdb, "JUDGE_RETRY_SECONDS", 0.0)
    calls = []
    replies = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("requests.post", post)
    return calls, replies


def _label(tag):
    return _Response(content=f'{{"behaviour": ["{tag}"], "reasoning": "x"}}')


def _doc(task="user_interruption"):
    transcript = json.dumps({"text": "hello there", "chunks": [{"text": "hello", "timestamp": [0.0, 0.4]}, {"text": "there", "timestamp": [0.4, 0.8]}]})
    return {"sample_id": f"{task}/1", "fdb_task": task, "accepted_labels": fdb_task.ACCEPTED_LABELS[task], "input_clean": transcript, "input_noisy": transcript}


def _output(clean_text="Sure.", noisy_text="Okay, the answer is four."):
    return json.dumps({"sessions": [[{"text": clean_text, "current_time": 1}], [{"text": "", "current_time": 1}, {"text": noisy_text, "current_time": 2}]], "preamble": 0.0})


# --- data preparation -------------------------------------------------------


def _zip(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return str(path)


def test_unzip_refuses_entries_that_leave_the_cache_dir(tmp_path):
    cache = tmp_path / "cache"
    archive = _zip(tmp_path / "evil.zip", {"v1.5/ok.txt": "ok", "../outside.txt": "pwned"})

    with pytest.raises(ValueError, match="leaves"):
        fdb_task._unzip_once(archive, str(cache))
    assert not (tmp_path / "outside.txt").exists()


def test_unzip_skips_macos_metadata_and_repairs_partial_files(tmp_path):
    cache = tmp_path / "cache"
    archive = _zip(tmp_path / "data.zip", {"v1.5/a/input.json": "0123456789", "__MACOSX/v1.5/a/._input.json": "junk", "v1.5/a/._input.json": "junk"})
    partial = cache / "v1.5" / "a" / "input.json"
    partial.parent.mkdir(parents=True)
    partial.write_text("0123")

    fdb_task._unzip_once(archive, str(cache))

    assert partial.read_text() == "0123456789"
    assert not (cache / "__MACOSX").exists()
    assert not (cache / "v1.5" / "a" / "._input.json").exists()


def test_build_rows_keeps_the_official_transcripts(tmp_path):
    sample = tmp_path / "v1.5" / "talking_to_other" / "7"
    sample.mkdir(parents=True)
    (sample / "metadata.json").write_text(json.dumps({"context_text": "c", "background_text": "b", "timestamps": [1.5, 3.0]}))
    (sample / "input.json").write_text(json.dumps({"text": "hi you", "chunks": [{"text": "hi", "timestamp": [0, 1]}]}))
    (sample / "clean_input.json").write_text(json.dumps([{"text": "hi", "timestamp": [0, 1]}]))

    (row,) = fdb_task._build_rows(str(tmp_path), "talking_to_other")

    assert json.loads(row["input_noisy"]) == {"text": "hi you", "chunks": [{"text": "hi", "timestamp": [0, 1]}]}
    assert json.loads(row["input_clean"]) == {"text": "hi", "chunks": [{"text": "hi", "timestamp": [0, 1]}]}
    assert row["accepted_labels"] == ["C_RESUME", "C_UNCERTAIN_HANDLING"]
    assert row["overlap_text"] == "b"


# --- judge protocol ---------------------------------------------------------


def test_judge_call_follows_the_official_protocol(judge):
    calls, replies = judge
    replies.append(_label("C_RESPOND"))

    fdb.fdb_process_results(_doc(), [_output()])

    (call,) = calls
    body = call["json"]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert body["model"] == "gpt-4o-2024-08-06"
    assert body["seed"] == 1
    assert "temperature" not in body
    assert body["messages"][0] == {"role": "system", "content": fdb.BEHAVIOR_INSTRUCTION}
    user = json.loads(body["messages"][1]["content"])
    assert set(user) == {"input_clean", "input_noisy", "output_clean", "output_noisy"}
    assert user["input_clean"]["text"] == "hello there"
    assert user["output_noisy"]["text"] == "Okay, the answer is four."
    assert user["output_noisy"]["chunks"][0] == {"text": "Okay,", "timestamp": [1.0, 1.2]}


def test_judge_retries_with_the_next_seed(judge):
    calls, replies = judge
    replies.extend([RuntimeError("timeout"), _Response(content="no json here"), _label("C_RESUME")])

    scores = fdb.fdb_process_results(_doc("user_backchannel"), [_output()])

    assert [call["json"]["seed"] for call in calls] == [1, 2, 3]
    assert scores["fdb_expected_behavior"] == 1.0
    assert scores["fdb_scoring_failed"] == 0.0


def test_judge_failure_after_all_attempts_leaves_the_sample_unscored(judge, monkeypatch):
    calls, replies = judge
    monkeypatch.setenv("FDB_JUDGE_MAX_ATTEMPTS", "2")
    replies.extend([RuntimeError("down"), _Response(status_code=503)])

    scores = fdb.fdb_process_results(_doc(), [_output()])

    assert len(calls) == 2
    assert scores["fdb_scoring_failed"] == 1.0
    assert scores["fdb_expected_behavior"] is None
    assert scores["fdb_c_respond"] is None


def test_missing_api_key_stops_the_run(judge, monkeypatch):
    monkeypatch.delenv("FDB_JUDGE_API_KEY")

    with pytest.raises(fdb.JudgeConfigError):
        fdb.fdb_process_results(_doc(), [_output()])


def test_rejected_credentials_stop_the_run(judge):
    calls, replies = judge
    replies.append(_Response(status_code=401, text="invalid key"))

    with pytest.raises(fdb.JudgeConfigError):
        fdb.fdb_process_results(_doc(), [_output()])
    assert len(calls) == 1


def test_generation_error_is_not_scored_and_not_judged(judge):
    calls, _replies = judge

    scores = fdb.fdb_process_results(_doc(), [json.dumps({"sessions": [[]], "preamble": 0.0, "error": "CUDA OOM"})])

    assert calls == []
    assert scores["fdb_scoring_failed"] == 1.0
    assert scores["fdb_expected_behavior"] is None


# --- scoring ----------------------------------------------------------------


@pytest.mark.parametrize(
    "task, tag, expected",
    [
        ("user_interruption", "C_RESPOND", 1.0),
        ("user_interruption", "C_UNCERTAIN_HANDLING", 0.0),
        ("user_backchannel", "C_RESUME", 1.0),
        ("user_backchannel", "C_UNCERTAIN_HANDLING", 0.0),
        ("talking_to_other", "C_UNCERTAIN_HANDLING", 1.0),
        ("background_speech", "C_RESUME", 1.0),
        ("background_speech", "C_RESPOND", 0.0),
    ],
)
def test_expected_behavior_follows_the_paper(judge, task, tag, expected):
    _calls, replies = judge
    replies.append(_label(tag))

    scores = fdb.fdb_process_results(_doc(task), [_output()])

    assert scores["fdb_expected_behavior"] == expected
    assert scores[fdb.LABEL_METRICS[tag]] == 1.0
    assert sum(scores[metric] for metric in fdb.LABEL_METRICS.values()) == 1.0


def test_aggregate_leaves_out_unscored_samples():
    assert fdb.fdb_aggregate_mean([1.0, None, 0.0]) == 0.5
    assert fdb.fdb_aggregate_mean([None]) == 0.0


def test_chunks_without_current_time_use_the_chunk_index():
    words = fdb.chunks_to_word_transcript([{"text": "a b", "chunk_idx": 2}])

    assert words == [{"text": "a", "timestamp": [2.0, 2.5]}, {"text": "b", "timestamp": [2.5, 3.0]}]


@pytest.mark.parametrize("yaml_name", sorted(p.name for p in TASK_DIR.glob("fdb_*.yaml")))
def test_task_configs_list_every_reported_metric(judge, yaml_name):
    _calls, replies = judge
    replies.append(_label("C_RESUME"))
    configured = re.findall(r"^\s*- metric: (\S+)", (TASK_DIR / yaml_name).read_text(), flags=re.MULTILINE)

    reported = fdb.fdb_process_results(_doc(), [_output()])

    assert sorted(configured) == sorted(reported)


def test_build_rows_skips_samples_without_transcripts(tmp_path):
    for sample, files in (("1", ("input.json", "clean_input.json")), ("2", ("input.json",))):
        folder = tmp_path / "v1.5" / "user_backchannel" / sample
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text("{}")
        for name in files:
            (folder / name).write_text(json.dumps({"text": "", "chunks": []}))

    rows = fdb_task._build_rows(str(tmp_path), "user_backchannel")

    assert [row["sample_id"] for row in rows] == ["user_backchannel/1"]
