import json
import shutil
import subprocess

import numpy as np
import pytest

from lmms_eval.tasks.omni_duplexeval import utils as omni

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not on PATH")


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _video(path, seconds, tone_hz=None):
    inputs = ["-f", "lavfi", "-i", f"testsrc=size=64x64:rate=2:duration={seconds}"]
    if tone_hz:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency={tone_hz}:sample_rate=16000:duration={seconds}"]
    _ffmpeg(*inputs, "-c:v", "mpeg4", *(["-c:a", "aac", "-shortest"] if tone_hz else []), str(path))
    return str(path)


def _question(path, seconds, speech_from, speech_to):
    """A track as long as the video, silent except where the user speaks."""
    voice = f"sine=frequency=1000:sample_rate=16000:duration={seconds},volume='between(t,{speech_from},{speech_to})':eval=frame"
    _ffmpeg("-f", "lavfi", "-i", voice, str(path))
    return str(path)


def _audio_rms_per_second(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True).stdout
    samples = np.frombuffer(raw, dtype=np.float32)
    seconds = len(samples) // 16000
    return [float(np.sqrt(np.mean(samples[i * 16000 : (i + 1) * 16000] ** 2))) for i in range(seconds)]


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,sample_rate,channels,nb_frames", "-of", "json", path], check=True, capture_output=True, text=True).stdout
    return json.loads(out)["streams"]


# --- protocol: the question is heard during the video -------------------------


@needs_ffmpeg
def test_question_audio_is_heard_at_its_own_time_in_a_silent_video(tmp_path):
    video = _video(tmp_path / "v.mp4", 4)
    question = _question(tmp_path / "q.wav", 4, 2, 3)

    mixed = omni.mix_question_into_video(video, question, str(tmp_path / "out.mp4"))

    rms = _audio_rms_per_second(mixed)
    assert len(rms) == 4
    assert rms[2] > 0.05 and max(rms[0], rms[1], rms[3]) < 0.01


@needs_ffmpeg
def test_question_audio_is_added_to_the_scene_sound(tmp_path):
    video = _video(tmp_path / "v.mp4", 4, tone_hz=300)
    question = _question(tmp_path / "q.wav", 4, 1, 2)

    mixed = omni.mix_question_into_video(video, question, str(tmp_path / "out.mp4"))

    rms = _audio_rms_per_second(mixed)
    assert len(rms) == 4
    assert min(rms) > 0.05  # the scene sound is kept
    assert rms[1] > 1.2 * max(rms[0], rms[2], rms[3])  # the user speaks in second 1
    streams = {s["codec_type"]: s for s in _probe(mixed)}
    original = {s["codec_type"]: s for s in _probe(video)}
    assert streams["video"]["codec_name"] == original["video"]["codec_name"]
    assert streams["video"]["nb_frames"] == original["video"]["nb_frames"]
    assert (streams["audio"]["sample_rate"], streams["audio"]["channels"]) == ("16000", 1)


@needs_ffmpeg
def test_doc_to_visual_returns_one_video_with_the_question(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    video = _video(tmp_path / "v.mp4", 3)
    question = _question(tmp_path / "q.wav", 3, 1, 2)
    doc = {"id": "pr_1", "video": {"bytes": open(video, "rb").read(), "path": None}, "question_audio": {"bytes": open(question, "rb").read(), "path": None}}

    (visual,) = omni.omni_duplexeval_doc_to_visual(doc)

    assert visual.endswith("video_with_question_pr_1.mp4")
    assert _audio_rms_per_second(visual)[1] > 0.05
    assert omni.omni_duplexeval_doc_to_visual(doc) == [visual]  # cached


def test_chunks_keep_the_video_timeline():
    raw = json.dumps({"sessions": [[{"text": "a", "current_time": 1}, {"text": " ", "current_time": 2}, {"text": "b", "current_time": 3}]], "preamble": 0.0})

    assert omni.normalize_chunks(raw) == [{"text": "a", "current_time": 1.0}, {"text": "b", "current_time": 3.0}]


def test_generation_error_scores_as_an_empty_response():
    assert omni.normalize_chunks(json.dumps({"sessions": [[]], "preamble": 0.0, "error": "OOM"})) == []


# --- judge client -------------------------------------------------------------


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"choices": [{"message": {"content": '{"success_score": 1}'}}]}
        self.text = f"status {status_code}"

    def json(self):
        return self._payload


class _Requests:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(omni, "JUDGE_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(omni, "_content_input_resolved", None)
    monkeypatch.delenv("OMNI_DUPLEXEVAL_CONTENT_INPUT", raising=False)

    def make(*replies):
        judge = omni.OpenAILLMClient(api_key="k", base_url="http://judge")
        judge.requests = _Requests(replies)
        return judge

    return make


def test_transient_judge_errors_are_retried(client):
    judge = client(RuntimeError("timeout"), _Response(429), _Response(200))

    assert "choices" in judge.chat_completion([{"role": "user", "content": "q"}], model="m")
    assert len(judge.requests.calls) == 3


def test_request_errors_are_not_retried(client):
    judge = client(_Response(400))

    assert judge.chat_completion([{"role": "user", "content": "q"}], model="m")["status_code"] == 400
    assert len(judge.requests.calls) == 1


def test_judge_gives_up_after_the_attempt_limit(client, monkeypatch):
    monkeypatch.setenv("OMNI_DUPLEXEVAL_JUDGE_MAX_ATTEMPTS", "2")
    judge = client(_Response(503), _Response(503))

    assert judge.chat_completion([{"role": "user", "content": "q"}], model="m")["status_code"] == 503
    assert len(judge.requests.calls) == 2


@pytest.mark.parametrize("reply, score", [('{"success_score": 1}', 1), ('{"success_score": "yes"}', 0), ("no json", 0)])
def test_success_score_parsing(reply, score):
    assert omni._parse_success_score(reply)["success_score"] == score


# --- content judge input mode -----------------------------------------------------


def _content_reply(score):
    return _Response(200, {"choices": [{"message": {"content": json.dumps({"content_score": score})}}]})


def _part_types(call):
    return [part["type"] for part in call["messages"][0]["content"]]


@pytest.fixture
def frames(monkeypatch):
    monkeypatch.setattr(omni, "_extract_uniform_frames", lambda path, n: [b"jpeg"] * n)


def test_auto_mode_keeps_the_official_video_part_when_accepted(client, frames):
    judge = client(_content_reply(2.5), _content_reply(1.0))

    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 2.5
    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 1.0
    assert [_part_types(call) for call in judge.requests.calls] == [["text", "video"], ["text", "video"]]


def test_auto_mode_switches_to_frames_once_the_video_part_is_rejected(client, frames):
    judge = client(_Response(400), _content_reply(2.0), _content_reply(3.0))

    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 2.0
    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 3.0
    types = [_part_types(call) for call in judge.requests.calls]
    assert types[0] == ["text", "video"]
    assert types[1] == types[2] == ["text"] + ["image_url"] * omni.CONTENT_FALLBACK_FRAMES


def test_auto_mode_does_not_switch_on_server_errors(client, frames, monkeypatch):
    monkeypatch.setenv("OMNI_DUPLEXEVAL_JUDGE_MAX_ATTEMPTS", "1")
    judge = client(_Response(503), _content_reply(2.0))

    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 0.0
    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 2.0
    assert [_part_types(call)[-1] for call in judge.requests.calls] == ["video", "video"]


def test_unreadable_video_verdict_scores_zero_without_fallback(client, frames):
    judge = client(_Response(200, {"choices": [{"message": {"content": "I cannot see a video."}}]}))

    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 0.0
    assert len(judge.requests.calls) == 1


def test_frames_mode_never_sends_the_video_part(client, frames, monkeypatch):
    monkeypatch.setenv("OMNI_DUPLEXEVAL_CONTENT_INPUT", "frames")
    judge = client(_content_reply(1.5))

    assert omni._judge_content_accuracy(judge, "v.mp4", "text", "q", [], "m") == 1.5
    assert "video" not in _part_types(judge.requests.calls[0])


def test_unknown_content_mode_is_rejected(client, monkeypatch):
    monkeypatch.setenv("OMNI_DUPLEXEVAL_CONTENT_INPUT", "gif")

    with pytest.raises(ValueError):
        omni._judge_content_accuracy(client(), "v.mp4", "text", "q", [], "m")
