import inspect
import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
from PIL import Image

from lmms_eval.api.model import lmms
from lmms_eval.models import get_model, get_model_manifest
from lmms_eval.models.chat.minicpm_o_45 import MiniCPM_O_45Chat
from lmms_eval.models.simple.minicpm_o_45 import MiniCPM_O_45
from lmms_eval.models.simple.minicpm_o_45_duplex import MiniCPM_O_45Duplex
from lmms_eval.protocol import ChatMessages


class _RecordingModel:
    """Stands in for model.chat: records the call or raises."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def chat(self, **kwargs):
        if self.fail:
            raise RuntimeError("chat failed")
        self.calls.append(kwargs)
        return "answer"


def _half_duplex(cls, model):
    wrapper = cls.__new__(cls)
    lmms.__init__(wrapper)
    wrapper._model = model
    wrapper._tokenizer = NS(encode=lambda text: text.split(), decode=lambda _ids: "</s>", eos_token_id=0)
    wrapper.batch_size_per_gpu = 1
    wrapper.system_prompt = None
    wrapper.init_audio = True
    wrapper.enable_thinking = False
    wrapper.max_slice_nums = None
    return wrapper


def _request(gen_kwargs, visual=None):
    return NS(args=("question", gen_kwargs, lambda doc: visual, 0, "task", "test"))


class TestMiniCPMO45Registration(unittest.TestCase):
    def test_registry_resolves_chat_by_default_and_simple_when_forced(self):
        manifest = get_model_manifest("minicpm_o_45")

        self.assertEqual(manifest.simple_class_path, "lmms_eval.models.simple.minicpm_o_45.MiniCPM_O_45")
        self.assertEqual(manifest.chat_class_path, "lmms_eval.models.chat.minicpm_o_45.MiniCPM_O_45Chat")
        self.assertIs(get_model("minicpm_o_45"), MiniCPM_O_45Chat)
        self.assertIs(get_model("minicpm_o_45", force_simple=True), MiniCPM_O_45)

    def test_registry_resolves_duplex_adapter(self):
        manifest = get_model_manifest("minicpm_o_45_duplex")

        self.assertEqual(manifest.simple_class_path, "lmms_eval.models.simple.minicpm_o_45_duplex.MiniCPM_O_45Duplex")
        self.assertIsNone(manifest.chat_class_path)

    def test_defaults_target_minicpm_o_4_5_checkpoint(self):
        params = inspect.signature(MiniCPM_O_45.__init__).parameters

        self.assertEqual(params["pretrained"].default, "openbmb/MiniCPM-o-4_5")
        self.assertTrue(params["init_vision"].default)
        self.assertTrue(params["init_audio"].default)
        self.assertFalse(params["init_tts"].default)
        self.assertFalse(params["enable_thinking"].default)


class TestMiniCPMO45Helpers(unittest.TestCase):
    def test_truncate_until_stops_at_earliest_marker(self):
        self.assertEqual(MiniCPM_O_45._truncate_until("A. extra", ["."]), "A")
        self.assertEqual(MiniCPM_O_45._truncate_until("no markers", ["."]), "no markers")

    def test_split_audio_chunks_pads_final_chunk(self):
        chunks = MiniCPM_O_45Duplex._split_audio_chunks(np.zeros(16000 * 2 + 5, dtype=np.float32))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) == 16000 for chunk in chunks))

    def test_duplex_classification(self):
        streams = MiniCPM_O_45Duplex._classify_media(["video.mp4", {"array": np.zeros(1600, dtype="float32"), "sampling_rate": 16000}])
        self.assertEqual([s["kind"] for s in streams], ["video", "audio"])


class TestMiniCPMO45ChatMedia(unittest.TestCase):
    """The chat path delivers in-memory media; none of it may be dropped."""

    def _infer(self, content):
        model = _RecordingModel()
        wrapper = _half_duplex(MiniCPM_O_45Chat, model)
        answer = wrapper._infer_one(ChatMessages(messages=[{"role": "user", "content": content}]), {"max_new_tokens": 8})
        return answer, model.calls

    def test_in_memory_image_reaches_the_model(self):
        image = Image.new("RGB", (8, 8), "red")
        answer, calls = self._infer([{"type": "image", "url": image}, {"type": "text", "text": "What colour?"}])

        self.assertEqual(answer, "answer")
        user_content = calls[0]["msgs"][-1]["content"]
        self.assertIsInstance(user_content[0], Image.Image)
        self.assertEqual(user_content[-1], "What colour?")

    def test_decoded_audio_dict_reaches_the_model_in_omni_mode(self):
        # lmms-eval labels a decoded audio dict without a "type" key as video.
        audio = {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}
        _answer, calls = self._infer([{"type": "video", "url": audio}, {"type": "text", "text": "Transcribe."}])

        user_content = calls[0]["msgs"][-1]["content"]
        self.assertIsInstance(user_content[0], np.ndarray)
        self.assertTrue(calls[0]["omni_mode"])

    def test_media_that_cannot_be_loaded_raises_instead_of_being_dropped(self):
        with self.assertRaises(FileNotFoundError):
            self._infer([{"type": "image", "url": "/nonexistent/image.png"}, {"type": "text", "text": "q"}])


class TestMiniCPMO45Failures(unittest.TestCase):
    def test_failed_request_returns_empty_answer_by_default(self):
        wrapper = _half_duplex(MiniCPM_O_45, _RecordingModel(fail=True))
        wrapper.task_dict = {"task": {"test": {0: {}}}}

        self.assertEqual(wrapper.generate_until([_request({"max_new_tokens": 8})]), [""])

    def test_raise_on_error_stops_at_the_first_failure(self):
        wrapper = _half_duplex(MiniCPM_O_45, _RecordingModel(fail=True))
        wrapper.task_dict = {"task": {"test": {0: {}}}}

        with self.assertRaises(RuntimeError):
            wrapper.generate_until([_request({"max_new_tokens": 8, "raise_on_error": True})])


def _duplex():
    wrapper = MiniCPM_O_45Duplex.__new__(MiniCPM_O_45Duplex)
    lmms.__init__(wrapper)
    wrapper.batch_size_per_gpu = 1
    wrapper.duplex_save_dir = None
    wrapper.task_dict = {"task": {"test": {0: {}}}}
    return wrapper


class TestMiniCPMO45Duplex(unittest.TestCase):
    def test_failed_request_returns_an_error_record(self):
        wrapper = _duplex()
        with mock.patch.object(MiniCPM_O_45Duplex, "_run_session", side_effect=RuntimeError("stream failed")):
            (answer,) = wrapper.generate_until([_request({}, visual=[{"array": np.zeros(160, dtype="float32"), "sampling_rate": 16000}])])

        self.assertEqual(json.loads(answer)["error"], "stream failed")

    def test_raise_on_error_stops_at_the_first_failure(self):
        wrapper = _duplex()
        with mock.patch.object(MiniCPM_O_45Duplex, "_run_session", side_effect=RuntimeError("stream failed")):
            with self.assertRaises(RuntimeError):
                wrapper.generate_until([_request({"raise_on_error": True}, visual=[{"array": np.zeros(160, dtype="float32"), "sampling_rate": 16000}])])

    def test_temporary_video_files_are_removed_after_the_request(self):
        wrapper = _duplex()
        seen = []

        def run_session(_wrapper, streams, gen_kwargs):
            seen.extend((stream["path"], os.path.exists(stream["path"])) for stream in streams)
            return [], 0.0

        with mock.patch.object(MiniCPM_O_45Duplex, "_run_session", run_session):
            wrapper.generate_until([_request({"duplex_multi_session": True}, visual=[{"bytes": b"\x00\x00\x00 ftypisom", "path": "clip.mp4"}])])

        self.assertEqual(len(seen), 1)
        path, existed_during_run = seen[0]
        self.assertTrue(existed_during_run)
        self.assertFalse(os.path.exists(path))

    def test_session_reports_the_audio_preamble(self):
        wrapper = _duplex()
        wrapper._model = NS(prepare=lambda **kwargs: None)
        wrapper.system_prompt = "Streaming Omni Conversation."
        wrapper.ref_audio_path = "ref.wav"
        wrapper._ref_audio_cache = np.zeros(16000, dtype=np.float32)
        streams = [{"kind": "audio", "audio": np.zeros(32000, dtype=np.float32)}, {"kind": "video", "path": "v.mp4"}]
        with mock.patch.object(MiniCPM_O_45Duplex, "_run_stream", return_value=[{"chunk_idx": 0}]):
            records, preamble = wrapper._run_session(streams, {})

        self.assertEqual(preamble, 2.0)
        self.assertEqual(len(records), 2)

    def test_video_stream_steps_like_the_model_card(self):
        # One step per audio segment; the stacked frame only with its main frame.
        wrapper = _duplex()
        wrapper.stack_frames = 1
        wrapper.use_ffmpeg = True
        steps = []
        segments = [np.full(16000, i, dtype=np.float32) for i in range(4)]
        loader = lambda path, **kwargs: (["f0", None, "f2"], segments, ["s0", "s1", "s2", "s3"])  # noqa: E731

        with (
            mock.patch("lmms_eval.models.simple.minicpm_o_45_duplex._load_minicpmo_utils", return_value=loader),
            mock.patch.object(MiniCPM_O_45Duplex, "_step", side_effect=lambda audio, frames: steps.append((int(audio[0]), list(frames))) or {}),
        ):
            wrapper._run_stream({"kind": "video", "path": "v.mp4"}, np.zeros(1), {})

        self.assertEqual(steps, [(0, ["f0", "s0"]), (1, []), (2, ["f2", "s2"]), (3, [])])


if __name__ == "__main__":
    unittest.main()
