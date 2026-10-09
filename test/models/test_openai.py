from __future__ import annotations

import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from lmms_eval.models.chat.openai import OpenAICompatible as ChatOpenAICompatible
from lmms_eval.models.simple.openai import OpenAICompatible as SimpleOpenAICompatible


def _fake_response(content: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    return SimpleNamespace(choices=[choice], usage=None)


class _CaptureCompletions:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def create(self, **payload):
        self.payloads.append(payload)
        return _fake_response()


def _request(*args) -> SimpleNamespace:
    return SimpleNamespace(args=args)


def _chat_request(gen_kwargs: dict) -> SimpleNamespace:
    return _request(
        "",
        lambda _doc: [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Respond to this"}],
            }
        ],
        gen_kwargs,
        0,
        "demo",
        "test",
    )


def _configure_openai_model(model, completions: _CaptureCompletions, *, model_version: str = "gpt-4o") -> None:
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model.model_version = model_version
    model.max_retries = 1
    model.retry_backoff_s = 0
    model.num_concurrent = 1
    model.adaptive_concurrency = False
    model.adaptive_config = SimpleNamespace(max_concurrency=1)
    model.prefix_aware_queue = False
    model.prefix_hash_chars = 256
    model.max_frames_num = 1
    model.video_fps = None
    model.pass_video_url = False
    model.video_target_frames = None
    model.video_max_frames = None
    model.enable_thinking_kwarg = None
    model._rank = 0
    model.task_dict = {"demo": {"test": [{"id": 0}]}}


class TestOpenAICompatibleGenerationParameters(unittest.TestCase):
    def test_simple_backend_preserves_requested_max_new_tokens(self):
        completions = _CaptureCompletions()
        model = SimpleOpenAICompatible.__new__(SimpleOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until(
            [
                _request(
                    "Describe the image",
                    {"max_new_tokens": 8192, "temperature": 0},
                    lambda _doc: None,
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(completions.payloads[0]["max_tokens"], 8192)

    def test_chat_backend_preserves_requested_max_new_tokens(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Describe this"}],
                        }
                    ],
                    {"max_new_tokens": 32768, "temperature": 0},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(completions.payloads[0]["max_tokens"], 32768)

    def test_chat_reasoning_models_use_requested_completion_tokens(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions, model_version="gpt-5")

        model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Reason carefully"}],
                        }
                    ],
                    {"max_new_tokens": 32768, "temperature": 0.7},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertNotIn("max_tokens", completions.payloads[0])
        self.assertEqual(completions.payloads[0]["max_completion_tokens"], 32768)

    def test_chat_backend_preserves_per_request_sampling_parameters(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)
        generation_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.4,
            "top_p": 0.8,
            "seed": 42,
            "presence_penalty": 0.1,
            "frequency_penalty": -0.2,
        }
        original_generation_kwargs = dict(generation_kwargs)

        model.generate_until([_chat_request(generation_kwargs)])

        payload = completions.payloads[0]
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["presence_penalty"], 0.1)
        self.assertEqual(payload["frequency_penalty"], -0.2)
        self.assertEqual(generation_kwargs, original_generation_kwargs)

    def test_chat_backend_accepts_one_response_choice(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until([_chat_request({"n": 1})])

        self.assertEqual(completions.payloads[0]["n"], 1)

    def test_chat_backend_rejects_unsupported_response_choice_counts(self):
        for invalid_n in (0, 2, -1, 1.0, True, "1"):
            with self.subTest(n=invalid_n):
                completions = _CaptureCompletions()
                model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
                _configure_openai_model(model, completions)

                with self.assertRaisesRegex(ValueError, "n.*integer 1"):
                    model.generate_until([_chat_request({"n": invalid_n})])

                self.assertEqual(completions.payloads, [])


def _write_video(path: str, *, frames: int = 20, fps: int = 10) -> None:
    import av
    import numpy as np

    with av.open(path, "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def _video_request(*urls: str) -> SimpleNamespace:
    content = [{"type": "video", "url": url} for url in urls] + [{"type": "text", "text": "Describe this"}]
    return _request("", lambda _doc: [{"role": "user", "content": content}], {"max_new_tokens": 8}, 0, "demo", "test")


class TestOpenAICompatibleVideoFrameControls(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.video = os.path.join(self.tmp.name, "clip.mp4")
        _write_video(self.video)  # 20 frames at 10 fps = 2 s

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _media_io_kwargs(self, request: SimpleNamespace, **attrs) -> dict:
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)
        model.pass_video_url = True
        model.max_frames_num = 32
        for name, value in attrs.items():
            setattr(model, name, value)
        model.generate_until([request])
        return completions.payloads[0]["extra_body"]["media_io_kwargs"]

    def test_default_payload_is_unchanged(self):
        self.assertEqual(self._media_io_kwargs(_video_request(self.video)), {"video": {"num_frames": 32}})

    def test_target_frames_sends_count_and_matching_fps(self):
        kwargs = self._media_io_kwargs(_video_request(self.video), video_target_frames=8)
        self.assertEqual(kwargs["video"]["num_frames"], 8)
        self.assertAlmostEqual(kwargs["video"]["fps"], 4.0, delta=1e-4)
        self.assertNotIn("max_frames", kwargs["video"])

    def test_target_frames_fps_does_not_round_to_one_frame_fewer(self):
        # 19 frames at 10 fps: 1.9 * (8 / 1.9) == 7.999..., so floor() and int()
        # samplers would return 7 frames for an unadjusted fps.
        from lmms_eval.models.model_utils.load_video import _probe_video_metadata

        clip = os.path.join(self.tmp.name, "rounding.mp4")
        _write_video(clip, frames=19)
        frames, rate = _probe_video_metadata(clip, count_frames=True)
        self.assertEqual((frames, rate), (19, 10.0))
        duration = frames / rate
        self.assertLess(duration * (8 / duration), 8)
        fps = self._media_io_kwargs(_video_request(clip), video_target_frames=8)["video"]["fps"]
        self.assertEqual(math.floor(duration * fps), 8)
        self.assertEqual(int(frames / rate * fps), 8)
        self.assertLess(duration * fps, 8.001)

    def test_target_frames_without_a_video_sends_the_default(self):
        request = _chat_request({"max_new_tokens": 8})
        self.assertEqual(self._media_io_kwargs(request, video_target_frames=8), {"video": {"num_frames": 32}})

    def test_target_frames_accepts_the_same_video_twice(self):
        kwargs = self._media_io_kwargs(_video_request(self.video, self.video), video_target_frames=8)
        self.assertEqual(kwargs["video"]["num_frames"], 8)

    def test_target_frames_reports_unreadable_videos_with_the_doc(self):
        broken = os.path.join(self.tmp.name, "broken.mp4")
        with open(broken, "wb") as handle:
            handle.write(b"not a video")
        with self.assertRaisesRegex(ValueError, "demo/test doc 0: Cannot read the duration"):
            self._media_io_kwargs(_video_request(broken), video_target_frames=8)

    def test_max_frames_is_passed_through(self):
        kwargs = self._media_io_kwargs(_video_request(self.video), video_max_frames=16)
        self.assertEqual(kwargs, {"video": {"num_frames": 32, "max_frames": 16}})

    def test_target_and_max_frames_combine(self):
        kwargs = self._media_io_kwargs(_video_request(self.video), video_target_frames=8, video_max_frames=16)
        self.assertEqual(kwargs["video"]["max_frames"], 16)
        self.assertAlmostEqual(kwargs["video"]["fps"], 4.0, delta=1e-4)

    def test_target_frames_rejects_remote_videos(self):
        for url in ("https://example.com/clip.mp4", "data:video/mp4;base64,AAAA"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "local videos"):
                self._media_io_kwargs(_video_request(url), video_target_frames=8)

    def test_target_frames_rejects_several_videos_per_request(self):
        other = os.path.join(self.tmp.name, "other.mp4")
        _write_video(other, frames=10)
        with self.assertRaisesRegex(ValueError, "one video per request"):
            self._media_io_kwargs(_video_request(self.video, other), video_target_frames=8)

    def test_frame_controls_are_validated(self):
        common = {"model_version": "demo", "base_url": "http://127.0.0.1:1/v1", "api_key": "EMPTY"}
        for name in ("video_target_frames", "video_max_frames"):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                    ChatOpenAICompatible(pass_video_url=True, **{name: value}, **common)
            with self.subTest(name=name, pass_video_url=False), self.assertRaisesRegex(ValueError, "pass_video_url"):
                ChatOpenAICompatible(**{name: 8}, **common)
        model = ChatOpenAICompatible(pass_video_url=True, video_target_frames=8, video_max_frames=16, **common)
        self.assertEqual((model.video_target_frames, model.video_max_frames), (8, 16))


if __name__ == "__main__":
    unittest.main()
