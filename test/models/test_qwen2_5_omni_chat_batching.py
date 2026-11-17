"""CPU-only contract tests for Qwen2.5-Omni chat native batching."""

from __future__ import annotations

import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages

if TYPE_CHECKING:
    from lmms_eval.models.chat.qwen2_5_omni import Qwen2_5_Omni


class _Inputs(dict):
    def to(self, value):
        return self


class _Processor:
    def __init__(self) -> None:
        self.templates = []
        self.calls = []
        self.decode_calls = []
        self.tokenizer = SimpleNamespace(padding_side="right", eos_token_id=2, pad_token_id=0)

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return messages[-1]["content"][-1]["text"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        ids = [int(text.rsplit("-", 1)[-1]) if text.startswith("question-") else 0 for text in kwargs["text"]]
        return _Inputs(input_ids=torch.tensor(ids, dtype=torch.long).unsqueeze(1))

    def batch_decode(self, cont, **kwargs):
        self.decode_calls.append((cont, kwargs))
        return [f"prompt assistant\nanswer-{int(row)} assistant\nfinal-{int(row)}" for row in cont]


class _Model:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls = []
        self.tuple_output = True

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        rows = kwargs["input_ids"].flatten()
        return (rows, "audio-placeholder") if self.tuple_output else rows


class _Cache:
    def __init__(self) -> None:
        self.entries = []

    def add_partial(self, method, key, answer) -> None:
        self.entries.append((method, key, answer))


def _request(index: int, kwargs: dict, media=None) -> Instance:
    def messages(doc):
        content = []
        if media is not None:
            content.extend(media)
        content.append({"type": "text", "text": doc["text"]})
        return [{"role": "user", "content": content}]

    return Instance(
        request_type="generate_until",
        arguments=(f"ctx-{index}", messages, kwargs, index, "task", "test"),
        idx=index,
        metadata={"task": "task", "doc_id": index, "repeats": 1},
    )


class _QwenOmniTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        missing = [name for name in ("librosa", "moviepy", "qwen_omni_utils") if find_spec(name) is None]
        if missing:
            raise unittest.SkipTest(f"Qwen2.5-Omni runtime dependencies unavailable: {', '.join(missing)}")

        from lmms_eval.models.chat.qwen2_5_omni import Qwen2_5_Omni
        from lmms_eval.models.simple import qwen2_5_omni as simple_omni

        cls.chat_adapter = Qwen2_5_Omni
        cls.simple_omni = simple_omni


class TestQwenProcessorLoading(_QwenOmniTestCase):
    def test_processor_uses_model_source_in_simple_and_chat_adapters(self) -> None:
        default_source = "Qwen/Qwen2.5-Omni-7B"
        local_source = "/models/qwen-omni-snapshot"
        simple_omni = self.simple_omni
        for adapter in (simple_omni.Qwen2_5_Omni, self.chat_adapter):
            for model_args, expected in (({}, default_source), ({"pretrained": local_source}, local_source)):
                with (
                    self.subTest(adapter=adapter.__module__, model_args=model_args),
                    patch.object(simple_omni, "Accelerator", return_value=SimpleNamespace(num_processes=1)),
                    patch.object(simple_omni.Qwen2_5OmniForConditionalGeneration, "from_pretrained") as load_model,
                    patch.object(simple_omni.Qwen2_5OmniProcessor, "from_pretrained") as load_processor,
                ):
                    wrapper = adapter(device="cpu", device_map="auto", **model_args)
                    self.assertEqual(load_model.call_args.args, (expected,))
                    load_processor.assert_called_once_with(expected)
                    self.assertIs(wrapper.tokenizer, load_processor.return_value.tokenizer)


class TestQwenNativeBatch(_QwenOmniTestCase):
    def _wrapper(self, batch_size: int) -> Qwen2_5_Omni:
        wrapper = self.chat_adapter.__new__(self.chat_adapter)
        wrapper.batch_size_per_gpu = batch_size
        wrapper._rank = 0
        wrapper.task_dict = {"task": {"test": [{"text": f"question-{i}"} for i in range(5)]}}
        wrapper.cache_hook = _Cache()
        wrapper.processor = _Processor()
        wrapper._tokenizer = wrapper.processor.tokenizer
        wrapper._model = _Model()
        wrapper.device_map = "auto"
        wrapper.use_cache = True
        wrapper.system_prompt = "system"
        return wrapper

    def test_public_batched_inference_media_padding_and_tuple_output(self) -> None:
        for size in (1, 2, 4):
            with self.subTest(size=size):
                wrapper = self._wrapper(size)
                kwargs = {"max_new_tokens": 11, "temperature": 0.5, "top_p": 0.8, "num_beams": 2}
                images = [Image.new("RGB", (2, 2)) for _ in range(3)]
                for marker, image in zip(("image-1", "image-2", "image-4"), images):
                    image.info["marker"] = marker
                request_media = [
                    None,
                    [{"type": "image", "url": images[0]}, {"type": "audio", "url": "audio-1.wav"}, {"type": "video", "url": "video-1.mp4"}],
                    [{"type": "image", "url": images[1]}],
                    [{"type": "audio", "url": "audio-3.wav"}, {"type": "video", "url": "video-3.mp4"}],
                    [{"type": "image", "url": images[2]}, {"type": "audio", "url": "audio-4.wav"}, {"type": "video", "url": "video-4.mp4"}],
                ]
                requests = [_request(i, kwargs, request_media[i]) for i in range(5)]
                expected_rows = [
                    ("question-0", [], [], []),
                    ("question-1", ["image-1"], ["audio-1.wav"], ["video-1.mp4"]),
                    ("question-2", ["image-2"], [], []),
                    ("question-3", [], ["audio-3.wav"], ["video-3.mp4"]),
                    ("question-4", ["image-4"], ["audio-4.wav"], ["video-4.mp4"]),
                ]
                media_calls = []

                def media_info(conversations, **options):
                    rows = []
                    for conversation in conversations:
                        content = [block for message in conversation for block in message["content"]]
                        rows.append(
                            (
                                content[-1]["text"],
                                [block["image"].info["marker"] for block in content if block["type"] == "image"],
                                [block["audio"] for block in content if block["type"] == "audio"],
                                [block["video"] for block in content if block["type"] == "video"],
                            )
                        )
                    media_calls.append((conversations, options, rows))
                    return (
                        [marker for _, _, audios, _ in rows for marker in audios],
                        [marker for _, images, _, _ in rows for marker in images],
                        [marker for _, _, _, videos in rows for marker in videos],
                    )

                with patch("lmms_eval.models.chat.qwen2_5_omni.process_mm_info", side_effect=media_info):
                    answers = wrapper.generate_until(requests)

                self.assertEqual(answers, [f"final-{i}" for i in range(5)])
                self.assertEqual(len(wrapper.model.calls), (5 + size - 1) // size)
                expected_chunks = [expected_rows[start : start + size] for start in range(0, 5, size)]
                self.assertEqual([rows for _, _, rows in media_calls], expected_chunks)
                self.assertTrue(all(options == {"use_audio_in_video": False} for _, options, _ in media_calls))
                self.assertTrue(all(batch[0]["role"] == "system" for conversations, _, _ in media_calls for batch in conversations))
                self.assertEqual(
                    [[block["type"] for message in batch for block in message["content"]] for conversations, _, _ in media_calls for batch in conversations],
                    [["text", *([item["type"] for item in request_media[i]] if request_media[i] else []), "text"] for i in range(5)],
                )
                for call, rows in zip(wrapper.processor.calls, expected_chunks):
                    self.assertEqual(call["text"], [row[0] for row in rows])
                    self.assertEqual(call["images"], [marker for _, images, _, _ in rows for marker in images])
                    self.assertEqual(call["audio"], [marker for _, _, audios, _ in rows for marker in audios])
                    self.assertEqual(call["videos"], [marker for _, _, _, videos in rows for marker in videos])
                self.assertTrue(all(call["padding"] is True and call["padding_side"] == "left" for call in wrapper.processor.calls))
                self.assertEqual(wrapper.processor.tokenizer.padding_side, "right")
                self.assertTrue(all(call["return_audio"] is False and call["use_audio_in_video"] is False and call["thinker_do_sample"] is False for call in wrapper.model.calls))
                self.assertTrue(all(call["do_sample"] is True and call["thinker_max_new_tokens"] == 11 and "max_new_tokens" not in call for call in wrapper.model.calls))
                self.assertEqual(kwargs, {"max_new_tokens": 11, "temperature": 0.5, "top_p": 0.8, "num_beams": 2})
                self.assertEqual(len(wrapper.cache_hook.entries), 5)
                self.assertCountEqual([(key[0], answer) for _, key, answer in wrapper.cache_hook.entries], [(f"ctx-{i}", f"final-{i}") for i in range(5)])

    def test_single_inference_keeps_kwargs_and_empty_batch_skips_model(self) -> None:
        wrapper = self._wrapper(2)
        messages = ChatMessages(messages=[{"role": "user", "content": [{"type": "text", "text": "single"}]}])
        kwargs = {"temperature": 0}
        with patch("lmms_eval.models.chat.qwen2_5_omni.process_mm_info", return_value=(None, None, None)):
            self.assertEqual(wrapper._infer_one(messages, kwargs), "final-0")
            self.assertEqual(wrapper._infer_batch([], kwargs), [])
        self.assertEqual(kwargs, {"temperature": 0})
        self.assertEqual(len(wrapper.model.calls), 1)
        self.assertEqual(wrapper.model.calls[0]["do_sample"], False)
        self.assertEqual(wrapper.model.calls[0]["thinker_max_new_tokens"], 4096)
        self.assertNotIn("max_new_tokens", wrapper.model.calls[0])
        self.assertEqual(wrapper.model.calls[0]["top_p"], None)
        self.assertEqual(wrapper.model.calls[0]["num_beams"], 1)

        wrapper.model.tuple_output = False
        wrapper.device_map = "cpu"
        with patch("lmms_eval.models.chat.qwen2_5_omni.process_mm_info", return_value=(None, None, None)):
            self.assertEqual(wrapper._infer_one(messages, kwargs), "final-0")
        self.assertEqual(len(wrapper.model.calls), 2)

    def test_public_limits_reach_native_thinker_without_weights(self) -> None:
        from transformers import Qwen2_5OmniForConditionalGeneration

        class Thinker:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["input_ids"]

        wrapper = self._wrapper(1)
        thinker = Thinker()
        native = SimpleNamespace(speaker_map={"Chelsie": 0}, has_talker=False, thinker=thinker, talker=SimpleNamespace(codec_pad_token=0))
        messages = ChatMessages(messages=[{"role": "user", "content": [{"type": "text", "text": "single"}]}])
        for public_kwargs, expected in (({"max_new_tokens": 1}, 1), ({"max_new_tokens": 16}, 16), ({"max_new_tokens": 11}, 11), ({}, 4096)):
            with self.subTest(public_kwargs=public_kwargs), patch("lmms_eval.models.chat.qwen2_5_omni.process_mm_info", return_value=(None, None, None)):
                wrapper._infer_one(messages, public_kwargs)
                model_kwargs = wrapper.model.calls[-1]
                self.assertNotIn("max_new_tokens", model_kwargs)
                Qwen2_5OmniForConditionalGeneration.generate(native, **model_kwargs, generation_mode="text")
                self.assertEqual(thinker.calls[-1]["max_new_tokens"], expected)
                self.assertEqual(public_kwargs, {} if expected == 4096 else {"max_new_tokens": expected})

    def test_decoded_cardinality_raises_before_cache(self) -> None:
        for decoded in ([], ["assistant\none"], ["assistant\none"] * 3):
            with self.subTest(decoded=decoded):
                wrapper = self._wrapper(2)
                with patch("lmms_eval.models.chat.qwen2_5_omni.process_mm_info", return_value=(None, None, None)), patch.object(wrapper.processor, "batch_decode", return_value=decoded):
                    with self.assertRaisesRegex(ValueError, "one string per request"):
                        wrapper.generate_until([_request(0, {}), _request(1, {})])
                self.assertEqual(wrapper.cache_hook.entries, [])


class TestQwenSimpleGenerationLimit(_QwenOmniTestCase):
    def test_public_limit_is_sent_to_thinker_for_explicit_and_default_values(self) -> None:
        class SimpleInputs(_Inputs):
            @property
            def input_ids(self):
                return self["input_ids"]

        class SimpleProcessor(_Processor):
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleInputs(input_ids=torch.tensor([[0]], dtype=torch.long))

        class SimpleModel(_Model):
            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return torch.cat((kwargs["input_ids"], torch.tensor([[1]], dtype=torch.long)), dim=1)

        adapter = self.simple_omni.Qwen2_5_Omni
        for supplied, expected in ((1, 1), (16, 16), (11, 11), (None, 4096)):
            with self.subTest(supplied=supplied):
                wrapper = adapter.__new__(adapter)
                wrapper.batch_size_per_gpu = 1
                wrapper._rank = 0
                wrapper.task_dict = {"task": {"test": [{"text": "question"}]}}
                wrapper.cache_hook = _Cache()
                wrapper.processor = SimpleProcessor()
                wrapper.processor.tokenizer.encode = lambda text: [0]
                wrapper.processor.tokenizer.decode = lambda token: ""
                wrapper.processor.batch_decode = lambda rows, **kwargs: ["answer"]
                wrapper._tokenizer = wrapper.processor.tokenizer
                wrapper._model = SimpleModel()
                wrapper.device_map = "cpu"
                wrapper.use_cache = True
                wrapper.system_prompt = "system"
                gen_kwargs = {} if supplied is None else {"max_new_tokens": supplied}
                request = Instance(
                    request_type="generate_until",
                    arguments=("question", gen_kwargs, lambda doc: [], 0, "task", "test"),
                    idx=0,
                    metadata={"task": "task", "doc_id": 0, "repeats": 1},
                )
                with patch.object(self.simple_omni, "process_mm_info", return_value=(None, None, None)):
                    self.assertEqual(wrapper.generate_until([request]), ["answer"])
                self.assertEqual(wrapper.model.calls[0]["thinker_max_new_tokens"], expected)
                self.assertNotIn("max_new_tokens", wrapper.model.calls[0])
