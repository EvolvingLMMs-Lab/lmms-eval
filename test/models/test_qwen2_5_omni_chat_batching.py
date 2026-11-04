"""CPU-only contract tests for Qwen2.5-Omni chat native batching."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.models.chat.qwen2_5_omni import Qwen2_5_Omni
from lmms_eval.models.simple import qwen2_5_omni as simple_omni
from lmms_eval.protocol import ChatMessages


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


class TestQwenProcessorLoading(unittest.TestCase):
    def test_processor_uses_model_source_in_simple_and_chat_adapters(self) -> None:
        default_source = "Qwen/Qwen2.5-Omni-7B"
        local_source = "/models/qwen-omni-snapshot"
        for adapter in (simple_omni.Qwen2_5_Omni, Qwen2_5_Omni):
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


class TestQwenNativeBatch(unittest.TestCase):
    def _wrapper(self, batch_size: int) -> Qwen2_5_Omni:
        wrapper = Qwen2_5_Omni.__new__(Qwen2_5_Omni)
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
                self.assertTrue(all(call["do_sample"] is True and call["max_new_tokens"] == 11 for call in wrapper.model.calls))
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
        self.assertEqual(wrapper.model.calls[0]["max_new_tokens"], 4096)
        self.assertEqual(wrapper.model.calls[0]["top_p"], None)
        self.assertEqual(wrapper.model.calls[0]["num_beams"], 1)

        wrapper.model.tuple_output = False
        wrapper.device_map = "cpu"
        with patch("lmms_eval.models.chat.qwen2_5_omni.process_mm_info", return_value=(None, None, None)):
            self.assertEqual(wrapper._infer_one(messages, kwargs), "final-0")
        self.assertEqual(len(wrapper.model.calls), 2)

    def test_decoded_cardinality_raises_before_cache(self) -> None:
        for decoded in ([], ["assistant\none"], ["assistant\none"] * 3):
            with self.subTest(decoded=decoded):
                wrapper = self._wrapper(2)
                with patch("lmms_eval.models.chat.qwen2_5_omni.process_mm_info", return_value=(None, None, None)), patch.object(wrapper.processor, "batch_decode", return_value=decoded):
                    with self.assertRaisesRegex(ValueError, "one string per request"):
                        wrapper.generate_until([_request(0, {}), _request(1, {})])
                self.assertEqual(wrapper.cache_hook.entries, [])
