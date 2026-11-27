"""Chat models must omit empty media lists when calling a Hugging Face processor.

Processors treat empty media lists differently from absent ones: on transformers 5.22 a Qwen-VL processor
crashes inside ``make_batched_videos([])`` when an image-only task passes ``videos=[]``, and an empty images
list breaks text-only requests (``IndexError`` in ``make_batched_images``). The audio kwarg follows the same
rule via ``audio_processor_kwargs``.
"""

from lmms_eval.models.model_utils.processor_kwargs import media_processor_kwargs


def test_empty_media_lists_are_omitted():
    assert media_processor_kwargs([], []) == {}


def test_present_media_are_passed_by_name():
    images = ["frame.png"]
    videos = ["clip.mp4"]
    assert media_processor_kwargs(images, []) == {"images": images}
    assert media_processor_kwargs([], videos) == {"videos": videos}
    assert media_processor_kwargs(images, videos) == {"images": images, "videos": videos}


class _StopAfterProcessor(Exception):
    pass


class _RecordingProcessor:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        raise _StopAfterProcessor


def _async_processor_kwargs(content):
    from types import SimpleNamespace

    import pytest

    from lmms_eval.models.chat.async_hf_model import AsyncHFModel, _WorkerResources

    model = AsyncHFModel.__new__(AsyncHFModel)
    model.task_dict = {"task": {"test": {0: {}}}}
    processor = _RecordingProcessor()
    worker = _WorkerResources(model=None, processor=processor, tokenizer=None, device="cpu")
    request = SimpleNamespace(args=("", lambda doc: [{"role": "user", "content": content}], {}, 0, "task", "test"))
    with pytest.raises(_StopAfterProcessor):
        model._run_single_request(worker, request)
    return processor.kwargs


def test_async_hf_model_omits_empty_media_lists():
    from PIL import Image

    text_only = _async_processor_kwargs([{"type": "text", "text": "hi"}])
    assert "images" not in text_only and "videos" not in text_only

    image = Image.new("RGB", (4, 4))
    with_image = _async_processor_kwargs([{"type": "image", "url": image}, {"type": "text", "text": "hi"}])
    assert with_image["images"] == [image] and "videos" not in with_image
