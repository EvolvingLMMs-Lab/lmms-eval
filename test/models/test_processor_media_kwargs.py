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
