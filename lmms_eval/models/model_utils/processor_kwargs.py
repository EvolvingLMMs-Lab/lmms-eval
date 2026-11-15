import inspect
from typing import Any


def media_processor_kwargs(images: list, videos: list) -> dict:
    """Return the ``images``/``videos`` keyword arguments for a Hugging Face processor, or ``{}`` per empty media.

    Processors treat empty media lists differently from absent ones: transformers 5.22 crashes inside
    ``make_batched_videos([])`` for Qwen-VL processors when the task carries no video, and an empty images
    list breaks text-only requests. Omit empty media so the processor only sees what the request carries.
    """
    kwargs = {}
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
    return kwargs


def audio_processor_kwargs(processor: Any, audios: list) -> dict:
    """Return the keyword argument for passing ``audios`` to a Hugging Face processor, or ``{}`` without audio.

    transformers renamed the processor argument from ``audios`` to ``audio``, and transformers 5 rejects ``audios``
    for every processor. Some older model-specific processors only accept ``audios``, so that name is kept when the
    processor's signature has it and no ``audio`` parameter.
    """
    if not audios:
        return {}
    params = inspect.signature(processor.__call__).parameters
    key = "audios" if "audios" in params and "audio" not in params else "audio"
    return {key: audios}
