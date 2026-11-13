import inspect
from typing import Any


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
