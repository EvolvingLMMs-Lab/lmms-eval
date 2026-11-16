from lmms_eval.models.model_utils.processor_kwargs import audio_processor_kwargs


class NewStyleProcessor:
    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        pass


class OldStyleProcessor:
    def __call__(self, text=None, audios=None, **kwargs):
        pass


class KwargsOnlyProcessor:
    def __call__(self, text=None, images=None, **kwargs):
        pass


def test_no_audio_passes_nothing():
    # transformers 5 rejects `audios`, so it must not be passed even as an empty list
    assert audio_processor_kwargs(NewStyleProcessor(), []) == {}
    assert audio_processor_kwargs(OldStyleProcessor(), []) == {}


def test_audio_uses_the_processor_argument_name():
    audios = ["waveform"]
    assert audio_processor_kwargs(NewStyleProcessor(), audios) == {"audio": audios}
    assert audio_processor_kwargs(KwargsOnlyProcessor(), audios) == {"audio": audios}
    assert audio_processor_kwargs(OldStyleProcessor(), audios) == {"audios": audios}
