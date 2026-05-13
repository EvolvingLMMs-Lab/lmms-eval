import torch

from lmms_eval.api.model import lmms


class _DummyLM(lmms):
    def loglikelihood(self, requests):
        return []

    def generate_until(self, requests):
        return []

    def generate_until_multi_round(self, requests):
        return []


class _FakeAccelerator:
    def __init__(self, model):
        self._models = [model]
        self.free_memory_calls = 0

    def free_memory(self):
        self.free_memory_calls += 1
        self._models = []


def test_clean_releases_accelerator_model_references():
    lm = _DummyLM()
    model = torch.nn.Linear(1, 1)
    accelerator = _FakeAccelerator(model)
    lm._model = model
    lm.accelerator = accelerator

    lm.clean()

    assert accelerator.free_memory_calls == 1
    assert accelerator._models == []
    assert not hasattr(lm, "_model")
