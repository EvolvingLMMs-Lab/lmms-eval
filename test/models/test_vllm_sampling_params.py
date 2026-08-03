from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch


def _install_vllm_stubs() -> None:
    class _FakeVLLMSimple:
        @property
        def rank(self):
            return self._rank

        def _run_tp_synced(self, local_inputs, run_fn):
            return run_fn(local_inputs)

    modules = {
        "lmms_eval.api.instance": types.SimpleNamespace(
            GenerationResult=lambda text, token_counts=None: types.SimpleNamespace(text=text, token_counts=token_counts),
            Instance=object,
            TokenCounts=object,
        ),
        "lmms_eval.api.registry": types.SimpleNamespace(register_model=lambda _name: (lambda cls: cls)),
        "lmms_eval.imports": types.SimpleNamespace(optional_import=lambda *_args: (None, False)),
        "lmms_eval.models.model_utils.gen_metrics": types.SimpleNamespace(log_metrics=lambda **_kwargs: None),
        "lmms_eval.models.simple.vllm": types.SimpleNamespace(VLLM=_FakeVLLMSimple),
        "lmms_eval.protocol": types.SimpleNamespace(ChatMessages=object),
    }
    for name, module in modules.items():
        sys.modules[name] = module if isinstance(module, types.ModuleType) else _namespace_module(name, module)


def _namespace_module(name: str, namespace) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(vars(namespace))
    return module


def _load_module(module_name: str, relative_path: str):
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_STUBBED_MODULES = (
    "lmms_eval.api.instance",
    "lmms_eval.api.registry",
    "lmms_eval.imports",
    "lmms_eval.models.model_utils.gen_metrics",
    "lmms_eval.models.simple.vllm",
    "lmms_eval.protocol",
    "lmms_eval.models.chat.vllm",
    "lmms_eval.models.chat.vllm_generate",
)
_original_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
try:
    _install_vllm_stubs()
    _vllm_chat = _load_module("lmms_eval.models.chat.vllm", "lmms_eval/models/chat/vllm.py")
    _vllm_generate = _load_module("lmms_eval.models.chat.vllm_generate", "lmms_eval/models/chat/vllm_generate.py")
finally:
    for name, module in _original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

VLLMChat = _vllm_chat.VLLM
VLLMGenerate = _vllm_generate.VLLMGenerate


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _CaptureClient:
    def __init__(self):
        self.calls = []

    def chat(self, *, messages, sampling_params, chat_template):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "chat_template": chat_template,
            }
        )
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"chat-{idx}")]) for idx, _ in enumerate(messages)]

    def generate(self, inputs, sampling_params):
        self.calls.append(
            {
                "inputs": inputs,
                "sampling_params": sampling_params,
            }
        )
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"generate-{idx}")]) for idx, _ in enumerate(inputs)]


def _request(name: str):
    return types.SimpleNamespace(name=name)


def _configure_model(model, client: _CaptureClient) -> None:
    model.client = client
    model.batch_size_per_gpu = 2
    model._rank = 0
    model._tp_world_size = 1
    model._tp_group_cpu = None
    model.disable_log_stats = True
    model.chat_template = None


class TestVLLMSamplingParams(unittest.TestCase):
    def test_chat_backend_keeps_per_request_sampling_params(self):
        client = _CaptureClient()
        model = VLLMChat.__new__(VLLMChat)
        _configure_model(model, client)

        params_by_request = {
            "short": {"max_tokens": 16, "temperature": 0, "top_p": 1.0},
            "long": {"max_tokens": 128, "temperature": 0.7, "top_p": 0.8},
        }

        def make_one_request(request):
            return [{"role": "user", "content": request.name}], params_by_request[request.name]

        model.make_one_request = make_one_request

        with patch.object(_vllm_chat, "SamplingParams", _FakeSamplingParams):
            results = model.generate_until([_request("short"), _request("long")])

        self.assertEqual([result.text for result in results], ["chat-0", "chat-1"])
        self.assertEqual(len(client.calls), 1)
        sent_params = client.calls[0]["sampling_params"]
        self.assertEqual([params.kwargs for params in sent_params], [params_by_request["short"], params_by_request["long"]])

    def test_generate_backend_keeps_per_request_sampling_params(self):
        client = _CaptureClient()
        model = VLLMGenerate.__new__(VLLMGenerate)
        _configure_model(model, client)

        params_by_request = {
            "ocr": {"max_tokens": 128, "temperature": 0, "top_p": 1.0},
            "vqa": {"max_tokens": 32, "temperature": 0.2, "top_p": 0.9},
        }

        def make_one_request(request):
            return {"prompt": request.name, "multi_modal_data": {}}, params_by_request[request.name]

        model.make_one_request = make_one_request

        with patch.object(_vllm_generate, "SamplingParams", _FakeSamplingParams):
            results = model.generate_until([_request("ocr"), _request("vqa")])

        self.assertEqual([result.text for result in results], ["generate-0", "generate-1"])
        self.assertEqual(len(client.calls), 1)
        sent_params = client.calls[0]["sampling_params"]
        self.assertEqual([params.kwargs for params in sent_params], [params_by_request["ocr"], params_by_request["vqa"]])


if __name__ == "__main__":
    unittest.main()
