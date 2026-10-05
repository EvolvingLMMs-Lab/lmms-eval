"""Causal likelihood scoring contracts for the simple and chat Qwen3 adapters."""

import math
from types import SimpleNamespace

import pytest
import torch

from lmms_eval.api.instance import Instance
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL as ChatQwen3VL
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


class TinyTokenizer:
    """Keep the prompt and choice tokens explicit, without automatic EOS."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        alphabet = {"P": 1, "Q": 2, "A": 3, "B": 4, "C": 5, "D": 6}
        return [alphabet[char] for char in text if not char.isspace()]


class BoundaryTokenizer(TinyTokenizer):
    """A continuation changes the final prompt token during joint encoding."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "PQ A":
            return [1, 7, 3]
        return super().encode(text, add_special_tokens=add_special_tokens)


class TinyProcessor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages: list, **kwargs: object) -> list[str]:
        self.calls.append({"messages": messages, **kwargs})
        return ["PQ"]


class TransitionModel(torch.nn.Module):
    """A real deterministic tensor forward pass with known next-token logits."""

    def __init__(self) -> None:
        super().__init__()
        table = torch.zeros(8, 8)
        table[1, 7] = 10.0  # Prompt prediction must not enter normal choice loss.
        table[2, 4] = 2.0  # After the prompt, B is the greedy choice.
        table[3, 4] = 1.0  # After A, the next B has a different probability.
        table[4, 6] = 3.0  # After B, D is greedy.
        self.register_buffer("table", table)
        self.calls: list[dict] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool, logits_to_keep: int) -> SimpleNamespace:
        assert not torch.is_grad_enabled()
        assert not use_cache
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        self.calls.append({"input_ids": input_ids.clone(), "logits_to_keep": logits_to_keep})
        return SimpleNamespace(logits=self.table[input_ids][:, -logits_to_keep:, :])


def make_model(model_class: type, tokenizer: TinyTokenizer | None = None) -> Qwen3_VL:
    model = model_class.__new__(model_class)
    model._model = TransitionModel()
    model._tokenizer = tokenizer or TinyTokenizer()
    model.processor = TinyProcessor()
    model.system_prompt = "Be precise."
    model.enable_thinking = False
    model._device = torch.device("cpu")
    model._rank = 0
    model.task_dict = {"likelihood-contract": {"test": [{"target": " B"}]}}
    model.cached: list[tuple] = []
    model.cache_hook = SimpleNamespace(add_partial=lambda *args: model.cached.append(args))
    return model


def request(continuation: str | object, visual: object = None, context: str = "The question and five-shot context") -> Instance:
    return Instance(
        request_type="loglikelihood",
        arguments=(context, continuation, visual or (lambda doc: []), 0, "likelihood-contract", "test"),
        idx=0,
        metadata={"task": "likelihood-contract", "doc_id": 0, "split": "test", "repeats": 1},
    )


@pytest.mark.parametrize("model_class", [Qwen3_VL, ChatQwen3VL])
@pytest.mark.parametrize(
    "continuation,expected_loss,greedy",
    [
        (" A", math.log(math.exp(2) + 7), False),
        (" B", math.log(math.exp(2) + 7) - 2, True),
        (" C", math.log(math.exp(2) + 7), False),
        (" D", math.log(math.exp(2) + 7), False),
        (" AB", math.log(math.exp(2) + 7) + math.log(math.e + 7) - 1, False),
        (" BD", math.log(math.exp(2) + 7) - 2 + math.log(math.exp(3) + 7) - 3, True),
    ],
)
def test_likelihood_scores_only_causally_shifted_continuation(model_class: type, continuation: str, expected_loss: float, greedy: bool) -> None:
    model = make_model(model_class)
    result = model.loglikelihood([request(continuation)])
    assert result[0][0] == pytest.approx(expected_loss)
    assert result[0][1] is greedy
    assert len(model.model.calls) == 1
    assert model.model.calls[0]["logits_to_keep"] == len(continuation.strip()) + 1
    assert model.cached[0][0] == "loglikelihood"
    assert model.cached[0][1] == ("The question and five-shot context", continuation)
    template = model.processor.calls[0]
    assert template["add_generation_prompt"] is True
    assert template["enable_thinking"] is False
    assert template["messages"][0][0]["role"] == "system"
    assert template["messages"][0][-1]["content"][0]["text"] == "The question and five-shot context"


@pytest.mark.parametrize("model_class", [Qwen3_VL, ChatQwen3VL])
def test_likelihood_callback_target_and_two_argument_unconditional_request(model_class: type) -> None:
    model = make_model(model_class)
    conditional = request(lambda doc: doc["target"])
    unconditional = request(" B")
    unconditional.arguments = ("", " B")
    results = model.loglikelihood([conditional, unconditional])
    assert len(results) == 2
    for loss, greedy in results:
        assert loss == pytest.approx(math.log(math.exp(2) + 7) - 2)
        assert greedy is True
    assert model.processor.calls[1]["messages"][0][-1]["content"][0]["text"] == ""


@pytest.mark.parametrize("model_class", [Qwen3_VL, ChatQwen3VL])
def test_likelihood_preserves_joint_encoding_at_prompt_boundary(model_class: type) -> None:
    model = make_model(model_class, BoundaryTokenizer())
    loss, greedy = model.loglikelihood([request(" A")])[0]
    assert loss == pytest.approx(math.log(math.exp(10) + 7) - 10 + math.log(8))
    assert not greedy
    assert model.model.calls[0]["input_ids"].tolist() == [[1, 7, 3]]


@pytest.mark.parametrize("model_class", [Qwen3_VL, ChatQwen3VL])
def test_empty_requests_and_continuations_need_no_forward_pass(model_class: type) -> None:
    model = make_model(model_class)
    assert model.loglikelihood([]) == []
    assert model.loglikelihood([request("")]) == [(0.0, True)]
    assert not model.model.calls


@pytest.mark.parametrize("model_class", [Qwen3_VL, ChatQwen3VL])
def test_likelihood_rejects_media_instead_of_ignoring_it(model_class: type) -> None:
    model = make_model(model_class)
    with pytest.raises(NotImplementedError, match="text-only"):
        model.loglikelihood([request(" A", visual=lambda doc: ["image.png"])])
    assert not model.model.calls
