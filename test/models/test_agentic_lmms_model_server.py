from __future__ import annotations

from lmms_eval.agentic.model_server.lmms import LmmsModelServer
from lmms_eval.agentic.types import AgentInput, ContentBlock


class _FakeLm:
    is_simple = False

    def __init__(self):
        self.calls = []
        self.task_dict = {"game": {"test": [{"id": 0}, {"id": 1}]}}

    def generate_until(self, instances):
        self.calls.append(instances)
        return [f"answer-{idx}" for idx, _instance in enumerate(instances)]


def _request(text, doc_id, **generation_kwargs):
    return AgentInput(
        content=[ContentBlock.text(text)],
        generation_kwargs=generation_kwargs,
        metadata={
            "lmms_eval": {
                "doc_id": doc_id,
                "task_name": "game",
                "split": "test",
                "request_metadata": {"task": "game", "doc_id": doc_id, "repeats": 1},
            }
        },
    )


def test_lmms_model_server_batches_agent_inputs_into_one_generate_until_call():
    lm = _FakeLm()
    server = LmmsModelServer(lm=lm, generation_kwargs={"max_new_tokens": 4})

    outputs = server.generate_batch(
        [
            _request("obs 0", 0, temperature=0),
            _request("obs 1", 1, max_new_tokens=7),
        ]
    )

    assert [output.first_text() for output in outputs] == ["answer-0", "answer-1"]
    assert len(lm.calls) == 1
    instances = lm.calls[0]
    assert len(instances) == 2
    assert instances[0].args[0] == "obs 0"
    assert instances[0].args[2] == {"max_new_tokens": 4, "temperature": 0}
    assert instances[0].args[3:6] == (0, "game", "test")
    assert instances[0].idx == 0
    assert instances[1].args[0] == "obs 1"
    assert instances[1].args[2] == {"max_new_tokens": 7}
    assert instances[1].idx == 0
    assert instances[1].metadata["doc_id"] == 1
