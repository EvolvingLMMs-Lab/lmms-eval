from __future__ import annotations

import base64

from PIL import Image

from lmms_eval.agentic.model_server import VllmModelServer
from lmms_eval.agentic.registry import build_model_server
from lmms_eval.agentic.types import AgentInput, ContentBlock


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCompletion:
    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.outputs = [FakeCompletion(text)]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(
        self,
        *,
        messages,
        sampling_params,
        use_tqdm=True,
        chat_template=None,
        chat_template_content_format="auto",
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        chat_template_kwargs=None,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "chat_template": chat_template,
                "chat_template_content_format": chat_template_content_format,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
                "tools": tools,
                "chat_template_kwargs": chat_template_kwargs,
                "tokenization_kwargs": tokenization_kwargs,
                "mm_processor_kwargs": mm_processor_kwargs,
            }
        )
        return [FakeResponse("MOVE_FORWARD") for _ in messages]


def test_vllm_model_server_converts_agent_input_to_chat_request():
    llm = FakeLLM()
    server = VllmModelServer(
        model="fake-model",
        llm=llm,
        sampling_params_cls=FakeSamplingParams,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
        generation_kwargs={"max_new_tokens": 7, "temperature": 0, "top_p": 0, "do_sample": False, "num_beams": 1, "until": ["\n\n"]},
    )

    output = server.generate(
        AgentInput(
            content=[
                ContentBlock.text("go east"),
                ContentBlock(type="state_features", data={"position": [1, 1]}),
            ],
            generation_kwargs={"max_new_tokens": 3},
        )
    )

    assert output.first_text() == "MOVE_FORWARD"
    call = llm.calls[0]
    assert call["messages"] == [[{"role": "user", "content": [{"type": "text", "text": "go east"}]}]]
    assert call["sampling_params"][0].kwargs["max_tokens"] == 3
    assert call["sampling_params"][0].kwargs["top_p"] == 1.0
    assert call["sampling_params"][0].kwargs["stop"] == ["\n\n"]
    assert "until" not in call["sampling_params"][0].kwargs
    assert "do_sample" not in call["sampling_params"][0].kwargs
    assert "num_beams" not in call["sampling_params"][0].kwargs
    assert call["chat_template_kwargs"] == {"enable_thinking": False}
    assert call["use_tqdm"] is False


def test_vllm_model_server_encodes_video_frames_as_jpeg_sequence_data_url():
    llm = FakeLLM()
    server = VllmModelServer(model="fake-model", llm=llm, sampling_params_cls=FakeSamplingParams)
    frames = [Image.new("RGB", (4, 4), color="red"), Image.new("RGB", (4, 4), color="blue")]

    server.generate(AgentInput(content=[ContentBlock.text("play"), ContentBlock(type="video", data=frames)]))

    content = llm.calls[0]["messages"][0][0]["content"]
    video_url = content[1]["video_url"]["url"]
    assert content[1]["type"] == "video_url"
    assert video_url.startswith("data:video/jpeg;base64,")
    encoded_frames = video_url.removeprefix("data:video/jpeg;base64,").split(",")
    assert len(encoded_frames) == 2
    assert base64.b64decode(encoded_frames[0]).startswith(b"\xff\xd8")


def test_vllm_model_server_converts_conversation_history_to_chat_messages():
    llm = FakeLLM()
    server = VllmModelServer(model="fake-model", llm=llm, sampling_params_cls=FakeSamplingParams)
    frame = Image.new("RGB", (4, 4), color="green")

    server.generate(
        AgentInput(
            content=[ContentBlock.text("step 1"), ContentBlock(type="image", data=frame)],
            metadata={
                "conversation_history": [
                    {"role": "user", "content": [ContentBlock.text("step 0"), ContentBlock(type="video", data=[frame])]},
                    {"role": "assistant", "content": '{"buttons": ["ATTACK"], "tics": 1}'},
                ]
            },
        )
    )

    messages = llm.calls[0]["messages"][0]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"][1]["type"] == "video_url"
    assert messages[1]["content"] == '{"buttons": ["ATTACK"], "tics": 1}'
    assert messages[2]["content"][0] == {"type": "text", "text": "step 1"}
    assert messages[2]["content"][1]["type"] == "image_url"


def test_agentic_registry_builds_vllm_from_dict_spec():
    llm = FakeLLM()

    server = build_model_server(
        {
            "name": "vllm",
            "model": "fake-model",
            "llm": llm,
            "sampling_params_cls": FakeSamplingParams,
        },
        generation_kwargs={"max_new_tokens": 5},
    )

    assert isinstance(server, VllmModelServer)
    assert server._build_sampling_params_dict(AgentInput())["max_tokens"] == 5


def test_vllm_model_server_uses_data_parallel_size_as_rollout_concurrency():
    server = VllmModelServer(
        model="fake-model",
        llm=FakeLLM(),
        sampling_params_cls=FakeSamplingParams,
        data_parallel_size=8,
    )

    assert server.max_parallel_rollouts(16) == 8
