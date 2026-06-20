from __future__ import annotations

import inspect
import os
from typing import Any

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64

_DEFAULT_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
_AGENTIC_ONLY_KEYS = {"max_agentic_steps", "max_game_steps", "game_seed"}
_GENERATION_KEYS_TO_DROP = {"do_sample", "num_beams"}


class VllmModelServer(ModelServer):
    """Direct vLLM Python API model server for agentic/game loops."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        generation_kwargs: dict[str, Any] | None = None,
        llm: Any = None,
        sampling_params_cls: Any = None,
        chat_template: str | None = None,
        chat_template_content_format: str = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        model_impl: str | None = None,
        use_flashinfer_sampler: bool | str | None = False,
        seed: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        default_max_tokens: int = 64,
        lm: Any = None,
        doc_id: int | None = None,
        task_name: str | None = None,
        split: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        response_cache: Any = None,
        **llm_kwargs: Any,
    ) -> None:
        del lm, doc_id, task_name, split, request_metadata, response_cache

        self.model = model
        self.generation_kwargs = dict(generation_kwargs or {})
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        self.add_generation_prompt = bool(add_generation_prompt)
        self.continue_final_message = bool(continue_final_message)
        self.tools = tools
        self.chat_template_kwargs = chat_template_kwargs
        self.tokenization_kwargs = tokenization_kwargs
        self.mm_processor_kwargs = mm_processor_kwargs
        self.use_tqdm = use_tqdm
        self.default_max_tokens = int(default_max_tokens)
        if use_flashinfer_sampler is not None:
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1" if _as_bool(use_flashinfer_sampler) else "0")
        self.sampling_params_cls = sampling_params_cls or _require_sampling_params()
        self.client = llm or self._build_llm(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            model_impl=model_impl,
            seed=seed,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            llm_kwargs=llm_kwargs,
        )

    def generate(self, request: AgentInput) -> AgentOutput:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[AgentInput]) -> list[AgentOutput]:
        if not requests:
            return []

        messages = [self._request_to_openai_messages(request) for request in requests]
        sampling_params = [self._build_sampling_params(request) for request in requests]
        responses = self.client.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=self.use_tqdm,
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
            add_generation_prompt=self.add_generation_prompt,
            continue_final_message=self.continue_final_message,
            tools=self.tools,
            chat_template_kwargs=self.chat_template_kwargs,
            tokenization_kwargs=self.tokenization_kwargs,
            mm_processor_kwargs=self.mm_processor_kwargs,
        )
        return [self._response_to_agent_output(response) for response in responses]

    @staticmethod
    def _build_llm(
        *,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
        model_impl: str | None,
        seed: int,
        max_model_len: int | None,
        max_num_seqs: int | None,
        llm_kwargs: dict[str, Any],
    ):
        LLM, has_vllm = optional_import("vllm", "LLM")
        if not has_vllm:
            raise ImportError("The agentic vllm model_server requires vLLM. Install it with `pip install vllm` in a Python version supported by vLLM.")

        kwargs = dict(llm_kwargs)
        if model_impl is not None:
            kwargs["model_impl"] = model_impl
        if max_model_len is not None:
            kwargs["max_model_len"] = int(max_model_len)
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = int(max_num_seqs)
        return LLM(
            model=model,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            trust_remote_code=trust_remote_code,
            seed=int(seed),
            **kwargs,
        )

    def _request_to_openai_messages(self, request: AgentInput) -> list[dict[str, Any]]:
        if isinstance(request.metadata.get("messages"), list):
            return request.metadata["messages"]

        content = []
        for block in request.content:
            if block.type == "text" and block.data is not None:
                content.append({"type": "text", "text": str(block.data)})
            elif block.type in {"image", "image_url"}:
                content.append({"type": "image_url", "image_url": {"url": self._image_to_url(block.data)}})
            elif block.type in {"video", "video_url"}:
                content.append({"type": "video_url", "video_url": {"url": self._media_url(block.data, "video_url")}})
            elif block.type in {"audio", "audio_url"}:
                content.append({"type": "audio_url", "audio_url": {"url": self._media_url(block.data, "audio_url")}})

        if not content:
            content.append({"type": "text", "text": ""})
        return [{"role": request.metadata.get("role", "user"), "content": content}]

    def _image_to_url(self, data: Any) -> str:
        image = self._media_url(data, "image_url")
        if isinstance(image, str) and _is_url_like(image):
            return image

        image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
        mime_type = f"image/{'jpeg' if image_format == 'JPG' else image_format.lower()}"
        quality = int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85")) if image_format in {"JPEG", "JPG", "WEBP"} else None
        encoded = encode_image_to_base64(
            image,
            image_format=image_format,
            convert_rgb=image_format in {"JPEG", "JPG", "WEBP"},
            quality=quality,
            copy_if_pil=False,
            use_path_cache=True,
        )
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _media_url(data: Any, nested_key: str) -> Any:
        if isinstance(data, dict):
            if "url" in data:
                return data["url"]
            nested = data.get(nested_key)
            if isinstance(nested, dict) and "url" in nested:
                return nested["url"]
            if nested is not None:
                return nested
        return data

    def _build_sampling_params(self, request: AgentInput):
        return self.sampling_params_cls(**self._build_sampling_params_dict(request))

    def _build_sampling_params_dict(self, request: AgentInput) -> dict[str, Any]:
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(request.generation_kwargs or {})

        params: dict[str, Any] = {
            "max_tokens": int(generation_kwargs.pop("max_tokens", generation_kwargs.pop("max_new_tokens", self.default_max_tokens))),
            "temperature": generation_kwargs.pop("temperature", 0),
            "top_p": _normalize_top_p(generation_kwargs.pop("top_p", 1.0)),
        }
        for key in _AGENTIC_ONLY_KEYS | _GENERATION_KEYS_TO_DROP:
            generation_kwargs.pop(key, None)
        params.update(generation_kwargs)
        return _filter_for_sampling_params(self.sampling_params_cls, params)

    @staticmethod
    def _response_to_agent_output(response: Any) -> AgentOutput:
        text = ""
        outputs = getattr(response, "outputs", None)
        if outputs:
            text = getattr(outputs[0], "text", "")
        else:
            text = str(response)
        return AgentOutput(content=[ContentBlock.text(text)], metadata={"raw_response": response})


def _require_sampling_params():
    SamplingParams, has_vllm = optional_import("vllm", "SamplingParams")
    if not has_vllm:
        raise ImportError("The agentic vllm model_server requires vLLM. Install it with `pip install vllm` in a Python version supported by vLLM.")
    return SamplingParams


def _normalize_top_p(top_p: Any) -> Any:
    if isinstance(top_p, bool):
        return top_p
    try:
        numeric_top_p = float(top_p)
    except (TypeError, ValueError):
        return top_p
    return 1.0 if numeric_top_p == 0.0 else top_p


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _filter_for_sampling_params(sampling_params_cls: Any, params: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(sampling_params_cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params
    return {key: value for key, value in params.items() if key in signature.parameters}


def _is_url_like(value: str) -> bool:
    return value.startswith(("http://", "https://", "file://", "data:image/"))
