from __future__ import annotations

import base64
import json
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Any

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64

_AGENTIC_ONLY_KEYS = {"max_agentic_steps", "max_game_steps", "game_seed"}
_GENERATION_KEYS_TO_DROP = {"do_sample", "num_beams"}


class OpenAIModelServer(ModelServer):
    """OpenAI-compatible HTTP model server for agentic/game loops."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        client: Any = None,
        timeout: float | int = 600,
        default_max_tokens: int = 64,
        max_concurrent_requests: int | str | None = None,
        lm: Any = None,
        doc_id: int | None = None,
        task_name: str | None = None,
        split: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        response_cache: Any = None,
        **_: Any,
    ) -> None:
        del lm, doc_id, task_name, split, request_metadata, response_cache
        model = model or os.getenv("OPENAI_MODEL")
        if not model:
            raise ValueError("OpenAIModelServer requires model=... in --agentic_model_server_args or OPENAI_MODEL")
        self.model = model
        self.generation_kwargs = dict(generation_kwargs or {})
        self.default_max_tokens = int(default_max_tokens)
        self.max_concurrent_requests = max(1, int(max_concurrent_requests or 1))
        self._request_semaphore = BoundedSemaphore(self.max_concurrent_requests)
        if client is not None:
            self.client = client
        else:
            OpenAI, has_openai = optional_import("openai", "OpenAI")
            if not has_openai:
                raise ImportError("The agentic openai model_server requires the `openai` package.")
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
                base_url=(base_url or os.getenv("OPENAI_API_BASE") or "http://127.0.0.1:8000/v1").rstrip("/"),
                timeout=timeout,
            )

    def generate(self, request: Any) -> AgentOutput:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[Any]) -> list[AgentOutput]:
        if not requests:
            return []
        for request in requests:
            if not isinstance(request, AgentInput):
                raise TypeError(f"OpenAIModelServer requires AgentInput requests, got {type(request).__name__}")
        if len(requests) == 1 or self.max_concurrent_requests <= 1:
            return [self._generate_one(request) for request in requests]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent_requests, len(requests))) as executor:
            return list(executor.map(self._generate_one, requests))

    def _generate_one(self, request: AgentInput) -> AgentOutput:
        with self._request_semaphore:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._request_to_openai_messages(request),
                **self._build_openai_params(request),
            )
        return self._response_to_agent_output(response)

    def _build_openai_params(self, request: AgentInput) -> dict[str, Any]:
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(request.generation_kwargs or {})
        params: dict[str, Any] = {
            "max_tokens": int(generation_kwargs.pop("max_tokens", generation_kwargs.pop("max_new_tokens", self.default_max_tokens))),
            "temperature": generation_kwargs.pop("temperature", 0),
            "top_p": _normalize_top_p(generation_kwargs.pop("top_p", 1.0)),
        }
        _apply_stop_sequences(params, generation_kwargs)
        for key in _AGENTIC_ONLY_KEYS | _GENERATION_KEYS_TO_DROP:
            generation_kwargs.pop(key, None)
        params.update(generation_kwargs)
        return params

    def _request_to_openai_messages(self, request: AgentInput) -> list[dict[str, Any]]:
        if isinstance(request.metadata.get("messages"), list):
            return request.metadata["messages"]

        messages = []
        history = request.metadata.get("conversation_history")
        if isinstance(history, list):
            messages.extend(message for message in (self._history_turn_to_openai_message(turn) for turn in history) if message is not None)
        messages.append(self._request_to_openai_message(request))
        return messages

    def _request_to_openai_message(self, request: AgentInput) -> dict[str, Any]:
        return {"role": request.metadata.get("role", "user"), "content": self._content_blocks_to_openai_content(request.content)}

    def _content_blocks_to_openai_content(self, blocks: list[ContentBlock]) -> list[dict[str, Any]]:
        content = []
        for block in blocks:
            if block.type == "text" and block.data is not None:
                content.append({"type": "text", "text": str(block.data)})
            elif block.type in {"image", "image_url"}:
                content.append({"type": "image_url", "image_url": {"url": self._image_to_url(block.data)}})
            elif block.type in {"video", "video_url"}:
                content.append({"type": "video_url", "video_url": {"url": self._video_to_url(block.data)}})
            elif block.type in {"audio", "audio_url"}:
                content.append({"type": "audio_url", "audio_url": {"url": self._media_url(block.data, "audio_url")}})

        if not content:
            content.append({"type": "text", "text": ""})
        return content

    def _history_turn_to_openai_message(self, turn: Any) -> dict[str, Any] | None:
        if not isinstance(turn, dict):
            return None

        role = str(turn.get("role", "user"))
        content = turn.get("content", "")
        if role == "assistant":
            return {"role": role, "content": _history_text(content)}

        if isinstance(content, AgentInput):
            return {"role": role, "content": self._content_blocks_to_openai_content(content.content)}
        if _is_content_block_list(content):
            return {"role": role, "content": self._content_blocks_to_openai_content(content)}
        if isinstance(content, str):
            return {"role": role, "content": [{"type": "text", "text": content}]}
        if isinstance(content, list):
            return {"role": role, "content": content}
        return {"role": role, "content": [{"type": "text", "text": str(content)}]}

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

    def _video_to_url(self, data: Any) -> str:
        video = self._media_url(data, "video_url")
        if isinstance(video, str):
            if _is_url_like(video):
                return video
            if os.path.exists(video):
                return _file_to_data_url(video, default_mime_type="video/mp4")
            return video

        frames = _as_frames(video)
        quality = int(os.getenv("LMMS_VIDEO_JPEG_QUALITY", "85"))
        encoded_frames = [
            encode_image_to_base64(
                _frame_to_image(frame),
                image_format="JPEG",
                convert_rgb=True,
                quality=quality,
                copy_if_pil=False,
                use_path_cache=False,
            )
            for frame in frames
        ]
        return f"data:video/jpeg;base64,{','.join(encoded_frames)}"

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

    @staticmethod
    def _response_to_agent_output(response: Any) -> AgentOutput:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        text = getattr(message, "content", "") if message is not None else ""
        metadata: dict[str, Any] = {"raw_response": response}
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if tool_calls:
            metadata["tool_calls"] = [_tool_call_to_dict(tool_call) for tool_call in tool_calls]
        return AgentOutput(content=[ContentBlock.text(text or "")], metadata=metadata)


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    function = getattr(tool_call, "function", None)
    arguments = getattr(function, "arguments", {}) if function is not None else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_args": arguments}
    return {
        "name": getattr(function, "name", None) if function is not None else getattr(tool_call, "name", None),
        "arguments": arguments,
        "id": getattr(tool_call, "id", None),
        "type": getattr(tool_call, "type", None),
    }


def _normalize_top_p(top_p: Any) -> Any:
    if isinstance(top_p, bool):
        return top_p
    try:
        numeric_top_p = float(top_p)
    except (TypeError, ValueError):
        return top_p
    return 1.0 if numeric_top_p == 0.0 else top_p


def _apply_stop_sequences(params: dict[str, Any], generation_kwargs: dict[str, Any]) -> None:
    until = generation_kwargs.pop("until", None)
    if "stop" in generation_kwargs:
        return

    if until is None:
        return

    if isinstance(until, (list, tuple)):
        stop = [item for item in until if item is not None]
        if not stop:
            return
        params["stop"] = stop
        return

    params["stop"] = until


def _is_url_like(value: str) -> bool:
    return value.startswith(("http://", "https://", "file://", "data:image/", "data:video/"))


def _file_to_data_url(path: str, *, default_mime_type: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or default_mime_type
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _as_frames(video: Any) -> list[Any]:
    if video is None:
        return []
    if isinstance(video, (list, tuple)):
        return list(video)
    if hasattr(video, "ndim"):
        if video.ndim == 4:
            return list(video)
        if video.ndim == 3:
            return [video]
    return [video]


def _is_content_block_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, ContentBlock) for item in value)


def _history_text(content: Any) -> str:
    if isinstance(content, AgentOutput):
        return content.first_text() or ""
    if isinstance(content, str):
        return content
    if _is_content_block_list(content):
        return "\n".join(str(block.data) for block in content if block.type == "text" and block.data is not None)
    return "" if content is None else str(content)


def _frame_to_image(frame: Any) -> Any:
    if getattr(frame.__class__, "__module__", "").startswith("PIL."):
        return frame

    Image, has_pil = optional_import("PIL.Image")
    if not has_pil:
        return frame

    array = frame
    if hasattr(array, "ndim") and array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = array.transpose(1, 2, 0)
    return Image.fromarray(array)
