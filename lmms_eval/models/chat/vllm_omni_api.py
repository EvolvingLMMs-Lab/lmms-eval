"""HTTP client backend for vLLM-Omni video-generation servers.

This backend targets an already-running vLLM-Omni OpenAI-compatible server.
For image-to-video benchmarks such as VBVR it sends multipart requests to
``POST /v1/videos/sync``, stores the returned MP4, and returns the JSON contract
expected by VBVR:

    {"text": "", "videos": ["/abs/path/to/generated.mp4"]}
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple

import httpx
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: Any, default: str = "x") -> str:
    s = _SAFE_RE.sub("_", str(name)).strip("_") or default
    return s[:128]


def _generate_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


def _default_output_dir(model_name: str | None, base_url: str) -> str:
    slug = _safe(model_name or base_url.replace("://", "_"), default="server")
    return os.path.join("./logs/vllm_omni_api", slug, _generate_run_id())


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_json_dict(value: Any, arg_name: str) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{arg_name} must be a JSON object string") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{arg_name} must be a JSON object")
        return parsed
    raise TypeError(f"{arg_name} must be a dict or JSON object string")


@register_model("vllm_omni_api")
class VLLMOmniAPI(lmms):
    is_simple = False

    def __init__(
        self,
        model_version: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        base_urls: Optional[str | List[str]] = None,
        api_key: Optional[str] = None,
        timeout: int = 1200,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        num_cpus: Optional[int] = 1,
        batch_size: Optional[int] = None,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_frames: int = 81,
        height: int = 384,
        width: int = 384,
        fps: int = 16,
        seed: Optional[int] = 42,
        boundary_ratio: Optional[float] = None,
        flow_shift: Optional[float] = None,
        true_cfg_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        extra_params: Optional[dict[str, Any] | str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if model is not None:
            model_version = model
        if kwargs:
            eval_logger.warning(f"Unknown model_args ignored: {list(kwargs.keys())}. " "Check the supported parameters for the 'vllm_omni_api' backend.")

        self.model_version = model_version or None
        configured_base_urls = base_urls or os.getenv("VLLM_OMNI_API_BASES")
        if configured_base_urls:
            if isinstance(configured_base_urls, str):
                urls = [url.strip() for url in re.split(r"[|;\s]+", configured_base_urls) if url.strip()]
            else:
                urls = [str(url).strip() for url in configured_base_urls if str(url).strip()]
            if not urls:
                raise ValueError("base_urls must contain at least one URL")
        else:
            urls = [base_url or os.getenv("VLLM_OMNI_API_BASE") or os.getenv("OPENAI_API_BASE") or "http://localhost:8091"]
        self.base_urls = [url.rstrip("/") for url in urls]
        self.base_url = self.base_urls[0]
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.batch_size_per_gpu = int(batch_size or 1)
        if num_cpus is None:
            self.num_cpus = max(1, cpu_count() // 2)
        else:
            self.num_cpus = max(1, int(num_cpus))

        self.output_dir = os.path.abspath(os.path.expanduser(output_dir or _default_output_dir(self.model_version, self.base_url)))
        os.makedirs(self.output_dir, exist_ok=True)
        self.overwrite = _parse_bool(overwrite)

        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.guidance_scale_2 = None if guidance_scale_2 is None else float(guidance_scale_2)
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.fps = int(fps)
        self.seed = None if seed is None else int(seed)
        self.boundary_ratio = None if boundary_ratio is None else float(boundary_ratio)
        self.flow_shift = None if flow_shift is None else float(flow_shift)
        self.true_cfg_scale = None if true_cfg_scale is None else float(true_cfg_scale)
        self.negative_prompt = negative_prompt
        self.extra_params = _parse_json_dict(extra_params, "extra_params") or {}

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed are supported."
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} processes with API data parallelism")
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self.device = accelerator.device
        if len(self.base_urls) > 1 and accelerator.is_local_main_process:
            eval_logger.info(f"Using {len(self.base_urls)} vLLM-Omni API endpoints")

    @property
    def model(self):
        return self.model_version

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "vllm_omni_api does not support loglikelihood"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("vllm_omni_api does not support multi-round generation")

    def _endpoint_url(self, base_url: Optional[str] = None) -> str:
        url = base_url or self.base_url
        if url.endswith("/v1"):
            return f"{url}/videos/sync"
        return f"{url}/v1/videos/sync"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "video/mp4"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_first_image_and_text(chat_messages: ChatMessages) -> Tuple[Any | None, str]:
        images, _, _ = chat_messages.extract_media()
        first_image = images[0] if images else None
        texts: list[str] = []
        for msg in chat_messages.messages:
            if msg.role != "user":
                continue
            for content in msg.content:
                if content.type == "text":
                    texts.append(content.text)
        return first_image, "\n".join(t for t in texts if t).strip()

    @staticmethod
    def _try_parse_vbvr_layout(doc: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        path = doc.get("first_frame_path") or doc.get("final_frame_path") or doc.get("prompt_path") or doc.get("ground_truth_video_path")
        if not path:
            return None
        parts = [p for p in str(path).split("/") if p]
        if len(parts) < 3:
            return None
        return parts[0], parts[1], parts[2]

    def _build_output_path(self, task: str, doc_id: Any, doc: Dict[str, Any]) -> str:
        layout = self._try_parse_vbvr_layout(doc)
        if layout is not None:
            file_split, task_name, video_idx = layout
            out_dir = os.path.join(self.output_dir, _safe(file_split), _safe(task_name))
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, f"{_safe(video_idx)}.mp4")

        out_dir = os.path.join(self.output_dir, _safe(task), _safe(str(doc_id)))
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, "video.mp4")

    @staticmethod
    def _image_file_tuple(image: Any) -> tuple[str, bytes, str] | None:
        if isinstance(image, Image.Image):
            handle = BytesIO()
            image.convert("RGB").save(handle, format="PNG")
            return ("input.png", handle.getvalue(), "image/png")

        if isinstance(image, str):
            if image.startswith("data:image"):
                media, payload = image.split(",", 1)
                mime = media.split(";", 1)[0].split(":", 1)[1] or "image/png"
                ext = mime.split("/", 1)[-1] or "png"
                return (f"input.{ext}", base64.b64decode(payload), mime)
            if os.path.isfile(os.path.expanduser(image)):
                path = os.path.expanduser(image)
                with open(path, "rb") as handle:
                    data = handle.read()
                ext = os.path.splitext(path)[1].lower()
                mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
                return (os.path.basename(path) or "input.png", data, mime)
        return None

    @staticmethod
    def _image_reference_json(image: Any) -> str | None:
        if isinstance(image, str) and image.startswith(("http://", "https://", "data:image")):
            return json.dumps({"image_url": image})
        return None

    @staticmethod
    def _coerce_extra_params(value: Any) -> dict[str, Any]:
        return _parse_json_dict(value, "extra_params") or {}

    def _request_params(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        gen = dict(gen_kwargs or {})
        params: dict[str, Any] = {
            "width": int(gen.get("width", self.width)),
            "height": int(gen.get("height", self.height)),
            "num_frames": int(gen.get("num_frames", self.num_frames)),
            "fps": int(gen.get("fps", self.fps)),
            "num_inference_steps": int(gen.get("num_inference_steps", self.num_inference_steps)),
            "guidance_scale": float(gen.get("guidance_scale", self.guidance_scale)),
        }
        optional_defaults = {
            "guidance_scale_2": self.guidance_scale_2,
            "seed": self.seed,
            "boundary_ratio": self.boundary_ratio,
            "flow_shift": self.flow_shift,
            "true_cfg_scale": self.true_cfg_scale,
            "negative_prompt": self.negative_prompt,
        }
        for key, default in optional_defaults.items():
            value = gen.get(key, default)
            if value is not None:
                params[key] = value

        merged_extra = dict(self.extra_params)
        if "extra_params" in gen and gen["extra_params"] is not None:
            merged_extra.update(self._coerce_extra_params(gen["extra_params"]))
        if merged_extra:
            params["extra_params"] = json.dumps(merged_extra, separators=(",", ":"))
        if self.model_version:
            params["model"] = self.model_version
        return params

    def make_one_request(self, request: Instance) -> Dict[str, Any]:
        _, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        doc = self.task_dict[task][split][doc_id]
        chat_messages = ChatMessages(messages=doc_to_messages(doc))
        first_image, prompt = self._extract_first_image_and_text(chat_messages)
        return {
            "prompt": prompt,
            "image": first_image,
            "params": self._request_params(gen_kwargs),
            "output_path": self._build_output_path(task, doc_id, doc),
            "task": task,
            "split": split,
            "doc_id": doc_id,
        }

    @staticmethod
    def _empty_result(error: str | None = None) -> GenerationResult:
        payload: dict[str, Any] = {"text": "", "videos": []}
        if error:
            payload["error"] = error
        return GenerationResult(text=json.dumps(payload), token_counts=TokenCounts())

    @staticmethod
    def _pack_result(mp4_path: str, metadata: dict[str, Any]) -> GenerationResult:
        payload = {"text": "", "videos": [mp4_path], "metadata": metadata}
        return GenerationResult(text=json.dumps(payload), token_counts=TokenCounts())

    async def _post_one(self, client: httpx.AsyncClient, prep: Dict[str, Any], idx: int) -> tuple[GenerationResult, int]:
        output_path = prep["output_path"]
        prompt = prep["prompt"]
        base_url = self.base_urls[idx % len(self.base_urls)]
        if not prompt:
            return self._empty_result("empty_prompt"), idx
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0 and not self.overwrite:
            return self._pack_result(os.path.abspath(output_path), {"cached": True}), idx

        params = dict(prep["params"])
        data = {key: str(value) for key, value in params.items() if value is not None}
        data["prompt"] = prompt

        files = None
        image_tuple = self._image_file_tuple(prep.get("image"))
        if image_tuple is not None:
            files = {"input_reference": image_tuple}
        else:
            image_reference = self._image_reference_json(prep.get("image"))
            if image_reference is not None:
                data["image_reference"] = image_reference

        last_error = "unknown error"
        for attempt in range(self.max_retries):
            started_at = time.perf_counter()
            try:
                response = await client.post(
                    self._endpoint_url(base_url),
                    data=data,
                    files=files,
                    headers=self._headers(),
                )
                elapsed = time.perf_counter() - started_at
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as handle:
                    handle.write(response.content)
                metadata = {
                    "cached": False,
                    "elapsed_s": elapsed,
                    "request_id": response.headers.get("X-Request-Id"),
                    "server_inference_time_s": response.headers.get("X-Inference-Time-S"),
                    "stage_durations": response.headers.get("X-Stage-Durations"),
                    "peak_memory_mb": response.headers.get("X-Peak-Memory-MB"),
                    "server_url": base_url,
                }
                return self._pack_result(os.path.abspath(output_path), metadata), idx
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                eval_logger.info(f"vllm_omni_api attempt {attempt + 1}/{self.max_retries} failed " f"for {output_path}: {last_error[:300]}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_s)

        return self._empty_result(last_error[:500]), idx

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        with ThreadPoolExecutor(max_workers=max(1, min(cpu_count(), len(requests) or 1))) as executor:
            prepared = list(executor.map(self.make_one_request, requests))

        async def run() -> list[tuple[GenerationResult, int]]:
            timeout = httpx.Timeout(self.timeout, connect=30.0)
            connection_count = max(1, self.num_cpus, len(self.base_urls))
            limits = httpx.Limits(max_connections=connection_count, max_keepalive_connections=connection_count)
            semaphore = asyncio.Semaphore(max(1, self.num_cpus))
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

                async def bounded(idx: int, prep: Dict[str, Any]) -> tuple[GenerationResult, int]:
                    async with semaphore:
                        return await self._post_one(client, prep, idx)

                tasks = [asyncio.create_task(bounded(idx, prep)) for idx, prep in enumerate(prepared)]
                results: list[tuple[GenerationResult, int]] = []
                pbar = tqdm(total=len(tasks), disable=(self.rank != 0), desc="vLLM-Omni API generating")
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    pbar.update(1)
                pbar.close()
                return results

        eval_results = asyncio.run(run())
        eval_results.sort(key=lambda item: item[1])
        return [result for result, _ in eval_results]
