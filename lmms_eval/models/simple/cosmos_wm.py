"""Cosmos World Model backend for evaluating next-state simulation quality.

Supports NVIDIA Cosmos-Predict2, Cosmos-Predict2.5, and Cosmos3 (omnimodal)
for image-conditioned and video-conditioned video generation (I2V and V2V).

Usage:
  # Cosmos Predict2 (I2V only)
  python -m lmms_eval \
    --model cosmos_wm \
    --model_args "backend=nim,output_dir=./logs/cosmos_wm" \
    --tasks physics_iq_i2v \
    --batch_size 1 \
    --log_samples

  # Cosmos Predict2.5-2B (I2V and V2V)
  python -m lmms_eval \
    --model cosmos_wm \
    --model_args "backend=local,nim_model=nvidia/Cosmos-Predict2.5-2B,output_dir=./logs/cosmos25" \
    --tasks physics_iq_i2v \
    --batch_size 1 \
    --log_samples

Backends:
  nim         NVIDIA NIM API (requires NVIDIA_API_KEY)
  local       Local diffusers pipeline (Predict2, Predict2.5, or Cosmos3)
  vllm_omni   Remote Cosmos3 served by a vLLM-Omni OpenAI-compatible video server
  transformers_reasoner
              Local Transformers Cosmos3 Reasoner for text-answer VQA tasks
  passthrough Returns input image as static video (for baseline comparison)

Environment:
  NVIDIA_API_KEY  Required for nim backend.
"""

import base64
import io
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests as http_requests
from loguru import logger as eval_logger
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

_WORLD_MODEL_TASK_PREFIXES = ("physics_iq_",)


def _is_world_model_task(task: str) -> bool:
    return task.startswith(_WORLD_MODEL_TASK_PREFIXES)


def _lazy_export_to_video():
    """Import diffusers' export_to_video, with an actionable error if missing."""
    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("Cosmos local backend requires diffusers: `pip install diffusers imageio imageio-ffmpeg`") from exc
    return export_to_video


def _image_to_base64(image: Any, fmt: str = "JPEG") -> str:
    """Convert a PIL Image or file path to base64 data URI."""
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
        mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"
    raise TypeError(f"Cannot convert {type(image)} to base64")


def _file_to_data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


def _extract_last_frame(video_path: str) -> Image.Image:
    """Extract the last frame from a video file as a PIL Image."""
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # Seek near end and decode last frame
        last_frame = None
        for frame in container.decode(stream):
            last_frame = frame
        if last_frame is None:
            raise ValueError(f"No frames in video: {video_path}")
        return last_frame.to_image()


@register_model("cosmos_wm")
class CosmosWorldModel(lmms):
    """World model for next-state video simulation.

    Evaluates world model quality by generating videos from image+instruction
    and measuring simulation fidelity via downstream metrics.
    """

    is_simple = True

    def __init__(
        self,
        # Backend selection
        backend: str = "local",
        # Model ID (for local backend: HF model ID; for NIM: API model path)
        nim_model: str = "nvidia/Cosmos-Predict2-2B-Video2World",
        nim_base_url: str = "https://ai.api.nvidia.com/v1/cv",
        # Model revision (required for Cosmos 2.5 branches)
        revision: str = "diffusers/base/post-trained",
        # Local backend params
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        num_inference_steps: int = 30,
        # Cosmos 3 safety checker (loads cosmos_guardrail when True). Default
        # False so the cosmos_guardrail package is not required.
        enable_safety_checker: bool = False,
        # Scheduler flow_shift override (Cosmos 3 uses 10.0 per the model card).
        flow_shift: Optional[float] = None,
        # Generation params (default 81 = 5s at 16fps + 1 for PhysicsIQ compat)
        num_frames: int = 81,
        fps: int = 16,
        width: int = 1280,
        height: int = 704,
        seed: Optional[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: str = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, poor lighting, washed-out colors, lack of detail, blurry backgrounds, monotonous scenery, cluttered composition, unnatural colors, pixelated images, repetitive patterns, flat visuals, poor framing, overexposed highlights, underexposed shadows, choppy transitions, artifacts, lens flare, distracting elements, inconsistent focus.",
        # Infrastructure
        output_dir: str = "./logs/cosmos_wm_videos",
        num_concurrent: int = 2,
        batch_size: Union[int, str] = 1,
        poll_interval: int = 5,
        max_polls: int = 240,
        max_retries: int = 2,
        # vllm_omni backend params (Cosmos3 served via vLLM-Omni OpenAI-compatible API)
        server_url: str = "http://localhost:8000",
        server_endpoint: str = "/v1/videos/sync",
        reasoner_server_url: Optional[str] = None,
        reasoner_endpoint: str = "/v1/chat/completions",
        reasoner_model: Optional[str] = None,
        reasoner_api_key_env: Optional[str] = None,
        request_timeout: int = 1800,
        max_sequence_length: int = 4096,
        send_cond_video: bool = True,
        cond_fps: int = 16,
        strip_cond_seconds: float = 0.0,
        # Local Transformers Reasoner params (Cosmos3 text-output tasks)
        reasoner_min_pixels: int = 256 * 28 * 28,
        reasoner_max_pixels: int = 1605632,
        reasoner_max_num_frames: int = 32,
        reasoner_video_fps: float = 4.0,
        reasoner_attn_implementation: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.nim_model = nim_model
        self.nim_base_url = nim_base_url.rstrip("/")
        self.revision = revision
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        # Pin each rank to its own GPU. The evaluator reads lm.device to place
        # cross-rank sync tensors *before* _load_pipeline runs, so a bare "cuda"
        # (the default) would put every rank's tensor on cuda:0 and collide.
        # Mirror DiffusersWMBase: cuda:{LOCAL_RANK} when available, else mps/cpu.
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.enable_safety_checker = bool(enable_safety_checker)
        self.flow_shift = float(flow_shift) if flow_shift is not None else None
        self.num_inference_steps = int(num_inference_steps)
        self.num_frames = int(num_frames)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed) if seed is not None else None
        self.guidance_scale = float(guidance_scale)
        self.negative_prompt = negative_prompt
        self.output_dir = str(output_dir)
        self.num_concurrent = max(1, int(num_concurrent))
        self.batch_size_per_gpu = int(batch_size)
        self.poll_interval = max(1, int(poll_interval))
        self.max_polls = max(1, int(max_polls))
        self.max_retries = max(1, int(max_retries))
        # vllm_omni — multiple replicas (one per GPU/node) for throughput;
        # requests are round-robined across them. The replica list is read from
        # the COSMOS3_SERVER_URLS env var when set (so the comma-separated list
        # does NOT collide with lmms-eval's comma-delimited model_args parsing);
        # otherwise from the server_url arg. Entries may be ";"/","/space
        # separated and bare "host:port" (http:// is prepended).
        import re as _re

        _raw = os.environ.get("COSMOS3_SERVER_URLS", "") or str(server_url)

        def _norm(u: str) -> str:
            u = u.strip().rstrip("/")
            if u and not u.startswith("http"):
                u = "http://" + u
            return u

        def _split_urls(raw: str) -> List[str]:
            return [_norm(u) for u in _re.split(r"[;,\s]+", raw) if u.strip()]

        self.server_urls = _split_urls(_raw)
        self.server_url = self.server_urls[0] if self.server_urls else _norm(str(server_url))
        reasoner_raw = os.environ.get("COSMOS3_REASONER_SERVER_URLS", "")
        if not reasoner_raw and reasoner_server_url is not None:
            reasoner_raw = str(reasoner_server_url)
        self.reasoner_server_urls = _split_urls(reasoner_raw) if reasoner_raw else list(self.server_urls)
        self._url_counter = 0
        self._reasoner_url_counter = 0
        self._url_lock = threading.Lock()
        self.server_endpoint = "/" + str(server_endpoint).lstrip("/")
        self.reasoner_endpoint = "/" + str(reasoner_endpoint).lstrip("/")
        self.reasoner_model = reasoner_model or self._default_reasoner_model(str(nim_model))
        self._reasoner_headers: Dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
        if reasoner_api_key_env:
            reasoner_api_key = os.environ.get(str(reasoner_api_key_env), "")
            if reasoner_api_key:
                self._reasoner_headers["Authorization"] = f"Bearer {reasoner_api_key}"
        self.request_timeout = int(request_timeout)
        self.max_sequence_length = int(max_sequence_length)
        self.send_cond_video = str(send_cond_video).lower() in ("1", "true", "yes")
        self.cond_fps = int(cond_fps)
        self.strip_cond_seconds = float(strip_cond_seconds)
        self.reasoner_min_pixels = int(reasoner_min_pixels)
        self.reasoner_max_pixels = int(reasoner_max_pixels)
        self.reasoner_max_num_frames = int(reasoner_max_num_frames)
        self.reasoner_video_fps = float(reasoner_video_fps)
        self.reasoner_attn_implementation = reasoner_attn_implementation

        # Resolve torch dtype
        _dtype_map = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16"}
        self._torch_dtype_str = _dtype_map.get(torch_dtype, torch_dtype)

        # Cache version detection
        self._is_v2_5 = self._is_cosmos_2_5()
        self._is_v3 = self._is_cosmos_3()
        if self._is_v3:
            # Cosmos 3 and 2.5 are mutually exclusive; v3 wins.
            self._is_v2_5 = False

        # Backend-specific init
        self._pipe = None  # Lazy-loaded for local backend
        self._transformers_reasoner_model = None
        self._transformers_reasoner_processor = None
        self._transformers_reasoner_device = device
        self._pipe_lock = threading.Lock()  # diffusers pipelines are NOT thread-safe

        # Force sequential generation for in-process model backends to avoid
        # racing a single loaded model from multiple Python threads.
        if backend in ("local", "transformers_reasoner") and self.num_concurrent > 1:
            eval_logger.info(f"{backend} backend: overriding num_concurrent={self.num_concurrent}->1 " f"(in-process generation is not thread-safe)")
            self.num_concurrent = 1

        if backend == "nim":
            self.api_key = os.environ.get("NVIDIA_API_KEY", "")
            if not self.api_key:
                raise EnvironmentError("NVIDIA_API_KEY environment variable is required for cosmos_wm nim backend.")
            self._session = http_requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
            )
        elif backend == "local":
            variant = "Cosmos3" if self._is_v3 else ("Cosmos-Predict2.5" if self._is_v2_5 else "Cosmos-Predict2")
            eval_logger.info(f"Local backend selected ({variant}). Model {nim_model} will be loaded on first use.")
        elif backend == "vllm_omni":
            # Cosmos3 served via vLLM-Omni OpenAI-compatible video API. The model
            # lives on the remote server; this backend is a thin HTTP client, so
            # concurrent requests are fine (the server batches them).
            self._session = http_requests.Session()
            eval_logger.info(
                f"vLLM-Omni backend selected. {len(self.server_urls)} replica(s)={self.server_urls}, "
                f"endpoint={self.server_endpoint}, timeout={self.request_timeout}s, "
                f"send_cond_video={self.send_cond_video}, strip_cond_seconds={self.strip_cond_seconds}, "
                f"reasoner_model={self.reasoner_model}, reasoner_replicas={self.reasoner_server_urls}, "
                f"reasoner_endpoint={self.reasoner_endpoint}"
            )
        elif backend == "transformers_reasoner":
            eval_logger.info(f"Transformers Reasoner backend selected. model={self.nim_model}, " f"video_fps={self.reasoner_video_fps}, max_num_frames={self.reasoner_max_num_frames}, " f"max_pixels={self.reasoner_max_pixels}")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        eval_logger.info(f"CosmosWM initialized: backend={backend}, model={nim_model}, " f"v2_5={self._is_v2_5}, v3={self._is_v3}, output_dir={self.output_dir}")

    # ------------------------------------------------------------------
    # Version detection
    # ------------------------------------------------------------------

    def _is_cosmos_2_5(self) -> bool:
        """Check if the configured model is a Cosmos Predict2.5 variant."""
        return "2.5" in self.nim_model or "Predict2.5" in self.nim_model

    def _is_cosmos_3(self) -> bool:
        """Check if the configured model is a Cosmos 3 (omnimodal) variant.

        Matches nvidia/Cosmos3-Nano, Cosmos3-Super, and future Cosmos-Predict3
        checkpoints. Case-insensitive.
        """
        name = self.nim_model.lower()
        return "cosmos3" in name or "predict3" in name

    @staticmethod
    def _default_reasoner_model(nim_model: str) -> str:
        name = nim_model.lower()
        if "cosmos3" in name and "nano" in name:
            return "nvidia/cosmos3-nano-reasoner"
        if "cosmos3" in name and "super" in name:
            return "nvidia/cosmos3-super-reasoner"
        return str(nim_model).rstrip("/").split("/")[-1]

    # ------------------------------------------------------------------
    # NIM API helpers
    # ------------------------------------------------------------------

    def _nim_submit(self, image_b64: str, prompt: str) -> Dict:
        """Submit image-to-video generation to NIM.

        Returns the synchronous result dict when the video is returned directly,
        otherwise ``{"request_id": ..., "data": ...}`` for async polling.
        """
        url = f"{self.nim_base_url}/{self.nim_model}"
        payload = {
            "image": image_b64,
            "prompt": prompt,
            "cfg_scale": self.guidance_scale,
            "seed": self.seed if self.seed is not None else 42,
        }

        resp = self._session.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        request_id = data.get("reqId") or data.get("request_id", "")
        if not request_id:
            # Synchronous response — video returned directly
            return data

        return {"request_id": request_id, "data": data}

    def _nim_poll(self, request_id: str) -> Dict:
        """Poll NIM for job completion."""
        url = f"{self.nim_base_url}/status/{request_id}"
        for i in range(1, self.max_polls + 1):
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "").upper()

            if status in ("COMPLETED", "FULFILLED"):
                return data
            if status in ("FAILED", "ERROR"):
                raise RuntimeError(f"NIM job failed: {json.dumps(data)[:500]}")

            if i % 10 == 0:
                eval_logger.debug(f"NIM poll {i}/{self.max_polls}: status={status}")
            time.sleep(self.poll_interval)

        raise TimeoutError(f"NIM job did not complete in {self.max_polls * self.poll_interval}s")

    def _nim_download(self, result_data: Dict, output_path: str) -> str:
        """Download video from NIM result."""
        video_url = None
        # Try common response fields
        for key in ("video_url", "videoUrl", "output", "url"):
            if key in result_data and isinstance(result_data[key], str):
                video_url = result_data[key]
                break
        # Handle nested data
        if video_url is None and "video" in result_data:
            vid = result_data["video"]
            if isinstance(vid, str):
                video_url = vid
            elif isinstance(vid, dict) and "url" in vid:
                video_url = vid["url"]

        if not video_url:
            # Result may contain base64-encoded video
            b64_video = result_data.get("video_b64") or result_data.get("video")
            if isinstance(b64_video, str) and not b64_video.startswith("http"):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(b64_video))
                return output_path
            raise ValueError(f"No video in NIM result: {json.dumps(result_data)[:500]}")

        # Download from URL
        resp = self._session.get(video_url, stream=True, timeout=300)
        resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return output_path

    # ------------------------------------------------------------------
    # Local inference via diffusers
    # ------------------------------------------------------------------

    def _ensure_local_pipeline(self):
        """Lazy-load the Cosmos diffusers pipeline on first use."""
        if self._pipe is not None:
            return

        import sys
        import types

        import torch

        # Stub cosmos_guardrail so diffusers' Cosmos pipeline can init
        # without installing the heavy (and numpy-incompatible) package.
        # diffusers does `from cosmos_guardrail import CosmosSafetyChecker`
        # at module level in pipeline_cosmos2_video2world.py.
        if "cosmos_guardrail" not in sys.modules:
            _stub = types.ModuleType("cosmos_guardrail")

            class _NoopGuardrail:
                """Catch-all stub that passes any safety check.

                - __call__: returns (first_arg, False) for result unpacking
                - __iter__: empty, so `for name, p in comp.named_parameters()` skips
                - __getattr__: returns no-op callable for any method
                """

                def __init__(self, *a, **kw):
                    pass

                def __call__(self, *a, **kw):
                    return (a[0] if a else None, False)

                def __iter__(self):
                    return iter([])

                def __bool__(self):
                    return True

                def __getattr__(self, name):
                    # Return first positional arg (the content being checked)
                    # so check_video_safety(vid) returns vid, not self.
                    return lambda *a, **kw: a[0] if a else self

            _stub.CosmosSafetyChecker = _NoopGuardrail
            _stub.CosmosGuardrail = _NoopGuardrail
            sys.modules["cosmos_guardrail"] = _stub

        try:
            if self._is_v3:
                # Cosmos3OmniPipeline lives in diffusers git-main (not in any
                # released diffusers as of 0.38.0). Import lazily so the class
                # import does not require diffusers at module import time.
                from diffusers import Cosmos3OmniPipeline as PipeClass
            elif self._is_v2_5:
                from diffusers import Cosmos2_5_PredictBasePipeline as PipeClass
            else:
                from diffusers import Cosmos2VideoToWorldPipeline as PipeClass
        except ImportError as exc:
            raise ImportError("Cosmos local backend requires diffusers: `pip install diffusers imageio imageio-ffmpeg`") from exc

        # MPS (Apple Silicon) does not support bfloat16 → fall back to float16
        dtype = getattr(torch, self._torch_dtype_str, torch.bfloat16)
        actual_device = self.device
        # Each rank is pinned to cuda:{LOCAL_RANK} in __init__; set the current
        # CUDA device on the loading thread so implicitly-placed tensors don't
        # default to cuda:0 and collide under multi-GPU DP. MPS/CPU adjust dtype.
        if actual_device.startswith("cuda"):
            torch.cuda.set_device(torch.device(actual_device))
        elif actual_device == "mps":
            dtype = torch.float16  # MPS doesn't support bfloat16
        elif actual_device == "cpu":
            dtype = torch.float32
            eval_logger.warning("No GPU available, running Cosmos on CPU (very slow)")

        self._actual_device = actual_device
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        # Build extra kwargs for from_pretrained
        if self._is_v3:
            # Cosmos3OmniPipeline takes `enable_safety_checker` (loads the
            # cosmos_guardrail CosmosSafetyChecker when True) instead of the
            # 2.x-style `safety_checker=None`. No subfolder revision for v3.
            load_kwargs: Dict[str, Any] = {"enable_safety_checker": self.enable_safety_checker}
        else:
            load_kwargs = {"safety_checker": None}
            if self._is_v2_5 and self.revision:
                load_kwargs["revision"] = self.revision

        variant_name = "Cosmos3" if self._is_v3 else ("Cosmos-Predict2.5" if self._is_v2_5 else "Cosmos-Predict2")
        eval_logger.info(
            f"Loading {variant_name} pipeline: {self.nim_model} " f"(dtype={dtype}, device={actual_device}, " f"token={'set' if hf_token else 'NOT SET'}" f"{', revision=' + self.revision if load_kwargs.get('revision') else ''})"
        )
        # Resolve local snapshot path to bypass Xet CDN 403 in containers.
        local_snapshot = self._find_local_snapshot(self.nim_model)

        if local_snapshot:
            eval_logger.info(f"Loading from local snapshot: {local_snapshot}")
            try:
                self._pipe = PipeClass.from_pretrained(
                    local_snapshot,
                    torch_dtype=dtype,
                    local_files_only=True,
                    **load_kwargs,
                )
            except Exception as e:
                eval_logger.warning(f"Local snapshot load failed ({e}), falling back to hub download")
                local_snapshot = None  # Fall through to hub download

        if not local_snapshot:
            eval_logger.info("Downloading from HuggingFace Hub...")
            self._pipe = PipeClass.from_pretrained(
                self.nim_model,
                torch_dtype=dtype,
                token=hf_token,
                **load_kwargs,
            )

        self._pipe.to(actual_device)

        # Verify no parameters stuck on meta device (can happen with
        # interrupted downloads or keep_in_fp32_modules edge cases).
        # Diffusers pipelines don't have named_parameters() directly,
        # so iterate over .components sub-models.
        meta_params = []
        for comp_name, comp in self._pipe.components.items():
            if hasattr(comp, "named_parameters"):
                for pname, p in comp.named_parameters():
                    if p.device.type == "meta":
                        meta_params.append(f"{comp_name}.{pname}")
        if meta_params:
            eval_logger.error(f"{len(meta_params)} parameters still on meta device after .to({actual_device}). " f"First 5: {meta_params[:5]}. Retrying with device_map...")
            del self._pipe
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            self._pipe = PipeClass.from_pretrained(
                local_snapshot or self.nim_model,
                torch_dtype=dtype,
                token=hf_token if not local_snapshot else None,
                local_files_only=bool(local_snapshot),
                device_map=actual_device,
                **load_kwargs,
            )

        eval_logger.info(f"{variant_name} pipeline loaded on {actual_device}")

        # Cosmos 3: apply the UniPCMultistepScheduler flow_shift tweak from the
        # official model card. flow_shift defaults to 10.0 for v3 when unset.
        # Guarded so a scheduler-config mismatch logs a warning instead of
        # hard-failing the run.
        if self._is_v3:
            v3_flow_shift = self.flow_shift if self.flow_shift is not None else 10.0
            try:
                from diffusers import UniPCMultistepScheduler

                self._pipe.scheduler = UniPCMultistepScheduler.from_config(self._pipe.scheduler.config, flow_shift=v3_flow_shift)
                eval_logger.info(f"Cosmos3: set UniPCMultistepScheduler flow_shift={v3_flow_shift}")
            except Exception as e:
                eval_logger.warning(f"Cosmos3: failed to apply UniPCMultistepScheduler flow_shift={v3_flow_shift} ({e}); using default scheduler")

        # Fix numpy 2.x + PIL incompatibility: np.asanyarray(pil_img) fails
        # with "invalid __array_struct__" in diffusers post-processing.
        # numpy 2.x checks __array_struct__ (C-level) before __array__,
        # and PIL's deprecated C interface triggers ValueError.
        # Patch np.asanyarray + its local ref in shape_base (used by np.stack).
        try:
            import numpy as _np

            if int(_np.__version__.split(".")[0]) >= 2 and not getattr(_np, "_cosmos_pil_patched", False):
                _orig_asanyarray = _np.asanyarray

                def _safe_asanyarray(a, dtype=None, order=None, **kw):
                    # Duck-type check: avoids isinstance failures from
                    # multiple PIL installs in containers
                    if hasattr(a, "tobytes") and hasattr(a, "size") and hasattr(a, "getbands"):
                        arr = _np.frombuffer(a.tobytes(), dtype=_np.uint8)
                        w, h = a.size
                        ch = len(a.getbands())
                        arr = arr.reshape((h, w, ch) if ch > 1 else (h, w))
                        return arr.astype(dtype) if dtype is not None else arr
                    try:
                        return _orig_asanyarray(a, dtype=dtype, order=order, **kw)
                    except ValueError:
                        # Last resort: if __array_struct__ fails on any object
                        if hasattr(a, "tobytes") and hasattr(a, "size"):
                            return _np.array(a)
                        raise

                _np.asanyarray = _safe_asanyarray
                # Patch local imports in shape_base (used by np.stack)
                import numpy._core.shape_base as _sb

                _sb.asanyarray = _safe_asanyarray
                _np._core.asanyarray = _safe_asanyarray
                try:
                    import numpy.core.shape_base as _sb2

                    _sb2.asanyarray = _safe_asanyarray
                    _np.core.asanyarray = _safe_asanyarray
                except (ImportError, AttributeError):
                    pass
                _np._cosmos_pil_patched = True
                eval_logger.info("Patched np.asanyarray for numpy 2.x + PIL compat")
        except Exception:
            pass

    @staticmethod
    def _find_local_snapshot(model_id: str) -> Optional[str]:
        """Find a locally cached snapshot directory for a HF model.

        Searches HF_HOME, HF_HUB_CACHE, and common paths to find a
        pre-downloaded snapshot, returning the full path or None.
        """
        # model_id like "nvidia/Cosmos-Predict2-2B-Video2World"
        safe_name = f"models--{model_id.replace('/', '--')}"
        search_dirs = []
        for env_var in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
            val = os.environ.get(env_var, "")
            if val:
                search_dirs.append(val)
                search_dirs.append(os.path.join(val, "hub"))
        # Common locations (including /root for containers with HOME override)
        search_dirs.extend(
            [
                os.path.expanduser("~/.cache/huggingface/hub"),
                "/root/.cache/huggingface/hub",
            ]
        )
        for cache_dir in search_dirs:
            model_dir = os.path.join(cache_dir, safe_name)
            refs_main = os.path.join(model_dir, "refs", "main")
            if os.path.exists(refs_main):
                with open(refs_main) as f:
                    commit = f.read().strip()
                snapshot = os.path.join(model_dir, "snapshots", commit)
                index_file = os.path.join(snapshot, "model_index.json")
                if os.path.exists(index_file):
                    return snapshot
        return None

    @staticmethod
    def _find_local_transformers_snapshot(model_id: str) -> Optional[str]:
        """Find a locally cached Transformers snapshot directory for a HF model."""

        def has_transformers_weights(snapshot: Path) -> bool:
            patterns = (
                "model.safetensors",
                "model-*.safetensors",
                "pytorch_model.bin",
                "pytorch_model-*.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
            return any(next(snapshot.glob(pattern), None) is not None for pattern in patterns)

        safe_name = f"models--{model_id.replace('/', '--')}"
        search_dirs = []
        for env_var in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
            val = os.environ.get(env_var, "")
            if val:
                search_dirs.append(val)
                search_dirs.append(os.path.join(val, "hub"))
        search_dirs.extend(
            [
                os.path.expanduser("~/.cache/huggingface/hub"),
                "/root/.cache/huggingface/hub",
            ]
        )
        for cache_dir in search_dirs:
            model_dir = Path(cache_dir) / safe_name
            refs_main = model_dir / "refs" / "main"
            if refs_main.exists():
                commit = refs_main.read_text().strip()
                snapshot = model_dir / "snapshots" / commit
                if (snapshot / "config.json").exists() and has_transformers_weights(snapshot):
                    return str(snapshot)
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = [p for p in snapshots_dir.iterdir() if (p / "config.json").exists() and has_transformers_weights(p)]
                if snapshots:
                    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return str(snapshots[0])
        return None

    def _local_generate(self, image: Image.Image, prompt: str, output_path: str) -> str:
        """Generate video using local diffusers pipeline (I2V mode)."""
        import torch

        export_to_video = _lazy_export_to_video()

        # Cosmos guardrail rejects empty prompts; use a neutral default.
        if not prompt or not prompt.strip():
            prompt = "Generate a natural video continuation."

        self._ensure_local_pipeline()
        actual_device = getattr(self, "_actual_device", self.device)

        # Ensure correct resolution and mode for Cosmos
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.LANCZOS)

        # MPS generator must be on CPU
        gen_device = "cpu" if actual_device == "mps" else actual_device
        generator = torch.Generator(device=gen_device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.manual_seed(42)

        call_kwargs: Dict[str, Any] = dict(
            image=image,
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_frames=self.num_frames,
            generator=generator,
            output_type="np",
        )

        with self._pipe_lock:
            output = self._pipe(**call_kwargs)

        # output.frames[0] is already numpy with output_type="np"
        frames_out = output.frames[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames_out, output_path, fps=self.fps)
        eval_logger.info(f"Local I2V generation saved: {output_path}")
        return output_path

    def _local_generate_v2v(self, video_frames: List[Image.Image], prompt: str, output_path: str) -> str:
        """Generate continuation video from conditioning video frames (V2V mode).

        Only Cosmos Predict2.5 supports native V2V. For Predict2, falls back
        to I2V using the last conditioning frame.
        """
        import torch

        export_to_video = _lazy_export_to_video()

        if not self._is_v2_5:
            # Fallback: use last frame as I2V conditioning
            eval_logger.info("V2V requested but model is Predict2; falling back to I2V with last frame")
            return self._local_generate(video_frames[-1], prompt, output_path)

        # Cosmos guardrail rejects empty prompts; use a neutral default.
        if not prompt or not prompt.strip():
            prompt = "Generate a natural video continuation."

        self._ensure_local_pipeline()
        actual_device = getattr(self, "_actual_device", self.device)

        # Resize all conditioning frames to expected resolution.
        # Keep as PIL Images — the diffusers pipeline's video_processor
        # handles PIL→tensor conversion. The __array_struct__ bug is
        # patched in the pipeline's output path via np.frombuffer.
        resized_frames = []
        for frame in video_frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if frame.size != (self.width, self.height):
                frame = frame.resize((self.width, self.height), Image.LANCZOS)
            resized_frames.append(frame)

        # MPS generator must be on CPU
        gen_device = "cpu" if actual_device == "mps" else actual_device
        generator = torch.Generator(device=gen_device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.manual_seed(42)

        with self._pipe_lock:
            output = self._pipe(
                video=resized_frames,
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                num_frames=self.num_frames,
                generator=generator,
                output_type="np",
            )

        # output.frames[0] is already numpy array (T, H, W, C) with output_type="np"
        frames_out = output.frames[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames_out, output_path, fps=self.fps)
        eval_logger.info(f"Local V2V generation saved: {output_path}")
        return output_path

    def _local_generate_v3(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        video: Optional[List[Image.Image]] = None,
        output_path: str = "",
    ) -> str:
        """Generate video with the Cosmos 3 omnimodal pipeline (Cosmos3OmniPipeline).

        Pass EITHER ``image`` (I2V) OR ``video`` (V2V) — they are mutually
        exclusive in the Cosmos3 ``__call__``. Conditioning frames are passed as
        PIL Images: the pipeline's VideoProcessor handles PIL→tensor conversion
        and resizing to (width, height), so we only normalize mode here and let
        the pipeline downscale (it never upscales).
        """
        import torch

        export_to_video = _lazy_export_to_video()

        if image is not None and video is not None:
            raise ValueError("Cosmos3 generation accepts either image (I2V) or video (V2V), not both.")

        # Cosmos guardrail (when enabled) rejects empty prompts; use a neutral
        # default. Harmless when the safety checker is disabled.
        if not prompt or not prompt.strip():
            prompt = "Generate a natural video continuation."

        self._ensure_local_pipeline()
        actual_device = getattr(self, "_actual_device", self.device)

        # Normalize conditioning visuals to RGB PIL. We deliberately do NOT
        # resize here: the Cosmos3 VideoProcessor resizes to (height, width).
        cond_image: Optional[Image.Image] = None
        cond_video: Optional[List[Image.Image]] = None
        if image is not None:
            cond_image = image if image.mode == "RGB" else image.convert("RGB")
        elif video is not None:
            cond_video = []
            for frame in video:
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                if frame.mode != "RGB":
                    frame = frame.convert("RGB")
                cond_video.append(frame)

        # MPS generator must be on CPU
        gen_device = "cpu" if actual_device == "mps" else actual_device
        generator = torch.Generator(device=gen_device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.manual_seed(42)

        # Only forward kwargs that exist in the Cosmos3OmniPipeline.__call__
        # signature (no Predict2-specific kwargs). Output is read via .video.
        call_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            fps=float(self.fps),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            enable_safety_check=self.enable_safety_checker,
            output_type="pil",
        )
        if cond_image is not None:
            call_kwargs["image"] = cond_image
        elif cond_video is not None:
            call_kwargs["video"] = cond_video

        with self._pipe_lock:
            result = self._pipe(**call_kwargs)

        # Cosmos3OmniPipelineOutput exposes .video (list of PIL frames for
        # output_type="pil"), NOT .frames.
        frames_out = result.video

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames_out, output_path, fps=self.fps)
        mode = "I2V" if cond_image is not None else "V2V"
        eval_logger.info(f"Local Cosmos3 {mode} generation saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # vLLM-Omni server backend (Cosmos3)
    # ------------------------------------------------------------------

    def _encode_frames_to_mp4(self, frames: List[Image.Image], path: str, fps: int) -> None:
        """Encode a list of PIL frames to an H.264 mp4 at the given fps."""
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError("Cosmos frame encoding requires imageio: `pip install imageio imageio-ffmpeg`") from exc
        import numpy as np

        arr = []
        for f in frames:
            if not isinstance(f, Image.Image):
                f = Image.fromarray(f)
            if f.mode != "RGB":
                f = f.convert("RGB")
            arr.append(np.asarray(f))
        iio.imwrite(path, np.stack(arr), fps=fps, codec="libx264")

    def _strip_leading_seconds(self, path: str, seconds: float) -> None:
        """Drop the first `seconds` of a video in-place (re-encode via ffmpeg)."""
        import subprocess

        tmp = path + ".strip.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seconds}", "-i", path, "-c:v", "libx264", "-an", tmp]
        subprocess.run(cmd, check=True)
        os.replace(tmp, path)

    def _vllm_omni_generate(self, image: Image.Image, prompt: str, output_path: str, video_frames: Optional[List[Image.Image]] = None) -> str:
        """Generate a video by POSTing to a vLLM-Omni Cosmos3 server.

        Conditioning:
          - V2V (multiframe): send the conditioning frames as an mp4 in
            ``input_reference`` (mime video/mp4) when ``send_cond_video`` is set,
            else fall back to the last frame as a single image.
          - I2V: send ``image`` as a single ``input_reference`` image.
          - T2V: no ``input_reference`` file.

        The server returns mp4 bytes (``/v1/videos/sync``). If
        ``strip_cond_seconds`` > 0, the leading conditioning window is dropped so
        the saved video is the prediction window only (Physics-IQ scoring compares
        the generated video directly against the 5s test clip).
        """
        if not prompt or not prompt.strip():
            prompt = "Generate a natural, physically plausible video continuation."

        # flow_shift is nullable on the shared __init__; vllm_omni still
        # defaults to 10.0 (unchanged behavior) when it is unset.
        _fs = self.flow_shift if self.flow_shift is not None else 10.0
        data = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "size": f"{self.width}x{self.height}",
            "num_frames": str(self.num_frames),
            "fps": str(self.fps),
            "num_inference_steps": str(self.num_inference_steps),
            "guidance_scale": str(self.guidance_scale),
            "max_sequence_length": str(self.max_sequence_length),
            "flow_shift": str(_fs),
            "extra_params": json.dumps(
                {
                    "use_resolution_template": False,
                    "use_duration_template": False,
                    "guardrails": self.enable_safety_checker,
                }
            ),
            "seed": str(self.seed if self.seed is not None else 42),
        }

        # Round-robin across replicas for throughput.
        with self._url_lock:
            base = self.server_urls[self._url_counter % len(self.server_urls)]
            self._url_counter += 1
        url = f"{base}{self.server_endpoint}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        tmp_cond = None
        files = None
        cond_file = None
        try:
            if video_frames and self.send_cond_video:
                tmp_cond = output_path + ".cond.mp4"
                self._encode_frames_to_mp4(video_frames, tmp_cond, self.cond_fps)
                cond_file = open(tmp_cond, "rb")
                files = {"input_reference": ("conditioning.mp4", cond_file, "video/mp4")}
            elif image is not None:
                buf = io.BytesIO()
                img = image.convert("RGB") if image.mode != "RGB" else image
                img.save(buf, format="JPEG", quality=95)
                buf.seek(0)
                files = {"input_reference": ("input.jpg", buf, "image/jpeg")}

            resp = self._session.post(url, data=data, files=files, headers={"Accept": "video/mp4"}, timeout=self.request_timeout)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                raise RuntimeError(f"Server returned JSON (not video): {resp.text[:500]}")
            with open(output_path, "wb") as f:
                f.write(resp.content)
        finally:
            if cond_file is not None:
                cond_file.close()
            if tmp_cond and os.path.exists(tmp_cond):
                os.remove(tmp_cond)

        if self.strip_cond_seconds > 0:
            self._strip_leading_seconds(output_path, self.strip_cond_seconds)

        eval_logger.info(f"vLLM-Omni generation saved: {output_path}")
        return output_path

    def _visual_to_openai_content(self, visual: Any) -> Optional[Dict[str, Any]]:
        if isinstance(visual, Image.Image):
            return {"type": "image_url", "image_url": {"url": _image_to_base64(visual)}}

        if isinstance(visual, str):
            video_exts = (".mp4", ".mov", ".webm", ".mkv")
            lower = visual.lower().split("?", 1)[0]
            if lower.startswith(("http://", "https://", "file://")):
                if lower.endswith(video_exts):
                    return {"type": "video_url", "video_url": {"url": visual}}
                return {"type": "image_url", "image_url": {"url": visual}}

            if os.path.exists(visual):
                if lower.endswith(video_exts):
                    return {"type": "video_url", "video_url": {"url": _file_to_data_uri(visual, "video/mp4")}}
                try:
                    return {"type": "image_url", "image_url": {"url": _image_to_base64(visual)}}
                except (UnidentifiedImageError, OSError):
                    return {"type": "video_url", "video_url": {"url": _file_to_data_uri(visual, "video/mp4")}}

        return None

    def _vllm_omni_reason(self, prompt: str, gen_kwargs: Dict[str, Any], visuals: List[Any]) -> str:
        """Answer text tasks through the Cosmos3 Reasoner OpenAI-compatible API."""
        if self.backend != "vllm_omni" or not self._is_v3:
            raise ValueError("Cosmos text-answer tasks require Cosmos3 with backend='vllm_omni'.")
        if not self.reasoner_server_urls:
            raise ValueError("Cosmos text-answer tasks require reasoner_server_url or COSMOS3_REASONER_SERVER_URLS.")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for visual in visuals:
            media = self._visual_to_openai_content(visual)
            if media is not None:
                content.append(media)

        with self._url_lock:
            base = self.reasoner_server_urls[self._reasoner_url_counter % len(self.reasoner_server_urls)]
            self._reasoner_url_counter += 1
        url = f"{base}{self.reasoner_endpoint}"

        payload: Dict[str, Any] = {
            "model": self.reasoner_model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["text"],
            "max_tokens": int(gen_kwargs.get("max_new_tokens", 4096)),
            "temperature": float(gen_kwargs.get("temperature", 0)),
        }
        top_p = gen_kwargs.get("top_p")
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if self.seed is not None:
            payload["seed"] = self.seed

        resp = self._session.post(url, json=payload, headers=self._reasoner_headers, timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            message = data["choices"][0]["message"]
            answer = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Cosmos Reasoner returned an unexpected response: {str(data)[:500]}") from exc

        if isinstance(answer, str):
            return answer
        if isinstance(answer, list):
            text = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
            if text:
                return text
            content_types = [str(part.get("type", type(part).__name__)) for part in answer if isinstance(part, dict)]
            raise RuntimeError(
                "Cosmos Reasoner returned non-text content "
                f"types={content_types or [type(answer).__name__]} from {url}. "
                "This usually means reasoner_server_url points at a Cosmos3 generator/vLLM-Omni endpoint. "
                "Use NVIDIA Cosmos3 Reasoner NIM, e.g. model nvidia/cosmos3-nano-reasoner."
            )
        return str(answer)

    def _ensure_transformers_reasoner(self) -> None:
        """Lazy-load the local Transformers Cosmos3 Reasoner."""
        if self._transformers_reasoner_model is not None:
            return
        if self.backend != "transformers_reasoner" or not self._is_v3:
            raise ValueError("Cosmos text-answer tasks require Cosmos3 with backend='transformers_reasoner'.")

        import torch

        try:
            from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration
        except ImportError as exc:
            raise ImportError("backend='transformers_reasoner' requires transformers>=5.11.0, " "where Cosmos3OmniForConditionalGeneration is available.") from exc

        dtype = getattr(torch, self._torch_dtype_str, torch.bfloat16)
        actual_device = self.device
        if actual_device.startswith("cuda"):
            torch.cuda.set_device(torch.device(actual_device))
        elif actual_device == "mps":
            dtype = torch.float16  # MPS doesn't support bfloat16
        elif actual_device == "cpu":
            dtype = torch.float32
            eval_logger.warning("CUDA not available, loading Cosmos3 Reasoner on CPU (very slow)")

        self._transformers_reasoner_device = actual_device
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        local_snapshot = self._find_local_transformers_snapshot(self.nim_model)
        model_source = local_snapshot or self.nim_model
        local_only = bool(local_snapshot)

        eval_logger.info(f"Loading Cosmos3 Reasoner with Transformers: {model_source} " f"(dtype={dtype}, device={actual_device}, token={'set' if hf_token else 'NOT SET'})")
        self._transformers_reasoner_processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=True,
            token=None if local_only else hf_token,
            local_files_only=local_only,
            min_pixels=self.reasoner_min_pixels,
            max_pixels=self.reasoner_max_pixels,
        )

        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "device_map": actual_device,
            "trust_remote_code": True,
            "token": None if local_only else hf_token,
            "local_files_only": local_only,
        }
        if self.reasoner_attn_implementation:
            load_kwargs["attn_implementation"] = self.reasoner_attn_implementation
        try:
            model = Cosmos3OmniForConditionalGeneration.from_pretrained(model_source, **load_kwargs)
        except TypeError:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            model = Cosmos3OmniForConditionalGeneration.from_pretrained(model_source, **load_kwargs)

        self._transformers_reasoner_model = model.eval()
        eval_logger.info(f"Cosmos3 Reasoner loaded on {actual_device}")

    def _visual_to_hf_content(self, visual: Any) -> Optional[Dict[str, Any]]:
        if isinstance(visual, Image.Image):
            return {
                "type": "image",
                "image": visual,
            }

        if isinstance(visual, str):
            lower = visual.lower().split("?", 1)[0]
            video_exts = (".mp4", ".mov", ".webm", ".mkv", ".avi")
            if lower.endswith(video_exts):
                return {
                    "type": "video",
                    "path": visual,
                }
            if lower.startswith(("http://", "https://")) or os.path.exists(visual):
                return {
                    "type": "image",
                    "path": visual,
                }

        return None

    def _build_transformers_generate_kwargs(self, gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 4096))
        temperature = float(gen_kwargs.get("temperature", 0) or 0)
        do_sample = bool(gen_kwargs.get("do_sample", temperature > 0))
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            top_p = gen_kwargs.get("top_p")
            if top_p is not None:
                generate_kwargs["top_p"] = float(top_p)
        num_beams = gen_kwargs.get("num_beams")
        if num_beams is not None:
            generate_kwargs["num_beams"] = int(num_beams)
        return generate_kwargs

    def _transformers_reason(self, prompt: str, gen_kwargs: Dict[str, Any], visuals: List[Any]) -> str:
        """Answer text tasks by loading Cosmos3 Reasoner in-process."""
        self._ensure_transformers_reasoner()
        import torch

        processor = self._transformers_reasoner_processor
        model = self._transformers_reasoner_model
        if processor is None or model is None:
            raise RuntimeError("Cosmos3 Transformers Reasoner failed to initialize.")

        content: List[Dict[str, Any]] = []
        has_video = False
        for visual in visuals:
            media = self._visual_to_hf_content(visual)
            if media is not None:
                content.append(media)
                has_video = has_video or media["type"] == "video"
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        template_kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if has_video:
            template_kwargs["processor_kwargs"] = {"num_frames": max(1, self.reasoner_max_num_frames)}

        dtype = getattr(torch, self._torch_dtype_str, torch.bfloat16)
        if str(self._transformers_reasoner_device).startswith("cpu"):
            dtype = torch.float32

        inputs = processor.apply_chat_template(messages, **template_kwargs).to(self._transformers_reasoner_device, dtype)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **self._build_transformers_generate_kwargs(gen_kwargs))

        generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids, strict=True)]
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        until = gen_kwargs.get("until")
        if isinstance(until, str):
            until = [until]
        if isinstance(until, list):
            for term in until:
                if term:
                    answer = answer.split(term)[0]
        return answer

    # ------------------------------------------------------------------
    # Passthrough (baseline)
    # ------------------------------------------------------------------

    def _passthrough_generate(self, image: Image.Image, output_path: str) -> str:
        """Return input image as a static 1-second video (baseline)."""
        import av

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        container = av.open(output_path, mode="w")
        stream = container.add_stream("libx264", rate=self.fps)
        stream.width = image.width
        stream.height = image.height
        stream.pix_fmt = "yuv420p"

        frame = av.VideoFrame.from_image(image)
        for _ in range(self.fps):  # 1 second of static frames
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return output_path

    # ------------------------------------------------------------------
    # Core generation pipeline
    # ------------------------------------------------------------------

    def _generate_video(
        self,
        image: Image.Image,
        prompt: str,
        task: str,
        doc_id: Union[str, int],
        video_frames: Optional[List[Image.Image]] = None,
    ) -> str:
        """Full pipeline: image + prompt → video path.

        Args:
            image: Conditioning image (used for I2V and as fallback for V2V).
            prompt: Text instruction describing the desired video.
            task: Task name for output path organization.
            doc_id: Document ID for output path naming.
            video_frames: Optional list of conditioning frames for V2V mode.
                When provided and backend is "local", uses V2V generation.
        """
        import hashlib

        safe_task = str(task).replace("/", "_").replace(" ", "_")
        # Cache key includes all generation params to avoid serving stale outputs
        # when seed, guidance_scale, num_frames, revision, or V2V input changes.
        v2v_tag = ""
        if video_frames:
            # Hash first+last frame content for V2V cache isolation
            frame_bytes = video_frames[0].tobytes()[:512] + video_frames[-1].tobytes()[:512]
            v2v_tag = f":v2v:{len(video_frames)}:{hashlib.sha256(frame_bytes).hexdigest()[:8]}"
        content_hash = hashlib.sha256(
            f"{self.backend}:{self.nim_model}:{self.revision}:{self.seed}:" f"{self.guidance_scale}:{self.num_frames}:{self.num_inference_steps}:" f"{prompt}:{image.size}:{image.tobytes()[:1024]}{v2v_tag}".encode()
        ).hexdigest()[:12]
        output_path = os.path.join(self.output_dir, safe_task, f"{safe_task}_{doc_id}_{content_hash}.mp4")

        # Check cache
        if os.path.exists(output_path):
            eval_logger.debug(f"Cache hit: {output_path}")
            return output_path

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.backend == "passthrough":
                    return self._passthrough_generate(image, output_path)

                if self.backend == "local":
                    if self._is_v3:
                        # Cosmos 3: route multi-frame conditioning to V2V (video=)
                        # and single-image conditioning to I2V (image=).
                        if video_frames is not None:
                            return self._local_generate_v3(prompt, video=video_frames, output_path=output_path)
                        return self._local_generate_v3(prompt, image=image, output_path=output_path)
                    if video_frames is not None:
                        return self._local_generate_v2v(video_frames, prompt, output_path)
                    return self._local_generate(image, prompt, output_path)

                if self.backend == "vllm_omni":
                    return self._vllm_omni_generate(image, prompt, output_path, video_frames=video_frames)

                if self.backend == "nim":
                    if video_frames is not None:
                        raise NotImplementedError("NIM backend does not support V2V (video conditioning). " "Use backend='local' for V2V tasks like physics_iq_v2v.")
                    image_b64 = _image_to_base64(image)
                    result = self._nim_submit(image_b64, prompt)

                    if isinstance(result, dict) and "request_id" in result:
                        poll_result = self._nim_poll(result["request_id"])
                        return self._nim_download(poll_result, output_path)
                    else:
                        # Synchronous result
                        return self._nim_download(result, output_path)

                raise ValueError(f"Unknown backend: {self.backend}")

            except Exception as exc:
                last_error = exc
                tb = traceback.format_exc()
                eval_logger.warning(f"Attempt {attempt}/{self.max_retries} failed: " f"task={task} doc_id={doc_id}: {exc}\n{tb}")
                if attempt < self.max_retries:
                    time.sleep(min(attempt * 5, 30))

        error_msg = f"[GENERATION_FAILED] {last_error}"
        eval_logger.error(f"All retries exhausted: task={task} doc_id={doc_id}: {last_error}")
        return error_msg

    # ------------------------------------------------------------------
    # Public: extract last frame (for agentic use)
    # ------------------------------------------------------------------

    def simulate(
        self,
        image: Image.Image,
        prompt: str,
        task: str = "sim",
        doc_id: int = 0,
        video_frames: Optional[List[Image.Image]] = None,
    ) -> Tuple[str, Image.Image]:
        """Generate video and extract last frame. Returns (video_path, last_frame).

        Args:
            image: Conditioning image for I2V (also used as fallback for V2V).
            prompt: Text instruction describing the desired video.
            task: Task name for output organization.
            doc_id: Document ID.
            video_frames: Optional list of conditioning frames for V2V mode.
        """
        video_path = self._generate_video(image, prompt, task, doc_id, video_frames=video_frames)
        if video_path.startswith("["):
            raise RuntimeError(video_path)
        last_frame = _extract_last_frame(video_path)
        return video_path, last_frame

    # ------------------------------------------------------------------
    # lmms interface
    # ------------------------------------------------------------------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate videos for world-model tasks or text answers via Cosmos3 Reasoner."""
        results: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Cosmos WM/Reasoner",
        )

        def _process(index: int) -> Tuple[int, str]:
            req = requests[index]
            ctx, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            prompt = str(ctx).strip()
            gen_kwargs = dict(gen_kwargs or {})

            visuals = []
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    visuals = doc_to_visual(doc) or []
                except Exception as e:
                    eval_logger.warning(f"Failed to extract visual: {e}")

            if not _is_world_model_task(str(task)):
                if self.backend == "vllm_omni":
                    return index, self._vllm_omni_reason(prompt, gen_kwargs, visuals)
                if self.backend == "transformers_reasoner":
                    return index, self._transformers_reason(prompt, gen_kwargs, visuals)
                raise ValueError("Cosmos text-answer tasks require backend='vllm_omni' or backend='transformers_reasoner'.")

            image = None
            video_frames: Optional[List[Image.Image]] = None
            if visuals:
                if len(visuals) > 1:
                    # Multiple frames = V2V conditioning
                    video_frames = []
                    for v in visuals:
                        if isinstance(v, Image.Image):
                            video_frames.append(v)
                        elif isinstance(v, str) and os.path.exists(v):
                            video_frames.append(Image.open(v))
                    # Use last frame as fallback image for cache key / non-V2V paths
                    image = video_frames[-1] if video_frames else None
                else:
                    v = visuals[0]
                    if isinstance(v, Image.Image):
                        image = v
                    elif isinstance(v, str) and os.path.exists(v):
                        image = Image.open(v)

            if image is None:
                # Create blank image if no conditioning provided
                image = Image.new("RGB", (self.width, self.height), (128, 128, 128))
                eval_logger.warning(f"No conditioning image for task={task} doc_id={doc_id}")

            return index, self._generate_video(image, prompt, task, doc_id, video_frames=video_frames)

        with ThreadPoolExecutor(max_workers=self.num_concurrent) as pool:
            futures = {pool.submit(_process, i): i for i in range(len(requests))}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, result = future.result()
                    results[idx] = result
                except Exception as exc:
                    eval_logger.error(f"Generation failed (idx={idx}): {exc}\n{traceback.format_exc()}")
                    results[idx] = f"[GENERATION_FAILED] {exc}"
                pbar.update(1)

        pbar.close()

        generated = sum(1 for r in results if r and not r.startswith("["))
        failed = len(results) - generated
        eval_logger.info(f"Cosmos WM complete: {generated} succeeded, {failed} failed, " f"output_dir={self.output_dir}")

        return [r if r is not None else "[ERROR] Unknown" for r in results]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("CosmosWM does not support loglikelihood")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("CosmosWM does not support multi-round generation")
