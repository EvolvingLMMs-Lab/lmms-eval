import io
import mimetypes
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.concurrency_control import (
    AdaptiveConcurrencyConfig,
    decide_next_concurrency,
    is_rate_limit_error,
    make_prefix_hash,
    parse_bool,
)
from lmms_eval.models.model_utils.usage_metrics import is_budget_exceeded, log_usage

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
    eval_logger.warning("google-genai not installed. Install with: pip install google-genai")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".flv", ".wmv", ".webm", ".mkv"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}

# Gemini expects these exact audio MIME strings; mimetypes.guess_type would yield OS-dependent / x- variants.
_AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".aiff": "audio/aiff",
    ".m4a": "audio/mp4",
}


def _is_video_path(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in _VIDEO_EXTS)


def _is_audio_path(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in _AUDIO_EXTS)


def _audio_mime_type(path: str) -> str:
    """Map an audio file path to Gemini's canonical MIME type by extension."""
    ext = os.path.splitext(path)[1].lower()
    return _AUDIO_MIME_TYPES.get(ext) or mimetypes.guess_type(path)[0] or "audio/wav"


def _extract_safety_tag(response) -> str:
    """Build a [SAFETY_BLOCKED:...] tag from a blocked Gemini response."""
    parts = []
    try:
        for candidate in response.candidates:
            reason = getattr(candidate, "finish_reason", None)
            if reason is not None:
                parts.append(f"finish_reason={reason}")
            for rating in getattr(candidate, "safety_ratings", []):
                cat = getattr(rating, "category", "")
                prob = getattr(rating, "probability", "")
                if hasattr(cat, "name"):
                    cat = cat.name
                if hasattr(prob, "name"):
                    prob = prob.name
                if prob and prob not in ("NEGLIGIBLE", "LOW"):
                    parts.append(f"{cat}={prob}")
    except Exception:
        pass
    detail = ",".join(parts) if parts else "unknown"
    return f"[SAFETY_BLOCKED:{detail}]"


def _image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@register_model("gemini")
class Gemini(lmms):
    def __init__(
        self,
        model_version: str = "gemini-3.5-flash",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        vertexai: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        timeout: int = 120,
        retry_backoff_s: float = 1.0,
        max_retries: int = 5,
        interleave: bool = False,
        max_frames_num: int = 10,
        num_concurrent: int = 32,
        adaptive_concurrency: bool = False,
        adaptive_min_concurrency: int = 1,
        adaptive_max_concurrency: int = 128,
        adaptive_target_latency_s: float = 15.0,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_factor: float = 0.7,
        adaptive_failure_threshold: float = 0.05,
        prefix_aware_queue: bool = True,
        prefix_hash_chars: int = 256,
        safety_threshold: str = "BLOCK_NONE",
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if model is not None:
            model_version = model
        if kwargs:
            eval_logger.warning(f"Unknown model_args ignored: {list(kwargs.keys())}")

        self.model_version = model_version
        self.timeout = timeout
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.max_retries = max_retries
        self.interleave = parse_bool(interleave)
        self.max_frames_num = int(max_frames_num)
        self.num_concurrent = max(1, int(num_concurrent))
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.adaptive_config = AdaptiveConcurrencyConfig.from_raw(
            min_concurrency=adaptive_min_concurrency,
            max_concurrency=adaptive_max_concurrency,
            target_latency_s=adaptive_target_latency_s,
            increase_step=adaptive_increase_step,
            decrease_factor=adaptive_decrease_factor,
            failure_threshold=adaptive_failure_threshold,
        )
        self.prefix_aware_queue = parse_bool(prefix_aware_queue)
        self.prefix_hash_chars = max(32, int(prefix_hash_chars))
        self.safety_threshold = safety_threshold
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None else None

        # --- Client ---
        if genai is None:
            raise ImportError("google-genai is required. Install with: pip install google-genai")

        client_kwargs = {}
        if parse_bool(vertexai):
            client_kwargs["vertexai"] = True
            if project:
                client_kwargs["project"] = project
            if location:
                client_kwargs["location"] = location
        else:
            client_kwargs["api_key"] = api_key or os.getenv("GOOGLE_API_KEY")

        self.client = genai.Client(**client_kwargs)

        # --- Distributed ---
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type"
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        # --- File cache for video uploads (thread-safe) ---
        self._file_cache: dict = {}
        self._file_cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Safety settings
    # ------------------------------------------------------------------

    def _build_safety_settings(self) -> list:
        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
        return [types.SafetySetting(category=cat, threshold=self.safety_threshold) for cat in categories]

    # ------------------------------------------------------------------
    # Generation config
    # ------------------------------------------------------------------

    def _build_generation_config(self, gen_kwargs: dict) -> "types.GenerateContentConfig":
        max_output_tokens = gen_kwargs.get("max_new_tokens", 1024)
        temperature = gen_kwargs.get("temperature", 0)

        config_kwargs = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "safety_settings": self._build_safety_settings(),
        }
        if self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)

        return types.GenerateContentConfig(**config_kwargs)

    # ------------------------------------------------------------------
    # Video file upload (with cache + double-checked locking)
    # ------------------------------------------------------------------

    def _upload_video(self, video_path: str):
        with self._file_cache_lock:
            if video_path in self._file_cache:
                return self._file_cache[video_path]

        # Upload outside lock (IO-bound)
        uploaded = self.client.files.upload(file=video_path)

        # Poll for ACTIVE state
        for _ in range(60):  # max ~2 min wait
            state = getattr(uploaded, "state", None)
            if state is None:
                break
            # state may be a FileState enum whose str() is "FileState.ACTIVE"; take the bare member name
            s = getattr(state, "name", None) or str(state)
            s = s.upper().rsplit(".", 1)[-1]
            if s in ("ACTIVE", "STATE_UNSPECIFIED"):
                break
            if s == "FAILED":
                raise RuntimeError(f"Video upload failed: {video_path}")
            time.sleep(2)
            uploaded = self.client.files.get(name=uploaded.name)

        with self._file_cache_lock:
            if video_path not in self._file_cache:
                self._file_cache[video_path] = uploaded
            return self._file_cache[video_path]

    def _cleanup_files(self):
        with self._file_cache_lock:
            for f in self._file_cache.values():
                try:
                    self.client.files.delete(name=f.name)
                except Exception as e:
                    eval_logger.debug(f"Failed to delete uploaded file {f.name}: {e}")
            self._file_cache.clear()

    # ------------------------------------------------------------------
    # Build Gemini-native contents from visuals + text
    # ------------------------------------------------------------------

    def _build_contents(self, context: str, visuals: list) -> list:
        """Build a flat list of content parts for generate_content()."""
        parts = []

        if self.interleave:
            parts = self._build_interleaved_contents(context, visuals)
        else:
            # Text first, then media
            parts.append(context)
            for v in visuals:
                parts.append(self._visual_to_part(v))

        return parts

    def _visual_to_part(self, visual):
        """Convert a single visual (PIL Image, video path, audio dict) to a Gemini part."""
        if isinstance(visual, Image.Image):
            return types.Part.from_bytes(data=_image_to_bytes(visual), mime_type="image/png")
        if isinstance(visual, str):
            if _is_video_path(visual):
                uploaded = self._upload_video(visual)
                return types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
            if _is_audio_path(visual):
                with open(visual, "rb") as f:
                    audio_bytes = f.read()
                return types.Part.from_bytes(data=audio_bytes, mime_type=_audio_mime_type(visual))
            # Assume image path
            return types.Part.from_bytes(data=_image_to_bytes(Image.open(visual).convert("RGB")), mime_type="image/png")
        if isinstance(visual, dict) and "sampling_rate" in visual:
            # Audio dict from HF datasets
            import soundfile as sf

            audio_io = io.BytesIO()
            sf.write(audio_io, visual["array"], visual["sampling_rate"], format="WAV")
            audio_io.seek(0)
            return types.Part.from_bytes(data=audio_io.read(), mime_type="audio/wav")
        # Fallback: return as-is (let SDK handle it)
        return visual

    def _build_interleaved_contents(self, context: str, visuals: list) -> list:
        """Build interleaved text + media parts from <media_N> tags in context."""
        import re

        pattern = r"<media_(\d+)>"
        split_parts = re.split(pattern, context)
        result = []
        for i, part in enumerate(split_parts):
            if i % 2 == 0:
                if part:
                    result.append(part)
            else:
                idx = int(part)
                if idx < len(visuals):
                    result.append(self._visual_to_part(visuals[idx]))
        return result

    # ------------------------------------------------------------------
    # Flatten helper
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(nested):
        flat = []
        for item in nested:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    # ------------------------------------------------------------------
    # generate_until — concurrent, following OpenAI pattern
    # ------------------------------------------------------------------

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        from lmms_eval import utils

        def _collate(x):
            return -len(str(x[0])), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        ordered_requests = []
        for single_request in re_ords.get_batched(n=1, batch_fn=None):
            ordered_requests.extend(single_request)

        if not ordered_requests:
            return []

        pbar = tqdm(total=len(ordered_requests), disable=(self.rank != 0), desc="Model Responding")
        reordered_responses: List[Union[GenerationResult, None]] = [None] * len(ordered_requests)

        current_concurrency = min(self.num_concurrent, self.adaptive_config.max_concurrency)
        dispatch_order = list(range(len(ordered_requests)))
        if self.prefix_aware_queue:
            dispatch_order.sort(
                key=lambda idx: (make_prefix_hash(str(ordered_requests[idx][0]), self.prefix_hash_chars), idx),
            )

        cursor = 0
        failed_requests = 0
        rate_limited_requests = 0
        request_latencies: List[float] = []
        completed_since_adapt = 0
        in_flight = {}
        max_workers = max(1, self.adaptive_config.max_concurrency if self.adaptive_concurrency else current_concurrency)

        # ---- Inner functions ----

        def build_payload_for_index(global_index: int):
            context, gen_kwargs, doc_to_visual_fn, doc_id, task_name, split_name = ordered_requests[global_index]

            visuals = [doc_to_visual_fn(self.task_dict[task_name][split_name][doc_id])]
            if None in visuals:
                visuals = []
            else:
                visuals = self._flatten(visuals)

            contents = self._build_contents(context, visuals)
            config = self._build_generation_config(dict(gen_kwargs))
            return contents, config, task_name

        def process_single_request(local_index: int, contents: list, config, task_name: str):
            started_at = time.time()
            rate_limited = False
            last_error_msg = "unknown error"

            for attempt in range(self.max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_version,
                        contents=contents,
                        config=config,
                    )

                    # --- Extract token counts ---
                    token_counts = None
                    meta = getattr(response, "usage_metadata", None)
                    if meta:
                        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
                        output_tokens = getattr(meta, "candidates_token_count", 0) or 0
                        reasoning_tokens = getattr(meta, "thoughts_token_count", 0) or 0
                        log_usage(
                            model_name=self.model_version,
                            task_name=task_name,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=reasoning_tokens,
                            source="model",
                        )
                        token_counts = TokenCounts(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=reasoning_tokens,
                        )

                    # --- Extract text (may raise on safety block) ---
                    try:
                        response_text = response.text
                    except (ValueError, AttributeError):
                        response_text = None

                    # google-genai SDK returns None for blocked/truncated responses.
                    # Try to salvage partial thinking text before falling back to a tag.
                    if response_text is None:
                        # Attempt to extract partial text from candidates
                        try:
                            for candidate in response.candidates:
                                for part in getattr(candidate.content, "parts", []):
                                    text = getattr(part, "text", None)
                                    if text:
                                        response_text = text
                                        break
                                if response_text:
                                    break
                        except Exception:
                            pass

                    if response_text is None:
                        response_text = _extract_safety_tag(response)

                    latency = time.time() - started_at
                    return response_text, local_index, True, rate_limited, latency, token_counts

                except Exception as exc:
                    error_msg = str(exc)
                    last_error_msg = error_msg

                    # Check for safety block in exception message
                    if "finish_reason" in error_msg and ("SAFETY" in error_msg or "is 2" in error_msg):
                        # Safety block — don't retry
                        latency = time.time() - started_at
                        try:
                            tag = _extract_safety_tag(response)
                        except Exception:
                            tag = f"[SAFETY_BLOCKED:{error_msg[:100]}]"
                        return tag, local_index, True, False, latency, token_counts if "token_counts" in dir() else None

                    rate_limited = rate_limited or is_rate_limit_error(error_msg)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_backoff_s * (attempt + 1))
                    else:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")

            latency = time.time() - started_at
            error_preview = last_error_msg.replace("\n", " ")[:200]
            return f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}", local_index, False, rate_limited, latency, None

        def maybe_update_concurrency(force: bool = False):
            nonlocal current_concurrency, failed_requests, rate_limited_requests, request_latencies, completed_since_adapt

            if not self.adaptive_concurrency:
                return
            sample_threshold = max(4, current_concurrency)
            if not force and completed_since_adapt < sample_threshold:
                return
            if completed_since_adapt <= 0:
                return

            decision = decide_next_concurrency(
                current_concurrency=current_concurrency,
                total_requests=completed_since_adapt,
                failed_requests=failed_requests,
                rate_limited_requests=rate_limited_requests,
                latencies=request_latencies,
                config=self.adaptive_config,
            )
            if decision.next_concurrency != decision.current_concurrency:
                eval_logger.info(f"Adaptive concurrency: {decision.current_concurrency} -> {decision.next_concurrency} " f"(fail={decision.failure_rate:.3f}, rl={decision.rate_limit_rate:.3f}, p95={decision.p95_latency_s:.3f}s)")
            current_concurrency = decision.next_concurrency
            failed_requests = 0
            rate_limited_requests = 0
            request_latencies = []
            completed_since_adapt = 0

        # ---- Dispatch loop ----

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    if is_budget_exceeded():
                        reordered_responses[dispatch_order[cursor]] = GenerationResult(text="", token_counts=None)
                        pbar.update(1)
                        cursor += 1
                        continue

                    request_index = dispatch_order[cursor]
                    contents, config, task_name = build_payload_for_index(request_index)
                    future = executor.submit(process_single_request, request_index, contents, config, task_name)
                    in_flight[future] = request_index
                    cursor += 1

                if not in_flight:
                    break

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    response_text, local_index, success, rate_limited, latency, token_counts = future.result()
                    in_flight.pop(future, None)
                    reordered_responses[local_index] = GenerationResult(text=response_text, token_counts=token_counts)
                    if not success:
                        failed_requests += 1
                    if rate_limited:
                        rate_limited_requests += 1
                    request_latencies.append(latency)
                    completed_since_adapt += 1
                    pbar.update(1)
                    maybe_update_concurrency(force=False)

        maybe_update_concurrency(force=True)
        pbar.close()

        # Cleanup uploaded video files
        self._cleanup_files()

        completed = [r if r is not None else GenerationResult(text="", token_counts=None) for r in reordered_responses]
        return re_ords.get_original(completed)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Gemini API does not support loglikelihood")
