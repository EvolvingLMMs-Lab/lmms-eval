"""Video generation model backend using fal.ai API.

Two modes, selected via ``mode=`` in model_args:

- ``t2v`` (default): text-to-video / text-to-image. Each request submits the
  doc's text prompt, polls the fal.ai queue, downloads the generated video,
  and returns the local file path as the generation result.
- ``v2v``: video-conditioned generation (extend / edit / reference-to-video).
  Each request uploads the
  doc's conditioning video to the fal CDN, calls the endpoint, downloads the
  result, trims the leading conditioning segment when the endpoint returns
  input++continuation, and returns the local path of the continuation video.

Usage (t2v):
  python -m lmms_eval \
    --model generation_api \
    --model_args "model=wan/v2.6/text-to-video,output_dir=./output_videos" \
    --tasks videogen_test \
    --batch_size 1 \
    --limit 4 \
    --log_samples

Usage (v2v):
  python -m lmms_eval \
    --model generation_api \
    --model_args "mode=v2v,model=grok_extend,output_dir=./output_videos/grok_extend" \
    --tasks videogen_v2v_test \
    --batch_size 1 \
    --log_samples

Environment:
  FAL_KEY or FAL_API_KEY must be set for any request that reaches the API.
  Runs where every request resumes from an existing output file complete
  without a key (and without billing).

Resume / no-double-billing contract (v2v):
  The output path for a doc is ``output_dir/<task>/<doc name>.mp4`` (falling
  back to ``<task>_<doc_id>`` when the doc has no ``name``). If that file
  already exists the API call is skipped and the path is returned as-is, so
  re-runs never re-bill completed clips. Externally generated videos can be
  pre-seeded into that layout to skip regeneration.

Supported t2v endpoints (pass via model= in model_args):
  Wan:        wan/v2.6/text-to-video
  LTX:        fal-ai/ltx-video
  Hunyuan:    fal-ai/hunyuan-video
  Hunyuan v1.5: fal-ai/hunyuan-video-v1.5/text-to-video

Supported v2v presets (pass via model= in model_args; empirical input
constraints):
  grok_extend      xai/grok-imagine-video/extend-video — true extend; 2-15s
                   input, <=921,600 px^2 (auto-downscaled); output = input ++
                   continuation
  gemini_edit      google/gemini-omni-flash/edit — edit-style; output is the
                   continuation only (same duration as input); flat $1/call
  seedance_ref     bytedance/seedance-2.0/reference-to-video — re-enacts the
                   reference from t=0 then continues; video param is an array
                   referenced as @Video1 in the prompt
  kling_ref        fal-ai/kling-video/o3/standard/video-to-video/reference —
                   re-enact + continue; REJECTS inputs < 3.0s (2s conditioning
                   fails; retiming workarounds distort dynamics,
                   so the backend fails fast instead of silently retiming)
  pixverse_extend  fal-ai/pixverse/v6/extend — extend; severe hallucination
                   observed, baseline only
Any other value is passed through as a raw fal.ai endpoint path; describe its
request shape via video_param / video_param_is_array / output_includes_input /
max_input_area / min_input_seconds model_args.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests as http_requests
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# ---------------------------------------------------------------------------
# Endpoint presets keyed by short alias -> fal.ai endpoint path
# ---------------------------------------------------------------------------
ENDPOINT_ALIASES: Dict[str, str] = {
    "wan_t2v": "wan/v2.6/text-to-video",
    "wan_i2v": "wan/v2.6/image-to-video",
    "ltx_t2v": "fal-ai/ltx-video",
    "hunyuan_t2v": "fal-ai/hunyuan-video",
    "hunyuan_t2v_v15": "fal-ai/hunyuan-video-v1.5/text-to-video",
    "hunyuan_i2v_v15": "fal-ai/hunyuan-video-v1.5/image-to-video",
}

# v2v presets: endpoint + payload shape + empirical input constraints
# (empirical field notes).
#   video_param            name of the endpoint's video input field
#   video_param_is_array   endpoint expects a list of URLs (seedance)
#   output_includes_input  output = conditioning ++ continuation, so the
#                          leading cond_seconds are trimmed off the response
#   max_input_area         max input pixel area; larger inputs are downscaled
#   min_input_seconds      endpoint rejects shorter conditioning clips
#   payload                endpoint-specific defaults (types matter: grok
#                          duration is an int, kling duration a string enum)
#   prompt                 fixed scene-agnostic prompt (per-scene descriptive
#                          prompts leak ground-truth information)
V2V_PRESETS: Dict[str, Dict[str, Any]] = {
    "grok_extend": {
        "endpoint": "xai/grok-imagine-video/extend-video",
        "video_param": "video_url",
        "output_includes_input": True,
        "max_input_area": 921_600,
        # Hard 2.0s minimum server-side; 59.94fps source clips land at 1.985s
        # and get clone-padded back over the line (see _prepare_input).
        "min_input_seconds": 2.0,
        "payload": {"duration": 2},
        "prompt": "Please continue to generate this video.",
    },
    "gemini_edit": {
        "endpoint": "google/gemini-omni-flash/edit",
        "video_param": "video_url",
        "output_includes_input": False,
        "payload": {},
        "prompt": "Please continue to generate this video.",
    },
    "seedance_ref": {
        "endpoint": "bytedance/seedance-2.0/reference-to-video",
        "video_param": "video_urls",
        "video_param_is_array": True,
        "output_includes_input": True,
        # Undocumented: rejects references above ~720p area with a generic
        # "Invalid input in 'video_urls'" 422 (measured 2026-07-04: every
        # 1280x720 input passed, every 1920x1080/2560x1440/3840x2160 failed).
        "max_input_area": 921_600,
        # Same hard 2.0s stream-duration minimum as grok (422
        # video_duration_too_short on a 1.9853s clip); enables the clone-pad.
        "min_input_seconds": 2.0,
        "payload": {"duration": 4, "generate_audio": False},
        "prompt": "Please continue to generate the video @Video1.",
    },
    "kling_ref": {
        "endpoint": "fal-ai/kling-video/o3/standard/video-to-video/reference",
        "video_param": "video_url",
        "output_includes_input": True,
        "min_input_seconds": 3.0,
        "payload": {"duration": "4"},
        "prompt": "Please continue to generate the video @Video1.",
    },
    "pixverse_extend": {
        "endpoint": "fal-ai/pixverse/v6/extend",
        "video_param": "video_url",
        "output_includes_input": True,
        "payload": {"duration": 2},
        "prompt": "Please continue to generate this video.",
    },
}

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")

_RESOLUTION_RE = re.compile(r"(\d{2,5})x(\d{2,5})[ ,]")
_DURATION_RE = re.compile(r"Duration: (\d+):(\d+):(\d+\.?\d*)")

# Inputs missing an endpoint's minimum duration by at most this much are
# clone-padded on the last frame; anything shorter fails fast (a whole missing
# second is a protocol mismatch, not encoder rounding).
_PAD_TOLERANCE_SECONDS = 0.1


class MissingApiKeyError(RuntimeError):
    """No FAL key configured — never heals on retry, so fail requests fast."""


def _resolve_endpoint(model: str) -> str:
    """Resolve a short alias or pass through a full fal.ai endpoint path."""
    return ENDPOINT_ALIASES.get(model, model)


def _ffmpeg_exe() -> str:
    """ffmpeg binary: PATH first, imageio-ffmpeg's bundled binary as fallback."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("ffmpeg is required for v2v input preparation and output trimming; install ffmpeg or `uv pip install imageio-ffmpeg`") from e


def _probe_video(path: str) -> Tuple[int, int, float]:
    """Return (width, height, duration_seconds) of the VIDEO STREAM.

    fal endpoints validate the stream duration, but the ``Duration:`` line in
    ``ffmpeg -i`` output is the container duration, which muxers round up
    (a 1.9853s stream prints as 2.00) — that rounding hid sub-minimum clips
    from the pad logic. Prefer ffprobe's per-stream numbers; fall back to the
    ``ffmpeg -i`` parse only when ffprobe is unavailable.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        parts = out.split(",")
        if len(parts) >= 3:
            try:
                return int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError:
                pass  # e.g. duration "N/A" — fall through to the ffmpeg parse
    out = subprocess.run([_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True).stderr
    m = _RESOLUTION_RE.search(out)
    if not m:
        raise RuntimeError(f"cannot parse resolution of {path}")
    w, h = int(m.group(1)), int(m.group(2))
    d = _DURATION_RE.search(out)
    duration = (int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))) if d else 0.0
    return w, h, duration


def _pick_video_url(node: Any) -> Optional[str]:
    """Recursively extract the first video URL from a fal.ai response dict."""
    if isinstance(node, dict):
        # Direct url field
        if "url" in node and isinstance(node["url"], str):
            u = node["url"]
            if u.startswith("http") and any(ext in u.lower() for ext in VIDEO_EXTENSIONS):
                return u
        # Priority keys
        for key in ("video", "videos", "output", "result", "data"):
            if key in node:
                found = _pick_video_url(node[key])
                if found:
                    return found
        # Fallback: scan all values
        for v in node.values():
            found = _pick_video_url(v)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _pick_video_url(item)
            if found:
                return found
    elif isinstance(node, str):
        if node.startswith("http") and any(ext in node.lower() for ext in VIDEO_EXTENSIONS):
            return node
    return None


@register_model("generation_api")
class GenerationApi(lmms):
    """fal.ai queue-based API generation backend."""

    is_simple = True

    def __init__(
        self,
        # ── infrastructure ──
        model: str = "wan/v2.6/text-to-video",
        mode: str = "t2v",
        output_dir: str = "./logs/generation_api_videos",
        num_concurrent: int = 4,
        batch_size: Union[int, str] = 1,
        poll_interval: int = 5,
        max_polls: int = 240,
        max_retries: int = 2,
        # ── v2v options (defaults come from the preset when model= matches) ──
        video_key: str = "conditioning_video",
        cond_seconds: float = 2.0,
        prompt: Optional[str] = None,
        video_param: Optional[str] = None,
        video_param_is_array: Optional[bool] = None,
        output_includes_input: Optional[bool] = None,
        max_input_area: Optional[int] = None,
        min_input_seconds: Optional[float] = None,
        # ── shared generation params (None = t2v defaults / absent in v2v) ──
        duration: Optional[Union[int, str]] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_frames: Optional[int] = None,
        enable_prompt_expansion: Optional[bool] = None,
        enable_safety_checker: Optional[bool] = None,
        # ── extra kwargs forwarded to fal.ai as-is ──
        **kwargs,
    ) -> None:
        super().__init__()

        # DP wiring for accelerate-launched runs: the evaluator reads lm.device,
        # lm.rank, lm.world_size for cross-rank sync (evaluator.py:950). Mirror
        # the sibling API backends (gemini_api / gpt4v).
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self.device = accelerator.device

        self.mode = str(mode).lower()
        if self.mode not in ("t2v", "v2v"):
            raise ValueError(f"generation_api mode must be 't2v' or 'v2v', got {mode!r}")

        preset = V2V_PRESETS.get(model, {}) if self.mode == "v2v" else {}
        self.endpoint = preset.get("endpoint") or _resolve_endpoint(model)
        self.output_dir = str(output_dir)
        self.num_concurrent = max(1, int(num_concurrent))
        self.batch_size_per_gpu = int(batch_size)
        self.poll_interval = max(1, int(poll_interval))
        self.max_polls = max(1, int(max_polls))
        self.max_retries = max(1, int(max_retries))

        # v2v request shape: explicit model_args override the preset.
        self.video_key = str(video_key)
        self.cond_seconds = float(cond_seconds)
        self.v2v_prompt = str(prompt) if prompt is not None else str(preset.get("prompt", ""))
        self.video_param = str(video_param) if video_param is not None else str(preset.get("video_param", "video_url"))
        self.video_param_is_array = bool(video_param_is_array if video_param_is_array is not None else preset.get("video_param_is_array", False))
        self.output_includes_input = bool(output_includes_input if output_includes_input is not None else preset.get("output_includes_input", False))
        self.max_input_area = int(max_input_area if max_input_area is not None else preset.get("max_input_area", 0))
        self.min_input_seconds = float(min_input_seconds if min_input_seconds is not None else preset.get("min_input_seconds", 0.0))
        self._v2v_payload_defaults: Dict[str, Any] = dict(preset.get("payload", {}))

        # API auth is checked lazily at the first real network call, so a run
        # where every request resumes from an existing output file completes
        # without a key (and without billing).
        self.api_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY", "")
        if not self.api_key:
            eval_logger.warning("generation_api: FAL_KEY/FAL_API_KEY not set — resume-only mode; any request that misses an existing output file will fail.")

        # Build the default generation payload template.
        # t2v keeps its historical defaults (applied when the arg is None).
        # v2v starts from the preset payload and folds in only the params the
        # caller explicitly set — unconditional t2v defaults (string duration,
        # resolution, aspect_ratio, ...) would 422 on extend/edit endpoints.
        explicit_gen_params: Dict[str, Any] = {
            k: v
            for k, v in {
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": num_frames,
                "enable_prompt_expansion": enable_prompt_expansion,
                "enable_safety_checker": enable_safety_checker,
            }.items()
            if v is not None
        }
        if negative_prompt:
            explicit_gen_params["negative_prompt"] = str(negative_prompt)

        self._gen_defaults: Dict[str, Any] = {}
        if self.mode == "t2v":
            self._gen_defaults["duration"] = str(explicit_gen_params.pop("duration", "5"))
            self._gen_defaults["resolution"] = str(explicit_gen_params.pop("resolution", "720p"))
            self._gen_defaults["aspect_ratio"] = str(explicit_gen_params.pop("aspect_ratio", "16:9"))
            self._gen_defaults["enable_prompt_expansion"] = bool(explicit_gen_params.pop("enable_prompt_expansion", True))
            self._gen_defaults["enable_safety_checker"] = bool(explicit_gen_params.pop("enable_safety_checker", True))
            for k, caster in (("seed", int), ("num_inference_steps", int), ("guidance_scale", float), ("num_frames", int)):
                if k in explicit_gen_params:
                    self._gen_defaults[k] = caster(explicit_gen_params.pop(k))
            self._gen_defaults.update(explicit_gen_params)
        else:
            # Explicit values keep their parsed types: grok wants int duration,
            # kling a string enum — the preset defaults model the difference.
            self._v2v_payload_defaults.update(explicit_gen_params)

        # Absorb any extra kwargs the user passes via model_args.
        # t2v: forwarded with the defaults. v2v: layered over the preset payload
        # so power users can pass endpoint-specific params directly.
        _known_infra_keys = {
            "model",
            "mode",
            "output_dir",
            "num_concurrent",
            "batch_size",
            "poll_interval",
            "max_polls",
            "max_retries",
            "device",
            "max_batch_size",
            "video_key",
            "cond_seconds",
            "prompt",
            "video_param",
            "video_param_is_array",
            "output_includes_input",
            "max_input_area",
            "min_input_seconds",
        }
        for k, v in kwargs.items():
            if k in _known_infra_keys:
                continue
            if self.mode == "v2v":
                self._v2v_payload_defaults[k] = v
            else:
                self._gen_defaults[k] = v

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._session = http_requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if self.api_key:
            self._session.headers.update({"Authorization": f"Key {self.api_key}"})
        self._upload_cache: Dict[str, str] = {}
        self._upload_lock = threading.Lock()
        self._request_log_lock = threading.Lock()

        eval_logger.info(f"GenerationApi initialized: mode={self.mode}, endpoint={self.endpoint}, output_dir={self.output_dir}, concurrency={self.num_concurrent}")

    # ------------------------------------------------------------------
    # fal.ai queue API helpers
    # ------------------------------------------------------------------

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise MissingApiKeyError("FAL_KEY or FAL_API_KEY environment variable is required for fal.ai calls (this request found no existing output file to resume from). Get an API key at https://fal.ai/dashboard/keys")

    def _upload_video(self, path: str) -> str:
        """Upload a local video to the fal CDN, memoized per source path."""
        with self._upload_lock:
            cached = self._upload_cache.get(path)
        if cached:
            return cached
        self._require_api_key()
        try:
            import fal_client
        except ImportError as e:
            raise ImportError("fal_client is required for v2v generation_api uploads: `uv pip install fal-client`") from e
        os.environ.setdefault("FAL_KEY", self.api_key)
        url = fal_client.upload_file(path)
        with self._upload_lock:
            self._upload_cache[path] = url
        return url

    def _submit_payload(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Submit a generation job to the fal.ai queue. Returns job metadata."""
        self._require_api_key()
        url = f"https://queue.fal.run/{self.endpoint}"

        resp = self._session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        request_id = data.get("request_id", "")
        if not request_id:
            raise ValueError(f"fal.ai submit response missing request_id: {data}")

        return {
            "request_id": request_id,
            "status_url": data.get(
                "status_url",
                f"{url}/requests/{request_id}/status",
            ),
            "response_url": data.get(
                "response_url",
                f"{url}/requests/{request_id}",
            ),
        }

    def _poll_until_done(self, status_url: str) -> None:
        """Block until job reaches COMPLETED. Raises on failure/timeout."""
        for i in range(1, self.max_polls + 1):
            resp = self._session.get(status_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            state = (data.get("status") or data.get("state") or "").upper()

            if state == "COMPLETED":
                return
            if state in ("FAILED", "CANCELED"):
                detail = json.dumps(data, indent=2)[:500]
                raise RuntimeError(f"fal.ai job {state}: {detail}")

            # Log progress occasionally
            if i % 10 == 0:
                eval_logger.debug(f"fal.ai poll {i}/{self.max_polls} state={state}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"fal.ai job did not complete within {self.max_polls * self.poll_interval}s")

    def _fetch_result(self, response_url: str) -> Tuple[str, Dict]:
        """Fetch completed result and return (video_url, raw_result_dict)."""
        resp = self._session.get(response_url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        video_url = _pick_video_url(data)
        if not video_url:
            raise ValueError(f"No video URL in fal.ai result: {json.dumps(data)[:500]}")

        return video_url, data

    def _download_video(self, video_url: str, output_path: str) -> str:
        """Download video from CDN URL to local file, atomically.

        Stream into a ``.part`` sidecar and ``os.replace`` into place only on
        clean completion, so an interrupted download never leaves a truncated
        file that ``_resume_hit`` would mistake for a finished cache entry.
        """
        resp = self._session.get(video_url, stream=True, timeout=300)
        resp.raise_for_status()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = f"{path}.part"
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        os.replace(part, path)
        return str(path)

    def _infer_extension(self, video_url: str) -> str:
        """Guess file extension from video URL."""
        parsed_path = urllib.parse.urlparse(video_url).path.lower()
        for ext in VIDEO_EXTENSIONS:
            if parsed_path.endswith(ext):
                return ext
        return ".mp4"

    def _append_request_log(self, record: Dict[str, Any]) -> None:
        """Append one line per completed API call to <output_dir>/requests.jsonl.

        Local observability only; the fal.ai dashboard is billing ground truth.
        """
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
        log_path = os.path.join(self.output_dir, "requests.jsonl")
        with self._request_log_lock:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Output naming + resume
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_stem(value: str) -> str:
        return str(value).replace("/", "_").replace(" ", "_")

    def _final_output_path(self, task: str, doc_id: Union[str, int], doc: Optional[dict], ext: str = ".mp4") -> str:
        """Canonical per-doc output path: output_dir/<task>/<doc name><ext>.

        Doc ``name`` keys the output layout, so name-keyed outputs stay stable
        across dataset re-indexing and are directly seedable. Docs without a
        name fall back to the ``<task>_<doc_id>`` stem.
        """
        safe_task = self._safe_stem(task)
        name = doc.get("name") if isinstance(doc, dict) else None
        stem = self._safe_stem(name) if name else f"{safe_task}_{doc_id}"
        return os.path.join(self.output_dir, safe_task, f"{stem}{ext}")

    @staticmethod
    def _resume_hit(path: str) -> bool:
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Input preparation (v2v)
    # ------------------------------------------------------------------

    def _prepare_input(self, src: str, task: str) -> Tuple[str, str]:
        """Validate + adapt a conditioning clip to the endpoint's limits.

        Two adaptations, both semantics-preserving:
        - downscale when the pixel area exceeds ``max_input_area`` (spatial
          only — retiming would invalidate dynamics comparisons);
        - clone-pad the last frame when the clip misses ``min_input_seconds``
          by at most ``_PAD_TOLERANCE_SECONDS`` (dataset clips encoded at
          59.94fps land at 1.985s and trip grok's hard 2.0s minimum). A real
          protocol mismatch (e.g. 2s conditioning into kling's 3s minimum)
          still fails fast.

        Returns (prepared_path, prep_note).
        """
        w, h, dur = _probe_video(src)
        short_by = (self.min_input_seconds - dur) if (self.min_input_seconds and dur) else 0.0
        if short_by > _PAD_TOLERANCE_SECONDS:
            raise ValueError(f"{self.endpoint} rejects inputs shorter than {self.min_input_seconds}s (got {dur:.2f}s from {src}); retiming workarounds distort dynamics, refusing")
        need_scale = bool(self.max_input_area) and w * h > self.max_input_area
        need_pad = short_by > 0
        if not need_scale and not need_pad:
            return src, f"{w}x{h}"

        vf_parts = []
        nw, nh = w, h
        note = f"{w}x{h}"
        if need_scale:
            scale = (self.max_input_area / (w * h)) ** 0.5
            nw, nh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
            vf_parts.append(f"scale={nw}:{nh}")
            note = f"{w}x{h}->{nw}x{nh}"
        if need_pad:
            # Pad slightly past the minimum so container rounding can't re-trip it.
            pad_seconds = short_by + 0.02
            vf_parts.append(f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}")
            note += f"+pad{pad_seconds:.3f}s"

        prepared_dir = os.path.join(self.output_dir, self._safe_stem(task), "prepared_inputs")
        os.makedirs(prepared_dir, exist_ok=True)
        suffix = f"_{nw}x{nh}" + ("_pad" if need_pad else "")
        dst = os.path.join(prepared_dir, f"{Path(src).stem}{suffix}.mp4")
        if not self._resume_hit(dst):
            subprocess.run(
                [_ffmpeg_exe(), "-v", "error", "-i", src, "-vf", ",".join(vf_parts), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", dst],
                check=True,
            )
        return dst, note

    def _trim_leading(self, full_path: str, out_path: str) -> None:
        """Strip the first cond_seconds so the response is the continuation only."""
        subprocess.run(
            [_ffmpeg_exe(), "-v", "error", "-ss", str(self.cond_seconds), "-i", full_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", out_path],
            check=True,
        )

    # ------------------------------------------------------------------
    # Single-request pipelines
    # ------------------------------------------------------------------

    def _generate_one(
        self,
        prompt: str,
        task: str,
        doc_id: Union[str, int],
    ) -> str:
        """t2v pipeline for one request: submit -> poll -> download. Returns local path."""
        # Resume: reruns must not re-bill docs that already have output.
        safe_task = self._safe_stem(task)
        for ext in VIDEO_EXTENSIONS:
            probe_path = os.path.join(self.output_dir, safe_task, f"{safe_task}_{doc_id}{ext}")
            if self._resume_hit(probe_path):
                eval_logger.debug(f"Cache hit: {probe_path}")
                return probe_path

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                job = self._submit_payload({**self._gen_defaults, "prompt": prompt})
                eval_logger.debug(f"Submitted fal.ai job: request_id={job['request_id']} task={task} doc_id={doc_id}")
                self._poll_until_done(job["status_url"])
                video_url, result_data = self._fetch_result(job["response_url"])

                # Build output path
                ext = self._infer_extension(video_url)
                filename = f"{safe_task}_{doc_id}{ext}"
                output_path = os.path.join(self.output_dir, safe_task, filename)
                self._download_video(video_url, output_path)

                # Persist metadata alongside video
                meta_path = output_path.rsplit(".", 1)[0] + ".meta.json"
                meta = {
                    "prompt": prompt,
                    "endpoint": self.endpoint,
                    "video_url": video_url,
                    "request_id": job["request_id"],
                    "seed": result_data.get("seed"),
                }
                Path(meta_path).write_text(json.dumps(meta, indent=2))
                self._append_request_log({"endpoint": self.endpoint, "request_id": job["request_id"], "task": task, "doc_id": doc_id, "output": output_path})

                eval_logger.info(f"Video saved: {output_path}")
                return output_path

            except MissingApiKeyError as exc:
                last_error = exc
                break  # a missing key never heals on retry; fail the doc fast
            except Exception as exc:
                last_error = exc
                eval_logger.warning(f"Attempt {attempt}/{self.max_retries} failed for task={task} doc_id={doc_id}: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(attempt * 5, 30))

        error_msg = f"[GENERATION_FAILED] {last_error}"
        eval_logger.error(f"All retries exhausted for task={task} doc_id={doc_id}: {last_error}")
        return error_msg

    def _generate_one_v2v(
        self,
        doc: dict,
        prompt: str,
        task: str,
        doc_id: Union[str, int],
    ) -> str:
        """v2v pipeline: resume-check -> upload cond video -> call -> trim. Returns local path."""
        final_path = self._final_output_path(task, doc_id, doc)
        if self._resume_hit(final_path):
            eval_logger.debug(f"Cache hit: {final_path}")
            return final_path

        cond_video = doc.get(self.video_key, "") if isinstance(doc, dict) else ""
        if not cond_video or not os.path.exists(cond_video):
            return f"[ERROR] conditioning video not found (doc key {self.video_key!r}): {cond_video!r}"

        # The task's doc_to_text is usually empty for v2v benchmarks; fall back
        # to the fixed scene-agnostic preset prompt so no GT detail leaks in.
        effective_prompt = prompt.strip() or self.v2v_prompt

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Key check precedes any input preparation so a missing key
                # surfaces as itself, not as a downstream ffmpeg/upload error.
                self._require_api_key()
                prepared, res_note = self._prepare_input(cond_video, task)
                video_url = self._upload_video(prepared)
                payload: Dict[str, Any] = dict(self._v2v_payload_defaults)
                if effective_prompt:
                    payload["prompt"] = effective_prompt
                payload[self.video_param] = [video_url] if self.video_param_is_array else video_url

                job = self._submit_payload(payload)
                eval_logger.debug(f"Submitted fal.ai job: request_id={job['request_id']} task={task} doc_id={doc_id}")
                self._poll_until_done(job["status_url"])
                out_url, result_data = self._fetch_result(job["response_url"])

                Path(final_path).parent.mkdir(parents=True, exist_ok=True)
                src_ext = self._infer_extension(out_url)
                if self.output_includes_input:
                    full_dir = os.path.join(os.path.dirname(final_path), "full")
                    os.makedirs(full_dir, exist_ok=True)
                    full_path = os.path.join(full_dir, Path(final_path).stem + src_ext)
                    self._download_video(out_url, full_path)
                    self._trim_leading(full_path, final_path)
                elif src_ext == ".mp4":
                    self._download_video(out_url, final_path)
                    full_path = ""
                else:
                    # Keep the ".mp4" naming contract honest: transcode foreign
                    # containers instead of mislabeling their bytes.
                    eval_logger.warning(f"{self.endpoint} returned {src_ext}; transcoding to mp4 for {final_path}")
                    tmp_path = final_path + f".src{src_ext}"
                    self._download_video(out_url, tmp_path)
                    subprocess.run([_ffmpeg_exe(), "-v", "error", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", final_path], check=True)
                    os.remove(tmp_path)
                    full_path = ""

                meta = {
                    "source": "fal.ai",
                    "endpoint": self.endpoint,
                    "request_id": job["request_id"],
                    "video_url": out_url,
                    "prompt": effective_prompt,
                    "payload_keys": sorted(payload.keys()),
                    "conditioning_video": cond_video,
                    "input_resolution": res_note,
                    "cond_seconds_trimmed": self.cond_seconds if self.output_includes_input else 0.0,
                    "full_output": full_path,
                    "seed": result_data.get("seed"),
                }
                Path(final_path + ".meta.json").write_text(json.dumps(meta, indent=2))
                self._append_request_log({"endpoint": self.endpoint, "request_id": job["request_id"], "task": task, "doc_id": doc_id, "output": final_path})

                eval_logger.info(f"V2V video saved: {final_path}")
                return final_path

            except MissingApiKeyError as exc:
                last_error = exc
                break  # a missing key never heals on retry; fail the doc fast
            except Exception as exc:
                last_error = exc
                eval_logger.warning(f"Attempt {attempt}/{self.max_retries} failed for task={task} doc_id={doc_id}: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(attempt * 5, 30))

        error_msg = f"[GENERATION_FAILED] {last_error}"
        eval_logger.error(f"All retries exhausted for task={task} doc_id={doc_id}: {last_error}")
        return error_msg

    # ------------------------------------------------------------------
    # lmms interface
    # ------------------------------------------------------------------

    def _resolve_doc(self, task: str, split: str, doc_id: Union[str, int]) -> Optional[dict]:
        try:
            return self.task_dict[task][split][doc_id]
        except Exception as e:
            eval_logger.warning(f"Failed to resolve doc task={task} split={split} doc_id={doc_id}: {e}")
            return None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate videos for all requests using concurrent fal.ai calls."""
        results: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Generating Videos",
        )

        # v2v outputs are keyed by doc name; two docs sharing a name would
        # silently alias to one file, so fail loudly before any API spend.
        if self.mode == "v2v":
            path_owner: Dict[str, Any] = {}
            for req in requests:
                _, _, _, doc_id, task, split = req.args
                doc = self._resolve_doc(task, split, doc_id)
                path = self._final_output_path(task, doc_id, doc)
                owner = path_owner.setdefault(path, doc_id)
                if owner != doc_id:
                    raise ValueError(f"generation_api: output path collision for {path} (doc_ids {owner} and {doc_id}); doc 'name' values must be unique within a task")

        def _process(index: int) -> Tuple[int, str]:
            req = requests[index]
            ctx, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            prompt = str(ctx).strip()
            if self.mode == "v2v":
                doc = self._resolve_doc(task, split, doc_id)
                if doc is None:
                    return index, "[ERROR] doc lookup failed"
                return index, self._generate_one_v2v(doc, prompt, task, doc_id)
            if not prompt:
                return index, "[ERROR] Empty prompt"
            return index, self._generate_one(prompt, task, doc_id)

        with ThreadPoolExecutor(max_workers=self.num_concurrent) as pool:
            futures = {pool.submit(_process, i): i for i in range(len(requests))}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                # Cache
                req = requests[idx]
                ctx = req.args[0]
                gen_kwargs = req.args[1]
                self.cache_hook.add_partial(
                    "generate_until",
                    (ctx, gen_kwargs),
                    result,
                )
                pbar.update(1)

        pbar.close()

        # Summary
        generated = sum(1 for r in results if r and not r.startswith("["))
        failed = len(results) - generated
        eval_logger.info(f"Video generation complete: {generated} succeeded, {failed} failed, output_dir={self.output_dir}")

        return [r if r is not None else "[ERROR] Unknown" for r in results]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("GenerationApi does not support loglikelihood")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("GenerationApi does not support multi-round generation")
