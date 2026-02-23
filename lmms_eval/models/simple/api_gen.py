"""Video generation model backend using fal.ai API.

Supports Wan, LTX, and HunyuanVideo model families via fal.ai queue-based
inference. Each request submits a text prompt, polls for completion, downloads
the generated video, and returns the local file path as the generation result.

Usage:
  python -m lmms_eval \
    --model api_gen \
    --model_args "model=wan/v2.6/text-to-video,output_dir=./output_videos" \
    --tasks videogen_test \
    --batch_size 1 \
    --limit 4 \
    --log_samples

Environment:
  FAL_KEY or FAL_API_KEY must be set.

Supported fal.ai endpoints (pass via model= in model_args):
  Wan:        wan/v2.6/text-to-video
  LTX:        fal-ai/ltx-video
  Hunyuan:    fal-ai/hunyuan-video
  Hunyuan v1.5: fal-ai/hunyuan-video-v1.5/text-to-video
"""

import json
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests as http_requests
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

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")


def _resolve_endpoint(model: str) -> str:
    """Resolve a short alias or pass through a full fal.ai endpoint path."""
    return ENDPOINT_ALIASES.get(model, model)


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


@register_model("api_gen")
class ApiGen(lmms):
    """fal.ai queue-based API generation backend."""

    is_simple = True

    def __init__(
        self,
        # ── infrastructure ──
        model: str = "wan/v2.6/text-to-video",
        output_dir: str = "./logs/api_gen_videos",
        num_concurrent: int = 4,
        batch_size: Union[int, str] = 1,
        poll_interval: int = 5,
        max_polls: int = 240,
        max_retries: int = 2,
        # ── common generation params (forwarded to fal.ai) ──
        duration: str = "5",
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        negative_prompt: str = "",
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_frames: Optional[int] = None,
        enable_prompt_expansion: bool = True,
        enable_safety_checker: bool = True,
        # ── extra kwargs forwarded to fal.ai as-is ──
        **kwargs,
    ) -> None:
        super().__init__()
        self.endpoint = _resolve_endpoint(model)
        self.output_dir = str(output_dir)
        self.num_concurrent = max(1, int(num_concurrent))
        self.batch_size_per_gpu = int(batch_size)
        self.poll_interval = max(1, int(poll_interval))
        self.max_polls = max(1, int(max_polls))
        self.max_retries = max(1, int(max_retries))

        # API auth
        self.api_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("FAL_KEY or FAL_API_KEY environment variable is required for api_gen model. " "Get an API key at https://fal.ai/dashboard/keys")

        # Build the default generation payload template.
        # Only include params that are not None so each model endpoint
        # receives only the params it understands.
        self._gen_defaults: Dict[str, Any] = {}
        if duration is not None:
            self._gen_defaults["duration"] = str(duration)
        if resolution is not None:
            self._gen_defaults["resolution"] = str(resolution)
        if aspect_ratio is not None:
            self._gen_defaults["aspect_ratio"] = str(aspect_ratio)
        if negative_prompt:
            self._gen_defaults["negative_prompt"] = str(negative_prompt)
        if seed is not None:
            self._gen_defaults["seed"] = int(seed)
        if num_inference_steps is not None:
            self._gen_defaults["num_inference_steps"] = int(num_inference_steps)
        if guidance_scale is not None:
            self._gen_defaults["guidance_scale"] = float(guidance_scale)
        if num_frames is not None:
            self._gen_defaults["num_frames"] = int(num_frames)
        self._gen_defaults["enable_prompt_expansion"] = bool(enable_prompt_expansion)
        self._gen_defaults["enable_safety_checker"] = bool(enable_safety_checker)

        # Absorb any extra kwargs the user passes via model_args.
        # This lets power users pass model-specific fal.ai params directly.
        _known_infra_keys = {
            "model",
            "output_dir",
            "num_concurrent",
            "batch_size",
            "poll_interval",
            "max_polls",
            "max_retries",
            "device",
            "max_batch_size",
        }
        for k, v in kwargs.items():
            if k not in _known_infra_keys:
                self._gen_defaults[k] = v

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._session = http_requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        eval_logger.info(f"ApiGen initialized: endpoint={self.endpoint}, " f"output_dir={self.output_dir}, concurrency={self.num_concurrent}")

    # ------------------------------------------------------------------
    # fal.ai queue API helpers
    # ------------------------------------------------------------------

    def _submit(self, prompt: str) -> Dict[str, str]:
        """Submit a generation job to the fal.ai queue. Returns job metadata."""
        url = f"https://queue.fal.run/{self.endpoint}"
        payload = {**self._gen_defaults, "prompt": prompt}

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
        """Download video from CDN URL to local file."""
        resp = self._session.get(video_url, stream=True, timeout=300)
        resp.raise_for_status()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return str(path)

    def _infer_extension(self, video_url: str) -> str:
        """Guess file extension from video URL."""
        parsed_path = urllib.parse.urlparse(video_url).path.lower()
        for ext in VIDEO_EXTENSIONS:
            if parsed_path.endswith(ext):
                return ext
        return ".mp4"

    # ------------------------------------------------------------------
    # Single-request pipeline
    # ------------------------------------------------------------------

    def _generate_one(
        self,
        prompt: str,
        task: str,
        doc_id: Union[str, int],
    ) -> str:
        """Full pipeline for one request: submit -> poll -> download. Returns local path."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                job = self._submit(prompt)
                eval_logger.debug(f"Submitted fal.ai job: request_id={job['request_id']} " f"task={task} doc_id={doc_id}")
                self._poll_until_done(job["status_url"])
                video_url, result_data = self._fetch_result(job["response_url"])

                # Build output path
                safe_task = str(task).replace("/", "_").replace(" ", "_")
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

                eval_logger.info(f"Video saved: {output_path}")
                return output_path

            except Exception as exc:
                last_error = exc
                eval_logger.warning(f"Attempt {attempt}/{self.max_retries} failed for " f"task={task} doc_id={doc_id}: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(attempt * 5, 30))

        error_msg = f"[GENERATION_FAILED] {last_error}"
        eval_logger.error(f"All retries exhausted for task={task} doc_id={doc_id}: {last_error}")
        return error_msg

    # ------------------------------------------------------------------
    # lmms interface
    # ------------------------------------------------------------------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate videos for all requests using concurrent fal.ai calls."""
        results: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Generating Videos",
        )

        def _process(index: int) -> Tuple[int, str]:
            req = requests[index]
            ctx, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            prompt = str(ctx).strip()
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
        eval_logger.info(f"Video generation complete: {generated} succeeded, {failed} failed, " f"output_dir={self.output_dir}")

        return [r if r is not None else "[ERROR] Unknown" for r in results]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("ApiGen does not support loglikelihood")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("ApiGen does not support multi-round generation")
