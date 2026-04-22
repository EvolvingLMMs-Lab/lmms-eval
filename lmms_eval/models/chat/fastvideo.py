"""FastVideo chat-model wrapper for video-generation benchmarks.

Uses `fastvideo.VideoGenerator` under the hood, pointed at the original
Wan2.2 weights by default. FastVideo's training-free acceleration knobs
(VSA sparse attention, STA sliding tile, TeaCache, torch.compile, FP8,
TP/SP parallelism) are exposed as constructor arguments so the same
checkpoint can be benchmarked across speedup configurations.

Output contract — matches the image-generation convention used by WISE and
the new VBVR task:

    GenerationResult.text = json.dumps({
        "text": "",
        "videos": ["/abs/path/to/generated.mp4"],
    })
"""

from __future__ import annotations

import json
import multiprocessing as _mp
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.protocol import ChatMessages

VideoGenerator, _has_fastvideo = optional_import("fastvideo", "VideoGenerator")

# Env vars that break fastvideo/accelerate when inherited from a k8s or
# torchrun launcher. The DP worker must start fresh — otherwise accelerate
# inside fastvideo hangs on rendezvous waiting for a non-existent rank 0.
_DIST_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
)

WORKERS = int(os.getenv("WORKERS", "8"))

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: str, default: str = "x") -> str:
    s = _SAFE_RE.sub("_", str(name)).strip("_") or default
    return s[:128]


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _fastvideo_dp_worker(
    rank: int,
    cuda_devices: str,
    task_q: "_mp.Queue",
    result_q: "_mp.Queue",
    config: Dict[str, Any],
) -> None:
    """Per-worker loop. Loads a FastVideo generator pinned to the given
    CUDA devices (comma-separated) and serves generate_video calls from the
    task queue until it gets a None sentinel. Runs in a spawn-mode subprocess
    so torch is imported fresh with CUDA_VISIBLE_DEVICES already scoped."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    for var in _DIST_ENV_VARS:
        os.environ.pop(var, None)
    # Quiet per-step tqdm + most per-sample INFO chatter so the main
    # process's DP progress bar stays readable.
    if config.get("quiet", True):
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("LOGURU_LEVEL", "WARNING")

    try:
        import torch as _torch
        from fastvideo import VideoGenerator as _VG

        dtype = _DTYPES.get(config["torch_dtype"], _torch.bfloat16)
        gen_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "num_gpus": config["num_gpus_per_worker"],
            "tp_size": config["tp_size_per_worker"],
            "sp_size": config["sp_size_per_worker"],
            "VSA_sparsity": config["VSA_sparsity"],
            "enable_torch_compile": config["enable_torch_compile"],
            "dit_cpu_offload": config["dit_cpu_offload"],
            "text_encoder_cpu_offload": config["text_encoder_cpu_offload"],
            "image_encoder_cpu_offload": config["image_encoder_cpu_offload"],
            "vae_cpu_offload": config["vae_cpu_offload"],
            "trust_remote_code": config["trust_remote_code"],
        }
        gen_kwargs.update(config.get("extra_kwargs", {}))
        generator = _VG.from_pretrained(config["model_path"], **gen_kwargs)
    except Exception as e:  # noqa: BLE001
        result_q.put(("init_error", rank, f"{type(e).__name__}: {str(e)[:500]}"))
        return

    result_q.put(("ready", rank, cuda_devices))

    while True:
        msg = task_q.get()
        if msg is None:
            break
        task_id, prompt, call_kwargs = msg
        try:
            result = generator.generate_video(prompt=prompt, **call_kwargs)
            result_q.put(("done", task_id, result))
        except Exception as e:  # noqa: BLE001
            result_q.put(("error", task_id, f"{type(e).__name__}: {str(e)[:500]}"))

    try:
        shutdown = getattr(generator, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


@register_model("fastvideo")
class FastVideo(lmms):
    """FastVideo-backed video generator for image+text → mp4 benchmarks."""

    is_simple = False

    def __init__(
        self,
        model: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        torch_dtype: str = "bfloat16",
        device: Optional[str] = None,
        # Sampling parameters
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        fps: int = 16,
        seed: int = 42,
        negative_prompt: Optional[str] = None,
        # Parallelism (FastVideo intra-sample: tp/sp split one sample across GPUs)
        num_gpus: int = 1,
        tp_size: int = -1,
        sp_size: int = -1,
        # Data parallelism (this wrapper: N independent generators, 1 sample each)
        data_parallel: int = 1,
        # FastVideo training-free acceleration knobs (work on original Wan2.2)
        VSA_sparsity: float = 0.0,
        enable_teacache: bool = False,
        enable_torch_compile: bool = False,
        # CPU offload
        dit_cpu_offload: bool = True,
        text_encoder_cpu_offload: bool = True,
        image_encoder_cpu_offload: bool = True,
        vae_cpu_offload: bool = True,
        # Misc
        trust_remote_code: bool = True,
        output_dir: str = "./fastvideo_generated_videos",
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        if not _has_fastvideo or VideoGenerator is None:
            raise ImportError("FastVideo is not installed. `pip install fastvideo` first.")

        self.model_path = model
        self.torch_dtype = _DTYPES.get(torch_dtype, torch.bfloat16)
        self.device = device

        # Sampling
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.seed = seed
        self.negative_prompt = negative_prompt

        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self._tmp_img_dir = tempfile.mkdtemp(prefix="fastvideo_inputs_")

        self.batch_size_per_gpu = int(batch_size)
        self._extra_kwargs = kwargs
        self.enable_teacache = enable_teacache

        self.data_parallel = int(data_parallel)
        self.generator = None
        self._workers: List[_mp.Process] = []
        self._task_q: Optional[_mp.Queue] = None
        self._result_q: Optional[_mp.Queue] = None

        if self.data_parallel > 1:
            self._init_dp_workers(
                torch_dtype=torch_dtype,
                num_gpus_per_worker=num_gpus,
                tp_size_per_worker=tp_size,
                sp_size_per_worker=sp_size,
                VSA_sparsity=VSA_sparsity,
                enable_torch_compile=enable_torch_compile,
                dit_cpu_offload=dit_cpu_offload,
                text_encoder_cpu_offload=text_encoder_cpu_offload,
                image_encoder_cpu_offload=image_encoder_cpu_offload,
                vae_cpu_offload=vae_cpu_offload,
                trust_remote_code=trust_remote_code,
                extra_kwargs=kwargs,
            )
        else:
            gen_kwargs: Dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "num_gpus": num_gpus,
                "tp_size": tp_size,
                "sp_size": sp_size,
                "VSA_sparsity": VSA_sparsity,
                "enable_torch_compile": enable_torch_compile,
                "dit_cpu_offload": dit_cpu_offload,
                "text_encoder_cpu_offload": text_encoder_cpu_offload,
                "image_encoder_cpu_offload": image_encoder_cpu_offload,
                "vae_cpu_offload": vae_cpu_offload,
                "trust_remote_code": trust_remote_code,
            }
            if device:
                gen_kwargs["device"] = device
            gen_kwargs.update(kwargs)

            eval_logger.info(f"Loading FastVideo VideoGenerator from {self.model_path}")
            self.generator = VideoGenerator.from_pretrained(self.model_path, **gen_kwargs)

        # Ranks/world are single-process by default (diffusion models typically
        # use FastVideo's own num_gpus for intra-sample parallelism instead of
        # accelerate-style data parallelism).
        self._rank = 0
        self._world_size = 1

    # ---------------------------------------------------------------- DP init
    def _init_dp_workers(
        self,
        torch_dtype: str,
        num_gpus_per_worker: int,
        tp_size_per_worker: int,
        sp_size_per_worker: int,
        VSA_sparsity: float,
        enable_torch_compile: bool,
        dit_cpu_offload: bool,
        text_encoder_cpu_offload: bool,
        image_encoder_cpu_offload: bool,
        vae_cpu_offload: bool,
        trust_remote_code: bool,
        extra_kwargs: Dict[str, Any],
    ) -> None:
        parent_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if parent_vis:
            visible = [d.strip() for d in parent_vis.split(",") if d.strip()]
        else:
            visible = [str(i) for i in range(torch.cuda.device_count())]

        gpus_per_worker = max(
            int(num_gpus_per_worker),
            int(tp_size_per_worker) if tp_size_per_worker > 0 else 1,
            int(sp_size_per_worker) if sp_size_per_worker > 0 else 1,
        )
        needed = self.data_parallel * gpus_per_worker
        if len(visible) < needed:
            raise ValueError(
                f"data_parallel={self.data_parallel} × {gpus_per_worker} GPUs/worker = {needed} GPUs needed, "
                f"but only {len(visible)} visible (CUDA_VISIBLE_DEVICES={parent_vis or 'unset'})."
            )
        # One contiguous chunk of physical GPU ids per worker.
        worker_gpus: List[str] = [
            ",".join(visible[r * gpus_per_worker : (r + 1) * gpus_per_worker])
            for r in range(self.data_parallel)
        ]
        eval_logger.info(
            f"FastVideo DP: spawning {self.data_parallel} workers "
            f"(num_gpus={num_gpus_per_worker}, tp={tp_size_per_worker}, sp={sp_size_per_worker}) "
            f"on GPU groups {worker_gpus}"
        )

        config: Dict[str, Any] = {
            "model_path": self.model_path,
            "torch_dtype": torch_dtype,
            "num_gpus_per_worker": int(num_gpus_per_worker),
            "tp_size_per_worker": int(tp_size_per_worker),
            "sp_size_per_worker": int(sp_size_per_worker),
            "VSA_sparsity": VSA_sparsity,
            "enable_torch_compile": enable_torch_compile,
            "dit_cpu_offload": dit_cpu_offload,
            "text_encoder_cpu_offload": text_encoder_cpu_offload,
            "image_encoder_cpu_offload": image_encoder_cpu_offload,
            "vae_cpu_offload": vae_cpu_offload,
            "trust_remote_code": trust_remote_code,
            "extra_kwargs": extra_kwargs,
            "quiet": os.environ.get("FASTVIDEO_DP_QUIET", "1") != "0",
        }

        ctx = _mp.get_context("spawn")
        self._task_q = ctx.Queue()
        self._result_q = ctx.Queue()
        # spawn's bootstrap imports the target's module (fastvideo) before our
        # worker code runs, and fastvideo's import path touches CUDA. To pin each
        # subprocess to the right GPU group we have to poke CUDA_VISIBLE_DEVICES
        # into the parent env around each .start() call.
        saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for rank, devs in enumerate(worker_gpus):
                os.environ["CUDA_VISIBLE_DEVICES"] = devs
                # Non-daemon: FastVideo's VideoGenerator spawns its own
                # MultiprocExecutor child inside each worker, and daemonic
                # processes cannot have children.
                p = ctx.Process(
                    target=_fastvideo_dp_worker,
                    args=(rank, devs, self._task_q, self._result_q, config),
                    daemon=False,
                )
                p.start()
                self._workers.append(p)
        finally:
            if saved_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

        pending = set(range(self.data_parallel))
        while pending:
            tag, worker_rank, payload = self._result_q.get()
            if tag == "ready":
                pending.discard(worker_rank)
                eval_logger.info(f"FastVideo DP worker {worker_rank} ready on GPUs {payload}")
            elif tag == "init_error":
                self._shutdown_workers()
                raise RuntimeError(f"FastVideo DP worker {worker_rank} failed to init: {payload}")
            else:
                eval_logger.warning(f"Unexpected DP init message: {(tag, worker_rank, payload)}")

    def _shutdown_workers(self) -> None:
        if self._task_q is not None:
            for _ in self._workers:
                try:
                    self._task_q.put(None)
                except Exception:
                    pass
        for p in self._workers:
            try:
                p.join(timeout=30)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        self._workers = []

    # ------------------------------------------------------------------ utils
    def _save_pil(self, img: Image.Image, tag: str) -> str:
        path = os.path.join(self._tmp_img_dir, f"{tag}.png")
        img.convert("RGB").save(path, format="PNG")
        return path

    def _extract_first_image_and_text(self, chat_messages: ChatMessages) -> Tuple[Optional[Image.Image], str]:
        images, _, _ = chat_messages.extract_media()
        first_image = images[0] if images else None
        texts: List[str] = []
        for msg in chat_messages.messages:
            if msg.role != "user":
                continue
            for content in msg.content:
                if content.type == "text":
                    texts.append(content.text)
        return first_image, "\n".join(t for t in texts if t).strip()

    def _try_parse_vbvr_layout(self, doc: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        path = doc.get("first_frame_path") or doc.get("final_frame_path") or doc.get("prompt_path")
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
        out_dir = os.path.join(self.output_dir, _safe(task))
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{_safe(str(doc_id))}.mp4")

    def _resolve_mp4(self, result: Any, expected_path: str) -> Optional[str]:
        """FastVideo occasionally renames the file (prompt-derived slug).
        Trust `expected_path` first; otherwise look at the returned dict."""
        if os.path.isfile(expected_path):
            return os.path.abspath(expected_path)
        # Scan the sibling directory for a recently written mp4
        parent = os.path.dirname(expected_path)
        candidates = []
        if os.path.isdir(parent):
            for f in os.listdir(parent):
                if f.endswith(".mp4"):
                    candidates.append(os.path.join(parent, f))
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            return os.path.abspath(candidates[0])
        if isinstance(result, dict):
            for k in ("video_path", "output_path", "path"):
                v = result.get(k)
                if isinstance(v, str) and v.endswith(".mp4") and os.path.isfile(v):
                    return os.path.abspath(v)
        return None

    # ----------------------------------------------------------- request prep
    def make_one_request(self, request: Instance) -> Dict[str, Any]:
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        doc = self.task_dict[task][split][doc_id]
        raw_messages = doc_to_messages(doc)
        chat_messages = ChatMessages(messages=raw_messages)
        first_image, prompt_text = self._extract_first_image_and_text(chat_messages)

        output_path = self._build_output_path(task, doc_id, doc)
        image_path = None
        if isinstance(first_image, Image.Image):
            tag = f"{_safe(task)}_{_safe(str(doc_id))}"
            image_path = self._save_pil(first_image, tag)
        elif isinstance(first_image, str) and os.path.isfile(first_image):
            image_path = first_image

        per_sample: Dict[str, Any] = {
            "prompt": prompt_text,
            "image_path": image_path,
            "output_path": output_path,
            "save_video": True,
            "num_inference_steps": int(gen_kwargs.get("num_inference_steps", self.num_inference_steps)),
            "guidance_scale": float(gen_kwargs.get("guidance_scale", self.guidance_scale)),
            "num_frames": int(gen_kwargs.get("num_frames", self.num_frames)),
            "height": int(gen_kwargs.get("height", self.height)),
            "width": int(gen_kwargs.get("width", self.width)),
            "fps": int(gen_kwargs.get("fps", self.fps)),
            "seed": int(gen_kwargs.get("seed", self.seed)),
            "enable_teacache": bool(gen_kwargs.get("enable_teacache", self.enable_teacache)),
        }
        if self.negative_prompt:
            per_sample["negative_prompt"] = self.negative_prompt
        return per_sample

    # ---------------------------------------------------------------- driver
    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        # Prepare all requests up front (image decode/save is CPU-bound).
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            prepared = list(executor.map(self.make_one_request, requests))

        if self.data_parallel > 1:
            return self._generate_until_parallel(prepared)
        return self._generate_until_single(prepared)

    def _empty_result(self) -> GenerationResult:
        return GenerationResult(text=json.dumps({"text": "", "videos": []}))

    def _pack_result(self, mp4_path: Optional[str]) -> GenerationResult:
        if mp4_path is None:
            return self._empty_result()
        return GenerationResult(text=json.dumps({"text": "", "videos": [mp4_path]}))

    def _generate_until_single(self, prepared: List[Dict[str, Any]]) -> List[GenerationResult]:
        res: List[GenerationResult] = []
        pbar = tqdm(total=len(prepared), disable=(self.rank != 0), desc="FastVideo generating")
        for prep in prepared:
            output_path = prep["output_path"]
            prompt = prep["prompt"]

            if not prompt:
                eval_logger.warning(f"FastVideo: empty prompt, skipping sample → {output_path}")
                res.append(self._empty_result())
                pbar.update(1)
                continue
            if not prep.get("image_path"):
                eval_logger.warning(f"FastVideo: no input image for → {output_path}")
                res.append(self._empty_result())
                pbar.update(1)
                continue

            call_kwargs = {k: v for k, v in prep.items() if k != "prompt"}
            try:
                result = self.generator.generate_video(prompt=prompt, **call_kwargs)
                mp4_path = self._resolve_mp4(result, output_path)
                if mp4_path is None:
                    eval_logger.error(f"FastVideo: no mp4 produced at {output_path}")
                res.append(self._pack_result(mp4_path))
            except Exception as e:
                eval_logger.error(f"FastVideo generation failed ({output_path}): {str(e)[:300]}")
                res.append(self._empty_result())
            pbar.update(1)

        pbar.close()
        return res

    def _generate_until_parallel(self, prepared: List[Dict[str, Any]]) -> List[GenerationResult]:
        """Fan out samples across self.data_parallel worker processes.
        Each worker owns one GPU and keeps the model hot between samples."""
        assert self._task_q is not None and self._result_q is not None

        results: List[Optional[GenerationResult]] = [None] * len(prepared)
        dispatched: Dict[int, Dict[str, Any]] = {}

        for task_id, prep in enumerate(prepared):
            output_path = prep["output_path"]
            prompt = prep["prompt"]
            if not prompt:
                eval_logger.warning(f"FastVideo DP: empty prompt, skipping → {output_path}")
                results[task_id] = self._empty_result()
                continue
            if not prep.get("image_path"):
                eval_logger.warning(f"FastVideo DP: no input image for → {output_path}")
                results[task_id] = self._empty_result()
                continue
            call_kwargs = {k: v for k, v in prep.items() if k != "prompt"}
            self._task_q.put((task_id, prompt, call_kwargs))
            dispatched[task_id] = prep

        pbar = tqdm(
            total=len(dispatched),
            desc=f"FastVideo DP×{self.data_parallel} generating",
            dynamic_ncols=True,
        )
        ok = err = 0
        remaining = len(dispatched)
        while remaining > 0:
            tag, task_id, payload = self._result_q.get()
            prep = dispatched.get(task_id)
            output_path = prep["output_path"] if prep else "?"
            short = os.path.basename(os.path.dirname(output_path)) + "/" + os.path.basename(output_path)
            if tag == "done":
                mp4_path = self._resolve_mp4(payload, output_path) if prep else None
                if mp4_path is None:
                    eval_logger.error(f"FastVideo DP: no mp4 produced at {output_path}")
                    err += 1
                else:
                    ok += 1
                results[task_id] = self._pack_result(mp4_path)
            elif tag == "error":
                eval_logger.error(f"FastVideo DP: sample {task_id} ({output_path}) failed: {payload}")
                results[task_id] = self._empty_result()
                err += 1
            else:
                eval_logger.warning(f"FastVideo DP: unexpected message {(tag, task_id, payload)}")
                results[task_id] = self._empty_result()
                err += 1
            pbar.set_postfix_str(f"last={short} ok={ok} err={err}", refresh=False)
            pbar.update(1)
            remaining -= 1
        pbar.close()

        return [r if r is not None else self._empty_result() for r in results]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("FastVideo is a generative video model; loglikelihood is not supported.")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not supported for FastVideo.")

    # lifecycle -----------------------------------------------------------
    def __del__(self):
        try:
            if self.data_parallel > 1 and getattr(self, "_workers", None):
                self._shutdown_workers()
            else:
                shutdown = getattr(self.generator, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        except Exception:
            pass
        try:
            if getattr(self, "_tmp_img_dir", None) and os.path.isdir(self._tmp_img_dir):
                shutil.rmtree(self._tmp_img_dir, ignore_errors=True)
        except Exception:
            pass
