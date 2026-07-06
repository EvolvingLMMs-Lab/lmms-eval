"""MAGI-1 World Model backend for evaluating video generation quality.

Sand AI's MAGI-1 is an autoregressive video generation model that generates
video chunk-by-chunk with block-causal attention.  Supports T2V, I2V, and V2V.

For Physics-IQ evaluation the V2V mode is the primary path: a 3-second
conditioning video is extended into a 5-second continuation for motion-mask
comparison against ground truth.

Requirements
    - Clone the MAGI-1 repo (``magi_root`` model arg).
    - Download weights from HuggingFace ``sand-ai/MAGI-1``.
    - Provide a config JSON (from ``MAGI-1/example/``).
    - CUDA-enabled torch.  ffmpeg recommended (for fps resampling).
    - 4.5B variant: 1× GPU (≥24 GB VRAM).
      24B variant: 8× H100/H800 (model-parallel — NOT supported under the
      stock lmms-eval evaluator; see the usage note below).

Performance note
    Pass ``kv_offload=False`` (model arg) to keep the KV cache resident on the
    GPU.  Leaving offload on idles the GPU near 0% util as KV blocks ping-pong
    to host RAM every chunk; turning it off pins the GPU near 100%.  Only 24B
    model-parallel configs that are memory-bound should keep offload on.

Usage (4.5B, single GPU)::

    python -m lmms_eval \\
        --model magi1_wm \\
        --model_args "magi_root=/path/to/MAGI-1,config_file=/path/to/4.5B_base_config.json,output_dir=./logs/magi1,kv_offload=False" \\
        --tasks physics_iq_v2v \\
        --batch_size 1 --log_samples

Model parallelism (24B, ``cp_size * pp_size > 1``) is NOT supported under the
stock lmms-eval evaluator: it shards docs across all ranks by RANK/WORLD_SIZE,
which desyncs MAGI-1's model-parallel collectives (each rank would try to
generate a different doc). Use the single-GPU 4.5B config above, or pure
data-parallel (``cp_size=pp_size=1``, one full replica per rank via
``torchrun --nproc_per_node=N``). ``__init__`` raises if model parallel is
combined with a multi-rank world.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


def _find_free_port() -> int:
    """Find a free TCP port for distributed rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass(frozen=True)
class _ParallelLayout:
    global_rank: int
    local_rank: int
    world_size: int
    model_parallel_world_size: int
    data_parallel_replicas: int
    replica_rank: int


def _read_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _compute_parallel_layout(
    *,
    global_rank: int,
    local_rank: int,
    world_size: int,
    cp_size: int,
    pp_size: int,
) -> _ParallelLayout:
    model_parallel_world_size = max(1, int(cp_size) * int(pp_size))
    world_size = max(1, int(world_size))

    if world_size % model_parallel_world_size != 0:
        raise RuntimeError(f"MAGI-1 parallel layout is invalid: world_size={world_size} is not divisible by cp_size*pp_size={model_parallel_world_size}")

    data_parallel_replicas = world_size // model_parallel_world_size

    return _ParallelLayout(
        global_rank=int(global_rank),
        local_rank=int(local_rank),
        world_size=world_size,
        model_parallel_world_size=model_parallel_world_size,
        data_parallel_replicas=data_parallel_replicas,
        replica_rank=int(global_rank) // model_parallel_world_size,
    )


@contextmanager
def _mask_default_group_to(mp_group):
    """Temporarily reroute torch.distributed's default group to *mp_group*.

    Inside the context every ``dist.*()`` call without an explicit ``group=``
    argument operates against *mp_group*.  On exit the original default group
    is restored so that lmms-eval's evaluator collectives (``gather_object``,
    ``barrier``, ``broadcast_object_list``) run on the real outer world.

    Used to satisfy MAGI-1's ``world_size == cp_size*pp_size`` assertion
    during ``MagiPipeline(...)`` construction.  ``mp_group=None`` is a no-op
    so the single-GPU path is unchanged.
    """
    import torch.distributed.distributed_c10d as c10d

    if mp_group is None:
        yield
        return
    orig_pg = c10d._get_default_group()
    c10d._world.default_pg = mp_group
    try:
        yield
    finally:
        c10d._world.default_pg = orig_pg


@register_model("magi1_wm")
class Magi1WorldModel(lmms):
    """MAGI-1 world model backend for video continuation generation.

    Evaluates world-model quality by generating video continuations from
    conditioning video (V2V) or a single image (I2V) and returning video
    paths for downstream metric computation (e.g. Physics-IQ).
    """

    is_simple = True

    def __init__(
        self,
        # ── MAGI-1 paths ──
        magi_root: str = "",
        config_file: str = "",
        # ── Generation overrides (applied on top of the JSON config) ──
        generation_seconds: float = 5.0,
        num_frames: Optional[int] = None,
        seed: Optional[int] = None,
        num_steps: Optional[int] = None,
        kv_offload: Optional[bool] = None,
        # ── Output / eval ──
        eval_fps: int = 16,
        output_dir: str = "./logs/magi1_wm_videos",
        # ── Infrastructure ──
        batch_size: Union[int, str] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        import torch

        if not magi_root:
            raise ValueError("magi_root (path to MAGI-1 repo) is required")
        if not config_file:
            raise ValueError("config_file (path to MAGI-1 config JSON) is required")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"MAGI-1 config not found: {config_file}")

        self.magi_root = os.path.abspath(str(magi_root))
        self.config_file = os.path.abspath(str(config_file))
        self.generation_seconds = float(generation_seconds)
        self._num_frames_override = int(num_frames) if num_frames is not None else None
        self._seed_override = int(seed) if seed is not None else None
        self._num_steps_override = int(num_steps) if num_steps is not None else None
        # Accept "false"/"0"/"no" from the model_args string, not just bool.
        if isinstance(kv_offload, str):
            kv_offload = kv_offload.strip().lower() not in ("false", "0", "no", "off", "")
        self._kv_offload_override = bool(kv_offload) if kv_offload is not None else None
        self.eval_fps = int(eval_fps)
        self.output_dir = str(output_dir)
        self.batch_size_per_gpu = int(batch_size)
        self._global_rank = _read_env_int("RANK", 0)
        self._local_rank = _read_env_int("LOCAL_RANK", 0)
        self._world_size = _read_env_int("WORLD_SIZE", 1)
        self._rank = self._global_rank
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{self._local_rank}")
            torch.cuda.set_device(self._device)
        else:
            self._device = torch.device("cpu")

        # Lazy-loaded
        self._pipeline = None
        self._native_fps: int = 24
        self._config_data: Optional[dict] = None
        self._modified_config_path: Optional[str] = None
        self._load_lock = threading.Lock()
        self._mp_group = None  # set in _load_pipeline for DP>1
        self._parallel_layout: Optional[_ParallelLayout] = None  # computed in _load_pipeline

        # MAGI-1 model parallelism (cp_size * pp_size > 1) needs every MP rank to
        # co-run each generation, but the stock lmms-eval evaluator shards docs
        # across all WORLD_SIZE ranks by env RANK/WORLD_SIZE. Under model parallel
        # the two collide: each MP rank would be handed a different disjoint slice
        # of docs and desync on MAGI's collectives. Single-GPU 4.5B and pure
        # data-parallel (cp_size=pp_size=1, one full replica per rank) are fine.
        with open(self.config_file) as f:
            _engine = json.load(f).get("engine_config", {})
        _mp_size = max(1, int(_engine.get("cp_size", 1)) * int(_engine.get("pp_size", 1)))
        if self._world_size > 1 and _mp_size > 1:
            raise ValueError(
                f"Model-parallel MAGI-1 (cp_size*pp_size={_mp_size} with "
                f"WORLD_SIZE={self._world_size}) is not supported under the stock "
                "lmms-eval evaluator: it shards docs across all ranks by "
                "RANK/WORLD_SIZE, which desyncs the model-parallel collectives. "
                "Use the single-GPU 4.5B config, or pure data-parallel "
                "(cp_size=pp_size=1, one full replica per rank)."
            )

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        eval_logger.info(
            "MAGI-1 WM init: magi_root={}, config={}, eval_fps={}, output_dir={}, rank={}, local_rank={}, world_size={}, device={}",
            self.magi_root,
            self.config_file,
            eval_fps,
            self.output_dir,
            self._global_rank,
            self._local_rank,
            self._world_size,
            self._device,
        )

    @property
    def device(self):
        return self._device

    def _is_mp_leader(self) -> bool:
        """Return True on the first rank of this replica's MP subgroup.

        Post-process side-effects (shutil.move, os.remove, ffmpeg) are local
        filesystem operations that must run exactly once per generated video;
        MP peers would otherwise race on the same path and spurious-log
        ``V2V failed`` when the losing rank tries to move an already-moved
        file.
        """
        mp_size = self._parallel_layout.model_parallel_world_size
        return self._global_rank % mp_size == 0

    def _mp_barrier(self):
        """Barrier across this replica's MP subgroup (no-op if mp_size == 1).

        Uses ``self._mp_group`` when multiple DP replicas are carved out
        (``data_parallel_replicas > 1``); otherwise the outer world *is* the
        MP subgroup, so the default group barrier is correct.
        """
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        if self._parallel_layout.model_parallel_world_size == 1:
            return
        if self._mp_group is not None:
            dist.barrier(group=self._mp_group)
        else:
            dist.barrier()

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            self._load_pipeline()

    def _load_pipeline(self):
        import torch.distributed as dist

        # ── sys.path ──
        if not os.path.isdir(self.magi_root):
            raise RuntimeError(f"MAGI-1 repo not found: {self.magi_root}")
        if self.magi_root not in sys.path:
            sys.path.insert(0, self.magi_root)
            eval_logger.info(f"Added {self.magi_root} to sys.path")

        # ── Load & patch config ──
        with open(self.config_file) as f:
            cfg = json.load(f)
        self._config_data = cfg

        runtime = cfg.setdefault("runtime_config", {})
        engine = cfg.setdefault("engine_config", {})

        self._native_fps = runtime.get("fps", 24)

        # Compute num_frames for target duration
        if self._num_frames_override is not None:
            runtime["num_frames"] = self._num_frames_override
        elif self.generation_seconds > 0:
            runtime["num_frames"] = int(self.generation_seconds * self._native_fps)

        if self._seed_override is not None:
            runtime["seed"] = self._seed_override
        if self._num_steps_override is not None:
            runtime["num_steps"] = self._num_steps_override

        # kv_offload lives in engine_config. Leaving it on (the MAGI default for
        # some configs) idles the GPU at ~0% util because KV blocks ping-pong to
        # host RAM every chunk; kv_offload=False keeps them resident and pins the
        # GPU near 100%. Override only when explicitly set so memory-bound 24B
        # model-parallel configs can keep offload on if they need it.
        if self._kv_offload_override is not None:
            engine["kv_offload"] = self._kv_offload_override

        num_frames = runtime.get("num_frames", 96)
        num_steps = runtime.get("num_steps", 64)
        seed = runtime.get("seed", 42)

        # ── Distributed setup ──
        cp_size = engine.get("cp_size", 1)
        pp_size = engine.get("pp_size", 1)
        self._parallel_layout = _compute_parallel_layout(
            global_rank=self._global_rank,
            local_rank=self._local_rank,
            world_size=self._world_size,
            cp_size=cp_size,
            pp_size=pp_size,
        )
        world_needed = self._parallel_layout.model_parallel_world_size

        if not dist.is_initialized():
            if world_needed == 1:
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                os.environ.setdefault("LOCAL_RANK", "0")
            else:
                raise RuntimeError(f"MAGI-1 config needs {world_needed} GPUs (cp_size={cp_size}, pp_size={pp_size}) but torch.distributed is not initialised.  Launch with:\n  torchrun --nproc_per_node={world_needed} -m lmms_eval …")
        else:
            actual = dist.get_world_size()
            if actual < world_needed:
                raise RuntimeError(f"MAGI-1 needs {world_needed} GPUs but distributed world_size is {actual}")

        # Prevent double init_process_group (e.g. accelerate already called it)
        _orig_init_pg = dist.init_process_group

        def _safe_init_pg(*args, **kwargs):
            if dist.is_initialized():
                return
            return _orig_init_pg(*args, **kwargs)

        dist.init_process_group = _safe_init_pg

        # MAGI-1 recommended env vars
        os.environ.setdefault("PAD_HQ", "1")
        os.environ.setdefault("PAD_DURATION", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Write modified config to a stable path
        config_dir = os.path.join(self.output_dir, ".config")
        os.makedirs(config_dir, exist_ok=True)
        self._modified_config_path = os.path.join(config_dir, f"magi1_config_rank{self.rank}.json")
        with open(self._modified_config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        eval_logger.info(
            "Loading MAGI-1 pipeline: fps={}, num_frames={}, num_steps={}, seed={}, model_parallel_world_size={}, data_parallel_replicas={}, replica_rank={}",
            self._native_fps,
            num_frames,
            num_steps,
            seed,
            self._parallel_layout.model_parallel_world_size,
            self._parallel_layout.data_parallel_replicas,
            self._parallel_layout.replica_rank,
        )

        # ── DP subgroup routing ──
        # When the outer torchrun world spans multiple DP replicas, carve out
        # this replica's MP subgroup so MAGI sees world=mp_size inside init.
        if self._parallel_layout.data_parallel_replicas > 1:
            self._mp_group, _ = dist.new_subgroups(group_size=self._parallel_layout.model_parallel_world_size)
            eval_logger.info(
                "MAGI-1 DP layout: replica {}/{}, mp_size={}",
                self._parallel_layout.replica_rank,
                self._parallel_layout.data_parallel_replicas,
                self._parallel_layout.model_parallel_world_size,
            )

        # MAGI-1 uses relative paths (e.g. example/assets/special_tokens.npz)
        # so cwd must be the repo root during pipeline creation.
        original_cwd = os.getcwd()
        os.chdir(self.magi_root)
        try:
            from inference.pipeline.pipeline import MagiPipeline

            with _mask_default_group_to(self._mp_group):
                self._pipeline = MagiPipeline(self._modified_config_path)
        finally:
            os.chdir(original_cwd)
            dist.init_process_group = _orig_init_pg

        eval_logger.info("MAGI-1 pipeline loaded successfully")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_data_path(doc: dict, field: str) -> str:
        """Resolve a data path from *doc* using the generic media resolver."""
        from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

        return resolve_media_reference(
            doc.get(field, ""),
            media_type="video",
            env_vars=("PHYSICS_IQ_DATA_DIR",),
        )

    def _output_path_for(
        self,
        input_key: str,
        prompt: str,
        task: str,
        doc_id: Union[str, int],
    ) -> str:
        safe_task = str(task).replace("/", "_").replace(" ", "_")
        runtime = (self._config_data or {}).get("runtime_config", {})
        cache_hash = hashlib.sha256(f"{self.config_file}:{runtime.get('seed', 42)}:{runtime.get('num_frames', 96)}:{runtime.get('num_steps', 64)}:{self.eval_fps}:{input_key}:{prompt[:100]}".encode()).hexdigest()[:12]
        return os.path.join(
            self.output_dir,
            safe_task,
            f"{safe_task}_{doc_id}_{cache_hash}.mp4",
        )

    # ------------------------------------------------------------------
    # Video fps resampling
    # ------------------------------------------------------------------

    def _resample_video(self, src: str, dst: str, target_fps: int) -> str:
        """Resample *src* to *target_fps*, writing to *dst*.

        Tries ffmpeg first (fast, reliable), then falls back to PyAV.
        """
        # ── ffmpeg path ──
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    src,
                    "-filter:v",
                    f"fps={target_fps}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-loglevel",
                    "error",
                    dst,
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0 and os.path.exists(dst):
                return dst
            eval_logger.warning(f"ffmpeg resample failed (rc={result.returncode}): {result.stderr.decode()[:200]}")
        except FileNotFoundError:
            eval_logger.debug("ffmpeg not found; falling back to PyAV")
        except Exception as exc:
            eval_logger.warning(f"ffmpeg error: {exc}")

        # ── PyAV fallback ──
        return self._resample_video_av(src, dst, target_fps)

    @staticmethod
    def _resample_video_av(src: str, dst: str, target_fps: int) -> str:
        try:
            import av

            with av.open(src) as inp:
                stream = inp.streams.video[0]
                src_fps = float(stream.average_rate or 24)
                frames = list(inp.decode(video=0))

            if not frames:
                return src

            duration = len(frames) / src_fps
            target_n = max(1, int(duration * target_fps))
            indices = [min(int(i * len(frames) / target_n), len(frames) - 1) for i in range(target_n)]

            with av.open(dst, "w") as out:
                s = out.add_stream("libx264", rate=target_fps)
                s.width = frames[0].width
                s.height = frames[0].height
                s.pix_fmt = "yuv420p"
                for idx in indices:
                    f = frames[idx].reformat(format="yuv420p")
                    for pkt in s.encode(f):
                        out.mux(pkt)
                for pkt in s.encode():
                    out.mux(pkt)

            return dst
        except Exception as exc:
            eval_logger.warning(f"PyAV resample failed ({exc}); using raw video")
            return src

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _generate_v2v(
        self,
        video_path: str,
        prompt: str,
        task: str,
        doc_id: Union[str, int],
    ) -> str:
        """V2V: extend a conditioning video into a continuation."""
        self._ensure_loaded()

        output_path = self._output_path_for(video_path, prompt, task, doc_id)
        if os.path.exists(output_path):
            eval_logger.debug(f"Cache hit: {output_path}")
            return output_path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            raw_path = output_path.replace(".mp4", "_raw.mp4")
            with _mask_default_group_to(self._mp_group):
                self._pipeline.run_video_to_video(
                    prompt=prompt,
                    prefix_video_path=video_path,
                    output_path=raw_path,
                )

            # Sync MP peers so every rank sees the committed raw_path, then
            # let exactly one rank do the post-process (shutil.move / ffmpeg
            # resample / os.remove). Peers wait at the second barrier and
            # return by inspecting the deterministic output_path.
            self._mp_barrier()

            if self._is_mp_leader():
                if not os.path.exists(raw_path):
                    eval_logger.error(f"V2V raw missing: task={task} doc_id={doc_id}")
                else:
                    if self._native_fps != self.eval_fps:
                        self._resample_video(raw_path, output_path, self.eval_fps)
                        if os.path.exists(output_path) and os.path.exists(raw_path):
                            os.remove(raw_path)
                    else:
                        shutil.move(raw_path, output_path)
                    eval_logger.info(f"V2V generated: {output_path}")

            self._mp_barrier()

            if os.path.exists(output_path):
                return output_path
            return "[GENERATION_FAILED] MAGI-1 produced no output"

        except Exception as exc:
            eval_logger.error(f"V2V failed: task={task} doc_id={doc_id}: {exc}")
            return f"[GENERATION_FAILED] {exc}"

    def _generate_i2v(
        self,
        image_path: str,
        prompt: str,
        task: str,
        doc_id: Union[str, int],
    ) -> str:
        """I2V: generate video from a single conditioning image."""
        self._ensure_loaded()

        output_path = self._output_path_for(image_path, prompt, task, doc_id)
        if os.path.exists(output_path):
            eval_logger.debug(f"Cache hit: {output_path}")
            return output_path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            raw_path = output_path.replace(".mp4", "_raw.mp4")
            with _mask_default_group_to(self._mp_group):
                self._pipeline.run_image_to_video(
                    prompt=prompt,
                    image_path=image_path,
                    output_path=raw_path,
                )

            self._mp_barrier()

            if self._is_mp_leader():
                if not os.path.exists(raw_path):
                    eval_logger.error(f"I2V raw missing: task={task} doc_id={doc_id}")
                else:
                    if self._native_fps != self.eval_fps:
                        self._resample_video(raw_path, output_path, self.eval_fps)
                        if os.path.exists(output_path) and os.path.exists(raw_path):
                            os.remove(raw_path)
                    else:
                        shutil.move(raw_path, output_path)
                    eval_logger.info(f"I2V generated: {output_path}")

            self._mp_barrier()

            if os.path.exists(output_path):
                return output_path
            return "[GENERATION_FAILED] MAGI-1 produced no output"

        except Exception as exc:
            eval_logger.error(f"I2V failed: task={task} doc_id={doc_id}: {exc}")
            return f"[GENERATION_FAILED] {exc}"

    def _save_frames_as_video(
        self,
        frames: list,
        task: str,
        doc_id: Union[str, int],
    ) -> str:
        """Write PIL frames to a temporary MP4 (fallback when the raw path is unavailable)."""
        import av
        from PIL import Image

        safe_task = str(task).replace("/", "_")
        tmp_dir = os.path.join(self.output_dir, ".tmp_cond")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"cond_{safe_task}_{doc_id}.mp4")

        if os.path.exists(tmp_path):
            return tmp_path

        pil_frames = []
        for f in frames:
            if isinstance(f, Image.Image):
                pil_frames.append(f)
            elif isinstance(f, str) and os.path.exists(f):
                pil_frames.append(Image.open(f))

        if not pil_frames:
            return ""

        with av.open(tmp_path, "w") as container:
            stream = container.add_stream("libx264", rate=self.eval_fps)
            stream.width = pil_frames[0].width
            stream.height = pil_frames[0].height
            stream.pix_fmt = "yuv420p"
            for img in pil_frames:
                frame = av.VideoFrame.from_image(img)
                for pkt in stream.encode(frame):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)

        return tmp_path

    # ------------------------------------------------------------------
    # lmms interface
    # ------------------------------------------------------------------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="MAGI-1 Generation",
        )

        for i, req in enumerate(requests):
            ctx, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            prompt = str(ctx).strip()
            doc = self.task_dict[task][split][doc_id]

            # ── Detect mode from doc fields ──
            cond_video = self._resolve_data_path(doc, "conditioning_video")
            switch_frame = self._resolve_data_path(doc, "switch_frame")

            if cond_video and os.path.exists(cond_video):
                # V2V: conditioning video exists → video-to-video
                results[i] = self._generate_v2v(cond_video, prompt, task, doc_id)

            elif switch_frame and os.path.exists(switch_frame):
                # I2V: single conditioning image → image-to-video
                results[i] = self._generate_i2v(switch_frame, prompt, task, doc_id)

            else:
                # Fallback: try doc_to_visual → save frames as temp video
                visuals = []
                if doc_to_visual is not None:
                    try:
                        visuals = doc_to_visual(doc) or []
                    except Exception as exc:
                        eval_logger.warning(f"doc_to_visual failed: {exc}")

                if not visuals:
                    results[i] = "[ERROR] No conditioning input"
                    pbar.update(1)
                    continue

                try:
                    if len(visuals) == 1:
                        # Single frame → save as temp image, run I2V
                        from PIL import Image

                        img = visuals[0]
                        if isinstance(img, str):
                            img = Image.open(img)
                        tmp_dir = os.path.join(self.output_dir, ".tmp_cond")
                        os.makedirs(tmp_dir, exist_ok=True)
                        safe_task = str(task).replace("/", "_")
                        tmp_img = os.path.join(tmp_dir, f"img_{safe_task}_{doc_id}.jpg")
                        if not os.path.exists(tmp_img):
                            img.save(tmp_img)
                        results[i] = self._generate_i2v(tmp_img, prompt, task, doc_id)
                    else:
                        # Multiple frames → save as temp video, run V2V
                        tmp_video = self._save_frames_as_video(visuals, task, doc_id)
                        if tmp_video:
                            results[i] = self._generate_v2v(tmp_video, prompt, task, doc_id)
                        else:
                            results[i] = "[ERROR] Failed to save conditioning frames"
                except Exception as exc:
                    eval_logger.error(f"Fallback conditioning failed: task={task} doc_id={doc_id}: {exc}")
                    results[i] = f"[ERROR] Fallback conditioning failed: {exc}"

            pbar.update(1)

        pbar.close()

        generated = sum(1 for r in results if r and not r.startswith("["))
        failed = len(results) - generated
        eval_logger.info(f"MAGI-1 complete: {generated} succeeded, {failed} failed, output_dir={self.output_dir}")

        return [r if r is not None else "[ERROR] Unknown" for r in results]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("MAGI-1 WM does not support loglikelihood")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("MAGI-1 WM does not support multi-round generation")
