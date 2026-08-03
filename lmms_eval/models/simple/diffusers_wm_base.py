"""Diffusers WM Base — shared skeleton for diffusers-backed generation models.

Consolidates per-rank device selection, lazy pipeline load, dual-expert
device-placement fix, scheduler device patch, cache-path hashing, and the
DP iteration loop from ``wan2_2``, ``wan2_2_t2v``, ``ltx_video``.

Subclass contract
-----------------
Required (set ``_pipeline_cls`` one of two ways):
    (a) Class attribute — ``_pipeline_cls = SomePipeline``.  Simple, but
        the ``from diffusers import ...`` must happen at module import time,
        which breaks in environments that don't have diffusers installed
        but still want to import the ``lmms_eval`` package.
    (b) ``_patch_pipeline_cls_before_load`` hook — assign
        ``type(self)._pipeline_cls`` inside the hook.  Preserves the
        original "no diffusers at module-import time" contract.

Required (methods):
    ``_generation_signature(prompt, visuals, extras) -> str``
    ``_invoke_pipeline(prompt, visuals, generator, **extras)`` -> output object

Optional overrides:
    ``_patch_pipeline_cls_before_load()``   class-level monkeypatch hook.
    ``_post_to_device(pipe, device)``       extra per-pipeline device fixups.
    ``_extract_visuals(req)``               I2V / V2V conditioning extraction.
    ``_export(output, out_path)``           swap MP4 export for images / other.

Parallelism
-----------
Currently DP-only; ``DiffusersWMBase`` enforces ``tp_size == 1`` and will be
extended in a follow-up PR to mirror ``vllm.py``'s
``world_size == tp_size * dp_size`` validation.
"""

from __future__ import annotations

import functools
import hashlib
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Tuple, Union

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms


@dataclass(frozen=True)
class ParallelPlan:
    """Launcher-level topology read from torch.distributed env vars.

    DP-only helper: reads ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` and pins
    each rank to its own CUDA device.
    """

    global_rank: int
    local_rank: int
    world_size: int
    tp_size: int = 1

    @classmethod
    def from_env(cls, tp_size: int = 1) -> "ParallelPlan":
        return cls(
            global_rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            tp_size=int(tp_size),
        )

    def device_str(self) -> str:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{self.local_rank}"


_DTYPE_ALIASES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def _resolve_torch_dtype(name: str):
    import torch

    key = _DTYPE_ALIASES.get((name or "").lower(), "bfloat16")
    return getattr(torch, key)


def _move_pipeline_to_device(pipe, device: str) -> None:
    """Pipeline-wide ``.to(device)`` with per-component fallback walk.

    Works around diffusers main's ``Pipeline.to()`` skipping dual-expert
    components like ``transformer_2`` (observed on Wan2.2-A14B), which would
    otherwise trigger torch.cat device-mismatch at the first denoising step.
    """
    pipe.to(device)
    components = getattr(pipe, "components", None) or {}
    for name, mod in components.items():
        if mod is None or not hasattr(mod, "to") or not callable(mod.to):
            continue
        try:
            mod.to(device)
        except Exception as exc:
            eval_logger.debug(f"skip .to on component {name}: {exc}")


def _patch_scheduler_keep_device(pipe, device: str) -> None:
    """Keep scheduler buffers on ``device`` after every ``set_timesteps`` call.

    ``UniPCMultistepScheduler`` pins ``sigmas`` to CPU on purpose (see
    scheduling_unipc_multistep.py: "to avoid too much CPU/GPU communication"),
    but that triggers torch.cat mismatch inside ``multistep_uni_p_bh_update``
    when concatenated with cuda model outputs. This wrapper migrates the
    known buffers back to ``device`` every time ``set_timesteps`` is called.

    ``functools.wraps`` is load-bearing: LTXPipeline's ``retrieve_timesteps``
    introspects ``scheduler.set_timesteps`` with ``inspect.signature(...)
    .parameters`` to decide whether custom ``sigmas`` are accepted. A naked
    ``def wrapper(*args, **kwargs)`` hides the real signature — Python then
    reports parameters = {"args", "kwargs"}, the ``"sigmas" in params`` check
    fails, and LTX aborts with ``ValueError: ... does not support custom
    sigmas schedules``. ``functools.wraps`` copies the original ``__wrapped__``
    so ``inspect.signature(..., follow_wrapped=True)`` (the default) sees the
    real parameters and the pipeline proceeds normally.
    """
    import torch

    sched = getattr(pipe, "scheduler", None)
    if sched is None or not hasattr(sched, "set_timesteps"):
        return
    if getattr(sched, "_lmms_eval_device_patched", False):
        return

    orig = sched.set_timesteps
    target = torch.device(device)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        result = orig(*args, **kwargs)
        for attr in ("sigmas", "timesteps", "alpha_t", "sigma_t", "lambda_t"):
            buf = getattr(sched, attr, None)
            if torch.is_tensor(buf) and buf.device != target:
                setattr(sched, attr, buf.to(target))
        return result

    sched.set_timesteps = wrapper
    sched._lmms_eval_device_patched = True


def _cache_path(output_dir: str, task: str, doc_id, signature: str, ext: str = "mp4") -> Path:
    safe = str(task).replace("/", "_").replace(" ", "_")
    h = hashlib.sha256(signature.encode()).hexdigest()[:12]
    return Path(output_dir) / safe / f"{safe}_{doc_id}_{h}.{ext}"


class DiffusersWMBase(lmms):
    """DP-only base class for diffusers-backed video / image generation models."""

    is_simple: ClassVar[bool] = True
    _pipeline_cls: ClassVar[Optional[type]] = None
    _output_ext: ClassVar[str] = "mp4"

    def __init__(
        self,
        *,
        pretrained: str,
        output_dir: str,
        seed: int = 42,
        dtype: str = "bfloat16",
        fps: int = 16,
        batch_size: Union[int, str] = 1,
        tp_size: int = 1,
        **_unused: Any,
    ) -> None:
        super().__init__()
        # _pipeline_cls is checked in _load_pipeline (after the
        # _patch_pipeline_cls_before_load hook), so subclasses may assign it
        # lazily to keep diffusers out of the module-import path.
        self.plan = ParallelPlan.from_env(tp_size=tp_size)
        if self.plan.tp_size != 1:
            raise NotImplementedError(f"{type(self).__name__}: tp_size={self.plan.tp_size} not yet supported. " "DiffusersWMBase is DP-only; TP support is planned in a follow-up PR.")

        self.pretrained = str(pretrained)
        self.output_dir = str(output_dir)
        self.seed = int(seed)
        self.dtype_name = str(dtype)
        self.fps = int(fps)
        self.batch_size_per_gpu = int(batch_size)

        self._rank = self.plan.global_rank
        self._world_size = self.plan.world_size

        self._pipe = None
        self._load_lock = threading.Lock()

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        eval_logger.info(
            f"{type(self).__name__} init: pretrained={self.pretrained}, "
            f"rank={self.plan.global_rank}/{self.plan.world_size}, "
            f"device={self.plan.device_str()}, dtype={self.dtype_name}, "
            f"seed={self.seed}, output_dir={self.output_dir}"
        )

    @property
    def device(self) -> str:
        """Per-rank device string (e.g. ``"cuda:0"``).

        ``lmms_eval.evaluator.evaluate`` reads ``lm.device`` as a bare
        attribute to place cross-rank sync tensors; the base's
        ``self.plan.device_str()`` is the canonical source, so this
        property just forwards. Lazy — doesn't require the pipeline to
        be loaded.
        """
        return self.plan.device_str()

    # ── Pipeline load + device placement ────────────────────────

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        with self._load_lock:
            if self._pipe is not None:
                return
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        self._patch_pipeline_cls_before_load()
        if self._pipeline_cls is None:
            raise TypeError(f"{type(self).__name__} must set _pipeline_cls before _load_pipeline runs " "(either as a class attribute or inside _patch_pipeline_cls_before_load)")
        device = self.plan.device_str()
        dtype = _resolve_torch_dtype(self.dtype_name)
        eval_logger.info(f"Loading {self._pipeline_cls.__name__} from {self.pretrained} on {device}")
        self._pipe = self._pipeline_cls.from_pretrained(self.pretrained, torch_dtype=dtype)
        _move_pipeline_to_device(self._pipe, device)
        _patch_scheduler_keep_device(self._pipe, device)
        self._post_to_device(self._pipe, device)
        self._log_component_devices()

    def _log_component_devices(self) -> None:
        comps = getattr(self._pipe, "components", None) or {}
        dev_map = {name: str(getattr(mod, "device", "?")) for name, mod in comps.items() if mod is not None and hasattr(mod, "device")}
        eval_logger.info(f"{type(self).__name__} loaded; components={dev_map}")

    # ── Subclass hooks ──────────────────────────────────────────

    def _patch_pipeline_cls_before_load(self) -> None:
        """Apply class-level monkeypatches before ``from_pretrained``. No-op by default."""
        return

    def _post_to_device(self, pipe, device: str) -> None:
        """Extra per-pipeline device fixups. No-op by default."""
        return

    def _extract_visuals(self, req: Instance) -> List:
        """Return conditioning visuals for one request. Empty for T2V / T2I by default.

        I2V subclasses override to pull visuals from ``doc_to_visual``.
        """
        return []

    def _generation_signature(self, prompt: str, visuals: List, extras: dict) -> str:
        """Return a deterministic string used for the cache key."""
        raise NotImplementedError(f"{type(self).__name__} must implement _generation_signature")

    def _invoke_pipeline(self, prompt: str, visuals: List, generator, **extras):
        """Call the diffusers pipeline; return the raw output object."""
        raise NotImplementedError(f"{type(self).__name__} must implement _invoke_pipeline")

    def _export(self, output_obj, out_path: Path) -> None:
        """Default: export first video frames to MP4. Override for image outputs."""
        from diffusers.utils import export_to_video

        export_to_video(output_obj.frames[0], str(out_path), fps=self.fps)

    # ── Core generation loop ────────────────────────────────────

    def _generate_one(
        self,
        prompt: str,
        visuals: List,
        doc_id,
        task: str,
        extras: Optional[dict] = None,
    ) -> str:
        import torch

        extras = extras or {}
        self._ensure_loaded()

        sig = self._generation_signature(prompt, visuals, extras)
        out_path = _cache_path(self.output_dir, task, doc_id, sig, ext=self._output_ext)
        if out_path.exists():
            eval_logger.debug(f"Cache hit: {out_path}")
            return str(out_path)

        try:
            generator = torch.Generator(device=self.plan.device_str()).manual_seed(self.seed)
            output = self._invoke_pipeline(prompt, visuals, generator, **extras)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._export(output, out_path)
            if out_path.exists():
                eval_logger.info(f"Generated: {out_path} ({out_path.stat().st_size} bytes)")
            else:
                eval_logger.error(f"Output NOT on disk after export: {out_path}")
            return str(out_path)
        except Exception as exc:
            eval_logger.error(f"Generation failed: task={task} doc_id={doc_id}: {exc}\n" f"{traceback.format_exc()}")
            return f"[GENERATION_FAILED] {exc}"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc=type(self).__name__)
        for i, req in enumerate(requests):
            ctx, gen_kwargs, _doc_to_visual, doc_id, task, _split = req.args
            prompt = str(ctx).strip()
            if not prompt:
                results[i] = "[ERROR] Empty prompt"
                pbar.update(1)
                continue
            visuals = self._extract_visuals(req)
            results[i] = self._generate_one(prompt, visuals, doc_id, task, extras=gen_kwargs or {})
            pbar.update(1)
        pbar.close()
        ok = sum(1 for r in results if r and not r.startswith("["))
        failed = len(results) - ok
        eval_logger.info(f"{type(self).__name__} complete: {ok} succeeded, {failed} failed")
        return [r if r is not None else "[ERROR] Unknown" for r in results]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(f"{type(self).__name__} does not support loglikelihood")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError(f"{type(self).__name__} does not support generate_until_multi_round")
