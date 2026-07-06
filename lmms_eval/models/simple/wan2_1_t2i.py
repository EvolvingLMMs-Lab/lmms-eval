"""Wan2.1-1.3B Text-to-Image backend for GenEval v1/v2 and DPG-Bench.

Wraps ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers`` via ``diffusers.WanPipeline`` with
``num_frames=1`` and writes the single decoded frame as a PNG. The output
layout matches the directory convention that the T2I task ``process_results``
hooks already glob:

    {output_dir}/{task}/{doc_id:04d}_s{seed_idx}.png

Single-DiT variant (no ``transformer_2``); ``DiffusersWMBase``'s component
walk handles the missing dual-expert gracefully.

Usage::

    python -m lmms_eval \\
      --model wan2_1_t2i \\
      --model_args "pretrained=Wan-AI/Wan2.1-T2V-1.3B-Diffusers,output_dir=./logs/wan2_1_t2i" \\
      --tasks geneval_v2 --batch_size 1 --log_samples
"""

from __future__ import annotations

import hashlib
import traceback
from pathlib import Path
from typing import List, Union

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase


@register_model("wan2_1_t2i")
class Wan2_1_T2I(DiffusersWMBase):
    """Wan2.1-1.3B T2I backend (T2V pipeline pinned to ``num_frames=1``)."""

    _output_ext = "png"

    def _patch_pipeline_cls_before_load(self) -> None:
        if type(self)._pipeline_cls is None:
            try:
                from diffusers import WanPipeline
            except ImportError as exc:
                raise ImportError("wan2_1_t2i requires diffusers: `pip install diffusers imageio imageio-ffmpeg`") from exc

            type(self)._pipeline_cls = WanPipeline

    def __init__(
        self,
        pretrained: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        num_frames: int = 1,
        seed: int = 42,
        dtype: str = "bfloat16",
        output_dir: str = "./logs/wan2_1_t2i",
        batch_size: Union[int, str] = 1,
        attn_backend: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            output_dir=output_dir,
            seed=seed,
            dtype=dtype,
            fps=1,
            batch_size=batch_size,
            **kwargs,
        )
        self.height = int(height)
        self.width = int(width)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.num_images_per_prompt = int(num_images_per_prompt)
        self.num_frames = int(num_frames)
        self.attn_backend = str(attn_backend).strip()

    def _invoke_pipeline(self, prompt, visuals, generator, **extras):
        def _run():
            return self._pipe(
                prompt=prompt,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_videos_per_prompt=1,
                generator=generator,
                # Force PIL frames so _save_first_frame can index as
                # frames[batch][frame_idx]. WanPipeline defaults to "np"
                # (returns ndarray, breaks truthy checks).
                output_type="pil",
            )

        if self.attn_backend:
            try:
                from diffusers.models.attention_dispatch import (
                    attention_backend,
                )

                with attention_backend(self.attn_backend):
                    return _run()
            except Exception as exc:
                eval_logger.warning(f"attn_backend='{self.attn_backend}' failed ({exc}); falling back to default")
        return _run()

    def _generation_signature(self, prompt, visuals, extras) -> str:
        # Unused (we override generate_until and bypass _cache_path), but the
        # base declares it abstract — keep a sane impl for completeness.
        return f"{self.pretrained}:{self.seed}:{self.num_inference_steps}:{self.guidance_scale}:{self.height}x{self.width}:{prompt[:200]}"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Per-doc loop that writes the per-seed T2I directory layout.

        T2I tasks (geneval_v1, geneval_v2, dpg_bench) read ``results[0]`` as a
        directory and glob ``{doc_id:04d}_s{seed}.png`` within it, so we
        bypass ``DiffusersWMBase._generate_one`` (which produces a single
        hashed file path) and write per-seed PNGs ourselves.
        """
        import torch

        self._ensure_loaded()

        results: List[str] = [""] * len(requests)
        device = self.plan.device_str()
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc=type(self).__name__)

        for i, req in enumerate(requests):
            ctx, gen_kwargs, _doc_to_visual, doc_id, task, _split = req.args
            prompt = str(ctx).strip()
            task_dir = Path(self.output_dir) / str(task).replace("/", "_")
            task_dir.mkdir(parents=True, exist_ok=True)

            if not prompt:
                results[i] = f"[ERROR] Empty prompt: doc_id={doc_id}"
                pbar.update(1)
                continue

            pid = int(doc_id) if isinstance(doc_id, int) else int(hashlib.sha1(str(doc_id).encode()).hexdigest(), 16) % 100000
            seed_paths = [task_dir / f"{pid:04d}_s{si}.png" for si in range(self.num_images_per_prompt)]

            if all(p.exists() and p.stat().st_size > 100 for p in seed_paths):
                eval_logger.debug(f"Cache hit: {task_dir}/{pid:04d}_s*.png")
                results[i] = str(task_dir)
                pbar.update(1)
                continue

            try:
                for si, out_path in enumerate(seed_paths):
                    if out_path.exists() and out_path.stat().st_size > 100:
                        continue
                    generator = torch.Generator(device=device).manual_seed(self.seed + si)
                    output = self._invoke_pipeline(prompt, [], generator, **(gen_kwargs or {}))
                    self._save_first_frame(output, out_path)
                results[i] = str(task_dir)
            except Exception as exc:
                eval_logger.error(f"Generation failed: task={task} doc_id={doc_id}: {exc}\n{traceback.format_exc()}")
                results[i] = f"[GENERATION_FAILED] {exc}"

            pbar.update(1)

        pbar.close()
        ok = sum(1 for r in results if r and not r.startswith("["))
        failed = len(results) - ok
        eval_logger.info(f"{type(self).__name__} complete: {ok} succeeded, {failed} failed")
        return results

    @staticmethod
    def _save_first_frame(output_obj, out_path: Path) -> None:
        """Save the first frame of a WanPipeline output as a PNG.

        With ``output_type="pil"``, ``frames`` is ``list[list[PIL.Image]]``
        keyed as ``frames[batch][frame_idx]``. Numpy / torch fallbacks are
        kept so this works if a future diffusers release changes the default.
        """
        import numpy as np
        from PIL import Image

        frames = getattr(output_obj, "frames", None)
        # Don't use ``not frames`` — numpy arrays raise on __bool__.
        if frames is None:
            raise RuntimeError("WanPipeline output has no .frames")

        first = frames[0]
        if isinstance(first, (list, tuple)):
            img = first[0]
        elif isinstance(first, Image.Image):
            img = first
        else:
            arr = first
            if hasattr(arr, "cpu"):  # torch tensor
                arr = arr.cpu().numpy()
            arr = np.asarray(arr)
            if arr.ndim == 4:  # (T, H, W, C)
                arr = arr[0]
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            img = Image.fromarray(arr)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        eval_logger.debug(f"Saved: {out_path}")
