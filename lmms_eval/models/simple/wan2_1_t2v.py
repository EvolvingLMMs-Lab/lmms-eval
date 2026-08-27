"""Wan2.1-1.3B Text-to-Video (T2V) backend for VBench evaluation.

Wraps ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers`` via ``diffusers.WanPipeline`` for
text-only video generation. Single-DiT variant (no ``transformer_2``);
``DiffusersWMBase``'s component walk handles the missing dual-expert
gracefully. Defaults match Wan2.1's canonical 480x832, 81-frame, 16-fps
configuration.

Usage::

    torchrun --nproc-per-node=8 -m lmms_eval \\
      --model wan2_1_t2v \\
      --model_args "pretrained=Wan-AI/Wan2.1-T2V-1.3B-Diffusers,output_dir=./logs/vbench_wan21" \\
      --tasks vbench \\
      --batch_size 1 \\
      --log_samples
"""

from typing import Union

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase


@register_model("wan2_1_t2v")
class Wan2_1_T2V(DiffusersWMBase):
    """Wan2.1-1.3B Text-to-Video backend for VBench evaluation."""

    def _patch_pipeline_cls_before_load(self) -> None:
        if type(self)._pipeline_cls is None:
            try:
                from diffusers import WanPipeline
            except ImportError as exc:
                raise ImportError("wan2_1_t2v requires diffusers: `pip install diffusers imageio imageio-ffmpeg`") from exc

            type(self)._pipeline_cls = WanPipeline

    def __init__(
        self,
        pretrained: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        fps: int = 16,
        seed: int = 42,
        dtype: str = "bfloat16",
        output_dir: str = "./logs/wan2_1_t2v_videos",
        batch_size: Union[int, str] = 1,
        attn_backend: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            output_dir=output_dir,
            seed=seed,
            dtype=dtype,
            fps=fps,
            batch_size=batch_size,
            **kwargs,
        )
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.attn_backend = str(attn_backend).strip()

    def _generation_signature(self, prompt, visuals, extras):
        return f"{self.pretrained}:{self.seed}:{self.num_inference_steps}:" f"{self.guidance_scale}:{self.num_frames}:{self.height}x{self.width}:" f"{prompt[:200]}"

    def _invoke_pipeline(self, prompt, visuals, generator, **extras):
        def _run():
            return self._pipe(
                prompt=prompt,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
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
