"""LTX-Video Text-to-Video backend for VBench evaluation.

Uses LTXPipeline (diffusers) for real-time text-to-video generation.  Supports
``Lightricks/LTX-Video`` and version variants (0.9.5, 0.9.7-dev, 0.9.8-dev).

Usage::

    torchrun --nproc-per-node=8 -m lmms_eval \\
      --model ltx_video \\
      --model_args "pretrained=Lightricks/LTX-Video,output_dir=./logs/vbench_ltx" \\
      --tasks vbench \\
      --batch_size 1 \\
      --log_samples
"""

from typing import Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase


@register_model("ltx_video")
class LTXVideo(DiffusersWMBase):
    """LTX-Video Text-to-Video backend for VBench evaluation."""

    def _patch_pipeline_cls_before_load(self) -> None:
        if type(self)._pipeline_cls is None:
            try:
                from diffusers import LTXPipeline
            except ImportError as exc:
                raise ImportError("ltx_video requires diffusers: `pip install diffusers imageio imageio-ffmpeg`") from exc

            type(self)._pipeline_cls = LTXPipeline

    def __init__(
        self,
        pretrained: str = "Lightricks/LTX-Video",
        num_frames: int = 97,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        decode_timestep: float = 0.03,
        decode_noise_scale: float = 0.025,
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        fps: int = 24,
        seed: int = 42,
        dtype: str = "bfloat16",
        output_dir: str = "./logs/ltx_video",
        batch_size: Union[int, str] = 1,
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
        self.decode_timestep = float(decode_timestep)
        self.decode_noise_scale = float(decode_noise_scale)
        self.negative_prompt = negative_prompt

    def _generation_signature(self, prompt, visuals, extras):
        return f"{self.pretrained}:{self.seed}:{self.num_inference_steps}:" f"{self.guidance_scale}:{self.num_frames}:{self.height}x{self.width}:" f"{prompt[:200]}"

    def _invoke_pipeline(self, prompt, visuals, generator, **extras):
        return self._pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            decode_timestep=self.decode_timestep,
            decode_noise_scale=self.decode_noise_scale,
            generator=generator,
        )
