"""Wan2.2 Text-to-Video (T2V) backend for VBench evaluation.

Uses WanPipeline (diffusers) for text-only video generation.  Dual-expert
device placement (``transformer`` + ``transformer_2``) and the
UniPCMultistepScheduler sigma-device patch are inherited from
``DiffusersWMBase``; this subclass only encodes T2V-specific defaults
and the generation signature / pipeline invocation.

Usage::

    torchrun --nproc-per-node=8 -m lmms_eval \\
      --model wan2_2_t2v \\
      --model_args "pretrained=Wan-AI/Wan2.2-T2V-A14B-Diffusers,output_dir=./logs/vbench_wan22" \\
      --tasks vbench \\
      --batch_size 1 \\
      --log_samples
"""

from typing import Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase

_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
_DEFAULT_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"


@register_model("wan2_2_t2v")
class Wan2_2_T2V(DiffusersWMBase):
    """Wan2.2 Text-to-Video backend for VBench evaluation."""

    def _patch_pipeline_cls_before_load(self) -> None:
        if type(self)._pipeline_cls is None:
            try:
                from diffusers import WanPipeline
            except ImportError as exc:
                raise ImportError("wan2_2_t2v requires diffusers: `pip install diffusers imageio imageio-ffmpeg`") from exc

            type(self)._pipeline_cls = WanPipeline

    def __init__(
        self,
        pretrained: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        revision: Optional[str] = _DEFAULT_REVISION,
        num_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        guidance_scale_2: float = 3.0,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        flow_shift: Optional[float] = 12.0,
        fps: int = 16,
        seed: int = 42,
        dtype: str = "bfloat16",
        output_dir: str = "./logs/wan2_2_t2v_videos",
        batch_size: Union[int, str] = 1,
        attn_backend: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            revision=revision,
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
        self.guidance_scale_2 = float(guidance_scale_2)
        self.negative_prompt = str(negative_prompt)
        self.flow_shift = None if flow_shift is None else float(flow_shift)
        # Attention kernel override. Empty = default SDPA. Supported values
        # per diffusers main's attention_dispatch: "flash" (FA2),
        # "_flash_3"/"_flash_3_hub" (FA3 Hopper-native, requires kernels pkg),
        # "sage", "xformers", "native". The dispatcher's "is not usable" check
        # fires on context-enter, not on import, so _invoke_pipeline wraps the
        # whole call in try/except and falls back to default on failure.
        self.attn_backend = str(attn_backend).strip()

    def _post_to_device(self, pipe, device: str) -> None:
        if self.flow_shift is not None:
            pipe.scheduler.register_to_config(flow_shift=self.flow_shift)
            eval_logger.info(f"Overrode scheduler flow_shift={self.flow_shift}")

    def _pipeline_load_kwargs(self, dtype) -> dict:
        import torch
        from diffusers import AutoencoderKLWan

        # Wan's Diffusers recipe keeps the VAE in fp32 while the experts and
        # text encoder use bf16. Loading it separately avoids a lossy
        # bf16-load-then-upcast cycle.
        revision_kwargs = {"revision": self.revision} if self.revision is not None else {}
        vae = AutoencoderKLWan.from_pretrained(self.pretrained, subfolder="vae", torch_dtype=torch.float32, **revision_kwargs)
        return {"torch_dtype": dtype, "vae": vae, **revision_kwargs}

    def _generation_signature(self, prompt, visuals, extras):
        request_seed = int(extras.get("_lmms_eval_seed", self.seed))
        return (
            f"{self.pretrained}@{self.revision}:{request_seed}:{self.num_inference_steps}:"
            f"{self.guidance_scale}:{self.guidance_scale_2}:{self.flow_shift}:"
            f"{self.num_frames}:{self.height}x{self.width}:{self.negative_prompt}:"
            f"{prompt[:200]}"
        )

    def _invoke_pipeline(self, prompt, visuals, generator, **extras):
        def _run():
            return self._pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                guidance_scale_2=self.guidance_scale_2,
                generator=generator,
            )

        if self.attn_backend:
            try:
                from diffusers.models.attention_dispatch import attention_backend

                with attention_backend(self.attn_backend):
                    return _run()
            except Exception as exc:  # backend unavailable / import miss / runtime failure
                eval_logger.warning(f"attn_backend='{self.attn_backend}' failed ({exc}); falling back to default")
        return _run()
