"""Wan2.2 Image-to-Video (I2V) backend for evaluating video generation quality.

Uses Wan2.2-I2V-A14B via HuggingFace diffusers to generate video continuations
from conditioning images.

Dual-expert (``transformer`` + ``transformer_2``) device placement and
UniPCMultistepScheduler sigma-device patches are inherited from
``DiffusersWMBase``.  I2V-specific behavior kept here: conditioning-image
preprocessing and doc→visual extraction fallback.

Default generation parameters: resolution=832x480, frames=81, steps=40,
guidance=3.5, fps=16, seed=42.

Usage::

    python -m lmms_eval \\
      --model wan2_2 \\
      --model_args "pretrained=Wan-AI/Wan2.2-I2V-A14B-Diffusers,output_dir=./logs/wan2_2" \\
      --tasks physics_iq_i2v \\
      --batch_size 1 \\
      --log_samples
"""

import os
from typing import Union

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase

_DEFAULT_PROMPT = "Generate a natural video continuation of this image."


@register_model("wan2_2")
class Wan2_2(DiffusersWMBase):
    """Wan2.2 Image-to-Video backend for evaluating video generation quality."""

    def __init__(
        self,
        pretrained: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        fps: int = 16,
        seed: int = 42,
        dtype: str = "bfloat16",
        output_dir: str = "./logs/wan2_2_videos",
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

    # ── DiffusersWMBase hooks ───────────────────────────────────

    def _patch_pipeline_cls_before_load(self) -> None:
        if type(self)._pipeline_cls is None:
            try:
                from diffusers import WanImageToVideoPipeline
            except ImportError as exc:
                raise ImportError("wan2_2 requires diffusers: `pip install diffusers imageio imageio-ffmpeg`") from exc

            type(self)._pipeline_cls = WanImageToVideoPipeline
        # Wan2.2-I2V-A14B conditions images through the VAE only, so its
        # model_index.json ships image_encoder / image_processor as [None, None].
        # Newer diffusers marks both as required at from_pretrained validation
        # time; widen _optional_components to accept the null-component layout.
        opts = list(getattr(self._pipeline_cls, "_optional_components", []) or [])
        for name in ("image_encoder", "image_processor"):
            if name not in opts:
                opts.append(name)
        self._pipeline_cls._optional_components = opts

    def _extract_visuals(self, req: Instance) -> list:
        _ctx, _kw, doc_to_visual, doc_id, task, split = req.args
        if doc_to_visual is None:
            return []
        # Standard tasks pass a real doc; agentic rollout rounds 2+ pass a
        # lambda that ignores the doc argument. Try doc first, then None.
        try:
            doc = self.task_dict[task][split][doc_id]
            raw = doc_to_visual(doc)
            if raw:
                return list(raw)
        except Exception as exc:
            eval_logger.debug(f"doc_to_visual(doc) failed; falling back to doc_to_visual(None): {exc}")
        try:
            raw = doc_to_visual(None)
            if raw:
                return list(raw)
        except Exception as exc:
            eval_logger.warning(f"Failed to extract visuals: {exc}")
        return []

    def _generation_signature(self, prompt, visuals, extras):
        return f"{self.pretrained}:{self.seed}:{self.num_inference_steps}:" f"{self.guidance_scale}:{self.num_frames}:{self.height}x{self.width}:" f"{len(visuals)}:{prompt[:100]}"

    def _invoke_pipeline(self, prompt, visuals, generator, **extras):
        if not visuals:
            raise RuntimeError("Wan2.2 I2V requires at least one conditioning image")
        image = self._prepare_image(visuals[0])
        if not prompt.strip():
            prompt = _DEFAULT_PROMPT
        return self._pipe(
            image=image,
            prompt=prompt,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )

    # ── I2V helpers ─────────────────────────────────────────────

    def _prepare_image(self, image):
        from PIL import Image

        if isinstance(image, str) and os.path.exists(image):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        return image.resize((self.width, self.height))
