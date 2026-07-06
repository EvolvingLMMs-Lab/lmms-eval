"""Wan2.2 Image-to-Video (I2V) backend for evaluating video generation quality.

Uses Wan2.2-I2V-A14B via HuggingFace diffusers to generate video continuations
from conditioning images.  Supports agentic multi-round rollout (last frame
chains into the next round).

Dual-expert (``transformer`` + ``transformer_2``) device placement and
UniPCMultistepScheduler sigma-device patches are inherited from
``DiffusersWMBase``.  I2V-specific behavior kept here: conditioning-image
preprocessing, doc→visual extraction fallback, and the rollout routine.

Default generation parameters: resolution=832x480, frames=81, steps=40,
guidance=3.5, fps=16, seed=114514.

Usage::

    python -m lmms_eval \\
      --model wan2_2 \\
      --model_args "pretrained=Wan-AI/Wan2.2-I2V-A14B-Diffusers,output_dir=./logs/wan2_2" \\
      --tasks physics_iq_i2v \\
      --batch_size 1 \\
      --log_samples
"""

import os
from typing import List, Optional, Union

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase, _cache_path

_DEFAULT_PROMPT = "Generate a natural video continuation of this image."
_DEFAULT_CONT_PROMPT = "Continue the video naturally."


@register_model("wan2_2")
class Wan2_2(DiffusersWMBase):
    """Wan2.2 Image-to-Video backend with agentic multi-round rollout support."""

    def __init__(
        self,
        pretrained: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        fps: int = 16,
        seed: int = 114514,
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
            from diffusers import WanImageToVideoPipeline

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

    # ── Agentic multi-round rollout ─────────────────────────────

    def generate_until_agentic(self, requests: List[Instance]) -> List[str]:
        """Multi-round rollout: last frame conditions the next round."""
        self._ensure_loaded()
        results: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Wan2.2 Rollout")
        for i, req in enumerate(requests):
            ctx, _kw, _dtv, doc_id, task, _split = req.args
            prompts = [str(p).strip() for p in ctx] if isinstance(ctx, list) else [str(ctx).strip()]
            visuals = self._extract_visuals(req)
            if not visuals:
                results[i] = "[ERROR] No conditioning visuals"
                pbar.update(1)
                continue
            if len(prompts) > 1:
                results[i] = self._rollout(visuals, prompts, task, doc_id)
            else:
                results[i] = self._generate_one(prompts[0], visuals, doc_id, task)
            pbar.update(1)
        pbar.close()
        return [r if r is not None else "[ERROR] Unknown" for r in results]

    def _rollout(self, visuals, prompts, task, doc_id) -> str:
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image

        sig = f"rollout:{self.pretrained}:{self.seed}:{self.num_inference_steps}:" f"{self.guidance_scale}:{self.num_frames}:{len(prompts)}:" f"{len(visuals)}:{'|'.join(p[:50] for p in prompts)}"
        out_path = _cache_path(self.output_dir, task, doc_id, sig, ext=self._output_ext)
        if out_path.exists():
            eval_logger.debug(f"Cache hit (rollout): {out_path}")
            return str(out_path)
        try:
            image = self._prepare_image(visuals[0])
            all_frames = []
            device = self.plan.device_str()
            for step_idx, prompt in enumerate(prompts):
                if not prompt:
                    prompt = _DEFAULT_CONT_PROMPT
                generator = torch.Generator(device=device).manual_seed(self.seed + step_idx)
                output = self._pipe(
                    image=image,
                    prompt=prompt,
                    num_frames=self.num_frames,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                )
                frames = output.frames[0]
                all_frames.extend(frames if step_idx == 0 else frames[1:])
                image = frames[-1]
                if isinstance(image, Image.Image):
                    image = image.resize((self.width, self.height))
                eval_logger.info(f"Rollout step {step_idx + 1}/{len(prompts)} done ({len(frames)} frames)")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(all_frames, str(out_path), fps=self.fps)
            eval_logger.info(f"Rollout video saved: {out_path} ({len(all_frames)} total frames)")
            return str(out_path)
        except Exception as exc:
            eval_logger.error(f"Rollout failed: task={task} doc_id={doc_id}: {exc}")
            return f"[ROLLOUT_FAILED] {exc}"
