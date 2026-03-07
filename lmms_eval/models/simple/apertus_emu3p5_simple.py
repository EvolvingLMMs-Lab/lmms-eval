"""
Apertus base model with EMU3.5 vision encoder (simple/non-chat version).

For evaluating base models without instruction tuning.
Uses direct text prompts instead of chat templates.
"""

from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from transformers import ApertusForCausalLM, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.emu_simple_model import EMU3p5SimpleModel


@register_model("apertus_emu3p5_simple")
class ApertusEmu3p5Simple(EMU3p5SimpleModel):
    """
    Apertus base model with EMU3.5 vision encoder (simple/non-chat version).

    For evaluating base models without instruction tuning.
    Uses direct text prompts instead of chat templates.
    """

    def __init__(
        self,
        model_descriptor: str,
        tokenizer_path: str,
        vq_hub: str = "BAAI/Emu3.5-VisionTokenizer",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        vision_tokenizer_dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        emu_min_pixels: int = 256 * 256,
        emu_max_pixels: int = 1400 * 1400,
        skip_text_only: bool = True,
        skip_multi_image: bool = True,
        debug_samples: bool = False,
        num_debug_samples: int = 5,
        **kwargs,
    ) -> None:
        """
        Initialize Apertus EMU3.5 simple model.

        Args:
            model_descriptor: Path or HF identifier for Apertus model
            tokenizer_path: Path or HF identifier for tokenizer
            vq_hub: Path or HF identifier for EMU3.5 IBQ vision tokenizer
            device: Device to load model on
            device_map: Device map for model
            batch_size: Batch size per GPU
            attn_implementation: Attention implementation
            trust_remote_code: Whether to trust remote code
            torch_dtype: Data type for model
            vision_tokenizer_dtype: Data type for vision tokenizer
            use_cache: Whether to use KV cache
            emu_min_pixels: Minimum pixels for image
            emu_max_pixels: Maximum pixels for image
            skip_text_only: Skip text-only samples
            skip_multi_image: Skip multi-image samples
            debug_samples: Print debug samples
            num_debug_samples: Number of debug samples to print
        """
        super().__init__(
            model_descriptor=model_descriptor,
            tokenizer_path=tokenizer_path,
            vq_hub=vq_hub,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            vision_tokenizer_dtype=vision_tokenizer_dtype,
            use_cache=use_cache,
            emu_min_pixels=emu_min_pixels,
            emu_max_pixels=emu_max_pixels,
            skip_text_only=skip_text_only,
            skip_multi_image=skip_multi_image,
            debug_samples=debug_samples,
            num_debug_samples=num_debug_samples,
            **kwargs,
        )

    def _load_tokenizer(self, tokenizer_path: str, **kwargs) -> AutoTokenizer:
        """Load Apertus tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            eval_logger.warning("No pad_token found, setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_llm(self, model_path: str, **kwargs) -> ApertusForCausalLM:
        """Load Apertus base model."""
        return ApertusForCausalLM.from_pretrained(model_path, **kwargs).eval()

    @property
    def image_placeholder(self) -> str:
        """Apertus uses <|image|> placeholder."""
        return "<|image|>"
