from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from transformers import ApertusForCausalLM, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.emu3p5_encoder_model import EMU3p5EncoderModel


@register_model("apertus_emu3p5")
class ApertusEmu3p5Chat(EMU3p5EncoderModel):
    """
    Apertus model integrated with EMU3.5 vision encoder.

    Uses swiss-ai Apertus language model with EMU3.5 IBQ vision tokenizer
    for multimodal chat capabilities.

    model_descriptor: path to model checkpoint (hf format) or HF hub identifier
    tokenizer_path: path to tokenizer checkpoint (hf format) or HF hub identifier
    vq_hub: path to IBQ vision tokenizer checkpoint
    """

    is_simple = False

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
        emu3_min_pixels: int = 512 * 512,
        emu3_max_pixels: int = 1024 * 1024,
        skip_text_only: bool = True,
        skip_multi_image: bool = True,
        debug_samples: bool = False,
        num_debug_samples: int = 5,
        max_length: Optional[int] = None,
        ignore_max_length: bool = True,
        **kwargs,
    ) -> None:
        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, " f"got {attn_implementation}")

        # Store Apertus-specific config before calling super
        self._max_length_override = max_length
        self.ignore_max_length = ignore_max_length

        # Call parent constructor (handles all initialization)
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
            emu3_min_pixels=emu3_min_pixels,
            emu3_max_pixels=emu3_max_pixels,
            skip_text_only=skip_text_only,
            skip_multi_image=skip_multi_image,
            debug_samples=debug_samples,
            num_debug_samples=num_debug_samples,
            **kwargs,
        )

        # Set max_length with sensible defaults
        if self._max_length_override is not None:
            self._max_length = self._max_length_override
        else:
            try:
                self._max_length = self.model.config.max_position_embeddings
                eval_logger.info(f"Using max_length from model config: {self._max_length}")
            except AttributeError:
                self._max_length = 8192
                eval_logger.warning(f"Could not infer max_length from model config, " f"using default: {self._max_length}")

        if self.ignore_max_length:
            eval_logger.warning("ignore_max_length=True: Truncation disabled. " "Long sequences may cause OOM or errors.")

    def _load_tokenizer(self, tokenizer_path: str, **kwargs) -> AutoTokenizer:
        """Load Apertus tokenizer and ensure pad token is set."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            eval_logger.warning("No pad_token found in tokenizer, setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_llm(self, model_path: str, **kwargs) -> ApertusForCausalLM:
        """Load Apertus causal language model."""
        return ApertusForCausalLM.from_pretrained(model_path, **kwargs).eval()

    @property
    def image_placeholder(self) -> str:
        """Apertus uses <|image|> as placeholder in chat template."""
        return "<|image|>"

    def _chat_transform(self, hf_messages: list[dict]) -> list[dict]:
        """
        Transform HF messages to Apertus format by wrapping content in parts.

        Apertus chat template expects user messages with multimodal content
        to have the format: {"role": "user", "content": {"parts": [...]}}
        instead of the standard HF format: {"role": "user", "content": [...]}
        """
        transformed = []
        for msg in hf_messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                # Wrap content list in "parts" key for Apertus format
                transformed_msg = {"role": msg["role"], "content": {"parts": msg["content"]}}
                transformed.append(transformed_msg)
            else:
                # Assistant and system messages don't need transformation
                transformed.append(msg)
        return transformed

    @property
    def max_length(self):
        """Return configured max sequence length."""
        return self._max_length

    @property
    def config(self):
        """Return model configuration."""
        return self.model.config
