from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from transformers import AutoTokenizer, LlamaForCausalLM

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.emu3p5_encoder_model import EMU3p5EncoderModel


@register_model("llama_emu3p5")
class LlamaEmu3p5Chat(EMU3p5EncoderModel):
    """
    Llama3.2-3B with EMU3.5 IBQ vision encoder integration.
    Uses early fusion with discrete vision tokens from EMU3.5.

    model_descriptor: path to model checkpoint (hf format) or HF hub identifier
    tokenizer_path: path to tokenizer checkpoint (hf format) or HF hub identifier
    vq_hub: path to IBQ vision tokenizer checkpoint
    """

    is_simple = False

    def __init__(
        self,
        model_descriptor: str,
        tokenizer_path: str,
        vq_hub: str,
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
            raise ValueError(
                f"attn_implementation must be one of {valid_attn_implementations}, "
                f"got {attn_implementation}"
            )

        # Store Llama-specific config before calling super
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
                eval_logger.warning(
                    f"Could not infer max_length from model config, "
                    f"using default: {self._max_length}"
                )

        if self.ignore_max_length:
            eval_logger.warning(
                "ignore_max_length=True: Truncation disabled. "
                "Long sequences may cause OOM or errors."
            )

    def _load_tokenizer(self, tokenizer_path: str, **kwargs) -> AutoTokenizer:
        """Load Llama tokenizer and ensure pad token is set."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            eval_logger.warning(
                "No pad_token found in tokenizer, setting pad_token to eos_token."
            )
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_llm(self, model_path: str, **kwargs) -> LlamaForCausalLM:
        """Load Llama causal language model."""
        return LlamaForCausalLM.from_pretrained(model_path, **kwargs).eval()

    def _load_vision_tokenizer(self, vq_hub: str, device: str, **kwargs):
        """Load IBQ vision tokenizer for EMU3.5."""
        # Import here to avoid dependency if not using this model
        try:
            from vision_tokenizer import build_vision_tokenizer
        except ImportError:
            raise ImportError(
                "vision_tokenizer package is required for EMU3.5 models. "
                "Please install it from the EMU3.5 repository."
            )

        return build_vision_tokenizer(
            type="ibq",
            model_path=vq_hub,
            device=device,
            **kwargs
        )

    @property
    def image_placeholder(self) -> str:
        """Llama uses <|image|> as placeholder in chat template."""
        return "<|image|>"

    def _chat_transform(self, hf_messages: list[dict]) -> list[dict]:
        """
        Llama doesn't require message transformation.
        Return messages unchanged.
        """
        return hf_messages

    @property
    def max_length(self):
        """Return configured max sequence length."""
        return self._max_length

    @property
    def config(self):
        """Return model configuration."""
        return self.model.config
