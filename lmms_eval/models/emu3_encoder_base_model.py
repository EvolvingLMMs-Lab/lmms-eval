"""
Shared base class for EMU3 encoder models (both chat and simple).

This module contains all shared logic for models using the EMU3 vision
tokenizer, including model loading, tokenizer setup, image processing,
and distributed training configuration.
"""

import abc
from typing import Optional, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import AutoModel

from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.emu3.emu3_image_processor import (
    Emu3VisionVQImageProcessor,
)
from lmms_eval.models.model_utils.emu3.emu3_input_processor import Emu3Processor
from lmms_eval.models.model_utils.memory_utils import print_memory_stats


class EMU3EncoderBaseModel(lmms):
    """
    Shared base class for EMU3 encoder models (both chat and simple).
    Subclasses define which language model is loaded and how results are generated.

    Handles vision tokenizer loading, image processing, and common
    utilities.

    This class provides all the infrastructure for:
    - Loading LLM and tokenizer
    - Loading EMU3 vision processor and tokenizer
    - Device management and distributed training setup
    - Memory reporting

    Subclasses must implement:
    - _load_llm(): Load the language model
    - _load_tokenizer(): Load the text tokenizer
    - image_placeholder property: Define image token placeholder
    - generate_until(): Model-specific generation logic
    """

    def __init__(
        self,
        model_descriptor: str,
        tokenizer_path: str,
        vq_hub: str = "BAAI/Emu3-VisionTokenizer",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "flash_attention_2",
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        image_tokenizer_dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        emu_min_pixels: int = 512 * 512,
        emu_max_pixels: int = 1024 * 1024,
        do_check_aspect_ratio: bool = False,
        skip_text_only: bool = True,
        skip_multi_image: bool = True,
        debug_samples: bool = False,
        num_debug_samples: int = 5,
        **kwargs,
    ):
        super().__init__()

        # Setup accelerator for distributed training
        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Load main model
        eval_logger.info(f"Loading Model from {model_descriptor}")
        self._model = self._load_llm(
            model_descriptor,
            device_map=device_map if accelerator.num_processes == 1 else None,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        ).eval()

        # Load tokenizer
        eval_logger.info(f"Loading EMU3 Text Tokenizer from {tokenizer_path}")
        self._tokenizer = self._load_tokenizer(tokenizer_path)

        # Load image processor and image tokenizer
        eval_logger.info(f"Loading EMU3 Vision Img Preprocessor from {vq_hub}")
        image_processor = Emu3VisionVQImageProcessor.from_pretrained(
            vq_hub,
            trust_remote_code=trust_remote_code,
            min_pixels=emu_min_pixels,
            max_pixels=emu_max_pixels,
            do_check_aspect_ratio=do_check_aspect_ratio,
        )
        eval_logger.info(f"Loading EMU3 Vision Tokenizer from {vq_hub}")
        image_tokenizer_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": trust_remote_code,
        }
        if image_tokenizer_dtype is not None:
            image_tokenizer_kwargs["torch_dtype"] = image_tokenizer_dtype
        image_tokenizer = AutoModel.from_pretrained(vq_hub, **image_tokenizer_kwargs).eval()

        # Set instance variables
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.skip_text_only = skip_text_only
        self.skip_multi_image = skip_multi_image
        self.debug_samples = debug_samples
        self.num_debug_samples = num_debug_samples
        self._debug_samples_printed = 0  # Counter for tracking printed samples

        # Prepare models for distributed training if needed
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
                image_tokenizer = accelerator.prepare(image_tokenizer)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
                image_tokenizer = accelerator.prepare_model(image_tokenizer, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        # Create EMU3 processor (after preparing image_tokenizer if multi-GPU)
        self.processor = Emu3Processor(image_processor, image_tokenizer, self._tokenizer)

        eval_logger.info(f"EMU3 model loaded successfully on rank {self.rank}/{self.world_size}")
        if self.debug_samples and self.rank == 0:
            eval_logger.info(f"Debug mode enabled: will print first {self.num_debug_samples} " "samples")

        # Report model sizes and GPU memory usage on each rank
        device_idx = self._device.index if self._device.type == "cuda" else None
        print_memory_stats(
            main_model=self._model,
            image_tokenizer=image_tokenizer,
            accelerator=self.accelerator,
            device_idx=device_idx,
            rank=self.rank,
        )

    @abc.abstractmethod
    def _load_tokenizer(self, tokenizer_path, **kwargs):
        """
        Load text tokenizer from path.

        Args:
            tokenizer_path: Path or HF identifier for tokenizer

        Returns:
            HuggingFace tokenizer object
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def _load_llm(self, model_path, **kwargs):
        """
        Load language model from path.

        Args:
            model_path: Path or HF identifier for model

        Returns:
            HuggingFace model object supporting generate()
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abc.abstractmethod
    def image_placeholder(self) -> str:
        """
        Image placeholder token used in prompts/templates.

        Examples: "<|image|>", "<image>", "[IMG]"

        Subclasses must define what placeholder their model uses for images.
        """
        raise NotImplementedError("Subclasses must define image_placeholder")

    @property
    def model(self):
        """Return unwrapped model (handles accelerator wrapping)."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def tokenizer(self):
        """Return text tokenizer."""
        return self._tokenizer

    @property
    def batch_size(self):
        """Return batch size per GPU."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return device object."""
        return self._device

    @property
    def rank(self):
        """Return process rank (for distributed training)."""
        return self._rank

    @property
    def world_size(self):
        """Return world size (for distributed training)."""
        return self._world_size
