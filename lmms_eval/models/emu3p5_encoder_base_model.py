"""
Shared base class for EMU3.5 encoder models (both chat and simple).

This module contains all shared logic for models using the EMU3.5 IBQ vision
tokenizer, including model loading, tokenizer setup, image processing,
and distributed training configuration.
"""

import abc
import sys
from pathlib import Path
from typing import Optional, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger

from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.emu3p5.download_utils import ensure_local_weights
from lmms_eval.models.model_utils.emu3p5.emu3p5_input_processor import (
    Emu3p5Processor,
)
from lmms_eval.models.model_utils.memory_utils import print_memory_stats

# Check if Emu3.5 submodule is initialized
_current_file = Path(__file__).resolve()
_repo_root = _current_file.parents[2]  # Go up to lmms-eval root
_emu35_src_path = _repo_root / "external" / "Emu3.5" / "src"
_emu35_modeling_file = _emu35_src_path / "emu3p5" / "modeling_emu3.py"

if not _emu35_modeling_file.exists():
    eval_logger.error(f"Emu3.5 submodule is not initialized in. {_repo_root / "external" / "Emu3.5"}. Please run the following commands:\n" f"  cd {_repo_root}\n" "  git submodule update --init --recursive external/Emu3.5\n")
    sys.exit(1)

# Add external Emu3.5 to path
if str(_emu35_src_path) not in sys.path:
    sys.path.insert(0, str(_emu35_src_path))

# Import Emu3 classes from external directory
from emu3p5 import Emu3Config, Emu3ForCausalLM  # noqa: E402
from vision_tokenizer import build_vision_tokenizer  # noqa: E402


class EMU3p5EncoderBaseModel(lmms):
    """
    Shared base class for EMU3.5 encoder models (both chat and simple).

    Handles IBQ vision tokenizer loading, image processing, and common
    utilities. Subclasses must implement model-specific generation logic.

    This class provides all the infrastructure for:
    - Loading LLM and tokenizer
    - Loading EMU3.5 IBQ vision tokenizer
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
        vq_hub: str = "BAAI/Emu3.5-VisionTokenizer",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "flash_attention_2",
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        vision_tokenizer_dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        emu_min_pixels: int = 512 * 512,
        emu_max_pixels: int = 1024 * 1024,
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
        eval_logger.info(f"Loading Text Tokenizer from {tokenizer_path}")
        self._tokenizer = self._load_tokenizer(tokenizer_path)

        # Load vision tokenizer (IBQ)
        eval_logger.info(f"Loading IBQ Vision Tokenizer from {vq_hub}")
        vision_tokenizer_kwargs = {}
        if vision_tokenizer_dtype is not None:
            vision_tokenizer_kwargs["torch_dtype"] = vision_tokenizer_dtype

        vision_tokenizer = self._load_vision_tokenizer(vq_hub, device=self.device_map, **vision_tokenizer_kwargs).eval()

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
                vision_tokenizer = accelerator.prepare(vision_tokenizer)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
                vision_tokenizer = accelerator.prepare_model(vision_tokenizer, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        # Create EMU3.5 processor (after preparing vision_tokenizer if multi-GPU)
        self.processor = Emu3p5Processor(
            vision_tokenizer=vision_tokenizer,
            tokenizer=self._tokenizer,
            min_pixels=emu_min_pixels,
            max_pixels=emu_max_pixels,
        )

        eval_logger.info(f"EMU3.5 model loaded successfully on rank {self.rank}/" f"{self.world_size}")
        if self.debug_samples and self.rank == 0:
            eval_logger.info(f"Debug mode enabled: will print first {self.num_debug_samples} " "samples")

        # Report model sizes and GPU memory usage on each rank
        device_idx = self._device.index if self._device.type == "cuda" else None
        print_memory_stats(
            main_model=self._model,
            image_tokenizer=vision_tokenizer,
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

    def _load_vision_tokenizer(self, vq_hub: str, device: str, **kwargs):
        """
        Load IBQ vision tokenizer for EMU3.5.

        Args:
            vq_hub: Path or HF identifier for vision tokenizer
            device: Device to load the tokenizer on

        Returns:
            Vision tokenizer object with encode() method
        """
        # Ensure vision tokenizer weights are available locally
        vq_hub = ensure_local_weights(vq_hub, "BAAI/Emu3.5-VisionTokenizer", accelerator=self.accelerator)

        # Map torch_dtype to dtype for build_vision_tokenizer
        if "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")

        return build_vision_tokenizer(
            type="ibq",
            model_path=vq_hub,
            device=device,
            **kwargs,
        )

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
