"""MiniCPM-OCR model for lmms-eval.

MiniCPM-OCR is a 1.3B document-understanding model from OpenBMB that
specializes in text-dense images such as scanned pages, tables, formulas,
charts, and receipts.

https://huggingface.co/openbmb/MiniCPM-OCR-1.0
https://huggingface.co/docs/transformers/en/model_doc/minicpm_ocr

The model is implemented natively in transformers, so this wrapper uses the
standard ``AutoProcessor`` + ``AutoModelForImageTextToText`` path. It requires
``transformers>=5.22.0``, the first release that ships the ``minicpm_ocr``
architecture.

Example usage:

    python -m lmms_eval --model minicpm_ocr \\
        --model_args pretrained=openbmb/MiniCPM-OCR-1.0 \\
        --tasks ocrbench --batch_size 1 --limit 8
"""

from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.protocol import ChatMessages

MIN_TRANSFORMERS_VERSION = "5.22.0"

# ``downsample_mode`` trades visual tokens for speed: "4x" keeps four times
# more tokens than "16x" and is what text-dense documents are read with.
DOWNSAMPLE_MODES = ("4x", "16x")
SMALL_IMAGE_UPSIZE_MODES = ("always", "never", "adaptive")


@register_model("minicpm_ocr")
class MiniCPMOcr(ChatMixin, lmms):
    """MiniCPM-OCR chat-style wrapper with structured message support."""

    is_simple = False

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-OCR-1.0",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
        downsample_mode: str = "4x",
        small_image_upsize: str = "always",
        enable_thinking: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        _require_transformers_version()
        # Import lazily so an older transformers fails with the actionable
        # version message above instead of a bare ImportError at module load.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if downsample_mode not in DOWNSAMPLE_MODES:
            raise ValueError(f"downsample_mode must be one of {DOWNSAMPLE_MODES}, got {downsample_mode!r}")
        if small_image_upsize not in SMALL_IMAGE_UPSIZE_MODES:
            raise ValueError(f"small_image_upsize must be one of {SMALL_IMAGE_UPSIZE_MODES}, got {small_image_upsize!r}")

        self.downsample_mode = downsample_mode
        self.small_image_upsize = small_image_upsize
        self.enable_thinking = enable_thinking
        self.system_prompt = self._resolve_system_prompt(system_prompt) if system_prompt else None
        self.use_cache = use_cache
        self.batch_size_per_gpu = int(batch_size)

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = None
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else None

        model_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to load {pretrained!r} as a MiniCPM-OCR checkpoint. Make sure the model id is correct and `transformers>={MIN_TRANSFORMERS_VERSION}` is installed.") from e
        if not self.device_map:
            self._model = self._model.to(self._device)

        self._processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self._processor.tokenizer

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

    @property
    def config(self):
        return self.model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @staticmethod
    def _normalize_until(until) -> List[str]:
        if until is None:
            return []
        if isinstance(until, str):
            until = [until]
        if not isinstance(until, (list, tuple)):
            raise ValueError(f"Expected `gen_kwargs['until']` to be a string or list, got {type(until)}")
        return [item for item in until if isinstance(item, str) and item]

    @staticmethod
    def _truncate_until(answer: str, until: List[str]) -> str:
        stop = min((answer.find(item) for item in until if item in answer), default=-1)
        return answer[:stop] if stop >= 0 else answer

    def _build_generate_kwargs(self, gen_kwargs: dict) -> dict:
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 1024))
        temperature = gen_kwargs.get("temperature", 0)
        do_sample = temperature is not None and float(temperature) > 0

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": self.use_cache,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(temperature)
            if gen_kwargs.get("top_p") is not None:
                generate_kwargs["top_p"] = float(gen_kwargs["top_p"])
            if gen_kwargs.get("top_k") is not None:
                generate_kwargs["top_k"] = int(gen_kwargs["top_k"])
        else:
            num_beams = gen_kwargs.get("num_beams")
            if num_beams is not None and int(num_beams) > 1:
                generate_kwargs["num_beams"] = int(num_beams)
        return generate_kwargs

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        hf_messages = chat_messages.to_hf_messages()
        if self.system_prompt:
            hf_messages = self._apply_system_prompt(hf_messages, self.system_prompt)

        template_kwargs = {"enable_thinking": self.enable_thinking}
        processor_kwargs = {
            "downsample_mode": self.downsample_mode,
            "small_image_upsize": self.small_image_upsize,
        }
        inputs = self.processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
            **template_kwargs,
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        until = self._normalize_until(gen_kwargs.get("until"))
        input_len = inputs["input_ids"].shape[1]
        generate_kwargs = self._build_generate_kwargs(gen_kwargs)
        if until:
            # Stop the model at the requested sequence instead of decoding to
            # max_new_tokens and truncating afterwards.
            generate_kwargs["stop_strings"] = until
            generate_kwargs["tokenizer"] = self.tokenizer
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                downsample_mode=self.downsample_mode,
                **generate_kwargs,
            )

        answer = self.processor.batch_decode(
            output[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._truncate_until(answer, until)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MiniCPM-OCR")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for MiniCPM-OCR")


def _require_transformers_version() -> None:
    """Raise a clear error when transformers predates native MiniCPM-OCR."""
    try:
        import transformers
        from packaging.version import Version
    except ImportError as e:  # pragma: no cover - transformers is a core dependency
        raise ImportError("MiniCPM-OCR requires `transformers>=5.22.0`.") from e

    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
        raise ImportError(f"MiniCPM-OCR requires `transformers>={MIN_TRANSFORMERS_VERSION}`, but {transformers.__version__} is installed. Upgrade with `pip install -U 'transformers>=5.22.0'`.")
