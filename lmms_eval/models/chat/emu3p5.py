import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages
from lmms_eval.models.model_utils.emu3p5.download_utils import (
    ensure_local_weights,
)
from lmms_eval.models.model_utils.emu3p5.emu3_tokenizer_loader import (
    load_emu3_tokenizer
)
from lmms_eval.models.model_utils.emu3p5.emu3p5_input_processor import Emu3p5Processor
from lmms_eval.models.model_utils.memory_utils import print_memory_stats

# Check if Emu3.5 submodule is initialized
_current_file = Path(__file__).resolve()
_repo_root = _current_file.parents[3]  # Go up to lmms-eval root
_emu35_src_path = _repo_root / "external" / "Emu3.5" / "src"
_emu35_modeling_file = _emu35_src_path / "emu3p5" / "modeling_emu3.py"

if not _emu35_modeling_file.exists():
    eval_logger.error(
        "Emu3.5 submodule is not initialized. Please run the following commands:\n"
        f"  cd {_repo_root}\n"
        "  git submodule update --init --recursive external/Emu3.5\n"
    )
    sys.exit(1)

# Add external Emu3.5 to path
if str(_emu35_src_path) not in sys.path:
    sys.path.insert(0, str(_emu35_src_path))

# Import Emu3 classes from external directory
from emu3p5 import Emu3Config, Emu3ForCausalLM
from vision_tokenizer import build_vision_tokenizer



@register_model("emu3p5")
class EMU3_5(lmms):
    """
    EMU3.5 Multimodal Model (34B parameters).
    Uses IBQ vision tokenizer for improved image understanding.
    https://github.com/baaivision/Emu3.5
    """

    is_simple = False  # Chat model

    def __init__(
        self,
        pretrained: str = "BAAI/Emu3.5",
        vq_hub: str = "BAAI/Emu3.5-VisionTokenizer",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "flash_attention_2",
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        image_tokenizer_dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        emu3_min_pixels: int = 512 * 512,
        emu3_max_pixels: int = 1024 * 1024,
        skip_text_only: bool = True,
        skip_multi_image: bool = True,
        debug_samples: bool = False,
        num_debug_samples: int = 5,
        **kwargs,
    ):
        super().__init__()

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Ensure main model weights are available locally
        pretrained = ensure_local_weights(
            pretrained, "BAAI/Emu3.5", accelerator=accelerator
        )

        # Load main model using Emu3ForCausalLM directly
        eval_logger.info(f"Loading EMU3.5 model from {pretrained}")
        model_config = Emu3Config.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
        )
        self._model = Emu3ForCausalLM.from_pretrained(
            pretrained,
            config=model_config,
            device_map=device_map if accelerator.num_processes == 1 else None,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()

        # Load tokenizer (with fallback handling for missing tokenization file)
        txt_tok_path = _emu35_src_path / "tokenizer_emu3_ibq"
        eval_logger.info(f"Loading EMU3.5 Text Tokenizer from {txt_tok_path}")
        self._txt_tokenizer = load_emu3_tokenizer(
            str(txt_tok_path), trust_remote_code=trust_remote_code, padding_side="left"
        )

        # Ensure vision tokenizer weights are available locally
        vq_hub = ensure_local_weights(
            vq_hub, "BAAI/Emu3.5-VisionTokenizer", accelerator=accelerator
        )

        eval_logger.info(f"Loading EMU3.5 Vision Tokenizer from {vq_hub}")
        # Use build_vision_tokenizer for loading the IBQ vision tokenizer
        vq_device = self.device_map if self.device_map != "auto" else self._device
        vq_kwargs = {}
        if image_tokenizer_dtype is not None:
            vq_kwargs["dtype"] = image_tokenizer_dtype
        image_tokenizer = build_vision_tokenizer(
            type="ibq",
            model_path=vq_hub,
            device=vq_device,
            **vq_kwargs,
        )

        self.processor = Emu3p5Processor(
            vision_tokenizer=image_tokenizer,
            tokenizer=self._txt_tokenizer,
            min_pixels=emu3_min_pixels,
            max_pixels=emu3_max_pixels,
        )

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
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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

        eval_logger.info(f"EMU3.5 model loaded successfully on rank {self.rank}/{self.world_size}")
        if self.debug_samples and self.rank == 0:
            eval_logger.info(f"Debug mode enabled: will print first {self.num_debug_samples} samples")

        # Report model sizes and GPU memory usage on each rank
        device_idx = self._device.index if self._device.type == "cuda" else None
        print_memory_stats(
            main_model=self._model,
            image_tokenizer=image_tokenizer,
            accelerator=self.accelerator,
            device_idx=device_idx,
            rank=self.rank,
        )

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def tokenizer(self):
        return self._txt_tokenizer

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

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # Initialize statistics counters
        text_only_count = 0
        multi_image_count = 0
        total_samples = 0
        skipped_text_only = 0
        skipped_multi_image = 0

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # Group requests by their generation_kwargs
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        # iterate over batches
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            # Get chat messages
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            # Convert to ChatMessages protocol
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            # Extract media and track per-sample image counts
            images_list = []
            sample_data = []
            for idx, messages in enumerate(chat_messages):
                total_samples += 1
                visual, video, audio = messages.extract_media()
                images_list.append(visual)

                # Extract text for this sample
                text = ""
                for message in messages.messages:
                    for content in message.content:
                        if content.type == "text":
                            text += content.text

                # Check for text-only samples (no images)
                if not visual or len(visual) == 0:
                    text_only_count += 1
                    if self.skip_text_only:
                        skipped_text_only += 1
                        # Add empty placeholder answer for this skipped sample
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[idx]), "")
                        pbar.update(1)
                        continue
                    else:
                        # If not skipping, we still can't process it (EMU3.5 requires images)
                        # So we add empty answer
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[idx]), "")
                        pbar.update(1)
                        continue

                # Check for multi-image samples (more than 1 image)
                if len(visual) > 1:
                    multi_image_count += 1
                    if self.skip_multi_image:
                        skipped_multi_image += 1
                        # Add empty placeholder answer for this skipped sample
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[idx]), "")
                        pbar.update(1)
                        continue
                    else:
                        # If not skipping, take only the first image
                        sample_data.append({"text": text, "image": visual[0], "context": ctx[idx]})
                else:
                    # Exactly 1 image - process normally
                    sample_data.append({"text": text, "image": visual[0], "context": ctx[idx]})

            # If all samples in batch were skipped, continue to next batch
            if len(sample_data) == 0:
                continue

            gen_kwargs = all_gen_kwargs[0]

            # Prepare inputs for EMU3.5 processor
            texts = [item["text"] for item in sample_data]
            images = [item["image"] for item in sample_data]

            # Convert image URLs to PIL Images if needed
            loaded_images = []
            for img in images:
                if isinstance(img, str):
                    loaded_images.append(Image.open(img))
                else:
                    loaded_images.append(img)

            # Process inputs for EMU3.5 - for EMU3.5 we only perform sequential processing for now
            inputs = self.processor(
                text=texts,
                image=loaded_images if loaded_images else None,
                return_tensors="pt",
                padding="longest",
            )

            # Move to device
            if self.device_map == "auto":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate conf with default values from EMU3.5 text sampling
            generation_config = GenerationConfig(
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                temperature=gen_kwargs.get("temperature", 1.0),
                do_sample=gen_kwargs.get("do_sample", False),
                top_k=gen_kwargs.get("top_k", 1024),
                top_p=gen_kwargs.get("top_p", 0.9),
                num_beams=gen_kwargs.get("num_beams", 1),
                num_return_sequences=gen_kwargs.get("num_beam_groups", 1),
                use_cache=self.use_cache,
            )

            # Filter inputs to only include keys accepted by model.generate()
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }

            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, generation_config=generation_config)

            # Trim input_ids from outputs
            outputs_trimmed = outputs[:, model_inputs["input_ids"].shape[-1] :]
            answers = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            for ans, item in zip(answers, sample_data):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (item["context"], gen_kwargs), ans)
                pbar.update(1)

                # Debug sample output (only on rank 0 to avoid duplicates in multi-GPU)
                if self.debug_samples and self._debug_samples_printed < self.num_debug_samples and self.rank == 0:
                    self._debug_samples_printed += 1
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"DEBUG SAMPLE {self._debug_samples_printed}/{self.num_debug_samples}")
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"PROMPT: {item['text']}")
                    eval_logger.info(f"ANSWER: {ans}")
                    eval_logger.info("=" * 80)

                eval_logger.debug(f"Question: {item['text']}")
                eval_logger.debug(f"Model Response: {ans}")

        # Reorder results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()

        # Print statistics at the end (warning mode)
        if self.rank == 0:  # Only print from main process
            eval_logger.warning(
                f"EMU3.5 Statistics: Found {text_only_count}/{total_samples} "
                f"text-only samples (no images). "
                f"Skipped: {skipped_text_only} "
                f"(skip_text_only={self.skip_text_only})"
            )
            eval_logger.warning(
                f"EMU3.5 Statistics: Found {multi_image_count}/{total_samples} "
                f"multi-image samples (>1 image). "
                f"Skipped: {skipped_multi_image} "
                f"(skip_multi_image={self.skip_multi_image})"
            )
            if text_only_count == 0 and multi_image_count == 0:
                eval_logger.info(
                    f"EMU3.5 Statistics: All {total_samples} samples had exactly 1 image. "
                    "No text-only or multi-image samples encountered."
                )

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for EMU3.5")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for EMU3.5")
