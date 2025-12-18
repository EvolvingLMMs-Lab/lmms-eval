"""
Model meant to use as base-class to enable models that use early fusion with discrete EMU3 tokenizer.
"""
import abc
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages
from lmms_eval.models.model_utils.emu3.emu3_input_processor import Emu3Processor
from lmms_eval.models.model_utils.emu3.emu3_image_processor import Emu3VisionVQImageProcessor
from lmms_eval.models.model_utils.memory_utils import print_memory_stats


class EMU3EncoderModel(lmms):
    """
    Baseclass for models using EMU3 Vision Tokenizer to encode images to discrete tokens for input.
    Inheriting models just overwrite the
    - _load_llm:        Given a model path / descriptor loads the model and returns HF transformers model object
    - _load_tokenizer:  Given tokenizer path / descriptor loads the tokenizer and returns HF transformers tokenizer object
    - _chat_transform:  Given a benchmark message in hf chat format - transform into format that works for the models chat template.

    Currently, this baseclass assumes the images aee given with <Image> placeholder in the string.
    EMU3 Vision Encoder:  https://github.com/baaivision/Emu3
    """

    is_simple = False  # Chat model

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
        emu3_min_pixels: int = 512 * 512,
        emu3_max_pixels: int = 1024 * 1024,
        do_check_aspect_ratio: bool = False,
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

        # Load image processor (local modified) and image tokenizer
        eval_logger.info(f"Loading EMU3 Vision Img Preprocessor from {vq_hub}")
        image_processor = Emu3VisionVQImageProcessor.from_pretrained(vq_hub,
                                                                     trust_remote_code=trust_remote_code,
                                                                     min_pixels=emu3_min_pixels,
                                                                     max_pixels=emu3_max_pixels,
                                                                     do_check_aspect_ratio=do_check_aspect_ratio)
        eval_logger.info(f"Loading EMU3 Vision Tokenizer from {vq_hub}")
        image_tokenizer_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": trust_remote_code,
        }
        if image_tokenizer_dtype is not None:
            image_tokenizer_kwargs["torch_dtype"] = image_tokenizer_dtype
        image_tokenizer = AutoModel.from_pretrained(
            vq_hub, **image_tokenizer_kwargs
        ).eval()

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

        # Create EMU3 processor (after preparing image_tokenizer if multi-GPU)
        self.processor = Emu3Processor(image_processor, image_tokenizer, self._tokenizer)

        eval_logger.info(f"EMU3 model loaded successfully on rank {self.rank}/{self.world_size}")
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

    @abc.abstractmethod
    def _load_tokenizer(self, tokenizer_path, **kwargs):
        """
        Take path to tokenizer/tokenizer-descriptor and load tokenizer
        Returns Hf tokenizer object.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def _load_llm(self, model_path, **kwargs):
        """
        Take path to llm/llm-descriptor and load the model as Hf Model
        Returns Hf model object (supporting generate method)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abc.abstractmethod
    def image_placeholder(self) -> str:
        """
        Image placeholder token used in chat template.

        Examples: "<|image|>", "<image>", "[IMG]"

        Subclasses must define this to specify what placeholder their
        chat template uses for images.
        """
        raise NotImplementedError("Subclasses must define image_placeholder")

    def _chat_transform(self, hf_messages: list[dict]) -> list[dict]:
        """
        Optional: Transform HF messages before applying chat template.

        Override this to customize message format for your model.
        Default implementation returns messages unchanged.

        Args:
            hf_messages: List of HF-formatted message dicts with
                         structure: [{"role": "user", "content": [...]}]

        Returns:
            Transformed message dicts in same format
        """
        return hf_messages

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

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
        """Main Method used for generating benchmark results"""
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

        # iterate through batches (1 chunk = 1 batch)
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            # Get chat messages (read samples from dataset)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            # Extract media and text per messages
            batch_data = []

            # Iterate through samples in a batch (1 chat message = 1 sample) to prepare input to model
            for idx, chat_message in enumerate(chat_messages):
                total_samples += 1

                # Extract media
                visual, _, _ = chat_message.extract_media()

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
                        # If not skipping, we still can't process it (EMU3 requires images) -> add empty answer
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
                        visual = [visual[0]]

                # Convert to HF messages
                hf_messages = chat_message.to_hf_messages()

                # Apply subclass transformation hook
                transformed_messages = self._chat_transform(hf_messages)

                # Convert images to PIL if needed
                pil_images = []
                for img in visual:
                    if isinstance(img, str):
                        img = Image.open(img)
                    pil_images.append(img)

                batch_data.append({
                    "messages": transformed_messages,
                    "images": pil_images,
                    "context": ctx[idx]
                })

            # If all samples in batch were skipped, continue to next batch
            if len(batch_data) == 0:
                continue

            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template with image placeholder (one messages object per sample -> becomes string)
            messages_list = [item["messages"] for item in batch_data]
            texts = [
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for msgs in messages_list
            ]

            # Prepare images list (one list per text)
            images_list = [item["images"] for item in batch_data]

            # Processor preprocesses, encodes images and adds images to message strings (one string per sample)
            # TODO: supports only batch size 1?
            inputs = self.processor.encode_and_inject_vision_tokens(
                texts=texts,
                images=images_list,
                image_placeholder=self.image_placeholder,
                return_tensors="pt",
                padding="longest",
            )

            # Move to device
            if self.device_map == "auto":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Create generation configuration
            generation_config = GenerationConfig(
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                temperature=gen_kwargs.get("temperature", 0.0),
                do_sample=gen_kwargs.get("do_sample", False),
                top_k=gen_kwargs.get("top_k", None),
                top_p=gen_kwargs.get("top_p", None),
                num_beams=gen_kwargs.get("num_beams", 1),
                use_cache=self.use_cache,
            )

            # Filter inputs to only include keys accepted by model.generate()
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, generation_config=generation_config)

            # Trim input_ids from outputs
            outputs_trimmed = outputs[:, model_inputs["input_ids"].shape[-1] :]
            answers = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            for ans, item, text in zip(answers, batch_data, texts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (item["context"], gen_kwargs), ans)
                pbar.update(1)

                # Debug sample output (only on rank 0 to avoid duplicates in multi-GPU)
                if self.debug_samples and self._debug_samples_printed < self.num_debug_samples and self.rank == 0:
                    self._debug_samples_printed += 1
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"DEBUG SAMPLE {self._debug_samples_printed}/{self.num_debug_samples}")
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"PROMPT: {text}")
                    eval_logger.info(f"ANSWER: {ans}")
                    eval_logger.info("=" * 80)

                eval_logger.debug(f"Question: {text}")
                eval_logger.debug(f"Model Response: {ans}")

        # Reorder results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()

        # Print statistics at the end (warning mode)
        if self.rank == 0:  # Only print from main process
            eval_logger.warning(
                f"EMU3 Statistics: Found {text_only_count}/{total_samples} "
                f"text-only samples (no images). "
                f"Skipped: {skipped_text_only} "
                f"(skip_text_only={self.skip_text_only})"
            )
            eval_logger.warning(
                f"EMU3 Statistics: Found {multi_image_count}/{total_samples} "
                f"multi-image samples (>1 image). "
                f"Skipped: {skipped_multi_image} "
                f"(skip_multi_image={self.skip_multi_image})"
            )
            if text_only_count == 0 and multi_image_count == 0:
                eval_logger.info(
                    f"EMU3 Statistics: All {total_samples} samples had exactly 1 image. "
                    "No text-only or multi-image samples encountered."
                )

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for EMU3")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for EMU3")