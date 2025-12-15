from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages
from lmms_eval.models.model_utils.emu3p5.emu3_5_image_processor import (
    Emu3_5VisionVQImageProcessor,
)
from lmms_eval.models.model_utils.emu3.emu3_input_processor import Emu3Processor
from lmms_eval.models.model_utils.emu3p5.emu3_tokenizer_loader import (
    load_emu3_tokenizer,
)



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
        use_cache: bool = True,
        emu3_min_pixels: int = 512 * 512,
        emu3_max_pixels: int = 1024 * 1024,
        do_check_aspect_ratio: bool = False,
        skip_text_only: bool = True,
        skip_multi_image: bool = True,
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
        eval_logger.info(f"Loading EMU3.5 model from {pretrained}")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map=device_map if accelerator.num_processes == 1 else None,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        ).eval()

        # Load tokenizer (with fallback handling for missing tokenization file)
        eval_logger.info(f"Loading EMU3.5 Text Tokenizer from {pretrained}")
        self._txt_tokenizer = load_emu3_tokenizer(
            pretrained, trust_remote_code=trust_remote_code, padding_side="left"
        )

        # Load image processor (local modified) and image tokenizer
        eval_logger.info(f"Loading EMU3.5 Vision Pre-Processor from {vq_hub}")
        image_processor = Emu3_5VisionVQImageProcessor.from_pretrained(
            vq_hub,
            trust_remote_code=trust_remote_code,
            min_pixels=emu3_min_pixels,
            max_pixels=emu3_max_pixels,
            do_check_aspect_ratio=do_check_aspect_ratio,
        )
        eval_logger.info(f"Loading EMU3.5 Vision Tokenizer from {vq_hub}")
        image_tokenizer = AutoModel.from_pretrained(vq_hub,
                                                    device_map=self.device_map,
                                                    trust_remote_code=trust_remote_code).eval()

        # Create EMU3.5 processor (Use same as EMU3 -> Same format)
        self.processor = Emu3Processor(image_processor, image_tokenizer, self._txt_tokenizer)

        # Set instance variables
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.skip_text_only = skip_text_only
        self.skip_multi_image = skip_multi_image

        # Prepare model for distributed training if needed
        if accelerator.num_processes > 1:
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"EMU3.5 model loaded successfully on rank {self.rank}/{self.world_size}")

    @property
    def model(self):
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

    def flatten(self, input):
        """Flatten nested lists for handling multiple images."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

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
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[0]), "")
                        pbar.update(1)
                        continue
                    else:
                        # If not skipping, we still can't process it (EMU3.5 requires images)
                        # So we add empty answer
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[0]), "")
                        pbar.update(1)
                        continue

                # Check for multi-image samples (more than 1 image)
                if len(visual) > 1:
                    multi_image_count += 1
                    if self.skip_multi_image:
                        skipped_multi_image += 1
                        # Add empty placeholder answer for this skipped sample
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[0]), "")
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
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    processed_images.append(Image.open(img))
                else:
                    processed_images.append(img)

            # Process inputs for EMU3.5
            inputs = self.processor(
                text=texts,
                image=processed_images if processed_images else None,
                mode="U",  # Understanding mode
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

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=generation_config)

            # Trim input_ids from outputs
            outputs_trimmed = outputs[:, inputs.input_ids.shape[-1] :]
            answers = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            for ans, item in zip(answers, sample_data):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (item["context"], gen_kwargs), ans)
                pbar.update(1)

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
