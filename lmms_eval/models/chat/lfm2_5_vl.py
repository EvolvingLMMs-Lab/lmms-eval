"""LFM2.5-VL model wrapper for lmms-eval.

LFM2.5-VL (LiquidAI/LFM2.5-VL-1.6B) is a compact vision-language model
with 1.6B parameters, supporting dynamic-resolution image processing via
tiling and a chat template interface.

Example usage:

    accelerate launch --num_processes=1 -m lmms_eval \\
        --model lfm2_5_vl \\
        --model_args pretrained=LiquidAI/LFM2.5-VL-1.6B \\
        --tasks docvqa_val \\
        --batch_size 8
"""

import time
from typing import List, Optional, Union

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


class LFM2_5_VL(lmms):
    """
    LFM2.5-VL model wrapper.

    Supports batched inference via processor(padding=True).
    Image-text only (no video/audio).
    """

    is_simple = False

    def __init__(
        self,
        pretrained: str = "LiquidAI/LFM2.5-VL-1.6B",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        max_new_tokens: int = 1024,
        use_cache: bool = True,
        attn_implementation: str = "sdpa",
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            eval_logger.warning(f"Ignoring unsupported kwargs: {sorted(kwargs.keys())}")

        self.pretrained = pretrained
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self._max_new_tokens = max_new_tokens
        self._use_cache = use_cache

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        eval_logger.info(f"Loading LFM2.5-VL from {pretrained} ...")
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                pretrained,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            .to(self._device)
            .eval()
        )

        # Left padding for batched generation
        self.processor.tokenizer.padding_side = "left"
        self._tokenizer = self.processor.tokenizer
        self._config = self.model.config

        # Required properties
        self._eot_token_id = self._tokenizer.eos_token_id
        self._max_length = getattr(self._config, "max_position_embeddings", 4096)

        eval_logger.info(f"LFM2.5-VL loaded on {self._device}")

    # ------------------------------------------------------------------
    # Properties required by lmms-eval
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return getattr(self, "_rank", 0)

    @rank.setter
    def rank(self, value):
        self._rank = value

    @property
    def world_size(self):
        return getattr(self, "_world_size", 1)

    @world_size.setter
    def world_size(self, value):
        self._world_size = value

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tok_encode(self, string: str, left_truncate_len: Optional[int] = None, add_special_tokens: Optional[bool] = None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self._tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _process_messages_to_lfm_format(self, chat_messages: ChatMessages) -> tuple:
        """Convert ChatMessages to LFM's expected format.

        Returns:
            Tuple of (formatted_text, list_of_pil_images)
        """
        hf_messages = chat_messages.to_hf_messages()

        # Convert image URLs to PIL images
        pil_images = []
        for msg in hf_messages:
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    url = content["image"]
                    if isinstance(url, str):
                        pil_images.append(Image.open(url).convert("RGB"))
                    elif isinstance(url, Image.Image):
                        pil_images.append(url.convert("RGB"))

        return hf_messages, pil_images

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        """Generate answers for all requests with batched inference.

        Uses processor(padding=True) to batch multiple samples
        into a single forward pass.
        """
        res = []

        def _collate(x):
            return x[0], x[0]

        # Group by gen_kwargs so we don't mix e.g. greedy and sampling
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = all_gen_kwargs[0]

            # Build messages + images for each sample in the batch
            batch_hf_messages = []
            batch_images = []
            valid_indices = []

            for i, ids in enumerate(doc_id):
                doc = self.task_dict[task][split][ids]
                raw_messages = doc_to_messages[i](doc)
                chat_messages = ChatMessages(messages=raw_messages)
                hf_messages, pil_images = self._process_messages_to_lfm_format(chat_messages)

                batch_hf_messages.append(hf_messages)
                batch_images.append(pil_images[0] if pil_images else None)
                valid_indices.append(i)

            if not valid_indices:
                continue

            # Apply chat template to get text prompts for all samples
            batch_texts = [
                self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_hf_messages
            ]

            # Filter out samples with no image
            valid_texts = [batch_texts[i] for i in valid_indices if batch_images[i] is not None]
            # Processor expects images as list-of-lists: one list per sample
            valid_imgs = [[batch_images[i]] for i in valid_indices if batch_images[i] is not None]

            if not valid_texts:
                continue

            # Batch process: single forward pass with padding
            inputs = self.processor(
                images=valid_imgs,
                text=valid_texts,
                padding=True,
                return_tensors="pt",
            ).to(dtype=torch.bfloat16, device=self._device)

            # Set generation kwargs
            max_new_tokens = gen_kwargs.get("max_new_tokens", self._max_new_tokens)
            temperature = gen_kwargs.get("temperature", 0)
            do_sample = temperature > 0
            top_p = gen_kwargs.get("top_p", None)
            num_beams = gen_kwargs.get("num_beams", 1)

            start_time = time.time()
            try:
                cont = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=self._use_cache,
                    pad_token_id=self._tokenizer.pad_token_id or self._eot_token_id,
                )
            except Exception as e:
                eval_logger.error(f"Error generating batch: {e}")
                cont = inputs["input_ids"]
            end_time = time.time()

            # Decode: strip input tokens from generated tokens
            input_length = inputs["input_ids"].shape[-1]
            cont = cont[:, input_length:]

            decoded = self.processor.batch_decode(cont, skip_special_tokens=True)

            total_elapsed_time += end_time - start_time

            for i, text_output in enumerate(decoded):
                token_count = len(cont[i]) if cont is not None else 0
                total_tokens += token_count
                res.append(
                    GenerationResult(
                        text=text_output,
                        token_counts=TokenCounts(output_tokens=token_count) if token_count > 0 else None,
                    )
                )
                self.cache_hook.add_partial("generate_until", (batch_texts[i], gen_kwargs), text_output)
                pbar.update(1)

        res = re_ords.get_original(res)

        # Log metrics
        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0,
            "additional_metrics": {"rank": self.rank},
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    # ------------------------------------------------------------------
    # Unused abstract methods
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: List[Instance]) -> List[tuple]:
        """Not implemented — LFM2.5-VL is generate-only for now."""
        raise NotImplementedError("LFM2.5-VL does not support loglikelihood scoring")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Not implemented — single-turn only."""
        raise NotImplementedError("LFM2.5-VL does not support multi-round generation")
