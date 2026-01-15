"""
Simple (non-chat) model base class using EMU3 Vision Tokenizer.

For base models without instruction tuning. Uses direct text prompts
instead of chat templates.
"""

import time
from typing import List, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.models.emu3_encoder_base_model import EMU3EncoderBaseModel
from lmms_eval.models.model_utils.gen_metrics import log_metrics


class EMU3SimpleModel(EMU3EncoderBaseModel):
    """
    Simple (non-chat) EMU3 encoder model base class.

    Works with plain text contexts instead of ChatMessages.
    For evaluating base models without instruction tuning.

    Inherits all vision processing logic from EMU3EncoderBaseModel.

    Subclasses must implement:
    - _load_llm: Load the language model
    - _load_tokenizer: Load the text tokenizer
    - image_placeholder: Property defining image placeholder token
    """

    is_simple = True  # Simple model

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for simple model with text prompts."""
        res = []

        # Initialize statistics
        text_only_count = 0
        multi_image_count = 0
        total_samples = 0

        # Collator for batching by context length
        def _collate(x):
            # x[0] is the context (text prompt)
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        # Group requests by generation_kwargs
        re_ords = utils.Collator([req.args for req in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0.0
        total_tokens = 0

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)

            # Extract visuals from dataset
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]

            # Build prompts with image placeholders
            prompts = []
            images_list = []
            batch_contexts = []  # Track contexts for non-skipped samples

            for context, visual_list in zip(contexts, visuals):
                total_samples += 1

                # Handle text-only samples
                if not visual_list or len(visual_list) == 0:
                    text_only_count += 1
                    if self.skip_text_only:
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                        pbar.update(1)
                        continue
                    else:
                        # Can't process without images
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                        pbar.update(1)
                        continue

                # Handle multi-image samples
                if len(visual_list) > 1:
                    multi_image_count += 1
                    if self.skip_multi_image:
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                        pbar.update(1)
                        continue
                    else:
                        # Take only first image
                        visual_list = [visual_list[0]]

                # Convert to PIL if needed
                pil_images = []
                for img in visual_list:
                    if isinstance(img, str):
                        img = Image.open(img)
                    pil_images.append(img)

                # Inject image placeholder
                image_tokens = [self.image_placeholder] * len(pil_images)
                prompt = " ".join(image_tokens) + "\n" + context
                prompts.append(prompt)
                images_list.append(pil_images)
                batch_contexts.append(context)  # Track this context

            # Skip if all filtered
            if len(prompts) == 0:
                continue

            # Encode images and inject vision tokens
            inputs = self.processor.encode_and_inject_vision_tokens(
                texts=prompts,
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

            model_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            start_time = time.time()
            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, generation_config=generation_config)
            end_time = time.time()

            # Trim input_ids from outputs
            outputs_trimmed = outputs[:, model_inputs["input_ids"].shape[-1] :]
            text_outputs = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in outputs_trimmed)

            # Decode with special tokens for debugging
            if self.debug_samples:
                prompts_with_tokens = self.processor.batch_decode(model_inputs["input_ids"], skip_special_tokens=False)
                answers_with_tokens = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=False)

            # Apply stopping sequences
            until = gen_kwargs.get("until", [])
            for i, output in enumerate(text_outputs):
                for term in until:
                    if term:
                        output = output.split(term)[0]
                text_outputs[i] = output

            for i, (ans, context) in enumerate(zip(text_outputs, batch_contexts)):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

                # Debug output
                if self.debug_samples and self._debug_samples_printed < self.num_debug_samples and self.rank == 0:
                    self._debug_samples_printed += 1
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"DEBUG SAMPLE {self._debug_samples_printed}/" f"{self.num_debug_samples}")
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"PROMPT (clean): {prompts[i]}")
                    eval_logger.info(f"PROMPT (with tokens): {prompts_with_tokens[i]}")
                    eval_logger.info(f"ANSWER (clean): {ans}")
                    eval_logger.info(f"ANSWER (with tokens): {answers_with_tokens[i]}")
                    eval_logger.info("=" * 80)

        # Reorder results
        res = re_ords.get_original(res)
        pbar.close()

        # Print statistics
        if self.rank == 0:
            eval_logger.warning(f"EMU3 Simple Statistics: Found {text_only_count}/" f"{total_samples} text-only samples (skipped: " f"{text_only_count if self.skip_text_only else 0})")
            eval_logger.warning(f"EMU3 Simple Statistics: Found {multi_image_count}/" f"{total_samples} multi-image samples (skipped: " f"{multi_image_count if self.skip_multi_image else 0})")

        # Log timing metrics
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Loglikelihood not implemented for vision models."""
        raise NotImplementedError("Loglikelihood not implemented for EMU3 simple")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round not implemented for simple models."""
        raise NotImplementedError("Multi-round not implemented for EMU3 simple")
