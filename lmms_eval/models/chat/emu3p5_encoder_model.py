"""
Chat model base class using EMU3.5 IBQ Vision Tokenizer with chat templates.

Inheriting models overwrite:
- _load_llm:               Load the language model
- _load_tokenizer:         Load the text tokenizer
- _load_vision_tokenizer:  Load the IBQ vision tokenizer
- _chat_transform:         Optional: Transform HF messages before chat template
- image_placeholder:       Property defining image placeholder token
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
from lmms_eval.models.emu3p5_encoder_base_model import EMU3p5EncoderBaseModel
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


class EMU3p5EncoderModel(EMU3p5EncoderBaseModel):
    """
    Chat-specific EMU3.5 encoder model.

    Handles ChatMessages and applies chat templates for instruction-tuned
    models. Inherits all vision processing logic from EMU3p5EncoderBaseModel.

    Subclasses must implement:
    - _load_llm: Load the language model
    - _load_tokenizer: Load the text tokenizer
    - _load_vision_tokenizer: Load the IBQ vision tokenizer
    - image_placeholder: Property defining image placeholder token

    Optionally override:
    - _chat_transform: Transform HF messages before chat template
    """

    is_simple = False  # Chat model

    def _chat_transform(self, hf_messages: list[dict]) -> list[dict]:
        """
        Optional: Transform HF messages before applying chat template.

        Override this to customize message format for your model.
        Default implementation returns messages unchanged.

        Args:
            hf_messages: List of HF-formatted message dicts

        Returns:
            Transformed message dicts
        """
        return hf_messages

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for chat model using chat template."""
        res = []

        # Initialize statistics counters
        text_only_count = 0
        multi_image_count = 0
        total_samples = 0
        skipped_text_only = 0
        skipped_multi_image = 0

        # Collate helper
        def _collate(x):
            return x[0], x[0]

        # Group requests by generation_kwargs
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0.0
        total_tokens = 0

        # Iterate through batches
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)

            # Get chat messages from dataset
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            # Extract media and prepare batch
            batch_data = []

            for idx, chat_message in enumerate(chat_messages):
                total_samples += 1

                # Extract media
                visual, _, _ = chat_message.extract_media()

                # Check for text-only samples
                if not visual or len(visual) == 0:
                    text_only_count += 1
                    if self.skip_text_only:
                        skipped_text_only += 1
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[idx]), "")
                        pbar.update(1)
                        continue
                    else:
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[idx]), "")
                        pbar.update(1)
                        continue

                # Check for multi-image samples
                if len(visual) > 1:
                    multi_image_count += 1
                    if self.skip_multi_image:
                        skipped_multi_image += 1
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (ctx[idx], all_gen_kwargs[idx]), "")
                        pbar.update(1)
                        continue
                    # else: process all images (multi-image supported)

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

                batch_data.append(
                    {
                        "messages": transformed_messages,
                        "images": pil_images,
                        "context": ctx[idx],
                    }
                )

            # Skip if all samples filtered
            if len(batch_data) == 0:
                continue

            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template with image placeholder
            messages_list = [item["messages"] for item in batch_data]
            texts = [self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_list]

            # Prepare images list
            images_list = [item["images"] for item in batch_data]

            # Encode images and inject vision tokens
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

            # Filter inputs for model.generate()
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
            answers = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in outputs_trimmed)

            # Decode with special tokens for debugging
            if self.debug_samples:
                prompts_with_tokens = self.processor.batch_decode(model_inputs["input_ids"], skip_special_tokens=False)
                answers_with_tokens = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=False)

            for i, (ans, item, text) in enumerate(zip(answers, batch_data, texts)):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (item["context"], gen_kwargs), ans)
                pbar.update(1)

                # Debug sample output
                if self.debug_samples and self._debug_samples_printed < self.num_debug_samples and self.rank == 0:
                    self._debug_samples_printed += 1
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"DEBUG SAMPLE {self._debug_samples_printed}/" f"{self.num_debug_samples}")
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"PROMPT (clean): {text}")
                    eval_logger.info(f"PROMPT (with tokens): {prompts_with_tokens[i]}")
                    eval_logger.info(f"ANSWER (clean): {ans}")
                    eval_logger.info(f"ANSWER (with tokens): {answers_with_tokens[i]}")
                    eval_logger.info("=" * 80)

                eval_logger.debug(f"Question: {text}")
                eval_logger.debug(f"Model Response: {ans}")

        # Reorder results
        res = re_ords.get_original(res)
        pbar.close()

        # Print statistics
        if self.rank == 0:
            eval_logger.warning(f"EMU3.5 Statistics: Found {text_only_count}/{total_samples} " f"text-only samples (no images). " f"Skipped: {skipped_text_only} " f"(skip_text_only={self.skip_text_only})")
            eval_logger.warning(f"EMU3.5 Statistics: Found {multi_image_count}/{total_samples} " f"multi-image samples (>1 image). " f"Skipped: {skipped_multi_image} " f"(skip_multi_image={self.skip_multi_image})")
            if text_only_count == 0 and multi_image_count == 0:
                eval_logger.info(f"EMU3.5 Statistics: All {total_samples} samples had exactly 1 " "image. No text-only or multi-image samples encountered.")

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
        """Loglikelihood not implemented for EMU3.5."""
        raise NotImplementedError("Loglikelihood not implemented for EMU3.5")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round generation not implemented for EMU3.5."""
        raise NotImplementedError("Multi-round generation not implemented for EMU3.5")
