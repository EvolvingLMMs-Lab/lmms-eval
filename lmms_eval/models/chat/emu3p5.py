"""
EMU3.5 Chat Model using EMU3p5EncoderBaseModel.

EMU3.5 Multimodal Model (34B parameters).
Uses IBQ vision tokenizer for improved image understanding.
https://github.com/baaivision/Emu3.5

This implementation uses direct processing without chat templates.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

# Import Emu3 classes (sys.path already set up by base class)
from emu3p5 import Emu3Config, Emu3ForCausalLM
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.emu3p5_encoder_base_model import EMU3p5EncoderBaseModel
from lmms_eval.models.model_utils.emu3p5.download_utils import ensure_local_weights
from lmms_eval.models.model_utils.emu3p5.emu3p5_tokenizer_loader import (
    load_emu3p5_tokenizer,
)
from lmms_eval.protocol import ChatMessages

# Path to EMU3.5 tokenizer directory
_current_file = Path(__file__).resolve()
_repo_root = _current_file.parents[3]  # Go up to lmms-eval root
_txt_tok_path = _repo_root / "external" / "Emu3.5" / "src" / "tokenizer_emu3_ibq"


@register_model("emu3p5")
class EMU3_5(EMU3p5EncoderBaseModel):
    """
    EMU3.5 Multimodal Model (34B parameters).

    Uses direct processing without chat templates.
    Inherits infrastructure from EMU3p5EncoderBaseModel.
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
        vision_tokenizer_dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        emu_min_pixels: int = 256 * 256,
        emu_max_pixels: int = 1400 * 1400,
        skip_text_only: bool = True,
        skip_multi_image: bool = True,
        debug_samples: bool = False,
        num_debug_samples: int = 5,
        **kwargs,
    ):
        # Store settings for use in abstract methods
        self._trust_remote_code = trust_remote_code
        self._pretrained = pretrained

        # Call parent constructor with mapped parameters
        super().__init__(
            model_descriptor=pretrained,
            tokenizer_path=str(_txt_tok_path),
            vq_hub=vq_hub,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            vision_tokenizer_dtype=vision_tokenizer_dtype,
            use_cache=use_cache,
            emu_min_pixels=emu_min_pixels,
            emu_max_pixels=emu_max_pixels,
            skip_text_only=skip_text_only,
            skip_multi_image=skip_multi_image,
            debug_samples=debug_samples,
            num_debug_samples=num_debug_samples,
            **kwargs,
        )

    def _load_tokenizer(self, tokenizer_path: str, **kwargs):
        """Load EMU3.5 text tokenizer with fallback handling."""
        return load_emu3p5_tokenizer(
            tokenizer_path,
            trust_remote_code=self._trust_remote_code,
            padding_side="left",
        )

    def _load_llm(self, model_path: str, **kwargs) -> Emu3ForCausalLM:
        """Load EMU3.5 causal language model with Emu3Config."""
        # Ensure main model weights are available locally
        model_path = ensure_local_weights(model_path, "BAAI/Emu3.5", accelerator=self.accelerator)

        # Extract kwargs for model loading
        trust_remote_code = kwargs.pop("trust_remote_code", self._trust_remote_code)

        # Load config
        model_config = Emu3Config.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        # Load model
        return Emu3ForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            **kwargs,
        )

    @property
    def image_placeholder(self) -> str:
        """Image placeholder token (not used in direct processing)."""
        return "<|image|>"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses using direct processing."""
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
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
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
            sample_data = []
            for idx, messages in enumerate(chat_messages):
                total_samples += 1
                visual, video, audio = messages.extract_media()

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
                        # EMU3.5 requires images - add empty answer
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

            # Process inputs for EMU3.5
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
                "attention_mask": inputs["attention_mask"],
            }

            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, generation_config=generation_config)

            # Trim input_ids from outputs
            outputs_trimmed = outputs[:, model_inputs["input_ids"].shape[-1] :]
            answers = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            # Decode with special tokens for debugging
            if self.debug_samples:
                prompts_with_tokens = self.processor.batch_decode(model_inputs["input_ids"], skip_special_tokens=False)
                answers_with_tokens = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=False)

            for i, (ans, item, text) in enumerate(zip(answers, sample_data, texts)):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (item["context"], gen_kwargs), ans)
                pbar.update(1)

                # Debug sample output (only on rank 0 to avoid duplicates)
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

        # Reorder results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()

        # Print statistics at the end (warning mode)
        if self.rank == 0:  # Only print from main process
            eval_logger.warning(f"EMU3.5 Statistics: Found {text_only_count}/{total_samples} " f"text-only samples (no images). " f"Skipped: {skipped_text_only} " f"(skip_text_only={self.skip_text_only})")
            eval_logger.warning(f"EMU3.5 Statistics: Found {multi_image_count}/{total_samples} " f"multi-image samples (>1 image). " f"Skipped: {skipped_multi_image} " f"(skip_multi_image={self.skip_multi_image})")
            if text_only_count == 0 and multi_image_count == 0:
                eval_logger.info(f"EMU3.5 Statistics: All {total_samples} samples had exactly 1 " "image. No text-only or multi-image samples encountered.")

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for EMU3.5")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for EMU3.5")
