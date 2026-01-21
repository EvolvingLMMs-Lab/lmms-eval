from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("ovis2_5")
class Ovis2_5(lmms):
    """
    Ovis2.5 Model
    https://huggingface.co/AIDC-AI/Ovis2.5-9B

    Ovis2.5 is a multimodal large language model supporting image understanding.
    It uses a NaViT vision encoder for variable resolution image processing.

    Example usage:
        python -m lmms_eval --model ovis2_5 \
            --model_args pretrained=AIDC-AI/Ovis2.5-9B \
            --tasks mme,mmmu_val --batch_size 1 --device cuda:0
    """

    def __init__(
        self,
        pretrained: str = "AIDC-AI/Ovis2.5-9B",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        device_map: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Determine device_map: use "auto" for multi-GPU, otherwise use specified device
        if device_map is not None:
            actual_device_map = device_map
        elif accelerator.num_processes > 1:
            actual_device_map = device
        else:
            actual_device_map = device

        # Load model
        eval_logger.info(f"Loading Ovis2.5 model from {pretrained}")
        eval_logger.info(f"Using device_map: {actual_device_map}")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map=actual_device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype if dtype != "auto" else torch.bfloat16,
        )

        # Ovis2.5 uses model.text_tokenizer instead of a separate tokenizer
        self._tokenizer = self._model.text_tokenizer  # type: ignore
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for Ovis2.5"
        self.use_cache = use_cache
        self.pretrained = pretrained

        # Setup distributed training
        if accelerator.num_processes > 1:
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert accelerator.distributed_type in distributed_type_list, (
                "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed supported"
            )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            # Only move to device if not using auto device_map
            if actual_device_map != "auto":
                self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

        # Store device_map for later use
        self._device_map = actual_device_map

    @property
    def config(self):
        """Return the model config."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def model(self):
        """Return the model, unwrapping it if using Accelerate."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        """Return the end of text token id."""
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        """Return the batch size."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return the device."""
        return self._device

    @property
    def rank(self):
        """Return the process rank."""
        return self._rank

    @property
    def world_size(self):
        """Return the world size."""
        return self._world_size

    def flatten(self, input_list):
        """Flatten a nested list."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood (not implemented for Ovis2.5)."""
        raise NotImplementedError("Loglikelihood not implemented for Ovis2.5")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until stopping criteria are met."""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        # Group requests by generation kwargs
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            # Get generation kwargs
            gen_kwargs = all_gen_kwargs[0]

            # Set default values
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Process each context
            assert len(contexts) == 1, "Batch size must be 1"
            context = contexts[0]

            # Prepare messages in Ovis2.5 format
            messages = [
                {
                    "role": "user",
                    "content": [],
                }
            ]

            # Add images to content
            for visual in visuals:
                if isinstance(visual, str):
                    # If visual is a path, load it
                    visual = Image.open(visual).convert("RGB")
                messages[0]["content"].append({"type": "image", "image": visual})

            # Remove <image> tokens from context if present
            # Some tasks (like geometry3k) include <image> tokens in the text
            # but Ovis2.5 uses messages format where images are separate
            clean_context = context.replace("<image>", "").strip()
            
            # Add text to content
            messages[0]["content"].append({"type": "text", "text": clean_context})

            # Preprocess inputs using model's built-in method
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
            )

            # Move to device - get correct device for auto device_map
            if self._device_map == "auto":
                # When using auto device_map, get the device from model's embedding
                target_device = self.model.get_llm().model.embed_tokens.weight.device
            else:
                target_device = self._device
            input_ids = input_ids.to(target_device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(target_device)
            if grid_thws is not None:
                grid_thws = grid_thws.to(target_device)

            # Set generation parameters
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # Prepare generation kwargs
            generate_kwargs = {
                "inputs": input_ids,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
            }

            # Add sampling parameters if temperature > 0
            if gen_kwargs["temperature"] > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = gen_kwargs["temperature"]
                if gen_kwargs["top_p"] is not None:
                    generate_kwargs["top_p"] = gen_kwargs["top_p"]
            else:
                generate_kwargs["do_sample"] = False

            if gen_kwargs["num_beams"] > 1:
                generate_kwargs["num_beams"] = gen_kwargs["num_beams"]

            # Generate response
            outputs = self.model.generate(**generate_kwargs)

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Debug: log first response to check if model is working
            if len(res) == 0:
                eval_logger.info(f"First response preview: {response[:500]}")

            # Clean up to free memory
            del outputs, input_ids, pixel_values, grid_thws
            torch.cuda.empty_cache()

            res.append(response)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), response
            )
            pbar.update(1)

        # Reorder results to original order
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate for multi-round conversations (not implemented)."""
        raise NotImplementedError("Multi-round generation not yet implemented")
