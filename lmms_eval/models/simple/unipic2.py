import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


# Add UniPic repository to Python path
# Expected: lmms-eval/UniPic/ directory at project root
wd = Path(__file__).parent.parent.parent.parent.resolve()
unipic_path = os.path.join(str(wd), "UniPic-2")
if os.path.exists(unipic_path):
    sys.path.insert(0, unipic_path)
    eval_logger.info(f"Added UniPic-2 path to sys.path: {unipic_path}")
else:
    # Try alternative path
    unipic_path_alt = os.path.join(str(wd), "UniPic", "UniPic-2")
    if os.path.exists(unipic_path_alt):
        sys.path.insert(0, unipic_path_alt)
        eval_logger.info(f"Added UniPic-2 path to sys.path: {unipic_path_alt}")
    else:
        eval_logger.warning(
            f"UniPic-2 repository not found at {unipic_path}. "
            f"Please clone it: cd {wd} && git clone https://github.com/SkyworkAI/UniPic.git"
        )


def fix_longer_edge(image: Image.Image, image_size: int = 512) -> Image.Image:
    """Resize image so the longer edge equals image_size while maintaining aspect ratio."""
    width, height = image.size
    if width > height:
        new_width = image_size
        new_height = int(height * image_size / width)
    else:
        new_height = image_size
        new_width = int(width * image_size / height)
    return image.resize((new_width, new_height), Image.LANCZOS)


@register_model("unipic2")
class UniPic2(lmms):
    """
    UniPic2-Metaquery Model
    https://huggingface.co/Skywork/UniPic2-Metaquery-9B

    UniPic2 is a unified multimodal model built on Qwen2.5-VL-Instruct and SD3.5-Medium.
    It supports image understanding, text-to-image generation, and image editing.

    This implementation focuses on image understanding for evaluation tasks.

    Example usage:
        python -m lmms_eval --model unipic2 \
            --model_args pretrained=../models/UniPic2-Metaquery-9B,lmm_model=Qwen/Qwen2.5-VL-7B-Instruct \
            --tasks mme,mmmu_val --batch_size 1 --device cuda:0

    Prerequisites:
        1. Clone UniPic repository: git clone https://github.com/SkyworkAI/UniPic.git
        2. Install requirements: pip install -r UniPic-2/requirements.txt
        3. Download local model to ../models/UniPic2-Metaquery-9B
    """

    def __init__(
        self,
        pretrained: str = "../models/UniPic2-Metaquery-9B",
        lmm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        max_new_tokens: int = 1024,
        attn_implementation: str = "flash_attention_2",
        image_size: int = 512,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        remove_system_prompt: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.lmm_model = lmm_model
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.remove_system_prompt = remove_system_prompt

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Determine dtype
        if dtype == "bfloat16" or dtype == "bf16":
            self._dtype = torch.bfloat16
        elif dtype == "float16" or dtype == "fp16":
            self._dtype = torch.float16
        elif dtype == "float32" or dtype == "fp32":
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

        # Load model components
        eval_logger.info(f"Loading UniPic2 model from {pretrained}")
        self._load_model(pretrained, lmm_model, attn_implementation, trust_remote_code)

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for UniPic2"

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
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(
                    self._model, evaluation_mode=True
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
            self._rank = 0
            self._world_size = 1

        eval_logger.info("UniPic2 model initialized successfully")

    def _load_model(
        self, pretrained: str, lmm_model: str, attn_implementation: str, trust_remote_code: bool
    ):
        """Load UniPic2 model components."""
        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLProcessor,
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers. Please install it: pip install transformers>=4.40.0\n"
                f"Error: {e}"
            )

        # For understanding tasks, we primarily use Qwen2.5-VL
        # Load the LMM (Language-Multimodal Model) component
        eval_logger.info(f"Loading Qwen2.5-VL model from {lmm_model}...")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            lmm_model,
            torch_dtype=self._dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=self._device,
        )

        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            lmm_model,
            trust_remote_code=trust_remote_code,
        )

        # Modify chat template as per UniPic2 requirements (optional)
        # Note: Removing system prompt may cause different results compared to standard Qwen2.5-VL
        if self.remove_system_prompt and self._processor.chat_template:
            self._processor.chat_template = self._processor.chat_template.replace(
                "{% if loop.first and message['role'] != 'system' %}"
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
                "",
            )

        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config

    @property
    def config(self):
        """Return the model config."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def processor(self):
        """Return the processor."""
        return self._processor

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
        """Compute log-likelihood (not implemented for UniPic2)."""
        raise NotImplementedError("Loglikelihood not implemented for UniPic2")

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
                gen_kwargs.pop("until")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Process each context
            assert len(contexts) == 1, "Batch size must be 1"
            context = contexts[0]

            # Prepare images
            images = []
            for visual in visuals:
                if isinstance(visual, str):
                    visual = Image.open(visual).convert("RGB")
                elif isinstance(visual, dict):
                    # Handle dict format - common in HuggingFace datasets
                    if "bytes" in visual:
                        from io import BytesIO
                        visual = Image.open(BytesIO(visual["bytes"])).convert("RGB")
                    elif "path" in visual:
                        visual = Image.open(visual["path"]).convert("RGB")
                    elif "image" in visual:
                        img = visual["image"]
                        if isinstance(img, str):
                            visual = Image.open(img).convert("RGB")
                        elif isinstance(img, Image.Image):
                            visual = img.convert("RGB")
                        else:
                            continue
                    else:
                        continue
                elif isinstance(visual, Image.Image):
                    visual = visual.convert("RGB")
                elif hasattr(visual, "convert"):
                    visual = visual.convert("RGB")
                else:
                    continue
                # Optionally resize image
                visual = fix_longer_edge(visual, self.image_size)
                images.append(visual)

            # Build conversation format for Qwen2.5-VL
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": context})

            messages = [{"role": "user", "content": content}]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            if images:
                inputs = self._processor(
                    text=[text],
                    images=images,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
            else:
                inputs = self._processor(
                    text=[text],
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)

            # Set generation parameters
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_p = gen_kwargs.get("top_p", None)
            num_beams = gen_kwargs.get("num_beams", 1)

            # Prepare generation kwargs
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "use_cache": self.use_cache,
            }

            # Add sampling parameters if temperature > 0
            if temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                if top_p is not None:
                    generate_kwargs["top_p"] = top_p
            else:
                generate_kwargs["do_sample"] = False

            if num_beams > 1:
                generate_kwargs["num_beams"] = num_beams

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            # Decode response - only the generated part
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Clean up to free memory
            del outputs, inputs
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
