import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add TokenFlow repository to Python path
# Expected: workspace has TokenFlow/ directory at the same level as lmms-eval/
wd = Path(__file__).parent.parent.parent.parent.resolve()  # lmms-eval/
tokenflow_i2t_path = (wd.parent / "TokenFlow" / "i2t").as_posix()
if os.path.exists(tokenflow_i2t_path):
    sys.path.append(tokenflow_i2t_path)
    eval_logger.info(f"Added TokenFlow path to sys.path: {tokenflow_i2t_path}")
else:
    eval_logger.warning(
        f"TokenFlow repository not found at {tokenflow_i2t_path}. "
        f"Please ensure TokenFlow is available in the workspace."
    )


@register_model("tokenflow")
class TokenFlow(lmms):
    """
    TokenFlow multimodal understanding model (image + text -> text).

    Example usage:
    accelerate launch -m lmms_eval \
        --model tokenflow \
        --model_args pretrained=/path/to/Tokenflow-llava-qwen2.5-14B-finetuning,tokenizer_path=/path/to/TokenFlow \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        conv_template: str = "qwen_2_5",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attn: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            eval_logger.warning(f"Unused kwargs in TokenFlow: {list(kwargs.keys())}")

        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot use both load_in_4bit and load_in_8bit")

        self.pretrained = pretrained
        self.tokenizer_path = tokenizer_path
        self.conv_template = conv_template
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_flash_attn = use_flash_attn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.seed = seed
        self.continual_mode = continual_mode

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/tokenflow_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, "tokenflow_response.json")
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup accelerator (only if in distributed mode)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.device_map = f"cuda:{accelerator.local_process_index}"
            else:
                self._device = torch.device(device)
                self.device_map = device_map
            self.accelerator = accelerator
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            # Single GPU mode - no accelerator needed
            self._device = torch.device(device)
            self.device_map = device_map
            self.accelerator = None
            self._rank = 0
            self._world_size = 1

        # Import TokenFlow (LLaVA) dependencies
        try:
            from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from llava.conversation import conv_templates
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.model.builder import load_pretrained_model
            from transformers import AutoTokenizer

            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.conv_templates = conv_templates
            self.get_model_name_from_path = get_model_name_from_path
            self.process_images = process_images
            self.tokenizer_image_token = tokenizer_image_token
            self.load_pretrained_model = load_pretrained_model
            self.AutoTokenizer = AutoTokenizer
        except Exception as e:
            raise ImportError(
                "Failed to import TokenFlow (LLaVA) dependencies. "
                "Please ensure TokenFlow/i2t is in sys.path. "
                f"Error: {e}"
            )

        eval_logger.info(f"Loading TokenFlow model from {self.pretrained}")
        self._load_model()
        eval_logger.info("TokenFlow model initialized successfully")

    def _load_model(self):
        model_name = self.get_model_name_from_path(self.pretrained)
        
        # TokenFlow understanding model uses tokenflow_siglip_32k.pt (not the enhanced version)
        # See: https://huggingface.co/ByteVisionLab/TokenFlow
        overwrite_config = None
        if self.tokenizer_path is not None:
            import os
            # Look for the correct vision tower checkpoint for understanding tasks
            vision_tower_path = os.path.join(self.tokenizer_path, "tokenflow_siglip_32k.pt")
            if os.path.exists(vision_tower_path):
                overwrite_config = {"mm_vision_tower": vision_tower_path}
                eval_logger.info(f"Setting vision_tower to {vision_tower_path}")
            else:
                eval_logger.warning(f"Vision tower not found at {vision_tower_path}")
        
        self._tokenizer, self._model, self._image_processor, self._max_length = self.load_pretrained_model(
            self.pretrained,
            None,
            model_name,
            load_8bit=self.load_in_8bit,
            load_4bit=self.load_in_4bit,
            device_map=self.device_map,
            device=str(self._device),
            use_flash_attn=self.use_flash_attn,
            overwrite_config=overwrite_config,
        )

        # Override tokenizer if specified
        if self.tokenizer_path is not None:
            try:
                tokenizer = self.AutoTokenizer.from_pretrained(self.tokenizer_path)
                # Add special tokens based on model config
                from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

                mm_use_im_start_end = getattr(self._model.config, "mm_use_im_start_end", False)
                mm_use_im_patch_token = getattr(self._model.config, "mm_use_im_patch_token", True)
                if mm_use_im_patch_token:
                    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
                if mm_use_im_start_end:
                    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
                self._model.resize_token_embeddings(len(tokenizer))
                self._tokenizer = tokenizer
                eval_logger.info(f"Loaded tokenizer from {self.tokenizer_path}")
            except Exception as e:
                eval_logger.warning(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")

        self._config = self._model.config
        self._model.eval()

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    def set_seed(self, seed: int):
        if seed > 0:
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def flatten(self, input_list):
        if not input_list or any(i is None for i in input_list):
            return []
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _build_prompt(self, context: str, image_count: int) -> str:
        if image_count > 0 and self.DEFAULT_IMAGE_TOKEN not in context:
            image_tokens = " ".join([self.DEFAULT_IMAGE_TOKEN] * image_count)
            question = image_tokens + "\n" + context
        else:
            question = context
        if "llama_3" in self.conv_template:
            conv = self.conv_templates[self.conv_template].copy()
        else:
            conv = self.conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="TokenFlow Generating")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            if doc_to_visual is None:
                eval_logger.warning(f"No image provided for TokenFlow understanding, doc_id={doc_id}")
                res.append("")
                pbar.update(1)
                continue

            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)]
            visuals = self.flatten(visuals)

            # CRITICAL DEBUG: Force print to verify image loading
            print(f"[TOKENFLOW DEBUG] doc_id={doc_id}, task={task}")
            print(f"[TOKENFLOW DEBUG] visuals count: {len(visuals)}")
            if visuals:
                for i, img in enumerate(visuals):
                    print(f"[TOKENFLOW DEBUG] Image {i}: size={img.size}, mode={img.mode}")
            else:
                print(f"[TOKENFLOW DEBUG] WARNING: No visuals loaded!")

            if not visuals or len(visuals) == 0:
                eval_logger.warning(f"No visual data found for doc_id={doc_id}, task={task}")
                res.append("")
                pbar.update(1)
                continue

            # Debug: Log successful image loading
            eval_logger.debug(f"Loaded {len(visuals)} images for doc_id={doc_id}, task={task}, sizes={[v.size for v in visuals]}")

            self.set_seed(self.seed)
            prompt = contexts
            prompt_question = self._build_prompt(prompt, len(visuals))

            # Debug: Log prompt details
            eval_logger.debug(f"Prompt length: {len(prompt_question)}, contains IMAGE_TOKEN: {self.DEFAULT_IMAGE_TOKEN in prompt_question}")
            eval_logger.debug(f"First 200 chars of prompt: {prompt_question[:200]}")

            image_tensor = self.process_images(visuals, self._image_processor, self._config)
            eval_logger.debug(f"Image tensor type: {type(image_tensor)}, shape: {image_tensor.shape if hasattr(image_tensor, 'shape') else 'N/A'}")

            # CRITICAL DEBUG: Force print to verify image tensor
            if isinstance(image_tensor, list):
                print(f"[TOKENFLOW DEBUG] Image tensor is list, length: {len(image_tensor)}")
                if len(image_tensor) > 0:
                    print(f"[TOKENFLOW DEBUG] First tensor shape: {image_tensor[0].shape}")
            else:
                print(f"[TOKENFLOW DEBUG] Image tensor shape: {image_tensor.shape}")

            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

            input_ids = self.tokenizer_image_token(
                prompt_question,
                self.tokenizer,
                self.IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
            temperature = gen_kwargs.get("temperature", self.temperature)
            top_p = gen_kwargs.get("top_p", self.top_p)
            num_beams = gen_kwargs.get("num_beams", self.num_beams)

            try:
                # CRITICAL DEBUG: Log generation parameters
                print(f"[TOKENFLOW DEBUG] Calling model.generate:")
                print(f"[TOKENFLOW DEBUG]   - input_ids shape: {input_ids.shape}")
                print(f"[TOKENFLOW DEBUG]   - images type: {type(image_tensor)}")
                print(f"[TOKENFLOW DEBUG]   - image_sizes: {[img.size for img in visuals]}")
                eval_logger.debug(f"Calling model.generate with:")
                eval_logger.debug(f"  - input_ids shape: {input_ids.shape}")
                eval_logger.debug(f"  - image_tensor shape: {image_tensor.shape if hasattr(image_tensor, 'shape') else [t.shape for t in image_tensor]}")
                eval_logger.debug(f"  - image_sizes: {[img.size for img in visuals]}")
                eval_logger.debug(f"  - max_new_tokens: {max_new_tokens}")

                cont = self.model.generate(
                    input_ids,
                    pad_token_id=pad_token_id,
                    images=image_tensor,
                    image_sizes=[img.size for img in visuals],
                    do_sample=True if temperature and temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                output_text = text_outputs[0] if text_outputs else ""
                eval_logger.debug(f"Generated output length: {len(output_text)} chars")
            except Exception as e:
                eval_logger.error(f"Error {e} in TokenFlow generation")
                output_text = ""

            res.append(output_text)

            # Save sample output (independent of continual_mode)
            # This ensures samples are always saved for analysis
            sample_output = {
                "doc_uuid": doc_uuid,
                "task": task,
                "doc_id": doc_id,
                "output": output_text,
                "prompt_length": len(prompt)
            }

            # Update cache if continual_mode is enabled
            if self.continual_mode:
                self.response_cache[doc_uuid] = output_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
            else:
                # Save to a separate samples file when continual_mode is disabled
                samples_file = os.path.join(self.response_persistent_folder, "tokenflow_samples.jsonl")
                os.makedirs(self.response_persistent_folder, exist_ok=True)
                with open(samples_file, "a") as f:
                    f.write(json.dumps(sample_output, ensure_ascii=False) + "\n")

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TokenFlow is a generation model and does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")