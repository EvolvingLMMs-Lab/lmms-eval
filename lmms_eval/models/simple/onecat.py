import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add OneCAT path to sys.path
wd = Path(__file__).parent.parent.parent.parent.resolve()
onecat_path = os.path.join(str(wd), "OneCAT")

# Try multiple possible locations for OneCAT repository
possible_paths = [
    onecat_path,  # /home/xinjiezhang/data/lei/lmms-eval/OneCAT
    os.path.join(str(wd.parent), "OneCAT"),  # /home/xinjiezhang/data/lei/OneCAT
]

onecat_found = False
for path in possible_paths:
    if os.path.exists(path):
        sys.path.append(path)
        eval_logger.info(f"Added OneCAT path to sys.path: {path}")
        onecat_found = True
        break

if not onecat_found:
    eval_logger.warning(
        f"OneCAT repository not found. Tried: {possible_paths}. "
        f"Please ensure it's in the correct location."
    )


@register_model("onecat")
class OneCAT(lmms):
    """
    OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation

    Supports visual understanding, text-to-image generation, and image editing.
    This integration focuses on visual understanding for evaluation tasks.

    Example usage:
    python -m lmms_eval \
        --model onecat \
        --model_args pretrained=/path/to/OneCAT-3B \
        --tasks illusionbench_arshia_icon_shape_test \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        max_new_tokens: int = 1000,
        do_sample: bool = False,
        num_beams: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        simple_output: bool = True,
        simple_output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.continual_mode = continual_mode
        self.device_str = device

        # Determine dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/onecat_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "onecat_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file) as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded cache: {len(self.response_cache)} records"
                )

        # Setup simple output
        self.simple_output = simple_output
        if simple_output_dir is None:
            self.simple_output_dir = "./logs/onecat_simple_output"
        else:
            self.simple_output_dir = simple_output_dir

        if self.simple_output:
            os.makedirs(self.simple_output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.simple_output_dir, "images"), exist_ok=True)
            self.simple_results = []

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Load model
        eval_logger.info(f"Loading OneCAT model from {pretrained}")
        self._load_model()

        eval_logger.info("OneCAT model initialized successfully")

    def _load_model(self):
        """Load OneCAT model and tokenizer"""
        try:
            from onecat.modeling_onecat import OneCatVLModel
            from onecat.smart_resize import smart_resize
            from onecat.util import build_transform

            self.OneCatVLModel = OneCatVLModel
            self.smart_resize = smart_resize
            self.build_transform = build_transform

        except Exception as e:
            raise ImportError(
                f"Failed to import OneCAT dependencies. "
                f"Please ensure:\n"
                f"  1. OneCAT repository is available at {onecat_path}\n"
                f"  2. Required dependencies are installed\n"
                f"Error: {e}"
            )

        # Load model
        self._model = self.OneCatVLModel.from_pretrained(self.pretrained)
        self._model = self._model.to(
            device=self.device_str, dtype=self.torch_dtype
        ).eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained)

        eval_logger.info(
            f"Model loaded with dtype={self.torch_dtype}, device={self.device_str}"
        )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def _load_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess image for OneCAT model

        Args:
            image: PIL Image

        Returns:
            Tuple of (pixel_values, pixel_values_thumbnail)
        """
        width, height = image.size

        # Smart resize
        resized_height, resized_width = self.smart_resize(height, width)
        transform = self.build_transform(input_size=(resized_height, resized_width))
        pixel_values = transform(image).unsqueeze(0)

        # Thumbnail (base size 448x448)
        transform_base = self.build_transform(input_size=(448, 448))
        pixel_values_thumbnail = transform_base(image).unsqueeze(0)

        return pixel_values, pixel_values_thumbnail

    def understand_image(self, prompt: str, image: Image.Image) -> str:
        """
        Understand image and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand

        Returns:
            Generated text answer
        """
        # Prepare image
        pixel_values, pixel_values_thumbnail = self._load_image(image)
        pixel_values = pixel_values.to(
            device=self.device_str, dtype=self.torch_dtype
        )
        pixel_values_thumbnail = pixel_values_thumbnail.to(
            device=self.device_str, dtype=self.torch_dtype
        )

        # Generation config
        generation_config = dict(
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Generate
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
            pixel_values_thumbnail=pixel_values_thumbnail,
            verbose=False,
        )

        return response

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="OneCAT Generating",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            prompt = contexts

            # Get image from doc_to_visual
            if doc_to_visual is None:
                eval_logger.warning(
                    f"No image provided for understanding mode, doc_id={doc_id}"
                )
                res.append("")
                pbar.update(1)
                continue

            # Get image from doc_to_visual
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if not visuals or len(visuals) == 0:
                eval_logger.warning(f"No visual data found for doc_id={doc_id}")
                res.append("")
                pbar.update(1)
                continue

            # Use first image for understanding
            image = visuals[0]
            output_text = self.understand_image(prompt, image)

            res.append(output_text)

            # Save simple output
            if self.simple_output:
                result_entry = {
                    "doc_id": str(doc_id),
                    "task": task,
                    "split": split,
                    "mode": "understanding",
                    "prompt": prompt,
                    "output": output_text
                }

                # Save input image
                image_filename = f"{doc_id}.jpg"
                image_path = os.path.join(self.simple_output_dir, "images", image_filename)
                image.save(image_path)
                result_entry["input_image"] = f"./images/{image_filename}"

                self.simple_results.append(result_entry)

                # Save results to JSON file
                results_file = os.path.join(self.simple_output_dir, "results.json")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(self.simple_results, f, ensure_ascii=False, indent=2)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = output_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(
                        self.response_cache, f, ensure_ascii=False, indent=2
                    )

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "OneCAT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation"
        )
