"""
OneCAT Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image and original image

Usage:
    python -m lmms_eval \
        --model onecat_visual_cot \
        --model_args pretrained=/path/to/OneCAT-3B,vae_path=/path/to/infinity_vae_d32reg.pth \
        --tasks illusionbench_arshia_icon_shape_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
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


@register_model("onecat_visual_cot")
class OneCATVisualCoT(lmms):
    """
    OneCAT Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using the generated image and original image
    """

    def __init__(
        self,
        pretrained: str,
        vae_path: str,
        # Stage 1: Image generation parameters
        stage1_t2i_stage: int = 3,
        stage1_h_div_w: float = 1.0,
        stage1_cfg: float = 20.0,
        stage1_top_k: int = 2,
        stage1_top_p: float = 0.97,
        stage1_max_input_tokens: int = 1024,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_do_sample: bool = False,
        stage2_num_beams: int = 1,
        stage2_top_k: Optional[int] = None,
        stage2_top_p: Optional[float] = None,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        simple_output: bool = True,
        simple_output_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        dtype: str = "bfloat16",
        device: str = "cuda",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.vae_path = vae_path
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template
        self.device_str = device

        # Stage 1 parameters
        self.stage1_t2i_stage = stage1_t2i_stage
        self.stage1_h_div_w = stage1_h_div_w
        self.stage1_cfg = stage1_cfg
        self.stage1_top_k = stage1_top_k
        self.stage1_top_p = stage1_top_p
        self.stage1_max_input_tokens = stage1_max_input_tokens

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_do_sample = stage2_do_sample
        self.stage2_num_beams = stage2_num_beams
        self.stage2_top_k = stage2_top_k
        self.stage2_top_p = stage2_top_p

        # Determine dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/onecat_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved under: {self.intermediate_dir}"
            )

        # Setup simple output
        self.simple_output = simple_output
        if simple_output_dir is None:
            self.simple_output_dir = "./logs/onecat_cot_simple_output"
        else:
            self.simple_output_dir = simple_output_dir

        if self.simple_output:
            os.makedirs(self.simple_output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.simple_output_dir, "generated_images"), exist_ok=True)
            self.simple_results = []

        # Setup rank and world size
        self._rank = 0
        self._world_size = 1

        # Load models
        eval_logger.info(f"Loading OneCAT model from {pretrained}")
        self._load_model()

        eval_logger.info("OneCATVisualCoT initialized successfully")

    def _load_model(self):
        """Load OneCAT model, tokenizer, and VAE"""
        try:
            from onecat.constants import (
                IMG_GEN_CONTEXT_TOKEN,
                IMG_GEN_START_TOKEN,
                SYSTEM_PROMPT,
            )
            from onecat.modeling_onecat import OneCatVLModel
            from onecat.smart_resize import smart_resize
            from onecat.util import build_transform
            from onecat.var_model.tools.run_infinity import load_visual_tokenizer

            self.IMG_GEN_CONTEXT_TOKEN = IMG_GEN_CONTEXT_TOKEN
            self.IMG_GEN_START_TOKEN = IMG_GEN_START_TOKEN
            self.SYSTEM_PROMPT = SYSTEM_PROMPT
            self.OneCatVLModel = OneCatVLModel
            self.smart_resize = smart_resize
            self.build_transform = build_transform
            self.load_visual_tokenizer = load_visual_tokenizer

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

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained)

        # Load VAE for image generation
        eval_logger.info(f"Loading Infinity VAE from {self.vae_path}")
        if self.stage1_t2i_stage == 2:
            pn = "0.25M"
        elif self.stage1_t2i_stage == 3:
            pn = "1M"
        else:
            raise ValueError(
                f"Expected t2i_stage 2 or 3. Got {self.stage1_t2i_stage}"
            )

        vae_args = argparse.Namespace(
            vae_type=32,
            vae_path=self.vae_path,
            apply_spatial_patchify=0,
            pn=pn,
        )
        vae_local = self.load_visual_tokenizer(vae_args)
        vae_local.eval()

        # Setup model for generation
        self._model.vae_local = vae_local
        self._model.vargpt_gen_args = vae_args
        img_gen_context_token_id = self.tokenizer.convert_tokens_to_ids(
            self.IMG_GEN_CONTEXT_TOKEN
        )
        img_gen_start_token_id = self.tokenizer.convert_tokens_to_ids(
            self.IMG_GEN_START_TOKEN
        )
        self._model.img_gen_context_token_id = img_gen_context_token_id
        self._model.img_gen_start_token_id = img_gen_start_token_id

        # Move model to device
        self._model = self._model.to(
            device=self.device_str, dtype=self.torch_dtype
        ).eval()

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
        """Load and preprocess image for OneCAT model"""
        width, height = image.size

        # Smart resize
        resized_height, resized_width = self.smart_resize(height, width)
        transform = self.build_transform(input_size=(resized_height, resized_width))
        pixel_values = transform(image).unsqueeze(0)

        # Thumbnail (base size 448x448)
        transform_base = self.build_transform(input_size=(448, 448))
        pixel_values_thumbnail = transform_base(image).unsqueeze(0)

        return pixel_values, pixel_values_thumbnail

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str
    ) -> List[str]:
        """
        Stage 1: Generate visualization image from prompt

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            List of image paths
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Prepare input
            system_message = self.SYSTEM_PROMPT
            user_message = f"<|im_start|>user\n{generation_prompt}<|im_end|>"
            assistant_message = "<|im_start|>assistant\n<img_gen>"

            batch = system_message + user_message + assistant_message
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.stage1_max_input_tokens,
                truncation=False,
                padding=False,
            )
            input_ids = model_inputs["input_ids"].to(self.device_str)
            attention_mask = model_inputs["attention_mask"].to(self.device_str)

            # CFG batch (empty prompt for classifier-free guidance)
            cfg_batch = (
                system_message
                + "<|im_start|>user\n<|im_end|><|im_start|>assistant\n<img_gen>"
            )
            model_inputs_cfg = self.tokenizer(
                cfg_batch,
                return_tensors="pt",
                max_length=self.stage1_max_input_tokens,
                truncation=False,
                padding=False,
            )
            input_ids_cfg = model_inputs_cfg["input_ids"].to(self.device_str)
            attention_mask_cfg = model_inputs_cfg["attention_mask"].to(
                self.device_str
            )

            # Generation config
            generation_config = dict(
                output_hidden_states=True,
                cfg=self.stage1_cfg,
                top_k=self.stage1_top_k,
                top_p=self.stage1_top_p,
                use_cache=True,
                return_dict=True,
                h_div_w=self.stage1_h_div_w,
            )

            # Generate image
            with torch.no_grad():
                img = self.model.generate_t2i(
                    input_ids=input_ids,
                    input_ids_cfg=input_ids_cfg,
                    attention_mask=attention_mask,
                    attention_mask_cfg=attention_mask_cfg,
                    **generation_config,
                )

            # Convert to PIL Image
            img_pil = Image.fromarray(
                img[0]
                .add_(1)
                .mul_(0.5)
                .to(dtype=torch.float32)
                .permute(1, 2, 0)
                .mul_(255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # Save image
            artifact_dir = os.path.join(self.intermediate_dir, task)
            os.makedirs(artifact_dir, exist_ok=True)
            image_path = os.path.join(
                artifact_dir, f"{doc_id}_stage1_generated.png"
            )
            img_pil.save(image_path)

            eval_logger.debug(f"Stage 1 - Generated image saved to {image_path}")
            return [image_path]

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return []
            else:
                raise

    def _stage2_answer_with_image(
        self,
        question: str,
        image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using generated image (and optionally original image)

        Args:
            question: Original question text
            image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (optional, used as primary reference)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # For OneCAT, we can pass both images by concatenating them
            # or use the auxiliary image as the primary input
            # Since OneCAT's chat method takes single pixel_values,
            # we'll use auxiliary image as primary and mention original in prompt

            if original_image is not None:
                eval_logger.debug(
                    "Stage 2 - Using auxiliary image (original referenced in prompt)"
                )
                # Use auxiliary image for visual input
                pixel_values, pixel_values_thumbnail = self._load_image(
                    auxiliary_image
                )
                # Enhance question to reference both images
                enhanced_question = (
                    f"{question}\n"
                    f"Note: An auxiliary visualization has been generated to help answer this question."
                )
            else:
                eval_logger.debug("Stage 2 - Using auxiliary image only")
                pixel_values, pixel_values_thumbnail = self._load_image(
                    auxiliary_image
                )
                enhanced_question = question

            pixel_values = pixel_values.to(
                device=self.device_str, dtype=self.torch_dtype
            )
            pixel_values_thumbnail = pixel_values_thumbnail.to(
                device=self.device_str, dtype=self.torch_dtype
            )

            # Generation config
            generation_config = dict(
                do_sample=self.stage2_do_sample,
                top_k=self.stage2_top_k,
                top_p=self.stage2_top_p,
                num_beams=self.stage2_num_beams,
                max_new_tokens=self.stage2_max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Generate answer
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=enhanced_question,
                generation_config=generation_config,
                pixel_values_thumbnail=pixel_values_thumbnail,
                verbose=False,
            )

            eval_logger.debug(f"Stage 2 - Generated answer: {response[:100]}...")
            return response

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            else:
                raise

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ) -> None:
        """Save intermediate artifacts for debugging"""
        if not self.save_intermediate:
            return

        artifact_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(artifact_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "task": task,
            "generation_prompt": generation_prompt,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Main inference method implementing two-stage visual CoT

        Stage 1: Generate visualization image from text prompt
        Stage 2: Answer question using the generated image
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="OneCATVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    # Get doc from task_dict
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                        eval_logger.debug(
                            f"Extracted original image for doc {doc_id}"
                        )
                except Exception as e:
                    eval_logger.warning(
                        f"Failed to extract original image for doc {doc_id}: {e}"
                    )

            # Parse contexts to extract generation_prompt if provided
            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
            )

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace(
                    "{question}", actual_question
                )
                # Extract just the question for stage 2
                contexts = actual_question
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image
            generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt, doc_id=doc_id, task=task
            )

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, returning empty answer"
                )
                res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer question using generated image (and original image if available)
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image,
            )

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                generated_images=generated_images,
                question=contexts,
                stage2_answer=final_answer,
            )

            # Save simple output if enabled
            if self.simple_output:
                result_entry = {
                    "doc_id": str(doc_id),
                    "task": task,
                    "split": split,
                    "mode": "visual_cot",
                    "generation_prompt": generation_prompt,
                    "question": contexts,
                    "output": final_answer,
                }

                # Copy generated image to simple output directory
                if generated_images and len(generated_images) > 0:
                    import shutil

                    gen_image_filename = f"{doc_id}_generated.jpg"
                    gen_image_dest = os.path.join(
                        self.simple_output_dir, "generated_images", gen_image_filename
                    )
                    shutil.copy2(generated_images[0], gen_image_dest)
                    result_entry["generated_image"] = f"./generated_images/{gen_image_filename}"

                self.simple_results.append(result_entry)

                # Save results to JSON file after each sample
                results_file = os.path.join(self.simple_output_dir, "results.json")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(self.simple_results, f, ensure_ascii=False, indent=2)

            # Return only final answer text
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "OneCATVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for OneCATVisualCoT"
        )
