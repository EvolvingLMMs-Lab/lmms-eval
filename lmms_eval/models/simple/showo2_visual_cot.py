# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Show-o2 Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image

Usage:
    python -m lmms_eval \
        --model showo2_visual_cot \
        --model_args pretrained=showlab/show-o2-7B \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import json
import os
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("showo2_visual_cot")
class Showo2VisualCoT(lmms):
    """
    Show-o2 Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using the generated image
    """

    def __init__(
        self,
        pretrained: str = "showlab/show-o2-7B",
        # Stage 1: Image generation parameters
        stage1_guidance_scale: float = 5.0,
        stage1_num_inference_steps: int = 50,
        stage1_resolution: int = 432,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_top_k: int = 1,
        stage2_temperature: float = 1.0,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        llm_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        vae_model_path: Optional[str] = None,
        weight_type: str = "bfloat16",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_resolution = stage1_resolution

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_top_k = stage2_top_k
        self.stage2_temperature = stage2_temperature

        # Model loading parameters
        self.llm_model_path = llm_model_path
        self.vae_model_path = vae_model_path
        self.weight_type = weight_type

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/showo2_visual_cot"
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

        # Import and initialize Show-o2 model
        eval_logger.info(f"Loading Show-o2 model from {pretrained}")
        self._load_showo2_model()

        eval_logger.info("Showo2VisualCoT initialized successfully")

    def _load_showo2_model(self):
        """Initialize model loading parameters (actual loading is deferred)"""
        # Don't load models here - we'll load them on demand to save memory
        self.showo2 = None
        self.current_mode = None
        eval_logger.info("Show-o2 model loading deferred (will load on demand)")

    def _load_model_for_mode(self, mode: str):
        """Load model for specific mode, unloading previous model if needed"""
        import gc
        import torch
        from lmms_eval.models.simple.showo2 import Showo2

        if self.current_mode == mode and self.showo2 is not None:
            return  # Already loaded for this mode

        # Unload previous model if exists
        if self.showo2 is not None:
            eval_logger.info(f"Unloading Show-o2 model (was in {self.current_mode} mode)")
            del self.showo2
            self.showo2 = None
            gc.collect()
            torch.cuda.empty_cache()

        # Load model for new mode
        eval_logger.info(f"Loading Show-o2 model for {mode} mode")
        self.showo2 = Showo2(
            pretrained=self.pretrained,
            mode=mode,
            llm_model_path=self.llm_model_path,
            vae_model_path=self.vae_model_path,
            resolution=self.stage1_resolution,
            weight_type=self.weight_type,
            output_image_dir=self.intermediate_dir,
            guidance_scale=self.stage1_guidance_scale,
            num_inference_steps=self.stage1_num_inference_steps,
            max_new_tokens=self.stage2_max_new_tokens,
            top_k=self.stage2_top_k,
            temperature=self.stage2_temperature,
            seed=self.seed,
            continual_mode=False,
        )
        self.current_mode = mode
        eval_logger.info(f"Show-o2 model loaded for {mode} mode")

    @property
    def rank(self):
        return 0  # Default rank

    @property
    def world_size(self):
        return 1  # Default world size

    @property
    def model(self):
        return self.showo2.model if self.showo2 else None

    @property
    def tokenizer(self):
        return self.showo2.tokenizer if self.showo2 else None

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Load generation model
            self._load_model_for_mode("generation")

            text, images = self.showo2.generate_image(
                prompt=generation_prompt,
                doc_id=f"{doc_id}_stage1",
                task=task,
            )
            eval_logger.debug(f"Stage 1 - Generated {len(images)} image(s)")
            return text, images
        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
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
            original_image: Original image (optional, will be used together with auxiliary)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load understanding model (will unload generation model first)
            self._load_model_for_mode("understanding")

            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # If original image is provided, use both images
            if original_image is not None:
                eval_logger.debug("Stage 2 - Using both original and auxiliary images")
                # Update question to provide context about the two images
                question_with_context = (
                    "You are given two images. The first image is the original image, "
                    "and the second image is an auxiliary visualization to help answer the question. "
                    + question
                )
                answer_text = self.showo2.understand_two_images(
                    prompt=question_with_context,
                    image1=original_image,
                    image2=auxiliary_image,
                    doc_id=doc_id,
                )
            else:
                # Use only auxiliary image
                eval_logger.debug("Stage 2 - Using auxiliary image only")
                answer_text = self.showo2.understand_image(
                    prompt=question,
                    image=auxiliary_image,
                    doc_id=doc_id,
                )

            eval_logger.debug(f"Stage 2 - Generated answer: {answer_text[:100]}...")
            return answer_text

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
        stage1_text: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ) -> None:
        """Save intermediate artifacts for debugging"""
        if not self.save_intermediate:
            return

        artifact_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(artifact_dir, exist_ok=True)

        metadata = {
            "doc_id": doc_id,
            "task": task,
            "generation_prompt": generation_prompt,
            "stage1_text": stage1_text,
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
            desc="Showo2VisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                        eval_logger.debug(f"Extracted original image for doc {doc_id}")
                except Exception as e:
                    eval_logger.warning(
                        f"Failed to extract original image for doc {doc_id}: {e}"
                    )

            # Parse contexts to extract generation_prompt if provided
            import re

            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
            )

            if gen_prompt_match and question_match:
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace(
                    "{question}", actual_question
                )
                contexts = contexts.replace(
                    f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", ""
                )
                contexts = contexts.replace(
                    f"[QUESTION]{question_match.group(1)}[/QUESTION]",
                    question_match.group(1),
                )
                eval_logger.info("Using custom generation prompt from task config")
            else:
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
            )

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, using stage 1 text as answer"
                )
                res.append(stage1_text if stage1_text else "")
                pbar.update(1)
                continue

            # Stage 2: Answer question using generated image
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
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=contexts,
                stage2_answer=final_answer,
            )

            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "Showo2VisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for Showo2VisualCoT"
        )
