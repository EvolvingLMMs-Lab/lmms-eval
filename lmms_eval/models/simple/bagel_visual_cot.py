"""
Bagel Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image

Usage:
    python -m lmms_eval \
        --model bagel_visual_cot \
        --model_args pretrained=/path/to/BAGEL-7B-MoT \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("bagel_visual_cot")
class BagelVisualCoT(lmms):
    """
    Bagel Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using the generated image
    """

    def __init__(
        self,
        pretrained: str,
        # Stage 1: Image generation parameters
        stage1_cfg_text_scale: float = 4.0,
        stage1_cfg_interval: float = 0.4,
        stage1_timestep_shift: float = 3.0,
        stage1_num_timesteps: int = 50,
        stage1_cfg_renorm_min: float = 0.0,
        stage1_cfg_renorm_type: str = "global",
        stage1_image_ratio: str = "1:1",
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
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
        self.stage1_cfg_text_scale = stage1_cfg_text_scale
        self.stage1_cfg_interval = stage1_cfg_interval
        self.stage1_timestep_shift = stage1_timestep_shift
        self.stage1_num_timesteps = stage1_num_timesteps
        self.stage1_cfg_renorm_min = stage1_cfg_renorm_min
        self.stage1_cfg_renorm_type = stage1_cfg_renorm_type
        self.stage1_image_ratio = stage1_image_ratio

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # Set image shapes based on ratio
        if stage1_image_ratio == "1:1":
            self.image_shapes = (1024, 1024)
        elif stage1_image_ratio == "4:3":
            self.image_shapes = (768, 1024)
        elif stage1_image_ratio == "3:4":
            self.image_shapes = (1024, 768)
        elif stage1_image_ratio == "16:9":
            self.image_shapes = (576, 1024)
        elif stage1_image_ratio == "9:16":
            self.image_shapes = (1024, 576)
        else:
            eval_logger.warning(
                f"Unknown image ratio {stage1_image_ratio}, using 1:1"
            )
            self.image_shapes = (1024, 1024)

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/bagel_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = os.path.join(
                self.output_dir, "intermediate_artifacts"
            )
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved to: {self.intermediate_dir}")

        # Import and initialize Bagel model
        eval_logger.info(f"Loading Bagel model from {pretrained}")
        self._load_bagel_model(load_in_4bit, load_in_8bit)

        eval_logger.info("BagelVisualCoT initialized successfully")

    def _load_bagel_model(self, load_in_4bit: bool, load_in_8bit: bool):
        """Load Bagel model with both generation and understanding capabilities"""
        # Import Bagel model class
        from lmms_eval.models.simple.bagel import Bagel

        # Initialize Bagel with both visual_gen and visual_und capabilities
        self.bagel = Bagel(
            pretrained=self.pretrained,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            output_image_dir=os.path.join(self.output_dir, "generated_images"),
            show_thinking=False,
            cfg_text_scale=self.stage1_cfg_text_scale,
            cfg_interval=self.stage1_cfg_interval,
            timestep_shift=self.stage1_timestep_shift,
            num_timesteps=self.stage1_num_timesteps,
            cfg_renorm_min=self.stage1_cfg_renorm_min,
            cfg_renorm_type=self.stage1_cfg_renorm_type,
            seed=self.seed,
            image_ratio=self.stage1_image_ratio,
            continual_mode=False,  # Disable caching for visual CoT
        )

        eval_logger.info("Bagel model loaded successfully")

    @property
    def rank(self):
        return self.bagel.rank

    @property
    def world_size(self):
        return self.bagel.world_size

    @property
    def model(self):
        return self.bagel.model

    @property
    def tokenizer(self):
        return self.bagel.tokenizer

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
            text, images = self.bagel.generate_text_and_image(
                prompt=generation_prompt, doc_id=f"{doc_id}_stage1", task=task
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
        self, question: str, image_path: str, doc_id: str
    ) -> str:
        """
        Stage 2: Answer question using generated image

        Args:
            question: Original question text
            image_path: Path to generated image
            doc_id: Document ID for logging

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load generated image
            pil_image = Image.open(image_path).convert("RGB")

            # Call Bagel inferencer with image input for visual understanding
            # Note: This assumes inferencer accepts 'images' parameter
            result = self.bagel.inferencer(
                text=question,
                images=[pil_image],
                think=False,
                max_new_tokens=self.stage2_max_new_tokens,
                temperature=self.stage2_temperature,
                do_sample=self.stage2_do_sample,
            )

            answer_text = result.get("text", "")
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

        # Save metadata
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
            desc="BagelVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Generate visualization prompt using template
            # The template can use {question} placeholder which will be replaced with contexts
            generation_prompt = self.generation_prompt_template.format(question=contexts)

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt, doc_id=doc_id, task=task
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
                question=contexts, image_path=generated_images[0], doc_id=doc_id
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

            # Return only final answer text (as per user requirement)
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "BagelVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for BagelVisualCoT"
        )
