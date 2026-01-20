"""
UniWorld Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image using UniWorld's generation mode
2. Stage 2: Answer question using UniWorld's understanding mode (Qwen2.5-VL)
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


@register_model("uniworld_visual_cot")
class UniWorldVisualCoT(lmms):
    """
    UniWorld Visual Chain-of-Thought Model
    
    Performs two-stage visual reasoning:
    1. Generate visualization using UniWorld's generation pipeline
    2. Answer question using Qwen2.5-VL understanding
    """

    def __init__(
        self,
        pretrained: str = "LanguageBind/UniWorld-V1",
        flux_path: str = "black-forest-labs/FLUX.1-dev",
        siglip_path: str = "google/siglip2-so400m-patch16-512",
        # Stage 1: Image generation parameters
        stage1_height: int = 1024,
        stage1_width: int = 1024,
        stage1_num_inference_steps: int = 28,
        stage1_guidance_scale: float = 3.5,
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
        min_pixels: int = 448 * 448,
        max_pixels: int = 448 * 448,
        no_joint_with_t5: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.flux_path = flux_path
        self.siglip_path = siglip_path
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_height = stage1_height
        self.stage1_width = stage1_width
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # UniWorld parameters
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.no_joint_with_t5 = no_joint_with_t5

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/uniworld_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved under: {self.intermediate_dir}")

        # Load UniWorld model
        eval_logger.info(f"Loading UniWorld model from {pretrained}")
        self._load_uniworld_model()

        eval_logger.info("UniWorldVisualCoT initialized successfully")

    def _load_uniworld_model(self):
        """Load UniWorld model for both generation and understanding"""
        from lmms_eval.models.simple.uniworld import UniWorld

        # Initialize UniWorld with full capabilities
        self.uniworld = UniWorld(
            pretrained=self.pretrained,
            flux_path=self.flux_path,
            siglip_path=self.siglip_path,
            mode="unified",  # Special mode for Visual CoT (both gen + und)
            height=self.stage1_height,
            width=self.stage1_width,
            num_inference_steps=self.stage1_num_inference_steps,
            guidance_scale=self.stage1_guidance_scale,
            max_new_tokens=self.stage2_max_new_tokens,
            do_sample=self.stage2_do_sample,
            temperature=self.stage2_temperature,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            no_joint_with_t5=self.no_joint_with_t5,
            image_output_dir=self.intermediate_dir,
        )

        eval_logger.info("UniWorld model loaded successfully")

    @property
    def rank(self):
        return self.uniworld.rank if hasattr(self.uniworld, 'rank') else 0

    @property
    def world_size(self):
        return self.uniworld.world_size if hasattr(self.uniworld, 'world_size') else 1

    @property
    def model(self):
        return self.uniworld.model

    @property
    def tokenizer(self):
        return self.uniworld.processor.tokenizer if hasattr(self.uniworld.processor, 'tokenizer') else None

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image=None
    ) -> Tuple[str, List[str]]:
        """Stage 1: Generate visualization image"""
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        
        try:
            # Use UniWorld's generation capability
            output = self.uniworld._generate_image_with_original(
                prompt_text=generation_prompt,
                original_image=original_image,
                doc_id=f"{doc_id}_stage1",
                task=task
            )
            
            # Parse output
            if isinstance(output, str):
                output_dict = json.loads(output)
                images = output_dict.get("images", [])
                text = output_dict.get("text", "")
            else:
                images = []
                text = ""
            
            eval_logger.debug(f"Stage 1 - Generated {len(images)} image(s)")
            return text, images
        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self, question: str, image_path: str, doc_id: str, original_image=None
    ) -> str:
        """Stage 2: Answer question using generated image"""
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        
        try:
            # Use UniWorld's understanding capability
            answer = self.uniworld._answer_with_images(
                question=question,
                images=[original_image, image_path] if original_image else [image_path],
                doc_id=doc_id
            )
            
            eval_logger.debug(f"Stage 2 - Generated answer: {answer[:100]}...")
            return answer
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

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method implementing two-stage visual CoT"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UniWorldVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                except Exception as e:
                    eval_logger.warning(f"Failed to extract original image: {e}")

            # Parse generation prompt
            import re
            gen_prompt_match = re.search(r'\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]', contexts, re.DOTALL)
            question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', contexts, re.DOTALL)

            if gen_prompt_match and question_match:
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace("{question}", actual_question)
                contexts = contexts.replace(f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", "")
                contexts = contexts.replace(f"[QUESTION]{question_match.group(1)}[/QUESTION]", question_match.group(1))
            else:
                generation_prompt = self.generation_prompt_template.format(question=contexts)

            # Stage 1: Generate visualization
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image,
            )

            if not generated_images:
                eval_logger.warning(f"No image generated for doc {doc_id}")
                res.append(stage1_text if stage1_text else "")
                pbar.update(1)
                continue

            # Stage 2: Answer with generated image
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image
            )

            # Save artifacts
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
        """Not supported"""
        raise NotImplementedError("UniWorldVisualCoT does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not yet implemented"""
        raise NotImplementedError("Multi-round not yet implemented for UniWorldVisualCoT")
