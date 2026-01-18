"""
JavisGPT Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image

Usage:
    python -m lmms_eval \
        --model javisgpt_visual_cot \
        --model_args pretrained=/path/to/JavisGPT-v0.1-7B-Instruct,base_model=/path/to/Qwen2.5-VL-7B-Instruct,beats_path=/path/to/BEATs.pt,avgen_cfg_path=/path/to/javisdit_v0.1_image.py \
        --tasks your_visual_cot_task \
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


@register_model("javisgpt_visual_cot")
class JavisGPTVisualCoT(lmms):
    """
    JavisGPT Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt (using JavisGPT image generation)
    2. Answer question using the generated image (using JavisGPT understanding)
    """

    def __init__(
        self,
        pretrained: str,
        base_model: str,
        beats_path: str,
        avgen_cfg_path: str,
        base_arch: str = "Qwen2_5_VL",
        av_gen_token_num: int = 377,
        # Stage 1: Image generation parameters
        stage1_output_type: str = "image",  # "image" or "video"
        stage1_num_frames: Optional[int] = None,
        stage1_image_size: Tuple[int, int] = (512, 512),
        stage1_max_new_tokens: int = 512,
        stage1_do_sample: bool = False,
        stage1_temperature: float = 0.0,
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
        save_results: bool = True,
        results_file: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        device: Optional[str] = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.base_model = base_model
        self.beats_path = beats_path
        self.avgen_cfg_path = avgen_cfg_path
        self.base_arch = base_arch
        self.save_intermediate = save_intermediate
        self.save_results = save_results
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_output_type = stage1_output_type
        self.stage1_num_frames = stage1_num_frames
        self.stage1_image_size = stage1_image_size
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_do_sample = stage1_do_sample
        self.stage1_temperature = stage1_temperature

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/javisgpt_visual_cot"
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
            eval_logger.info(
                f"Intermediate artifacts will be saved to: {self.intermediate_dir}"
            )

        # Setup results file for saving detailed results
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            if results_file is None:
                self.results_file = os.path.join(
                    self.output_dir, "javisgpt_visual_cot_results.jsonl"
                )
            else:
                self.results_file = results_file
            eval_logger.info(f"Results will be saved to: {self.results_file}")
            # Initialize results list
            self.results_list = []

        # Import and initialize JavisGPT model
        eval_logger.info(f"Loading JavisGPT model from {pretrained}")
        self._load_javisgpt_model(av_gen_token_num, device)

        eval_logger.info("JavisGPTVisualCoT initialized successfully")

    def _load_javisgpt_model(self, av_gen_token_num: int, device: Optional[str]):
        """Load JavisGPT model with both generation and understanding capabilities"""
        # Import JavisGPT model class
        from lmms_eval.models.simple.javisgpt import JavisGPT

        # Initialize JavisGPT with generation mode to ensure av_generator is loaded
        # We'll use both generation (Stage 1) and understanding (Stage 2) capabilities
        self.javisgpt = JavisGPT(
            pretrained=self.pretrained,
            base_model=self.base_model,
            beats_path=self.beats_path,
            mode="generation",  # Use generation mode to load av_generator
            base_arch=self.base_arch,
            avgen_cfg_path=self.avgen_cfg_path,
            av_gen_token_num=av_gen_token_num,
            output_type=self.stage1_output_type,
            num_frames=self.stage1_num_frames,
            image_size=self.stage1_image_size,
            output_image_dir=os.path.join(self.output_dir, "generated_images"),
            output_video_dir=os.path.join(self.output_dir, "generated_videos"),
            max_new_tokens=self.stage1_max_new_tokens,  # Use stage1 tokens for generation
            do_sample=self.stage1_do_sample,
            temperature=self.stage1_temperature,
            seed=self.seed,
            continual_mode=False,  # Disable caching for visual CoT
            device=device,
        )

        eval_logger.info("JavisGPT model loaded successfully")

    @property
    def rank(self):
        return self.javisgpt.rank

    @property
    def world_size(self):
        return self.javisgpt.world_size

    @property
    def model(self):
        return self.javisgpt.model

    @property
    def tokenizer(self):
        return self.javisgpt.tokenizer

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
            # Switch to generation mode temporarily
            original_mode = self.javisgpt.mode
            original_max_tokens = self.javisgpt.max_new_tokens
            original_do_sample = self.javisgpt.do_sample
            original_temperature = self.javisgpt.temperature

            self.javisgpt.mode = "generation"
            self.javisgpt.max_new_tokens = self.stage1_max_new_tokens
            self.javisgpt.do_sample = self.stage1_do_sample
            self.javisgpt.temperature = self.stage1_temperature

            text, content_paths = self.javisgpt.generate_content(
                prompt=generation_prompt,
                doc_id=f"{doc_id}_stage1",
                task=task,
            )

            # Restore original mode
            self.javisgpt.mode = original_mode
            self.javisgpt.max_new_tokens = original_max_tokens
            self.javisgpt.do_sample = original_do_sample
            self.javisgpt.temperature = original_temperature

            eval_logger.debug(
                f"Stage 1 - Generated {len(content_paths)} image(s)"
            )
            return text, content_paths
        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self,
        question: str,
        generated_image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using generated image (and optionally original image)

        Args:
            question: Original question text
            generated_image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (optional, used as primary reference)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load generated auxiliary image
            generated_image = Image.open(generated_image_path).convert("RGB")

            # Determine which image to use
            if original_image is not None:
                eval_logger.debug(
                    "Stage 2 - Using original image as primary reference"
                )
                # Use original image as the main input
                # The generated image provides auxiliary context
                # For now, we'll use the original image only
                # TODO: Support multi-image input if the model supports it
                primary_image = original_image
            else:
                eval_logger.debug("Stage 2 - Using generated image only")
                primary_image = generated_image

            # Call understanding mode
            answer_text = self.javisgpt.understand_audiovisual(
                prompt=question,
                image=primary_image,
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

    def save_result(
        self,
        doc_id: str,
        task: str,
        split: str,
        question: str,
        generation_prompt: str,
        stage1_text: str,
        generated_images: List[str],
        final_answer: str,
    ) -> None:
        """
        Save a single result to the results file

        Args:
            doc_id: Document ID
            task: Task name
            split: Dataset split
            question: Original question
            generation_prompt: Stage 1 generation prompt
            stage1_text: Stage 1 generated text
            generated_images: List of generated image paths
            final_answer: Final answer from stage 2
        """
        if not self.save_results:
            return

        result_entry = {
            "doc_id": str(doc_id),
            "task": task,
            "generation_prompt": generation_prompt,
            "stage1_text": stage1_text,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": final_answer,
        }

        # Append to results list
        self.results_list.append(result_entry)

        # Write to file incrementally (JSONL format)
        try:
            with open(self.results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            eval_logger.error(f"Failed to save result to {self.results_file}: {e}")

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
            desc="JavisGPTVisualCoT Generating",
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
            import re

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
                # Update contexts to be just the question for stage 2
                contexts = contexts.replace(
                    f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", ""
                )
                contexts = contexts.replace(
                    f"[QUESTION]{question_match.group(1)}[/QUESTION]",
                    question_match.group(1),
                )
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
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt, doc_id=doc_id, task=task
            )

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, using stage 1 text as answer"
                )

                # Save result even if generation failed
                self.save_result(
                    doc_id=doc_id,
                    task=task,
                    split=split,
                    question=contexts,
                    generation_prompt=generation_prompt,
                    stage1_text=stage1_text,
                    generated_images=[],
                    final_answer=stage1_text if stage1_text else "",
                )

                res.append(stage1_text if stage1_text else "")
                pbar.update(1)
                continue

            # Stage 2: Answer question using generated image (and original image if available)
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                generated_image_path=generated_images[0],
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

            # Save result
            self.save_result(
                doc_id=doc_id,
                task=task,
                split=split,
                question=contexts,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                final_answer=final_answer,
            )

            # Return only final answer text
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "JavisGPTVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for JavisGPTVisualCoT"
        )
