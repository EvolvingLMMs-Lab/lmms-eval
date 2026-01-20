"""
Qwen-Image-Edit Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization using Qwen-Image-Edit (editing mode)
2. Stage 2: Answer question using Qwen2.5-VL (understanding mode)
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


@register_model("qwen_image_edit_visual_cot")
class QwenImageEditVisualCoT(lmms):
    """
    Qwen-Image-Edit Visual Chain-of-Thought Model
    
    Two-stage visual reasoning:
    1. Stage 1: Use Qwen-Image-Edit to generate auxiliary visualization
    2. Stage 2: Use Qwen2.5-VL to answer question based on generated image
    """

    def __init__(
        self,
        pretrained_edit: str = "Qwen/Qwen-Image-Edit",
        pretrained_understand: str = "Qwen/Qwen2-VL-7B-Instruct",
        # Stage 1: Image editing/generation parameters
        stage1_num_inference_steps: int = 50,
        stage1_guidance_scale: float = 7.5,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        stage2_max_pixels: int = 1605632,
        stage2_min_pixels: int = 256 * 28 * 28,
        # Generation prompt template
        generation_prompt_template: str = "Based on this image and question, generate an annotated or highlighted version that helps answer: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained_edit = pretrained_edit
        self.pretrained_understand = pretrained_understand
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample
        self.stage2_max_pixels = stage2_max_pixels
        self.stage2_min_pixels = stage2_min_pixels

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/qwen_edit_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved to: {self.intermediate_dir}")

        # Load models
        eval_logger.info("Loading Qwen-Image-Edit Visual CoT models...")
        self._load_models()
        eval_logger.info("QwenImageEditVisualCoT initialized successfully")

    def _load_models(self):
        """Load both editing and understanding models"""
        from lmms_eval.models.simple.qwen_image_edit import QwenImageEdit

        # Stage 1: Editing model
        eval_logger.info(f"Loading editing model: {self.pretrained_edit}")
        self.edit_model = QwenImageEdit(
            pretrained=self.pretrained_edit,
            mode="editing",
            num_inference_steps=self.stage1_num_inference_steps,
            guidance_scale=self.stage1_guidance_scale,
            save_generated_images=True,
            generated_image_dir=os.path.join(self.intermediate_dir, "stage1_images"),
        )

        # Stage 2: Understanding model
        eval_logger.info(f"Loading understanding model: {self.pretrained_understand}")
        self.understand_model = QwenImageEdit(
            pretrained=self.pretrained_understand,
            mode="understanding",
            max_pixels=self.stage2_max_pixels,
            min_pixels=self.stage2_min_pixels,
        )

        eval_logger.info("Both models loaded successfully")

    @property
    def rank(self):
        return self.understand_model.rank

    @property
    def world_size(self):
        return self.understand_model.world_size

    @property
    def model(self):
        return self.understand_model.model

    @property
    def tokenizer(self):
        return self.understand_model.tokenizer

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image: Image.Image
    ) -> Tuple[str, str]:
        """
        Stage 1: Generate auxiliary visualization using editing model
        
        Args:
            generation_prompt: Instruction for editing
            doc_id: Document ID
            task: Task name
            original_image: Original input image
            
        Returns:
            Tuple of (response_text, generated_image_path)
        """
        eval_logger.debug(f"Stage 1 - Generating auxiliary image for doc {doc_id}")
        eval_logger.debug(f"Edit instruction: {generation_prompt}")

        try:
            # Create a mock request for the editing model
            class MockRequest:
                def __init__(self, doc, args):
                    self.doc = doc
                    self.arguments = args

            mock_doc = {
                "image": original_image,
                "prompt": generation_prompt,
                "task": task,
                "doc_id": doc_id,
                "num_inference_steps": self.stage1_num_inference_steps,
                "guidance_scale": self.stage1_guidance_scale,
            }
            
            mock_request = MockRequest(mock_doc, [generation_prompt])
            results = self.edit_model._generate_editing([mock_request])
            
            if results and results[0]:
                # Extract image path from response
                response = results[0]
                image_path = os.path.join(
                    self.edit_model.generated_image_dir,
                    f"{task}_{doc_id}.png"
                )
                eval_logger.debug(f"Stage 1 - Generated image at: {image_path}")
                return response, image_path
            else:
                eval_logger.warning(f"Stage 1 - No image generated for doc {doc_id}")
                return "", ""

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", ""
            else:
                raise

    def _stage2_answer_with_image(
        self, question: str, image_path: str, doc_id: str, original_image: Optional[Image.Image] = None
    ) -> str:
        """
        Stage 2: Answer question using understanding model
        
        Args:
            question: Question text
            image_path: Path to generated auxiliary image
            doc_id: Document ID
            original_image: Original image (included as context)
            
        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load auxiliary image
            if not os.path.exists(image_path):
                eval_logger.warning(f"Generated image not found: {image_path}")
                return ""
                
            auxiliary_image = Image.open(image_path).convert("RGB")

            # Create mock request for understanding model
            class MockRequest:
                def __init__(self, doc_id, task, split, args):
                    self.doc = None
                    self.args = (
                        question,  # contexts
                        {"max_new_tokens": self.stage2_max_new_tokens, "temperature": self.stage2_temperature, "do_sample": self.stage2_do_sample},  # gen_kwargs
                        lambda doc: [original_image, auxiliary_image] if original_image else [auxiliary_image],  # doc_to_visual
                        doc_id,
                        task,
                        split,
                    )

            # Need to set up task_dict for the understanding model
            mock_doc = {}
            if not hasattr(self.understand_model, 'task_dict'):
                self.understand_model.task_dict = {}
            if 'temp_task' not in self.understand_model.task_dict:
                self.understand_model.task_dict['temp_task'] = {}
            if 'test' not in self.understand_model.task_dict['temp_task']:
                self.understand_model.task_dict['temp_task']['test'] = {}
            self.understand_model.task_dict['temp_task']['test'][doc_id] = mock_doc

            mock_request = MockRequest(doc_id, 'temp_task', 'test', None)
            results = self.understand_model._generate_understanding([mock_request])
            
            if results and results[0]:
                answer = results[0]
                eval_logger.debug(f"Stage 2 - Answer: {answer[:100]}...")
                return answer
            else:
                return ""

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
        stage1_response: str,
        generated_image_path: str,
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
            "stage1_response": stage1_response,
            "generated_image_path": generated_image_path,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Two-stage visual CoT inference
        
        Stage 1: Generate auxiliary visualization (editing)
        Stage 2: Answer question using generated image (understanding)
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="QwenEditVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.understand_model.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                        eval_logger.debug(f"Extracted original image for doc {doc_id}")
                except Exception as e:
                    eval_logger.warning(f"Failed to extract original image: {e}")

            if original_image is None:
                eval_logger.warning(f"No original image for doc {doc_id}, skipping")
                res.append("")
                pbar.update(1)
                continue

            # Parse contexts for custom generation prompt
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

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate auxiliary image
            stage1_response, generated_image_path = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image,
            )

            if not generated_image_path or not os.path.exists(generated_image_path):
                eval_logger.warning(f"No image generated for doc {doc_id}")
                res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer using generated image
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                image_path=generated_image_path,
                doc_id=doc_id,
                original_image=original_image,
            )

            # Save intermediate artifacts
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                stage1_response=stage1_response,
                generated_image_path=generated_image_path,
                question=contexts,
                stage2_answer=final_answer,
            )

            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported"""
        raise NotImplementedError("Visual CoT does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not supported"""
        raise NotImplementedError("Multi-round not implemented for Visual CoT")
