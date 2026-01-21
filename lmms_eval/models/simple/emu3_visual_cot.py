"""
Emu3 Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image

Usage:
    python -m lmms_eval \
        --model emu3_visual_cot \
        --model_args pretrained=BAAI/Emu3-Chat-hf,gen_pretrained=BAAI/Emu3-Gen-hf \
        --tasks auxsolidmath_easy_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import json
import os
import re
from typing import List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("emu3_visual_cot")
class Emu3VisualCoT(lmms):
    """
    Emu3 Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt (using Emu3-Gen)
    2. Answer question using the generated image (using Emu3-Chat)
    """

    def __init__(
        self,
        pretrained: str = "BAAI/Emu3-Chat-hf",
        gen_pretrained: Optional[str] = None,
        # Stage 1: Image generation parameters
        stage1_image_height: int = 1024,
        stage1_image_width: int = 1024,
        stage1_cfg_scale: float = 4.0,
        stage1_max_new_tokens: int = 50000,
        stage1_negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.",
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 16384,
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
        device_map: str = "auto",
        use_flash_attention_2: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.gen_pretrained = gen_pretrained if gen_pretrained else "BAAI/Emu3-Gen-hf"
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template
        self.device_map = device_map
        self.use_flash_attention_2 = use_flash_attention_2

        # Stage 1 parameters
        self.stage1_image_height = stage1_image_height
        self.stage1_image_width = stage1_image_width
        self.stage1_cfg_scale = stage1_cfg_scale
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_negative_prompt = stage1_negative_prompt

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/emu3_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved under: {self.intermediate_dir}")

        # Load models
        eval_logger.info(f"Loading Emu3 models:")
        eval_logger.info(f"  - Chat model: {pretrained}")
        eval_logger.info(f"  - Gen model: {self.gen_pretrained}")
        self._load_models()

        eval_logger.info("Emu3VisualCoT initialized successfully")

    def _load_models(self):
        """Load both Emu3-Chat and Emu3-Gen models"""
        from transformers import Emu3ForConditionalGeneration, Emu3Processor

        dtype = torch.bfloat16

        # Setup attention implementation
        attn_implementation = "flash_attention_2" if self.use_flash_attention_2 else "eager"
        if self.use_flash_attention_2:
            try:
                import flash_attn
                eval_logger.info("Using Flash Attention 2")
            except ImportError:
                eval_logger.warning("flash_attn not installed, falling back to eager attention")
                attn_implementation = "eager"

        # Load Chat model (for understanding)
        eval_logger.info("Loading Emu3-Chat model...")
        self.chat_processor = Emu3Processor.from_pretrained(self.pretrained)
        self.chat_processor.tokenizer.padding_side = "left"
        self.chat_model = Emu3ForConditionalGeneration.from_pretrained(
            self.pretrained,
            torch_dtype=dtype,
            device_map=self.device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).eval()
        eval_logger.info("Emu3-Chat model loaded")

        # Load Gen model (for image generation)
        eval_logger.info("Loading Emu3-Gen model...")
        self.gen_processor = Emu3Processor.from_pretrained(self.gen_pretrained)
        self.gen_processor.tokenizer.padding_side = "left"
        self.gen_model = Emu3ForConditionalGeneration.from_pretrained(
            self.gen_pretrained,
            torch_dtype=dtype,
            device_map=self.device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).eval()
        eval_logger.info("Emu3-Gen model loaded")

    @property
    def batch_size(self):
        return 1  # Visual CoT processes one at a time

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image: Optional[Image.Image] = None
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to condition on (optional)

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Prepare inputs for image generation
            inputs = self.gen_processor(
                text=generation_prompt,
                padding=True,
                return_tensors="pt",
                return_for_image_generation=True,
            ).to(self.gen_model.device, dtype=torch.bfloat16)

            # Prepare negative prompt
            neg_inputs = self.gen_processor(
                text=self.stage1_negative_prompt,
                return_tensors="pt"
            ).to(self.gen_model.device)

            # Get image size
            image_sizes = inputs.pop("image_sizes")
            HEIGHT, WIDTH = self.stage1_image_height, self.stage1_image_width
            VISUAL_TOKENS = self.gen_model.vocabulary_mapping.image_tokens

            # Define prefix constraint function for image generation
            def prefix_allowed_tokens_fn(batch_id, input_ids):
                height, width = HEIGHT, WIDTH
                visual_tokens = VISUAL_TOKENS
                image_wrapper_token_id = torch.tensor([self.gen_processor.tokenizer.image_wrapper_token_id], device=self.gen_model.device)
                eoi_token_id = torch.tensor([self.gen_processor.tokenizer.eoi_token_id], device=self.gen_model.device)
                eos_token_id = torch.tensor([self.gen_processor.tokenizer.eos_token_id], device=self.gen_model.device)
                pad_token_id = torch.tensor([self.gen_processor.tokenizer.pad_token_id], device=self.gen_model.device)
                eof_token_id = torch.tensor([self.gen_processor.tokenizer.eof_token_id], device=self.gen_model.device)
                eol_token_id = self.gen_processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

                position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
                offset = input_ids.shape[0] - position
                if offset % (width + 1) == 0:
                    return (eol_token_id, )
                elif offset == (width + 1) * height + 1:
                    return (eof_token_id, )
                elif offset == (width + 1) * height + 2:
                    return (eoi_token_id, )
                elif offset == (width + 1) * height + 3:
                    return (eos_token_id, )
                elif offset > (width + 1) * height + 3:
                    return (pad_token_id, )
                else:
                    return visual_tokens

            # Generate
            with torch.no_grad():
                out = self.gen_model.generate(
                    **inputs,
                    max_new_tokens=self.stage1_max_new_tokens,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    return_dict_in_generate=True,
                    negative_prompt_ids=neg_inputs.input_ids,
                    negative_prompt_attention_mask=neg_inputs.attention_mask,
                )

            # Decode image tokens
            image = self.gen_model.decode_image_tokens(
                out.sequences[:, inputs.input_ids.shape[1]:],
                height=HEIGHT,
                width=WIDTH
            )
            images = self.gen_processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image")

            # Save generated images
            output_images = []
            artifact_dir = os.path.join(self.intermediate_dir, task)
            os.makedirs(artifact_dir, exist_ok=True)

            for i, img in enumerate(images['pixel_values']):
                safe_filename = f"{doc_id}_stage1_{i}.png"
                image_path = os.path.join(artifact_dir, safe_filename)
                img.save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Stage 1 - Saved generated image: {image_path}")

            return "", output_images

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self, question: str, image_path: str, doc_id: str, original_image: Optional[Image.Image] = None
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

            # If original image is provided, use both images
            # Emu3 doesn't natively support multiple images, so we'll use the auxiliary image
            # and mention the original in the prompt
            if original_image is not None:
                eval_logger.debug("Stage 2 - Using auxiliary image (original mentioned in prompt)")
                # Use auxiliary image with enhanced prompt
                enhanced_question = f"""You are given TWO images:
1) ORIGINAL DIAGRAM: The 3D solid geometry figure as given
2) AUXILIARY DIAGRAM (shown): The same figure with auxiliary constructions added

{question}"""
                image_to_use = auxiliary_image
            else:
                eval_logger.debug("Stage 2 - Using auxiliary image only")
                enhanced_question = question
                image_to_use = auxiliary_image

            # Prepare inputs
            inputs = self.chat_processor(
                images=image_to_use,
                text=enhanced_question,
                return_tensors="pt",
                padding=True,
            ).to(self.chat_model.device, dtype=torch.bfloat16)

            # Generate
            with torch.no_grad():
                output_ids = self.chat_model.generate(
                    **inputs,
                    max_new_tokens=self.stage2_max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature if self.stage2_do_sample else None,
                )

            # Decode
            answer_text = self.chat_processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

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
            desc="Emu3VisualCoT Generating",
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
                    eval_logger.warning(f"Failed to extract original image for doc {doc_id}: {e}")

            # Parse contexts to extract generation_prompt if provided
            gen_prompt_match = re.search(r'\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]', contexts, re.DOTALL)
            question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', contexts, re.DOTALL)

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace("{question}", actual_question)
                # Update contexts to be just the question for stage 2
                contexts = contexts.replace(f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", "")
                contexts = contexts.replace(f"[QUESTION]{question_match.group(1)}[/QUESTION]", question_match.group(1))
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(question=contexts)

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image,
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
                original_image=original_image
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

            # Return only final answer text
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "Emu3VisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for Emu3VisualCoT"
        )
