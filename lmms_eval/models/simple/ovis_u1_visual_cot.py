"""
Ovis-U1 Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Original image + prompt + question → Generate auxiliary visualization image
2. Stage 2: Original image + auxiliary image + prompt + question → Answer

Usage:
    python -m lmms_eval \
        --model ovis_u1_visual_cot \
        --model_args pretrained=AIDC-AI/Ovis-U1-3B \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import json
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.models.auto import configuration_auto

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@contextmanager
def patch_autoconfig_register():
    """Patch AutoConfig.register to allow re-registering existing configs.

    This is needed because some model repos (like Ovis-U1) try to register
    configs (e.g., 'aimv2') that are already built into newer transformers versions.
    """
    original_register = configuration_auto.CONFIG_MAPPING.register

    def patched_register(key, value, exist_ok=False):
        try:
            original_register(key, value, exist_ok=True)
        except TypeError:
            # Older transformers versions don't support exist_ok parameter
            try:
                original_register(key, value)
            except ValueError:
                # Config already exists, ignore
                pass

    configuration_auto.CONFIG_MAPPING.register = patched_register
    try:
        yield
    finally:
        configuration_auto.CONFIG_MAPPING.register = original_register


@register_model("ovis_u1_visual_cot")
class OvisU1VisualCoT(lmms):
    """
    Ovis-U1 Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Stage 1: Original image + prompt + question → Generate auxiliary visualization image
    2. Stage 2: Original image + auxiliary image + prompt + question → Answer
    """

    def __init__(
        self,
        pretrained: str = "AIDC-AI/Ovis-U1-3B",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        # Stage 1: Image generation parameters
        stage1_max_new_tokens: int = 4096,
        stage1_guidance_scale: float = 3.0,
        stage1_image_ratio: str = "1:1",
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        # Generation prompt template
        generation_prompt_template: str = (
            "Generate a detailed visual diagram or illustration to help "
            "answer this question: {question}"
        ),
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_guidance_scale = stage1_guidance_scale
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
            eval_logger.warning(f"Unknown image ratio {stage1_image_ratio}, using 1:1")
            self.image_shapes = (1024, 1024)

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/ovis_u1_visual_cot"
        else:
            self.output_dir = output_dir

        self.generated_images_dir = os.path.join(self.output_dir, "generated_images")
        os.makedirs(self.generated_images_dir, exist_ok=True)

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

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Load model with patched AutoConfig.register to handle duplicate config names
        eval_logger.info(f"Loading Ovis-U1 model from {pretrained}")
        with patch_autoconfig_register():
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
            )

        # Explicitly convert model to bfloat16 and eval mode (as per official code)
        self._model = self._model.eval().to(torch.bfloat16)

        # Ovis-U1 uses model.text_tokenizer
        self._tokenizer = self._model.text_tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported"
        self.use_cache = use_cache

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
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
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
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

        eval_logger.info("OvisU1VisualCoT initialized successfully")

    @property
    def config(self):
        """Return the model config."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

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

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        question: str,
        doc_id: str,
        task: str,
        original_image: Optional[Image.Image] = None,
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image using original image + prompt + question

        Args:
            generation_prompt: Text prompt for image generation
            question: The question to answer
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: The original input image to analyze

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")
        eval_logger.debug(f"Question: {question}")
        eval_logger.debug(f"Original image provided: {original_image is not None}")

        try:
            # Get tokenizers
            text_tokenizer = self.model.get_text_tokenizer()
            visual_tokenizer = self.model.get_visual_tokenizer()

            height, width = self.image_shapes

            # Use original image for conditional generation, blank image for unconditional
            uncond_image = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')

            # For conditional generation, use the original image if provided
            if original_image is not None:
                cond_image = original_image.convert('RGB')
                eval_logger.info(f"Using original image for conditional generation")
            else:
                cond_image = uncond_image
                eval_logger.warning(f"No original image provided, using blank image")

            # Build generation kwargs
            gen_kwargs = dict(
                max_new_tokens=self.stage1_max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=text_tokenizer.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=self.use_cache,
                height=height,
                width=width,
                num_steps=50,
                seed=42,
                img_cfg=0,
                txt_cfg=self.stage1_guidance_scale,
            )

            # Helper function to build inputs
            def build_inputs(prompt_text, pil_image, target_width, target_height):
                if pil_image is not None:
                    # Keep original PIL image for preprocess_inputs
                    original_pil = pil_image
                    
                    # Process image for VAE
                    target_size = (int(target_width), int(target_height))
                    processed_img, vae_pixel_values, cond_img_ids = self.model.visual_generator.process_image_aspectratio(
                        pil_image, target_size
                    )
                    cond_img_ids[..., 0] = 1.0
                    vae_pixel_values = vae_pixel_values.unsqueeze(0).to(
                        device=self.model.device, dtype=torch.bfloat16
                    )
                    
                    # Use original PIL image for visual tokenizer
                    img_width = original_pil.width
                    img_height = original_pil.height
                    resized_height, resized_width = visual_tokenizer.smart_resize(
                        img_height, img_width, max_pixels=visual_tokenizer.image_processor.min_pixels
                    )
                    resized_pil = original_pil.resize((resized_width, resized_height))
                    images_list = [resized_pil]
                else:
                    vae_pixel_values = None
                    images_list = []

                _, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                    prompt_text,
                    images_list if images_list else None,
                    generation_preface=None,
                    return_labels=False,
                    propagate_exception=False,
                    multimodal_type='single_image' if images_list else 'text_only',
                    fix_sample_overall_length_navit=False
                )
                
                attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
                input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
                attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device=self.model.device, dtype=torch.bfloat16)
                if grid_thws is not None:
                    grid_thws = grid_thws.to(device=self.model.device)
                return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values

            # Step 1: Generate unconditional baseline
            uncond_prompt = "<image>\nGenerate an image."
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
                uncond_prompt, uncond_image, width, height
            )
            with torch.inference_mode():
                no_both_cond = self.model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                    grid_thws=grid_thws, **gen_kwargs
                )

            # Step 2: Generate conditional with original image + prompt + question
            # Use cond_image (original image) so model can see and analyze it
            # Build prompt text - generation_prompt may already contain <image> tags
            # We need exactly 1 <image> tag for 1 image
            full_prompt = generation_prompt + "\n\nQuestion: " + question
            image_tag_count = full_prompt.count("<image>")
            
            if image_tag_count == 0:
                # No <image> tag, add one at the beginning
                prompt_text = "<image>\n" + full_prompt
            elif image_tag_count == 1:
                # Already has 1 <image> tag, use as is
                prompt_text = full_prompt
            else:
                # Multiple <image> tags - keep only the first one
                parts = full_prompt.split("<image>")
                # parts[0] is before first <image>, parts[1:] are after each <image>
                # Reconstruct with only one <image> tag
                prompt_text = parts[0] + "<image>" + "".join(parts[1:])
            
            input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs(
                prompt_text, cond_image, width, height
            )
            with torch.inference_mode():
                cond = self.model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                    grid_thws=grid_thws, **gen_kwargs
                )
            cond["vae_pixel_values"] = vae_pixel_values

            # Step 3: Generate the image
            with torch.inference_mode():
                generated_images_list = self.model.generate_img(
                    cond=cond, no_both_cond=no_both_cond, no_txt_cond=None, **gen_kwargs
                )

            # Save generated image (organized by task)
            generated_images = []
            if generated_images_list is not None and len(generated_images_list) > 0:
                # Create task-specific subdirectory
                task_image_dir = os.path.join(self.generated_images_dir, task)
                os.makedirs(task_image_dir, exist_ok=True)
                
                for idx, img in enumerate(generated_images_list):
                    if img is not None and hasattr(img, 'save'):
                        safe_filename = f"{task}_{doc_id}_stage1_{idx}.png"
                        image_path = os.path.join(task_image_dir, safe_filename)
                        img.save(image_path)
                        generated_images.append(image_path)
                        eval_logger.info(f"Saved generated image: {image_path}")

            eval_logger.debug(f"Stage 1 - Generated {len(generated_images)} image(s)")
            return "", generated_images

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self,
        question: str,
        generation_prompt: str,
        image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using original image + auxiliary image + prompt + question

        Args:
            question: Original question text
            generation_prompt: The generation prompt used in stage 1
            image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (required for proper inference)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # Prepare images list: original image + auxiliary image
            images = []

            if original_image is not None:
                images.append(original_image)
                images.append(auxiliary_image)
                expected_image_count = 2
            else:
                images.append(auxiliary_image)
                expected_image_count = 1

            # Build query text (without image placeholders first)
            query_text = generation_prompt + "\n\n" + "Question: " + question
            
            # Count existing <image> tags in the query
            existing_image_tags = query_text.count("<image>")
            
            # Adjust <image> tags to match the number of images
            if existing_image_tags == expected_image_count:
                # Perfect match, use as is
                query = query_text
            elif existing_image_tags < expected_image_count:
                # Need to add more <image> tags at the beginning
                tags_to_add = expected_image_count - existing_image_tags
                image_placeholder = "<image>" * tags_to_add
                query = image_placeholder + "\n" + query_text
            else:
                # Too many <image> tags, keep only the first expected_image_count
                parts = query_text.split("<image>")
                # Reconstruct with only expected_image_count tags
                query = parts[0]
                for i in range(1, min(expected_image_count + 1, len(parts))):
                    query += "<image>" + parts[i]
                # Add remaining parts without <image> tags
                for i in range(expected_image_count + 1, len(parts)):
                    query += parts[i]

            # Preprocess inputs
            _, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                text_or_conversations=query,
                images=images if images else None,
            )

            # Move to device
            input_ids = input_ids.unsqueeze(0).to(self._device)
            attention_mask = torch.ones_like(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.to(
                    device=self._device, dtype=self.model.dtype
                )
            if grid_thws is not None:
                grid_thws = grid_thws.to(self._device)

            # Prepare generation kwargs for understanding
            generate_kwargs = {
                "inputs": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
                "max_new_tokens": self.stage2_max_new_tokens,
                "use_cache": self.use_cache,
            }

            # Add sampling parameters
            if self.stage2_temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = self.stage2_temperature
            else:
                generate_kwargs["do_sample"] = False

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**generate_kwargs)

            # Decode response
            # Note: Ovis-U1 uses inputs_embeds internally, so we decode the full output
            answer_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up to free memory
            del outputs, input_ids, pixel_values, grid_thws
            torch.cuda.empty_cache()

            eval_logger.debug(f"Stage 2 - Generated answer: {answer_text[:100]}...")
            return answer_text

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            else:
                raise

    def _extract_image_from_various_formats(self, img_data) -> Optional[Image.Image]:
        """
        Extract PIL Image from various formats (HuggingFace datasets, file paths, etc.)
        
        Args:
            img_data: Image data in various formats
            
        Returns:
            PIL Image or None if extraction fails
        """
        try:
            if img_data is None:
                return None
            elif isinstance(img_data, Image.Image):
                return img_data.convert("RGB")
            elif isinstance(img_data, str):
                # File path
                return Image.open(img_data).convert("RGB")
            elif isinstance(img_data, dict):
                # HuggingFace dataset format
                if "bytes" in img_data:
                    from io import BytesIO
                    return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                elif "path" in img_data:
                    return Image.open(img_data["path"]).convert("RGB")
                elif "image" in img_data:
                    inner_img = img_data["image"]
                    return self._extract_image_from_various_formats(inner_img)
            else:
                # Try to open it directly
                return Image.open(img_data).convert("RGB")
        except Exception as e:
            eval_logger.debug(f"Failed to extract image from format {type(img_data)}: {e}")
            return None

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

        Stage 1: Original image + prompt + question → Generate auxiliary image
        Stage 2: Original image + auxiliary image + prompt + question → Answer
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="OvisU1VisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document
            original_image = None
            if doc_to_visual is not None:
                try:
                    # Get doc from task_dict
                    doc = self.task_dict[task][split][doc_id]
                    
                    # Try doc_to_visual first
                    original_visuals = doc_to_visual(doc)
                    
                    if original_visuals and len(original_visuals) > 0:
                        original_image = self._extract_image_from_various_formats(original_visuals[0])
                    
                    # If doc_to_visual didn't work, try direct field access
                    if original_image is None:
                        # Try common image field names
                        for field_name in ["image", "images", "original_image", "img"]:
                            if field_name in doc:
                                img_data = doc[field_name]
                                # Handle list of images
                                if isinstance(img_data, list):
                                    if len(img_data) > 0:
                                        img_data = img_data[0]
                                original_image = self._extract_image_from_various_formats(img_data)
                                if original_image is not None:
                                    eval_logger.debug(f"Extracted image from doc['{field_name}']")
                                    break
                    
                    if original_image is not None:
                        eval_logger.debug(f"Successfully extracted original image for doc {doc_id}")
                    else:
                        eval_logger.warning(f"No image found for doc {doc_id}")
                        
                except Exception as e:
                    import traceback
                    eval_logger.error(f"Failed to extract original image for doc {doc_id}: {e}")
                    eval_logger.debug(f"Traceback: {traceback.format_exc()}")

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
                # Use default template, contexts is the question
                actual_question = contexts
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'=' * 60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'=' * 60}")

            # Stage 1: Generate visualization image with original_image + prompt + question
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                question=actual_question,
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

            # Stage 2: Answer with original_image + auxiliary_image + prompt + question
            final_answer = self._stage2_answer_with_image(
                question=actual_question,
                generation_prompt=generation_prompt,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image,
            )

            # Build full question with image tags for metadata (matching bagel format)
            if original_image is not None:
                full_question = f"<image>{actual_question}"
            else:
                full_question = actual_question

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=full_question,
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
            "OvisU1VisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for OvisU1VisualCoT"
        )
