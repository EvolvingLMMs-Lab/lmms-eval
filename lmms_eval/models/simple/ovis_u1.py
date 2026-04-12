"""
Ovis-U1 Model (Unified: Understanding + Visual CoT)

Supports three modes, auto-detected from the prompt:
1. Standard understanding: direct question answering
2. Visual CoT (two-stage): [GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]
   Stage 1: Original image + prompt → Generate auxiliary visualization
   Stage 2: Original image + auxiliary image + question → Answer
3. Uni-MMMU interleaved generation: jigsaw/maze/sliding puzzles

Usage:
    # Standard understanding
    python -m lmms_eval --model ovis_u1 \
        --model_args pretrained=AIDC-AI/Ovis-U1-3B \
        --tasks unig2u --batch_size 1

    # Visual CoT (auto-detected from task config)
    python -m lmms_eval --model ovis_u1 \
        --model_args pretrained=AIDC-AI/Ovis-U1-3B \
        --tasks unig2u_GtA --batch_size 1
"""

import json
import os
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.models.auto import configuration_auto

from lmms_eval import utils
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


@register_model("ovis_u1")
class OvisU1(lmms):
    """
    Ovis-U1 Model (Unified Understanding + Visual CoT)
    https://huggingface.co/AIDC-AI/Ovis-U1-3B

    Supports standard understanding and Visual Chain-of-Thought (auto-detected).
    When the prompt contains [GEN_PROMPT]...[/GEN_PROMPT] tags, automatically
    switches to two-stage Visual CoT inference.
    """

    supports_visual_cot = True

    def __init__(
        self,
        pretrained: str = "AIDC-AI/Ovis-U1-3B",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        # Visual CoT parameters (used when auto-detected)
        stage1_max_new_tokens: int = 4096,
        stage1_guidance_scale: float = 3.0,
        stage1_image_ratio: str = "1:1",
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        generation_prompt_template: str = (
            "Generate a detailed visual diagram or illustration to help "
            "answer this question: {question}"
        ),
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        fail_gracefully: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.use_cache = use_cache
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template

        # Visual CoT stage parameters
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage1_image_ratio = stage1_image_ratio
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # Set image shapes based on ratio
        ratio_to_shape: Dict[str, Tuple[int, int]] = {
            "1:1": (1024, 1024),
            "4:3": (768, 1024),
            "3:4": (1024, 768),
            "16:9": (576, 1024),
            "9:16": (1024, 576),
        }
        self.image_shapes = ratio_to_shape.get(stage1_image_ratio, (1024, 1024))

        # Setup output directories for visual CoT
        if output_dir is None:
            self.output_dir = "./logs/ovis_u1_visual_cot"
        else:
            self.output_dir = output_dir
        self.generated_images_dir = os.path.join(self.output_dir, "generated_images")

        if intermediate_dir is None:
            self.intermediate_dir = os.path.join(
                self.output_dir, "intermediate_artifacts"
            )
        else:
            self.intermediate_dir = intermediate_dir

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(
                f"attn_implementation must be one of "
                f"{valid_attn_implementations}, got {attn_implementation}"
            )

        # Prepare model loading arguments
        model_kwargs = {
            "device_map": device,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype if dtype != "auto" else torch.bfloat16,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load model
        eval_logger.info(f"Loading Ovis-U1 model from {pretrained}")
        with patch_autoconfig_register():
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained, **model_kwargs
            )
        self._model = self._model.eval().to(torch.bfloat16)
        self._tokenizer = self._model.text_tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for Ovis-U1"

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

        eval_logger.info("OvisU1 initialized successfully")

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ── Helpers ─────────────────────────────────────────────────────────

    def flatten(self, input_list):
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def _extract_image(self, img_data) -> Optional[Image.Image]:
        """Extract PIL Image from various formats."""
        try:
            if img_data is None:
                return None
            elif isinstance(img_data, Image.Image):
                return img_data.convert("RGB")
            elif isinstance(img_data, str):
                return Image.open(img_data).convert("RGB")
            elif isinstance(img_data, dict):
                if "bytes" in img_data:
                    from io import BytesIO

                    return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                elif "path" in img_data:
                    return Image.open(img_data["path"]).convert("RGB")
                elif "image" in img_data:
                    return self._extract_image(img_data["image"])
            else:
                return Image.open(img_data).convert("RGB")
        except Exception as e:
            eval_logger.debug(
                f"Failed to extract image from {type(img_data)}: {e}"
            )
            return None

    def _prepare_images(self, visuals: list) -> List[Image.Image]:
        """Convert a list of visuals (various formats) to PIL Images."""
        images = []
        for visual in visuals:
            img = self._extract_image(visual)
            if img is not None:
                images.append(img)
            else:
                eval_logger.warning(
                    f"Skipping non-image type: {type(visual)}"
                )
        return images

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
        """Save intermediate artifacts for debugging."""
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

    # ── Visual CoT: Stage 1 ────────────────────────────────────────────

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        question: str,
        doc_id: str,
        task: str,
        original_image: Optional[Image.Image] = None,
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate auxiliary visualization image.

        Uses original image + prompt + question to generate an annotated /
        auxiliary image via Ovis-U1's conditional generation pipeline.
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")

        try:
            text_tokenizer = self.model.get_text_tokenizer()
            visual_tokenizer = self.model.get_visual_tokenizer()
            height, width = self.image_shapes

            uncond_image = Image.new(
                "RGB", (width, height), (255, 255, 255)
            ).convert("RGB")
            cond_image = (
                original_image.convert("RGB") if original_image else uncond_image
            )

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

            def build_inputs(prompt_text, pil_image, target_width, target_height):
                if pil_image is not None:
                    original_pil = pil_image
                    target_size = (int(target_width), int(target_height))
                    (
                        processed_img,
                        vae_pixel_values,
                        cond_img_ids,
                    ) = self.model.visual_generator.process_image_aspectratio(
                        pil_image, target_size
                    )
                    cond_img_ids[..., 0] = 1.0
                    vae_pixel_values = vae_pixel_values.unsqueeze(0).to(
                        device=self.model.device, dtype=torch.bfloat16
                    )
                    img_w = original_pil.width
                    img_h = original_pil.height
                    rh, rw = visual_tokenizer.smart_resize(
                        img_h,
                        img_w,
                        max_pixels=visual_tokenizer.image_processor.min_pixels,
                    )
                    resized_pil = original_pil.resize((rw, rh))
                    images_list = [resized_pil]
                else:
                    vae_pixel_values = None
                    images_list = []

                _, input_ids, pixel_values, grid_thws = (
                    self.model.preprocess_inputs(
                        prompt_text,
                        images_list if images_list else None,
                        generation_preface=None,
                        return_labels=False,
                        propagate_exception=False,
                        multimodal_type=(
                            "single_image" if images_list else "text_only"
                        ),
                        fix_sample_overall_length_navit=False,
                    )
                )
                attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
                input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
                attention_mask = attention_mask.unsqueeze(0).to(
                    device=self.model.device
                )
                if pixel_values is not None:
                    pixel_values = pixel_values.to(
                        device=self.model.device, dtype=torch.bfloat16
                    )
                if grid_thws is not None:
                    grid_thws = grid_thws.to(device=self.model.device)
                return (
                    input_ids,
                    pixel_values,
                    attention_mask,
                    grid_thws,
                    vae_pixel_values,
                )

            # Step 1: Unconditional baseline
            uncond_prompt = "<image>\nGenerate an image."
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
                uncond_prompt, uncond_image, width, height
            )
            with torch.inference_mode():
                no_both_cond = self.model.generate_condition(
                    input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    grid_thws=grid_thws,
                    **gen_kwargs,
                )

            # Step 2: Conditional with original image
            full_prompt = generation_prompt + "\n\nQuestion: " + question
            image_tag_count = full_prompt.count("<image>")
            if image_tag_count == 0:
                prompt_text = "<image>\n" + full_prompt
            elif image_tag_count == 1:
                prompt_text = full_prompt
            else:
                parts = full_prompt.split("<image>")
                prompt_text = parts[0] + "<image>" + "".join(parts[1:])

            (
                input_ids,
                pixel_values,
                attention_mask,
                grid_thws,
                vae_pixel_values,
            ) = build_inputs(prompt_text, cond_image, width, height)
            with torch.inference_mode():
                cond = self.model.generate_condition(
                    input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    grid_thws=grid_thws,
                    **gen_kwargs,
                )
            cond["vae_pixel_values"] = vae_pixel_values

            # Step 3: Generate the image
            with torch.inference_mode():
                generated_images_list = self.model.generate_img(
                    cond=cond,
                    no_both_cond=no_both_cond,
                    no_txt_cond=None,
                    **gen_kwargs,
                )

            # Save generated images
            generated_images: List[str] = []
            if generated_images_list:
                task_image_dir = os.path.join(self.generated_images_dir, task)
                os.makedirs(task_image_dir, exist_ok=True)
                for idx, img in enumerate(generated_images_list):
                    if img is not None and hasattr(img, "save"):
                        safe_filename = f"{task}_{doc_id}_stage1_{idx}.png"
                        image_path = os.path.join(task_image_dir, safe_filename)
                        img.save(image_path)
                        generated_images.append(image_path)
                        eval_logger.info(f"Saved generated image: {image_path}")

            return "", generated_images

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            raise

    # ── Visual CoT: Stage 2 ────────────────────────────────────────────

    def _stage2_answer_with_image(
        self,
        question: str,
        generation_prompt: str,
        image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using original + auxiliary image.
        """
        eval_logger.debug(f"Stage 2 - Answering for doc {doc_id}")

        try:
            auxiliary_image = Image.open(image_path).convert("RGB")

            images = []
            if original_image is not None:
                images.append(original_image)
            images.append(auxiliary_image)
            expected_image_count = len(images)

            query_text = generation_prompt + "\n\nQuestion: " + question
            existing_tags = query_text.count("<image>")

            if existing_tags == expected_image_count:
                query = query_text
            elif existing_tags < expected_image_count:
                tags_to_add = expected_image_count - existing_tags
                query = "<image>" * tags_to_add + "\n" + query_text
            else:
                parts = query_text.split("<image>")
                query = parts[0]
                for i in range(1, min(expected_image_count + 1, len(parts))):
                    query += "<image>" + parts[i]
                for i in range(expected_image_count + 1, len(parts)):
                    query += parts[i]

            _, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                text_or_conversations=query,
                images=images if images else None,
            )

            input_ids = input_ids.unsqueeze(0).to(self._device)
            attention_mask = torch.ones_like(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.to(
                    device=self._device, dtype=self.model.dtype
                )
            if grid_thws is not None:
                grid_thws = grid_thws.to(self._device)

            generate_kwargs = {
                "inputs": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
                "max_new_tokens": self.stage2_max_new_tokens,
                "use_cache": self.use_cache,
            }
            if self.stage2_temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = self.stage2_temperature
            else:
                generate_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = self.model.generate(**generate_kwargs)

            answer_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            del outputs, input_ids, pixel_values, grid_thws
            torch.cuda.empty_cache()

            return answer_text

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            raise

    # ── Standard understanding ──────────────────────────────────────────

    def _generate_standard(
        self,
        context: str,
        images: List[Image.Image],
        gen_kwargs: dict,
    ) -> str:
        """Standard single-stage understanding pipeline."""
        if len(images) > 0:
            image_placeholder = "<image>" * len(images)
            query = image_placeholder + context
        else:
            query = context

        try:
            result = self.model.preprocess_inputs(
                text_or_conversations=query,
                images=images if images else None,
            )
            _, input_ids, pixel_values, grid_thws = result
        except Exception as e:
            eval_logger.error(f"Error in preprocess_inputs: {e}")
            raise

        input_ids = input_ids.unsqueeze(0).to(self._device)
        attention_mask = torch.ones_like(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                device=self._device, dtype=self.model.dtype
            )
        if grid_thws is not None:
            grid_thws = grid_thws.to(self._device)

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0.0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        generate_kwargs = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            "use_cache": self.use_cache,
        }
        if gen_kwargs["temperature"] > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = gen_kwargs["temperature"]
            if gen_kwargs["top_p"] is not None:
                generate_kwargs["top_p"] = gen_kwargs["top_p"]
        else:
            generate_kwargs["do_sample"] = False
        if gen_kwargs["num_beams"] > 1:
            generate_kwargs["num_beams"] = gen_kwargs["num_beams"]

        outputs = self.model.generate(**generate_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        del outputs, input_ids, pixel_values, grid_thws
        torch.cuda.empty_cache()

        return response

    # ── Visual CoT pipeline ────────────────────────────────────────────

    def _generate_visual_cot(
        self,
        contexts: str,
        doc_to_visual,
        doc_id,
        task: str,
        split: str,
        gen_kwargs: dict,
    ) -> str:
        """Two-stage Visual CoT pipeline, auto-detected from [GEN_PROMPT] tags."""
        # Ensure output dirs exist
        os.makedirs(self.generated_images_dir, exist_ok=True)
        if self.save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)

        # Extract original image
        original_image = None
        if doc_to_visual is not None:
            try:
                doc = self.task_dict[task][split][doc_id]
                original_visuals = doc_to_visual(doc)
                if original_visuals and len(original_visuals) > 0:
                    original_image = self._extract_image(original_visuals[0])
                if original_image is None:
                    for field_name in [
                        "image",
                        "images",
                        "original_image",
                        "img",
                    ]:
                        if field_name in doc:
                            img_data = doc[field_name]
                            if isinstance(img_data, list) and len(img_data) > 0:
                                img_data = img_data[0]
                            original_image = self._extract_image(img_data)
                            if original_image is not None:
                                break
            except Exception as e:
                eval_logger.error(
                    f"Failed to extract original image for doc {doc_id}: {e}"
                )

        # Parse prompt tags
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
        else:
            actual_question = contexts
            generation_prompt = self.generation_prompt_template.format(
                question=contexts
            )

        eval_logger.info(f"Visual CoT for doc {doc_id}, task {task}")

        # Stage 1
        stage1_text, generated_images = self._stage1_generate_image(
            generation_prompt=generation_prompt,
            question=actual_question,
            doc_id=doc_id,
            task=task,
            original_image=original_image,
        )

        if not generated_images:
            eval_logger.warning(
                f"No image generated for doc {doc_id}, returning empty"
            )
            return stage1_text or ""

        # Stage 2
        final_answer = self._stage2_answer_with_image(
            question=actual_question,
            generation_prompt=generation_prompt,
            image_path=generated_images[0],
            doc_id=doc_id,
            original_image=original_image,
        )

        # Save intermediate artifacts
        self._save_intermediate_artifacts(
            doc_id=str(doc_id),
            task=task,
            generation_prompt=generation_prompt,
            stage1_text=stage1_text,
            generated_images=generated_images,
            question=actual_question,
            stage2_answer=final_answer,
        )

        return final_answer

    # ── Main entry point ────────────────────────────────────────────────

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text for each request, auto-detecting mode:
        - [GEN_PROMPT] tags → Visual CoT two-stage pipeline
        - bagel_interleaved config → Uni-MMMU interleaved generation
        - Otherwise → Standard understanding
        """
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = all_gen_kwargs[0]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            assert len(contexts) == 1, "Batch size must be 1"
            context = contexts[0]

            # ── Route: Uni-MMMU interleaved ──
            bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)
            if bagel_interleaved is not None:
                eval_logger.info(
                    f"Uni-MMMU interleaved mode for doc {doc_id[0]}"
                )
                doc = self.task_dict[task][split][doc_id[0]]
                input_images = []
                if doc_to_visual[0]:
                    visuals = doc_to_visual[0](doc)
                    if visuals:
                        input_images = (
                            visuals if isinstance(visuals, list) else [visuals]
                        )
                final_answer, generated_imgs = (
                    self.generate_uni_mmmu_interleaved(
                        input_images,
                        context,
                        str(doc_id[0]),
                        task,
                        bagel_interleaved,
                        doc,
                    )
                )
                self._save_intermediate_artifacts(
                    doc_id=str(doc_id[0]),
                    task=task,
                    generation_prompt=(
                        f"Interleaved: "
                        f"{bagel_interleaved.get('task_type', 'unknown')}"
                    ),
                    stage1_text="",
                    generated_images=generated_imgs,
                    question=context,
                    stage2_answer=final_answer,
                )
                res.append(final_answer)
                pbar.update(1)
                continue

            # ── Route: Visual CoT (explicit via gen_kwargs) ──
            if gen_kwargs.pop("visual_cot", False):
                answer = self._generate_visual_cot(
                    contexts=context,
                    doc_to_visual=doc_to_visual[0],
                    doc_id=doc_id[0],
                    task=task,
                    split=split,
                    gen_kwargs=gen_kwargs,
                )
                res.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), answer
                )
                pbar.update(1)
                continue

            # ── Route: Standard understanding ──
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in doc_id
            ]
            visuals = self.flatten(visuals)
            images = self._prepare_images(visuals)

            response = self._generate_standard(context, images, gen_kwargs)

            res.append(response)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), response
            )
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    # ── Uni-MMMU interleaved generation ─────────────────────────────────

    def generate_uni_mmmu_interleaved(
        self,
        input_images: List,
        prompt: str,
        doc_id: str,
        task: str,
        interleaved_config: dict,
        doc: dict = None,
    ) -> Tuple[str, List[str]]:
        """
        Uni-MMMU interleaved generation:
        - Jigsaw: gen_image(cand0) -> gen_image(cand1) -> gen_text(answer)
        - Maze/Sliding: [gen_text(plan) -> gen_image(step)]xk -> gen_text(answer)
        """
        task_type = interleaved_config.get("task_type", "jigsaw")
        num_images = interleaved_config.get("num_images", 2)

        # Get dynamic num_images from doc
        if doc is not None:
            if task_type == "maze":
                steps_str = doc.get("steps", "[]")
                steps = (
                    json.loads(steps_str)
                    if isinstance(steps_str, str)
                    else steps_str
                )
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                steps_str = doc.get("steps_words", "[]")
                steps = (
                    json.loads(steps_str)
                    if isinstance(steps_str, str)
                    else steps_str
                )
                if steps:
                    num_images = len(steps)

        original_image = None
        if input_images and len(input_images) > 0:
            original_image = self._extract_image(input_images[0])

        generated_images: List[str] = []
        task_output_dir = os.path.join(self.generated_images_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)

        if task_type == "jigsaw":
            # Generate 2 completed images then final answer
            for cand_idx in range(2):
                suffix = (
                    f"Output ONLY a single image with Candidate {cand_idx} "
                    f"placed in the bottom-right cell. No text."
                )
                gen_prompt = prompt + "\n\n" + suffix
                _, img_paths = self._stage1_generate_image(
                    generation_prompt=gen_prompt,
                    question="",
                    doc_id=f"{doc_id}_cand{cand_idx}",
                    task=task,
                    original_image=original_image,
                )
                if img_paths:
                    generated_images.extend(img_paths)

            # Final answer with all generated images
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>'
                '{"choice": 0 or 1, "rationale": "<=30 words"}'
                "</FINAL_ANSWER_JSON>\n"
                "Do not output any additional images."
            )
            final_text = self._answer_with_multiple_images(
                prompt + "\n\n" + final_suffix,
                original_image,
                generated_images,
            )
        else:
            # Maze/Sliding: iterative generation
            for i in range(1, num_images + 1):
                if task_type == "maze":
                    plan_suffix = (
                        f"Step {i}: Generate an image showing the next move "
                        f"(one step up/down/left/right)."
                    )
                else:
                    plan_suffix = (
                        f"Step {i}: Generate an image showing which tile "
                        f"to move and in which direction."
                    )
                gen_prompt = prompt + "\n\n" + plan_suffix
                _, img_paths = self._stage1_generate_image(
                    generation_prompt=gen_prompt,
                    question="",
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task,
                    original_image=original_image,
                )
                if img_paths:
                    generated_images.extend(img_paths)

            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY "
                "the final move list as <ANSWER_JSON>[...]</ANSWER_JSON>. "
                "No other text."
            )
            final_text = self._answer_with_multiple_images(
                prompt + "\n\n" + final_suffix,
                original_image,
                generated_images,
            )

        return final_text, generated_images

    def _answer_with_multiple_images(
        self,
        query_text: str,
        original_image: Optional[Image.Image],
        generated_image_paths: List[str],
    ) -> str:
        """Answer a question using original image + multiple generated images."""
        if not generated_image_paths:
            return ""

        images = []
        if original_image:
            images.append(original_image)
        for img_path in generated_image_paths:
            images.append(Image.open(img_path).convert("RGB"))

        image_placeholder = "<image>" * len(images)
        query = image_placeholder + "\n" + query_text

        _, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            text_or_conversations=query,
            images=images,
        )
        input_ids = input_ids.unsqueeze(0).to(self._device)
        attention_mask = torch.ones_like(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                device=self._device, dtype=self.model.dtype
            )
        if grid_thws is not None:
            grid_thws = grid_thws.to(self._device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=self.stage2_max_new_tokens,
                use_cache=self.use_cache,
                do_sample=False,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        del outputs, input_ids, pixel_values, grid_thws
        torch.cuda.empty_cache()
        return text

    # ── Not implemented ─────────────────────────────────────────────────

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for Ovis-U1")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round generation not yet implemented"
        )
