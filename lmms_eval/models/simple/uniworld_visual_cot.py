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

import torch
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

eval_logger = utils.eval_logger


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
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = True,
        # Hugging Face upload
        hf_repo: Optional[str] = None,
        hf_upload: bool = False,
        # Model loading
        min_pixels: int = 448 * 448,
        max_pixels: int = 448 * 448,
        no_joint_with_t5: bool = False,
        offload: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.flux_path = flux_path
        self.siglip_path = siglip_path
        self.save_intermediate = save_intermediate

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
        self.offload = offload

        # Hugging Face upload
        self.hf_upload = hf_upload
        self.hf_repo = hf_repo
        self.hf_api = None

        if self.hf_upload:
            if not self.hf_repo:
                eval_logger.warning("hf_upload=True but hf_repo not specified, disabling upload")
                self.hf_upload = False
            else:
                try:
                    from huggingface_hub import HfApi
                    self.hf_api = HfApi()
                    eval_logger.info(f"✅ Hugging Face upload enabled: {self.hf_repo}")
                except ImportError:
                    eval_logger.warning("huggingface_hub not installed, disabling upload")
                    self.hf_upload = False

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/uniworld_visual_cot"
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        eval_logger.info(f"Output directory: {self.output_dir}")

        # Load UniWorld models
        eval_logger.info(f"Loading UniWorld model from {pretrained}")
        self._load_uniworld_models()

        eval_logger.info("UniWorldVisualCoT initialized successfully")

    def _load_uniworld_models(self):
        """Load UniWorld model with full capabilities (generation + understanding)"""
        from lmms_eval.models.simple.uniworld import UniWorld

        # Load UniWorld with generation mode (includes all models: Qwen2.5-VL + FLUX + SigLIP)
        eval_logger.info("Loading UniWorld with full capabilities...")
        self.uniworld = UniWorld(
            pretrained=self.pretrained,
            flux_path=self.flux_path,
            siglip_path=self.siglip_path,
            mode="generation",  # Load all models
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
            offload=self.offload,
            image_output_dir=self.output_dir,
        )

        eval_logger.info("UniWorld loaded successfully (generation + understanding)")

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
    def batch_size(self):
        return 1  # Visual CoT processes one at a time

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method implementing two-stage visual CoT"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UniWorldVisualCoT",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            try:
                # Extract original image if available
                original_images = []
                if doc_to_visual is not None:
                    doc = self.task_dict[task][split][doc_id]
                    visuals = [doc_to_visual(doc)]
                    original_images = self.flatten(visuals)

                # Stage 1: Generate visualization
                eval_logger.info(f"[Doc {doc_id}] Stage 1: Generating visualization...")
                generated_image_path = self._stage1_generate(
                    prompt=contexts,
                    doc_id=doc_id,
                    task=task,
                    original_images=original_images,
                )

                if not generated_image_path:
                    eval_logger.warning(f"No image generated for doc {doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Stage 2: Answer with generated image
                eval_logger.info(f"[Doc {doc_id}] Stage 2: Understanding with visualization...")
                final_answer = self._stage2_understand(
                    prompt=contexts,
                    generated_image_path=generated_image_path,
                    original_images=original_images,
                    doc_id=doc_id,
                )

                res.append(final_answer)
                eval_logger.info(f"[Doc {doc_id}] ✅ Answer: {final_answer[:100]}...")

            except Exception as e:
                eval_logger.error(f"Error in visual CoT for doc_id={doc_id}: {e}")
                import traceback
                traceback.print_exc()
                res.append("")

            pbar.update(1)

        pbar.close()
        return res

    def _stage1_generate(
        self,
        prompt: str,
        doc_id: int,
        task: str,
        original_images: List,
    ) -> Optional[str]:
        """Stage 1: Generate visualization image"""
        # Create generation prompt
        gen_prompt = f"{prompt}\n\nGenerate a clear schematic visualization to help understand this problem."

        # Use UniWorld's generation capability
        try:
            # Call UniWorld's internal generation method
            output = self.uniworld._process_single_request(
                context=gen_prompt,
                doc_to_visual=None,  # No input images for pure generation
                doc_id=doc_id,
                task=task,
                split="",
                gen_kwargs={},
            )

            # Parse output to get image path
            if isinstance(output, str) and output.startswith("{"):
                output_dict = json.loads(output)
                images = output_dict.get("images", [])
                if images:
                    image_path = images[0]
                    # Upload to HF if enabled
                    if self.hf_upload and self.hf_api:
                        self._upload_to_hf(image_path, f"logs/{task}/images")
                    return image_path

            return None

        except Exception as e:
            eval_logger.error(f"Stage 1 generation failed: {e}")
            return None

    def _upload_to_hf(self, file_path: str, hf_path: str):
        """Upload file to Hugging Face"""
        try:
            self.hf_api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"{hf_path}/{os.path.basename(file_path)}",
                repo_id=self.hf_repo,
                repo_type="dataset",
            )
            eval_logger.debug(f"📤 Uploaded to HF: {file_path}")
        except Exception as e:
            eval_logger.warning(f"Failed to upload to HF: {e}")

    def _stage2_understand(
        self,
        prompt: str,
        generated_image_path: str,
        original_images: List,
        doc_id: int,
    ) -> str:
        """Stage 2: Understand with generated visualization"""
        try:
            # Load generated image
            generated_image = Image.open(generated_image_path).convert("RGB")

            # Combine original + generated images
            all_images = original_images + [generated_image]

            # Create understanding prompt
            und_prompt = f"{prompt}\n\nBased on the visualization, provide your answer."

            # Prepare messages for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in all_images],
                        {"type": "text", "text": und_prompt}
                    ]
                }
            ]

            # Process with Qwen2.5-VL (use UniWorld's processor and model)
            from qwen_vl_utils import process_vision_info

            text = self.uniworld.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.uniworld.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.uniworld._device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.uniworld.model.generate(
                    **inputs,
                    max_new_tokens=self.stage2_max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature if self.stage2_do_sample else None,
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.uniworld.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return output_text.strip()

        except Exception as e:
            eval_logger.error(f"Stage 2 understanding failed: {e}")
            return ""

    def flatten(self, item):
        """Flatten nested lists"""
        if isinstance(item, list):
            output = []
            for sub_item in item:
                output.extend(self.flatten(sub_item))
            return output
        else:
            return [item]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported"""
        raise NotImplementedError("UniWorldVisualCoT does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not yet implemented"""
        raise NotImplementedError("Multi-round not yet implemented for UniWorldVisualCoT")
