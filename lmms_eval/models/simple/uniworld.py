import os
import sys
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

eval_logger = utils.eval_logger

# Add UniWorld path to sys.path
UNIWORLD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "UniWorld", "UniWorld-V1")
if os.path.exists(UNIWORLD_PATH):
    sys.path.insert(0, UNIWORLD_PATH)
    eval_logger.info(f"Added UniWorld path to sys.path: {UNIWORLD_PATH}")

# Apply transformers compatibility patch for UniWorld
try:
    from transformers import modeling_utils
    import torch
    from contextlib import contextmanager
    
    if not hasattr(modeling_utils, 'restore_default_torch_dtype'):
        @contextmanager
        def restore_default_torch_dtype(*args, **kwargs):
            """
            Compatibility shim for restore_default_torch_dtype.
            Accepts any arguments for compatibility but ignores them.
            """
            original_dtype = torch.get_default_dtype()
            try:
                yield
            finally:
                torch.set_default_dtype(original_dtype)
        
        modeling_utils.restore_default_torch_dtype = restore_default_torch_dtype
        eval_logger.info("Applied transformers compatibility patch for UniWorld")
except Exception as e:
    eval_logger.warning(f"Failed to apply transformers patch: {e}")

try:
    from transformers import AutoProcessor, SiglipImageProcessor, SiglipVisionModel
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
    from univa.models.qwen2p5vl.configuration_univa_qwen2p5vl import UnivaQwen2p5VLConfig
    from univa.utils.flux_pipeline import FluxPipeline
    from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
    from qwen_vl_utils import process_vision_info
    from univa.utils.anyres_util import dynamic_resize
    
    # Fix UniWorld config missing text_config
    if "text_config" not in UnivaQwen2p5VLConfig.sub_configs:
        UnivaQwen2p5VLConfig.sub_configs["text_config"] = Qwen2Config
        eval_logger.info("Applied UniWorld config patch: added text_config to sub_configs")
except ImportError as e:
    eval_logger.error(f"Failed to import UniWorld dependencies: {e}")
    eval_logger.error("Please ensure UniWorld repository is cloned to UniWorld/UniWorld-V1/")
    raise


@register_model("uniworld")
class UniWorld(lmms):
    """
    UniWorld: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation
    
    Paper: https://arxiv.org/abs/2506.03147
    GitHub: https://github.com/PKU-YuanGroup/UniWorld
    Model: https://huggingface.co/LanguageBind/UniWorld-V1
    
    Architecture:
    - Main Model: UnivaQwen2p5VLForConditionalGeneration (custom Qwen2.5-VL)
    - Vision Tower: Qwen2.5-VL visual encoder
    - Denoise Tower: FLUX.1-dev denoiser
    - Task Head: MLP classifier (understanding vs generation)
    - SigLIP: Reference image encoder for editing
    """

    def __init__(
        self,
        pretrained: str = "LanguageBind/UniWorld-V1",
        flux_path: str = "black-forest-labs/FLUX.1-dev",
        siglip_path: str = "google/siglip2-so400m-patch16-512",
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        min_pixels: int = 448 * 448,
        max_pixels: int = 448 * 448,
        no_joint_with_t5: bool = False,
        image_output_dir: str = "./uniworld_generated_images",
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.pretrained = pretrained
        self.flux_path = flux_path
        self.siglip_path = siglip_path
        self._device = device
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self._batch_size = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.no_joint_with_t5 = no_joint_with_t5
        
        # Image output directory
        self.output_image_dir = image_output_dir
        Path(self.output_image_dir).mkdir(parents=True, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")
        
        # Accelerator setup
        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = torch.device(device)
        
        # Load models
        self._load_models()
        
        # Image generation counter
        self.gen_image_counter = 0

    def _load_models(self):
        """Load all required models"""
        eval_logger.info(f"Loading UniWorld from {self.pretrained}")
        
        # 1. Load main UniWorld model (Qwen2.5-VL + Denoise Tower)
        # Using same loading method as original UniWorld app.py
        # Note: flash_attention_2 disabled due to UnivaDenoiseTower compatibility
        self.model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
            self.pretrained,
            torch_dtype=self._dtype,
            # attn_implementation="flash_attention_2",  # Disabled: UnivaDenoiseTower doesn't support it
        ).to(self._device)
        
        # 2. Load task head (classifier for understanding vs generation)
        self.task_head = nn.Sequential(
            nn.Linear(3584, 10240),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(10240, 2)
        ).to(self._device)
        task_head_path = os.path.join(self.pretrained, 'task_head_final.pt')
        if os.path.exists(task_head_path):
            self.task_head.load_state_dict(torch.load(task_head_path))
            eval_logger.info("Loaded task head")
        else:
            eval_logger.warning(f"Task head not found at {task_head_path}, using random weights")
        self.task_head.eval()
        
        # 3. Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.pretrained,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        
        # 4. Load FLUX pipeline
        self.pipe = FluxPipeline.from_pretrained(
            self.flux_path,
            transformer=self.model.denoise_tower.denoiser,
            torch_dtype=self._dtype,
        ).to(self._device)
        
        self.tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
        self.text_encoders = [self.pipe.text_encoder, self.pipe.text_encoder_2]
        
        # 5. Load SigLIP for reference image encoding
        self.siglip_processor = SiglipImageProcessor.from_pretrained(self.siglip_path)
        self.siglip_model = SiglipVisionModel.from_pretrained(
            self.siglip_path,
            torch_dtype=self._dtype,
        ).to(self._device)
        
        self.model.eval()
        self.siglip_model.eval()
        
        eval_logger.info("UniWorld models loaded successfully")

    @property
    def batch_size(self):
        return self._batch_size

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main generation interface"""
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="UniWorld Processing")
        
        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            
            # Check for Uni-MMMU interleaved mode
            uniworld_interleaved = gen_kwargs.get("uniworld_interleaved", None)
            
            if uniworld_interleaved is not None:
                # Uni-MMMU interleaved generation
                try:
                    doc = self.task_dict[task][split][doc_id]
                    input_images = []
                    if doc_to_visual:
                        visuals = [doc_to_visual(doc)]
                        input_images = self.flatten(visuals)
                    
                    output_text, output_images = self.generate_uni_mmmu_interleaved(
                        input_images, context, str(doc_id), task, uniworld_interleaved, doc
                    )
                    output = json.dumps({"text": output_text, "images": output_images}, ensure_ascii=False)
                except Exception as e:
                    eval_logger.error(f"Error in Uni-MMMU interleaved for doc_id={doc_id}: {e}")
                    output = json.dumps({"text": "", "images": []}, ensure_ascii=False)
            else:
                # Normal processing
                try:
                    output = self._process_single_request(
                        context, doc_to_visual, doc_id, task, split, gen_kwargs
                    )
                except Exception as e:
                    eval_logger.error(f"Error processing doc_id={doc_id}: {e}")
                    output = ""
            
            res.append(output)
            pbar.update(1)
        
        pbar.close()
        return res

    def _process_single_request(
        self,
        context: str,
        doc_to_visual,
        doc_id: int,
        task: str,
        split: str,
        gen_kwargs: Dict
    ) -> str:
        """Process a single request"""
        # Get visual input
        doc = self.task_dict[task][split][doc_id]
        visuals = [doc_to_visual(doc)]
        visuals = self.flatten(visuals)
        
        # Prepare conversation
        content = [{"type": "text", "text": context}]
        history_image_paths = []
        
        for visual in visuals:
            if visual is not None:
                content.append({
                    "type": "image",
                    "image": visual,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })
                if isinstance(visual, str):
                    history_image_paths.append(visual)
        
        conversation = [{"role": "user", "content": content}]
        
        # Prepare inputs
        chat_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])  # drop system
        
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        
        # Get task classification
        with torch.inference_mode():
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]  # B L D
        assistant_mask = inputs.input_ids == 77091  # assistant token
        assistant_vectors = hidden_states[assistant_mask][-1:]
        task_result = self.task_head(assistant_vectors.float())[0]
        
        if task_result[0] < task_result[1]:
            # Generation task
            output = self._generate_image(
                inputs, context, history_image_paths, doc_id, task
            )
        else:
            # Understanding task
            output = self._generate_text(inputs)
        
        return output

    def _generate_image(
        self,
        inputs,
        prompt_text: str,
        history_image_paths: List[str],
        doc_id: int,
        task: str
    ) -> str:
        """Generate image using FLUX pipeline"""
        # Encode reference images with SigLIP if available
        siglip_hidden_states = None
        if len(history_image_paths) > 0:
            siglip_pixel_values = []
            for img_path in history_image_paths:
                pixel_value = self.siglip_processor.preprocess(
                    images=Image.open(img_path).convert('RGB'),
                    do_resize=True,
                    return_tensors="pt",
                    do_convert_rgb=True
                ).pixel_values
                siglip_pixel_values.append(pixel_value)
            siglip_pixel_values = torch.concat(siglip_pixel_values).to(self._device)
            siglip_hidden_states = self.siglip_model(siglip_pixel_values).last_hidden_state
        
        # Get LVLM embeddings
        with torch.no_grad():
            lvlm_embeds = self.model(
                inputs.input_ids,
                pixel_values=getattr(inputs, 'pixel_values', None),
                attention_mask=inputs.attention_mask,
                image_grid_thw=getattr(inputs, 'image_grid_thw', None),
                siglip_hidden_states=siglip_hidden_states,
                output_type="denoise_embeds",
            )
        
        input_embeds = lvlm_embeds
        
        # Encode text prompt with T5
        if not self.no_joint_with_t5:
            t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
                self.text_encoders,
                self.tokenizers,
                prompt_text,
                256,
                self._device,
                1,
            )
            input_embeds = torch.concat([t5_prompt_embeds, input_embeds], dim=1)
        else:
            _, pooled_prompt_embeds = encode_prompt(
                self.text_encoders,
                self.tokenizers,
                "",
                256,
                self._device,
                1,
            )
        
        # Generate image
        output_image = self.pipe(
            prompt_embeds=input_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator(device=self._device).manual_seed(42),
        ).images[0]
        
        # Save image
        image_path = os.path.join(
            self.output_image_dir,
            f"{task}_{doc_id}_{self.gen_image_counter}.png"
        )
        self.gen_image_counter += 1
        output_image.save(image_path)
        
        # Return JSON format
        output = {
            "text": f"Generated image: {image_path}",
            "images": [image_path]
        }
        return json.dumps(output, ensure_ascii=False)

    def _generate_text(self, inputs) -> str:
        """Generate text using Qwen2.5-VL"""
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 0.0,
            )
        
        # Decode only newly generated tokens
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        reply = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return reply

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError("UniWorld does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("Multi-round generation not yet implemented for UniWorld")

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
        Uni-MMMU interleaved generation using UniWorld
        
        Aligned with Bagel's implementation:
        - Jigsaw: gen_image(cand0) → gen_image(cand1) → gen_text(answer)
        - Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
        """
        task_type = interleaved_config.get("task_type", "jigsaw")
        num_images = interleaved_config.get("num_images", 2)
        
        # Get num_images from doc if available
        if doc is not None:
            if task_type == "maze":
                steps = json.loads(doc.get("steps", "[]")) if isinstance(doc.get("steps", "[]"), str) else doc.get("steps", [])
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                steps = json.loads(doc.get("steps_words", "[]")) if isinstance(doc.get("steps_words", "[]"), str) else doc.get("steps_words", [])
                if steps:
                    num_images = len(steps)
        
        generated_images = []
        
        if task_type == "jigsaw":
            # Save input images temporarily
            temp_imgs = []
            for idx, img in enumerate(input_images):
                if img:
                    temp_path = os.path.join(self.output_image_dir, f"{task}_{doc_id}_input_{idx}.png")
                    if hasattr(img, 'save'):
                        img.save(temp_path)
                    temp_imgs.append(temp_path)
            
            # Generate Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            full_prompt1 = f"{prompt}\n{suffix1}"
            
            img0_output = self._generate_image_with_original(
                prompt_text=full_prompt1,
                original_image=input_images[0] if input_images else None,
                doc_id=f"{doc_id}_cand0",
                task=task
            )
            img0_dict = json.loads(img0_output)
            img0_path = img0_dict.get("images", [])[0] if img0_dict.get("images") else None
            if img0_path:
                generated_images.append(img0_path)
            
            # Generate Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            full_prompt2 = f"{prompt}\n{suffix2}"
            
            img1_output = self._generate_image_with_original(
                prompt_text=full_prompt2,
                original_image=input_images[0] if input_images else None,
                doc_id=f"{doc_id}_cand1",
                task=task
            )
            img1_dict = json.loads(img1_output)
            img1_path = img1_dict.get("images", [])[0] if img1_dict.get("images") else None
            if img1_path:
                generated_images.append(img1_path)
            
            # Generate final answer using understanding mode
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_question = f"{prompt}\n{final_suffix}"
            
            # Prepare all images for final answer
            all_images = input_images + [img0_path, img1_path]
            final_text = self._answer_with_images(
                question=final_question,
                images=all_images,
                doc_id=doc_id
            )
        
        else:
            # Maze/Sliding: alternating text plan and image generation
            step_texts = []
            step_images = []
            
            for i in range(1, num_images + 1):
                # Generate planning text
                if task_type == "maze":
                    plan_suffix = f'Now planning for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right."'
                else:
                    plan_suffix = f'Now planning for step {i}, Please output a sentence describing which tile to move and in which direction.'
                
                # Use understanding mode to generate plan
                plan_question = f"{prompt}\n{plan_suffix}"
                plan_text = self._answer_with_images(
                    question=plan_question,
                    images=input_images + step_images,
                    doc_id=f"{doc_id}_plan_{i}"
                )
                step_texts.append(plan_text)
                eval_logger.info(f"Step {i} plan: {plan_text}")
                
                # Generate step image
                img_suffix = f"Now, generate the image for step {i}."
                img_prompt = f"{prompt}\n{' '.join(step_texts)}\n{img_suffix}"
                
                img_output = self._generate_image_with_original(
                    prompt_text=img_prompt,
                    original_image=input_images[0] if input_images else None,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task
                )
                img_dict = json.loads(img_output)
                img_path = img_dict.get("images", [])[0] if img_dict.get("images") else None
                if img_path:
                    generated_images.append(img_path)
                    step_images.append(img_path)
                    eval_logger.info(f"Saved step {i} image: {img_path}")
            
            # Generate final answer
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            final_question = f"{prompt}\n{' '.join(step_texts)}\n{final_suffix}"
            
            all_images = input_images + step_images
            final_text = self._answer_with_images(
                question=final_question,
                images=all_images,
                doc_id=doc_id
            )
            eval_logger.info(f"{task_type} final answer: {final_text}")
        
        return final_text, generated_images

    def _generate_image_with_original(
        self, prompt_text: str, original_image, doc_id: str, task: str
    ) -> str:
        """Helper for Visual CoT: Generate image conditioned on original image"""
        # Prepare inputs similar to _generate_image but with original image
        history_image_paths = []
        if original_image:
            if isinstance(original_image, str):
                history_image_paths.append(original_image)
            else:
                # Save temp image
                temp_path = os.path.join(self.output_image_dir, f"temp_{doc_id}.png")
                original_image.save(temp_path)
                history_image_paths.append(temp_path)
        
        # Create fake inputs for generation
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        for img_path in history_image_paths:
            conversation[0]["content"].append({
                "type": "image",
                "image": img_path,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels
            })
        
        chat_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])
        
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        
        # Generate image
        return self._generate_image(inputs, prompt_text, history_image_paths, doc_id, task)

    def _answer_with_images(self, question: str, images: List, doc_id: str) -> str:
        """Helper for Visual CoT: Answer question with multiple images"""
        # Prepare conversation with multiple images
        content = [{"type": "text", "text": question}]
        for img in images:
            if img is not None:
                content.append({
                    "type": "image",
                    "image": img if isinstance(img, str) else img,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })
        
        conversation = [{"role": "user", "content": content}]
        
        chat_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])
        
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        
        # Generate text answer
        return self._generate_text(inputs)

    def flatten(self, input_list):
        """Flatten nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output
