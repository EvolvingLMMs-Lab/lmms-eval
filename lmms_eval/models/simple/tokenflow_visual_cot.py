"""
TokenFlow Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt using TokenFlow-t2i
2. Stage 2: Answer question using the generated image with TokenFlow-i2t

Usage:
    python -m lmms_eval \
        --model tokenflow_visual_cot \
        --model_args pretrained_i2t=/path/to/TokenFlow-t2i,pretrained_understanding=/path/to/Tokenflow-llava-qwen2.5-14B-finetuning,tokenizer_path=/path/to/TokenFlow \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add TokenFlow repository to Python path
wd = Path(__file__).parent.parent.parent.parent.parent.resolve()  # from lmms-eval/lmms_eval/models/simple/
tokenflow_i2t_path = (wd / "TokenFlow" / "i2t").as_posix()
tokenflow_t2i_path = (wd / "TokenFlow" / "t2i").as_posix()

if os.path.exists(tokenflow_i2t_path):
    sys.path.insert(0, tokenflow_i2t_path)
    eval_logger.info(f"Added TokenFlow i2t path: {tokenflow_i2t_path}")
else:
    eval_logger.warning(f"TokenFlow i2t not found at {tokenflow_i2t_path}")

if os.path.exists(tokenflow_t2i_path):
    sys.path.insert(0, tokenflow_t2i_path)
    eval_logger.info(f"Added TokenFlow t2i path: {tokenflow_t2i_path}")
else:
    eval_logger.warning(f"TokenFlow t2i not found at {tokenflow_t2i_path}")


@register_model("tokenflow_visual_cot")
class TokenFlowVisualCoT(lmms):
    """
    TokenFlow Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt (t2i model)
    2. Answer question using the generated image (i2t understanding model)
    """

    def __init__(
        self,
        pretrained_i2t: Optional[str] = None,
        pretrained_understanding: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        # Stage 1: Image generation parameters (t2i)
        stage1_cfg: float = 7.5,
        stage1_loop: int = 1,
        stage1_batch_size: int = 1,
        stage1_mixed_precision: str = "bf16",
        # Stage 2: Visual understanding parameters (i2t)
        stage2_conv_template: str = "qwen_2_5",
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_top_p: Optional[float] = None,
        stage2_num_beams: int = 1,
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
        use_flash_attn: bool = False,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        # Multi-GPU configuration
        t2i_device: Optional[str] = None,  # Device for T2I model (default: cuda:0)
        i2t_device: Optional[str] = None,  # Device for I2T model (default: cuda:1)
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            eval_logger.warning(f"Unused kwargs in TokenFlowVisualCoT: {list(kwargs.keys())}")
        
        # Setup GPU configuration (default: single GPU)
        if t2i_device is None:
            self.t2i_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.t2i_device = torch.device(t2i_device)
        
        if i2t_device is None:
            # Default to same GPU as T2I (single GPU mode)
            self.i2t_device = self.t2i_device
        else:
            self.i2t_device = torch.device(i2t_device)
        
        if self.t2i_device == self.i2t_device:
            eval_logger.info(f"Single-GPU mode: Both T2I and I2T will use {self.t2i_device} (sequential loading)")
        else:
            eval_logger.info(f"Multi-GPU mode: T2I on {self.t2i_device}, I2T on {self.i2t_device}")

        # Auto-detect paths from models directory structure
        if pretrained_understanding is None and pretrained_i2t is None:
            raise ValueError("Either pretrained_understanding or pretrained_i2t must be provided")
        
        # If only pretrained_i2t provided, try to infer understanding path
        if pretrained_understanding is None and pretrained_i2t is not None:
            from pathlib import Path
            t2i_path = Path(pretrained_i2t)
            parent_dir = t2i_path.parent
            understanding_candidate = parent_dir / "Tokenflow-llava-qwen2.5-14B-finetuning"
            if understanding_candidate.exists():
                pretrained_understanding = str(understanding_candidate)
                eval_logger.info(f"Auto-detected pretrained_understanding: {pretrained_understanding}")
        
        # If only pretrained_understanding provided, try to infer t2i path
        if pretrained_i2t is None and pretrained_understanding is not None:
            from pathlib import Path
            understanding_path = Path(pretrained_understanding)
            parent_dir = understanding_path.parent
            t2i_candidate = parent_dir / "TokenFlow-t2i"
            if t2i_candidate.exists():
                pretrained_i2t = str(t2i_candidate)
                eval_logger.info(f"Auto-detected pretrained_i2t: {pretrained_i2t}")
        
        # Auto-detect tokenizer_path if not provided
        if tokenizer_path is None:
            from pathlib import Path
            if pretrained_understanding is not None:
                understanding_path = Path(pretrained_understanding)
                parent_dir = understanding_path.parent
                tokenizer_candidate = parent_dir / "TokenFlow"
                if tokenizer_candidate.exists():
                    tokenizer_path = str(tokenizer_candidate)
                    eval_logger.info(f"Auto-detected tokenizer_path: {tokenizer_path}")
        
        # Final validation
        if pretrained_i2t is None:
            raise ValueError("pretrained_i2t must be provided or auto-detected")
        if pretrained_understanding is None:
            raise ValueError("pretrained_understanding must be provided or auto-detected")
        if tokenizer_path is None:
            raise ValueError("tokenizer_path must be provided or auto-detected")

        self.pretrained_i2t = pretrained_i2t
        self.pretrained_understanding = pretrained_understanding
        self.tokenizer_path = tokenizer_path
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters (t2i)
        self.stage1_cfg = stage1_cfg
        self.stage1_loop = stage1_loop
        self.stage1_batch_size = stage1_batch_size
        self.stage1_mixed_precision = stage1_mixed_precision

        # Stage 2 parameters (i2t understanding)
        self.stage2_conv_template = stage2_conv_template
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_top_p = stage2_top_p
        self.stage2_num_beams = stage2_num_beams

        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_flash_attn = use_flash_attn
        self._device = torch.device(device)
        self.device_map = device_map

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/tokenflow_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved under: {self.intermediate_dir}")

        # Setup rank and world_size
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))

        # DO NOT load models here - they will be loaded on-demand in generate_until
        # This avoids OOM when both models are loaded simultaneously
        eval_logger.info("TokenFlow models will be loaded on-demand during inference")
        eval_logger.info(f"T2I model path: {pretrained_i2t}")
        eval_logger.info(f"I2T model path: {pretrained_understanding}")
        
        # Import dependencies
        self._load_dependencies()
        
        # Initialize model references as None
        self.t2i_model = None
        self.t2i_tokenizer = None
        self.i2t_model = None
        self.i2t_tokenizer = None
        self.i2t_image_processor = None

        eval_logger.info("TokenFlowVisualCoT initialized successfully")

    def _load_dependencies(self):
        """Load TokenFlow dependencies without loading models"""
        # Import dependencies
        try:
            # t2i dependencies
            from llava_t2i.model import LlavaLlamaForCausalLM as T2IModel
            from llava_t2i.dataset.process import crop_and_encode_text_and_img
            import transformers as t2i_transformers
            
            # i2t dependencies
            from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from llava.conversation import conv_templates
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.model.builder import load_pretrained_model
            from transformers import AutoTokenizer
            
            self.T2IModel = T2IModel
            self.crop_and_encode_text_and_img = crop_and_encode_text_and_img
            self.t2i_transformers = t2i_transformers
            
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.conv_templates = conv_templates
            self.get_model_name_from_path = get_model_name_from_path
            self.process_images = process_images
            self.tokenizer_image_token = tokenizer_image_token
            self.load_pretrained_model = load_pretrained_model
            self.AutoTokenizer = AutoTokenizer
            
        except Exception as e:
            raise ImportError(
                f"Failed to import TokenFlow dependencies. "
                f"Please ensure TokenFlow/i2t and TokenFlow/t2i are in sys.path. "
                f"Error: {e}"
            )

        # Multi-step inference strategy for t2i
        self.multi_step_infer_strategy = {
            1: {'topk_list': [600], 'topp_list': [0.6]},
            2: {'topk_list': [1200, 1], 'topp_list': [0.8, 0]},
            3: {'topk_list': [1200, 100, 1], 'topp_list': [0.8, 0.8, 0]},
        }

    def _load_t2i_model(self):
        """Load TokenFlow t2i model for image generation"""
        if self.t2i_model is not None:
            eval_logger.info("T2I model already loaded, skipping")
            return
            
        eval_logger.info(f"Loading TokenFlow t2i model on {self.t2i_device}...")
        
        # Vision tower path for t2i (enhanced version for generation)
        vision_tower_path_t2i = os.path.join(self.tokenizer_path, "tokenflow_clipb_32k_enhanced.pt")
        if not os.path.exists(vision_tower_path_t2i):
            raise FileNotFoundError(f"T2I vision tower not found at {vision_tower_path_t2i}")
        
        ptdtype = {
            'none': torch.float32,
            'bf16': torch.bfloat16,
            'fp16': torch.float16
        }[self.stage1_mixed_precision]
        
        # Load t2i model to specific device
        self.t2i_model = self.T2IModel.from_pretrained(
            self.pretrained_i2t,
            attn_implementation='eager',
            mm_vision_tower=vision_tower_path_t2i
        )
        self.t2i_model = self.t2i_model.eval()
        self.t2i_model = self.t2i_model.to(ptdtype).to(self.t2i_device)
        
        # Load vision tower explicitly
        vision_tower = self.t2i_model.get_vision_tower()
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=self.t2i_device, dtype=ptdtype)
            eval_logger.info(f"T2I vision tower loaded successfully on {self.t2i_device}")
        else:
            eval_logger.warning("T2I vision tower is None after model loading")
        
        self.t2i_model.config.mm_vision_vq_type = str(self.t2i_model.config.mm_vision_vq_type)
        mm_use_vq_token = getattr(self.t2i_model.config, "mm_use_vq_token", False)
        assert mm_use_vq_token, "T2I model must use VQ tokens"
        self.t2i_model.config.use_cache = False
        
        # Load t2i tokenizer
        self.t2i_tokenizer = self.t2i_transformers.AutoTokenizer.from_pretrained(
            self.pretrained_i2t,
            model_max_length=self.t2i_model.config.tokenizer_model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.t2i_model.reinit_image_token_start_end(self.t2i_tokenizer)
        
        eval_logger.info(f"T2I model loaded successfully on {self.t2i_device} (dtype: {ptdtype})")

    def _load_i2t_model(self):
        """Load TokenFlow i2t model for understanding"""
        if self.i2t_model is not None:
            eval_logger.info("I2T model already loaded, skipping")
            return
            
        eval_logger.info(f"Loading TokenFlow i2t model on {self.i2t_device}...")
        
        model_name = self.get_model_name_from_path(self.pretrained_understanding)
        
        # Vision tower path for i2t (standard version for understanding)
        vision_tower_path_i2t = os.path.join(self.tokenizer_path, "tokenflow_siglip_32k.pt")
        if not os.path.exists(vision_tower_path_i2t):
            raise FileNotFoundError(f"I2T vision tower not found at {vision_tower_path_i2t}")
        
        overwrite_config = {"mm_vision_tower": vision_tower_path_i2t}
        eval_logger.info(f"Setting i2t vision_tower to {vision_tower_path_i2t}")
        
        # Use device string directly instead of dict
        device_str = str(self.i2t_device)
        eval_logger.info(f"Loading i2t model with device: {device_str}")
        
        self.i2t_tokenizer, self.i2t_model, self.i2t_image_processor, self.i2t_max_length = self.load_pretrained_model(
            self.pretrained_understanding,
            None,
            model_name,
            load_8bit=self.load_in_8bit,
            load_4bit=self.load_in_4bit,
            device_map=device_str,
            device=device_str,
            use_flash_attn=self.use_flash_attn,
            overwrite_config=overwrite_config,
        )
        
        # Override i2t tokenizer from tokenizer_path
        try:
            tokenizer = self.AutoTokenizer.from_pretrained(self.tokenizer_path)
            from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            
            mm_use_im_start_end = getattr(self.i2t_model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(self.i2t_model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.i2t_model.resize_token_embeddings(len(tokenizer))
            self.i2t_tokenizer = tokenizer
            eval_logger.info(f"Loaded i2t tokenizer from {self.tokenizer_path}")
        except Exception as e:
            eval_logger.warning(f"Failed to load i2t tokenizer from {self.tokenizer_path}: {e}")
        
        self.i2t_model.eval()
        eval_logger.info(f"I2T model loaded successfully on {self.i2t_device}")

    def _unload_t2i_model(self):
        """Unload T2I model to free GPU memory"""
        if hasattr(self, 't2i_model') and self.t2i_model is not None:
            eval_logger.info("Unloading T2I model from GPU...")
            del self.t2i_model
            del self.t2i_tokenizer
            if hasattr(self, 't2i_processor'):
                del self.t2i_processor
            torch.cuda.empty_cache()
            eval_logger.info("T2I model unloaded successfully")
        else:
            eval_logger.warning("T2I model was not loaded, nothing to unload")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self.i2t_model

    @property
    def tokenizer(self):
        return self.i2t_tokenizer

    @property
    def device(self):
        return self._device

    def set_seed(self, seed: int):
        if seed > 0:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image=None
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt using TokenFlow t2i

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image (not used in t2i, kept for interface compatibility)

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Prepare negative prompt
            negative_prompt = (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, blurry."
            )

            # Encode prompts
            input_id, prefix_len = self.crop_and_encode_text_and_img(
                self.t2i_tokenizer, generation_prompt, image=None, max_text_token_num=128
            )
            uncondition_input_id, _ = self.crop_and_encode_text_and_img(
                self.t2i_tokenizer, negative_prompt, image=None, max_text_token_num=128
            )
            
            prefix_text_codes = [input_id, uncondition_input_id]
            
            # Get multi-step inference strategy
            topk_list = self.multi_step_infer_strategy[self.stage1_loop]['topk_list']
            topp_list = self.multi_step_infer_strategy[self.stage1_loop]['topp_list']
            
            # Generate image
            with torch.inference_mode():
                samples = self.t2i_model.autoregressive_infer_cfg(
                    B=1,
                    prefix_text_codes=prefix_text_codes,
                    cfg=self.stage1_cfg,
                    topk_list=topk_list,
                    topp_list=topp_list,
                    g_seed=None if self.seed == 0 else self.seed
                )
            
            # Save generated images
            generated_image_paths = []
            task_dir = os.path.join(self.intermediate_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            
            for iid, img in enumerate(samples):
                img_path = os.path.join(task_dir, f"{doc_id}_stage1_gen_{iid}.png")
                Image.fromarray(img.cpu().numpy().astype(np.uint8)).save(img_path)
                generated_image_paths.append(img_path)
                eval_logger.debug(f"Saved generated image to {img_path}")
            
            torch.cuda.empty_cache()
            
            return generation_prompt, generated_image_paths

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
        Stage 2: Answer question using generated image with TokenFlow i2t

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
        eval_logger.info(f"Stage 2 _stage2_answer_with_image called")

        try:
            # Load generated auxiliary image
            eval_logger.info(f"Loading auxiliary image from: {image_path}")
            auxiliary_image = Image.open(image_path).convert("RGB")
            eval_logger.info(f"Auxiliary image loaded: {auxiliary_image.size}")

            # Prepare visuals list
            if original_image is not None:
                eval_logger.info(f"🖼️  [Doc {doc_id}] Stage 2 - Using BOTH images: [Original, Generated]")
                eval_logger.info(f"  → Original: {type(original_image)}, size={original_image.size}")
                eval_logger.info(f"  → Generated: {type(auxiliary_image)}, size={auxiliary_image.size}")
                visuals = [original_image, auxiliary_image]
            else:
                eval_logger.info(f"🖼️  [Doc {doc_id}] Stage 2 - Using ONLY generated image (no original)")
                visuals = [auxiliary_image]

            # Build prompt
            prompt_question = self._build_prompt(question, len(visuals))
            eval_logger.info(f"Built prompt with {len(visuals)} images")
            eval_logger.info(f"Prompt preview: {prompt_question[:300]}...")

            # Process images
            eval_logger.info(f"Processing {len(visuals)} images")
            image_tensor = self.process_images(visuals, self.i2t_image_processor, self.i2t_model.config)
            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=self.i2t_device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.i2t_device)

            # Tokenize input and move to i2t device
            input_ids = self.tokenizer_image_token(
                prompt_question,
                self.i2t_tokenizer,
                self.IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.i2t_device)

            pad_token_id = self.i2t_tokenizer.pad_token_id if self.i2t_tokenizer.pad_token_id is not None else self.i2t_tokenizer.eos_token_id

            # Generate answer
            eval_logger.info(f"Starting model.generate with max_new_tokens={self.stage2_max_new_tokens}")
            with torch.inference_mode():
                cont = self.i2t_model.generate(
                    input_ids,
                    pad_token_id=pad_token_id,
                    images=image_tensor,
                    image_sizes=[img.size for img in visuals],
                    do_sample=True if self.stage2_temperature and self.stage2_temperature > 0 else False,
                    temperature=self.stage2_temperature,
                    top_p=self.stage2_top_p,
                    num_beams=self.stage2_num_beams,
                    max_new_tokens=self.stage2_max_new_tokens,
                    use_cache=True,
                )
            
            eval_logger.info(f"Generation completed, output shape: {cont.shape}")
            text_outputs = self.i2t_tokenizer.batch_decode(cont, skip_special_tokens=True)
            answer_text = text_outputs[0] if text_outputs else ""
            eval_logger.info(f"Decoded answer length: {len(answer_text)}")

            eval_logger.debug(f"Stage 2 - Generated answer: {answer_text[:100]}...")
            return answer_text

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            else:
                raise

    def _build_prompt(self, context: str, image_count: int) -> str:
        """Build prompt with image tokens"""
        if image_count > 0 and self.DEFAULT_IMAGE_TOKEN not in context:
            image_tokens = " ".join([self.DEFAULT_IMAGE_TOKEN] * image_count)
            question = image_tokens + "\n" + context
        else:
            question = context
        
        if "llama_3" in self.stage2_conv_template:
            conv = self.conv_templates[self.stage2_conv_template].copy()
        else:
            conv = self.conv_templates[self.stage2_conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
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
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def flatten(self, input_list):
        """Flatten nested lists"""
        if not input_list or any(i is None for i in input_list):
            return []
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Main inference method implementing two-stage visual CoT with sequential model loading

        Architecture:
        1. Load T2I model
        2. Process all Stage 1 requests (image generation)
        3. Unload T2I model, free GPU memory
        4. Load I2T model
        5. Process all Stage 2 requests (question answering)
        """
        eval_logger.info(f"Starting two-stage visual CoT with {len(requests)} requests")
        
        # Data structures to hold all request data
        stage1_data = []
        
        # Preprocessing: Extract all necessary data from requests
        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals:
                        original_visuals = self.flatten(original_visuals)
                        if len(original_visuals) > 0:
                            original_image = original_visuals[0]
                            eval_logger.info(f"✓ [Doc {doc_id}] Extracted original image: {type(original_image)}, size={getattr(original_image, 'size', 'N/A')}")
                        else:
                            eval_logger.warning(f"✗ [Doc {doc_id}] original_visuals is empty")
                    else:
                        eval_logger.warning(f"✗ [Doc {doc_id}] doc_to_visual returned None/empty")
                except Exception as e:
                    eval_logger.warning(f"✗ [Doc {doc_id}] Failed to extract original image: {e}")
            else:
                eval_logger.info(f"✗ [Doc {doc_id}] doc_to_visual is None - no original image available")

            # Parse contexts to extract generation_prompt if provided
            import re
            gen_prompt_match = re.search(r'\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]', contexts, re.DOTALL)
            question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', contexts, re.DOTALL)

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace("{question}", actual_question)
                # Update contexts to be just the question for stage 2
                stage2_contexts = contexts.replace(f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", "")
                stage2_contexts = stage2_contexts.replace(f"[QUESTION]{question_match.group(1)}[/QUESTION]", question_match.group(1))
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(question=contexts)
                stage2_contexts = contexts

            stage1_data.append({
                'doc_id': doc_id,
                'task': task,
                'split': split,
                'generation_prompt': generation_prompt,
                'stage2_contexts': stage2_contexts,
                'original_image': original_image,
                'gen_kwargs': gen_kwargs,
            })
        
        # ====================================================================
        # STAGE 1: Load T2I model and generate all images
        # ====================================================================
        eval_logger.info("="*70)
        eval_logger.info("STAGE 1: Loading T2I model and generating images")
        eval_logger.info("="*70)
        
        self._load_t2i_model()
        
        pbar_stage1 = tqdm(
            total=len(stage1_data),
            disable=(self.rank != 0),
            desc="Stage 1: Generating Images",
        )
        
        stage2_data = []
        for data in stage1_data:
            self.set_seed(self.seed)
            
            eval_logger.info(f"Stage 1: Processing doc {data['doc_id']}")
            
            # Generate visualization image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=data['generation_prompt'],
                doc_id=data['doc_id'],
                task=data['task'],
                original_image=data['original_image'],
            )
            
            # Store result for stage 2
            stage2_data.append({
                **data,
                'stage1_text': stage1_text,
                'generated_images': generated_images,
            })
            
            pbar_stage1.update(1)
        
        pbar_stage1.close()
        
        # ====================================================================
        # UNLOAD T2I MODEL
        # ====================================================================
        eval_logger.info("="*70)
        eval_logger.info("Unloading T2I model to free GPU memory")
        eval_logger.info("="*70)
        self._unload_t2i_model()
        
        # ====================================================================
        # STAGE 2: Load I2T model and answer all questions
        # ====================================================================
        eval_logger.info("="*70)
        eval_logger.info("STAGE 2: Loading I2T model and answering questions")
        eval_logger.info("="*70)
        
        self._load_i2t_model()
        
        pbar_stage2 = tqdm(
            total=len(stage2_data),
            disable=(self.rank != 0),
            desc="Stage 2: Answering Questions",
        )
        
        res = []
        for data in stage2_data:
            # Check if image was generated
            if not data['generated_images'] or len(data['generated_images']) == 0:
                eval_logger.warning(
                    f"No image generated for doc {data['doc_id']}, returning empty string"
                )
                res.append("")
                pbar_stage2.update(1)
                continue
            
            eval_logger.info(f"Stage 2: Processing doc {data['doc_id']}")
            eval_logger.info(f"  - Generated image path: {data['generated_images'][0]}")
            eval_logger.info(f"  - Original image: {'Yes ('+str(type(data['original_image']))+')' if data['original_image'] is not None else 'No'}")
            eval_logger.info(f"  - Question: {data['stage2_contexts'][:200]}...")
            
            # Answer question using generated image + original image
            final_answer = self._stage2_answer_with_image(
                question=data['stage2_contexts'],
                image_path=data['generated_images'][0],
                doc_id=data['doc_id'],
                original_image=data['original_image']
            )
            
            eval_logger.info(f"Stage 2: Completed doc {data['doc_id']}")
            eval_logger.info(f"  - Answer length: {len(final_answer)}")
            eval_logger.info(f"  - Answer preview: {final_answer[:200]}...")
            
            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=data['doc_id'],
                task=data['task'],
                generation_prompt=data['generation_prompt'],
                generated_images=data['generated_images'],
                question=data['stage2_contexts'],
                stage2_answer=final_answer,
            )
            
            res.append(final_answer)
            pbar_stage2.update(1)
        
        pbar_stage2.close()
        
        eval_logger.info("="*70)
        eval_logger.info(f"Two-stage visual CoT completed: {len(res)} results")
        eval_logger.info("="*70)
        
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "TokenFlowVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for TokenFlowVisualCoT"
        )
