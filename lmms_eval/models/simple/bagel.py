import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add lmms-engine to Python path for official Bagel implementation.
# (We don't require the separate upstream Bagel repo for this wrapper.)
wd = Path(__file__).parent.parent.parent.parent.resolve()
engine_src_path = Path(wd).parent / "lmms-engine" / "src"
if engine_src_path.exists():
    engine_src_str = str(engine_src_path.resolve())
    if engine_src_str not in sys.path:
        sys.path.append(engine_src_str)
        eval_logger.info(f"Added lmms-engine path to sys.path: {engine_src_str}")
else:
    eval_logger.warning(f"lmms-engine src directory not found at {engine_src_path}. Please ensure lmms-engine is available.")


@register_model("bagel")
class Bagel(lmms):
    """
    Bagel Multimodal Model
    Supports text-to-image generation and image editing with optional thinking process

    Example usage:
    # Text-to-Image Generation
    accelerate launch -m lmms_eval \
        --model bagel \
        --model_args pretrained=/path/to/BAGEL-7B-MoT,task_mode=generate \
        --tasks ueval \
        --batch_size 1 \
        --output_path ./logs/

    # Image Editing (e.g., GEdit-Bench)
    accelerate launch -m lmms_eval \
        --model bagel \
        --model_args pretrained=/path/to/BAGEL-7B-MoT,task_mode=edit \
        --tasks gedit_bench \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        output_image_dir: Optional[str] = None,
        show_thinking: bool = False,
        task_mode: str = "auto",  # "auto" selects edit/generate based on input images per-sample
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,  # Image guidance scale for edit tasks
        cfg_interval: float = 0.4,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        max_think_token_n: int = 1024,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        seed: int = 0,
        image_ratio: str = "1:1",
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Import Bagel dependencies
        try:
            from lmms_engine.models.bagel.bagel import Bagel as BagelModel
            from lmms_engine.models.bagel.bagel import BagelConfig
            from lmms_engine.models.bagel.data_utils import add_special_tokens
            from lmms_engine.models.bagel.inferencer import InterleaveInferencer
            from lmms_engine.models.bagel.qwen2 import Qwen2Tokenizer
            from lmms_engine.models.bagel.transforms import ImageTransform

            self.add_special_tokens = add_special_tokens
            self.ImageTransform = ImageTransform
            self.InterleaveInferencer = InterleaveInferencer
            self.BagelConfig = BagelConfig
            self.BagelModel = BagelModel
            self.Qwen2Tokenizer = Qwen2Tokenizer

        except Exception as e:
            raise ImportError("Failed to import lmms_engine Bagel dependencies. " "Please ensure lmms-engine is available under the project root " "or installed as a package.\n" f"Error: {e}")

        self.pretrained = pretrained
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.show_thinking = show_thinking
        self.continual_mode = continual_mode
        if task_mode not in ["generate", "edit", "auto"]:
            raise ValueError(f"Invalid task_mode: {task_mode}. Must be 'generate', 'edit', or 'auto'")
        self.task_mode = task_mode

        eval_logger.info(f"Bagel task_mode: {task_mode}")

        # Validate quantization settings
        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot use both load_in_4bit and load_in_8bit")

        # Determine precision mode
        if load_in_4bit:
            self.precision_mode = "4bit"
        elif load_in_8bit:
            self.precision_mode = "8bit"
        else:
            self.precision_mode = "bf16"

        # Generation hyperparameters
        self.cfg_text_scale = cfg_text_scale
        self.cfg_img_scale = cfg_img_scale  # For edit tasks
        self.cfg_interval = cfg_interval
        self.timestep_shift = timestep_shift
        self.num_timesteps = num_timesteps
        self.cfg_renorm_min = cfg_renorm_min
        # Use different default cfg_renorm_type based on task_mode
        if task_mode == "edit" and cfg_renorm_type == "global":
            self.cfg_renorm_type = "text_channel"  # Better for edit tasks
        else:
            self.cfg_renorm_type = cfg_renorm_type
        self.max_think_token_n = max_think_token_n
        self.do_sample = do_sample
        self.text_temperature = text_temperature
        self.seed = seed
        self.image_ratio = image_ratio

        # Set image shapes based on ratio
        if image_ratio == "1:1":
            self.image_shapes = (1024, 1024)
        elif image_ratio == "4:3":
            self.image_shapes = (768, 1024)
        elif image_ratio == "3:4":
            self.image_shapes = (1024, 768)
        elif image_ratio == "16:9":
            self.image_shapes = (576, 1024)
        elif image_ratio == "9:16":
            self.image_shapes = (1024, 576)
        else:
            eval_logger.warning(f"Unknown image ratio {image_ratio}, using 1:1")
            self.image_shapes = (1024, 1024)

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/bagel_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        # Setup output directory for generated images
        # Priority: output_image_dir param > BAGEL_OUTPUT_IMAGE_DIR env > default
        if output_image_dir is not None:
            self.output_image_dir = output_image_dir
        elif os.getenv("BAGEL_OUTPUT_IMAGE_DIR"):
            self.output_image_dir = os.getenv("BAGEL_OUTPUT_IMAGE_DIR")
            eval_logger.info(f"Using BAGEL_OUTPUT_IMAGE_DIR: {self.output_image_dir}")
        else:
            self.output_image_dir = os.path.join(self.response_persistent_folder, "bagel_generated_images")

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, "bagel_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning("Continual mode is not supported for distributed inference. " "Automatically disabling continual_mode.")
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Load model
        eval_logger.info(f"Loading Bagel model from {pretrained}")
        self._load_model()

        eval_logger.info("Bagel model initialized successfully")

    def _build_inference_hyper(self, *, use_edit_mode: bool) -> dict:
        """Build inferencer kwargs, keeping behavior consistent across tasks."""
        common = {
            "max_think_token_n": self.max_think_token_n if self.show_thinking else 1024,
            "do_sample": self.do_sample if self.show_thinking else False,
            "text_temperature": self.text_temperature if self.show_thinking else 0.3,
            "cfg_text_scale": self.cfg_text_scale,
            "timestep_shift": self.timestep_shift,
            "num_timesteps": self.num_timesteps,
            "cfg_renorm_min": self.cfg_renorm_min,
            "cfg_renorm_type": self.cfg_renorm_type,
        }
        if use_edit_mode:
            common.update(
                cfg_img_scale=self.cfg_img_scale,
                cfg_interval=[0.0, 1.0],
            )
        else:
            common.update(
                cfg_interval=[self.cfg_interval, 1.0],
                image_shapes=self.image_shapes,
            )
        return common

    @staticmethod
    def _make_doc_uuid(task: str, split: str, doc_id) -> str:
        return f"{task}___{split}___{doc_id}"

    def _cache_get(self, doc_uuid: str) -> Optional[str]:
        if not self.continual_mode or self.cache_mode != "resume":
            return None
        return self.response_cache.get(doc_uuid)

    def _cache_put(self, doc_uuid: str, formatted_output: str) -> None:
        if not self.continual_mode:
            return
        self.response_cache[doc_uuid] = formatted_output
        with open(self.response_persistent_file, "w") as f:
            json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

    def _load_model(self):
        """
        Load Bagel model components.

        This follows the pattern from lmms-engine's bagel_grpo_trainer._create_inferencer_from_model:
        - Load model using from_pretrained (handles both original and HF format)
        - Get vae_model from model.vae_model
        - Create inferencer with model, vae_model, tokenizer, transforms
        """
        model_path = self.pretrained

        # Determine device
        local_rank = self._rank
        if hasattr(self, "accelerator") and self.accelerator is not None:
            local_rank = self.accelerator.local_process_index
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        eval_logger.info(f"Loading model to {device}")

        # Determine dtype based on precision mode
        if self.precision_mode == "bf16":
            inference_dtype = torch.bfloat16
        elif self.precision_mode == "4bit" or self.precision_mode == "8bit":
            # For quantization, we'll handle it separately
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.bfloat16

        # Check model format: original (ema.safetensors) vs HuggingFace (config.json with sharded weights)
        ema_path = os.path.join(model_path, "ema.safetensors")

        if os.path.exists(ema_path):
            # Original Bagel format - use Bagel.from_pretrained with training_config
            eval_logger.info("Detected original Bagel format (ema.safetensors)")

            # Create base config
            config = self.BagelConfig()

            # Training config for from_pretrained
            training_config = {
                "visual_gen": True,
                "visual_und": True,
                "layer_module": "Qwen2MoTDecoderLayer",
                "llm_qk_norm": True,
                "tie_word_embeddings": False,
                "vit_select_layer": -2,
                "vit_rope": False,
            }

            model = self.BagelModel.from_pretrained(
                model_path,
                config,
                training_config=training_config,
                torch_dtype=inference_dtype,
            )
            model = model.to(device).eval()
            eval_logger.info(f"Loaded original Bagel model to {device} (dtype={inference_dtype})")
        else:
            # HuggingFace format - use standard from_pretrained
            eval_logger.info("Detected HuggingFace format (no ema.safetensors)")
            # Load config from config.json
            config = self.BagelConfig.from_pretrained(model_path)
            # Note: Don't use device_map, use .to(device) instead to ensure
            # all tensors are on the same device. device_map uses accelerate hooks
            # which can cause device mismatch issues with the inferencer.
            model = self.BagelModel.from_pretrained(
                model_path,
                config,
                torch_dtype=inference_dtype,
            )
            # Explicitly move model to device (don't rely on device_map)
            model = model.to(device).eval()
            eval_logger.info(f"Loaded HuggingFace Bagel model to {device} (dtype={inference_dtype})")

        # Get VAE model from the loaded model (like _create_inferencer_from_model does)
        if hasattr(model, "vae_model"):
            vae_model = model.vae_model.to(device)
            eval_logger.info("Got vae_model from model.vae_model")
        else:
            raise ValueError("Model does not have vae_model attribute. Cannot create inferencer.")

        # Load tokenizer and add special tokens
        tokenizer = self.Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = self.add_special_tokens(tokenizer)

        # Create transforms (same as in bagel_grpo_trainer)
        vae_transform = self.ImageTransform(1024, 512, 16)
        vit_transform = self.ImageTransform(980, 224, 14)

        # Handle quantization if requested (applied after model loading)
        if self.precision_mode == "4bit":
            eval_logger.warning("4-bit quantization with from_pretrained may require additional setup. " "Consider using load_in_4bit parameter if available.")
        elif self.precision_mode == "8bit":
            eval_logger.warning("8-bit quantization with from_pretrained may require additional setup. " "Consider using load_in_8bit parameter if available.")

        # Patch model's prepare_* methods to move tensors to device
        # (InterleaveInferencer doesn't handle this, but Bagel.chat does manually)
        self._patch_prepare_methods(model, device)

        # Create inferencer (following bagel_grpo_trainer._create_inferencer_from_model pattern)
        self.inferencer = self.InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def _patch_prepare_methods(self, model, device):
        """Patch model's prepare_* methods to auto-move tensors to device."""

        def make_wrapper(orig_fn):
            def wrapper(*args, **kwargs):
                result = orig_fn(*args, **kwargs)
                # Handle tuple returns: (generation_input, ...)
                if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict):
                    for k, v in result[0].items():
                        if torch.is_tensor(v):
                            result[0][k] = v.to(device)
                elif isinstance(result, dict):
                    for k, v in result.items():
                        if torch.is_tensor(v):
                            result[k] = v.to(device)
                return result

            return wrapper

        for name in ["prepare_prompts", "prepare_vae_images", "prepare_vit_images", "prepare_vae_latent", "prepare_vae_latent_cfg", "prepare_start_tokens"]:
            if hasattr(model, name):
                setattr(model, name, make_wrapper(getattr(model, name)))

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._device

    def _doc_get(self, doc, key, default=None):
        if doc is None:
            return default
        if isinstance(doc, Mapping):
            return doc.get(key, default)
        return getattr(doc, key, default)

    def _extract_input_image(self, visuals):
        if visuals is None:
            return None
        if isinstance(visuals, (list, tuple)):
            imgs = []
            for visual in visuals:
                img = self._convert_visual_to_image(visual)
                if img is not None:
                    imgs.append(img)
            if not imgs:
                return None
            # Return a list if there are multiple images (for multi-image conditioning tasks)
            return imgs[0] if len(imgs) == 1 else imgs
        return self._convert_visual_to_image(visuals)

    def _convert_visual_to_image(self, visual):
        if visual is None:
            return None
        if hasattr(visual, "convert"):
            return visual.convert("RGB")
        return None

    def _should_use_edit_mode(self, input_image):
        if self.task_mode == "auto":
            return input_image is not None
        return self.task_mode == "edit"

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
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

    def generate_text_and_image(self, prompt: str, doc_id: str, task: str, input_image=None, key: str = None, source_image=None) -> Tuple[str, List[str]]:
        """
        Generate or edit image based on prompt and optional input image.

        Images are saved to: {output_dir}/{task}/{key}.png
        Task's process_results is responsible for reorganizing to task-specific structure.

        Args:
            prompt: Input text prompt (for generation) or editing instruction (for edit)
            doc_id: Document ID for file naming
            task: Task name for subdirectory
            input_image: Optional PIL Image for editing tasks
            key: Unique key for naming (defaults to doc_id if not provided)
            source_image: Original source image to save as {key}_SRCIMG.png

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        use_edit_mode = self._should_use_edit_mode(input_image)
        if use_edit_mode and input_image is None:
            eval_logger.warning(f"Edit mode requested but no input image provided for doc_id {doc_id}; falling back to generation.")
            use_edit_mode = False

        inference_hyper = self._build_inference_hyper(use_edit_mode=use_edit_mode)

        # NOTE: lmms-engine InterleaveInferencer.__call__ returns output_list[0].
        # When think=True, output_list[0] is the think text (str), and the image is later in the list.
        # So we must use interleave_inference to retrieve both text and image.
        if self.show_thinking:
            input_list = []
            if use_edit_mode:
                if isinstance(input_image, (list, tuple)):
                    for img in input_image:
                        if hasattr(img, "convert"):
                            img = img.convert("RGB")
                        input_list.append(img)
                else:
                    if hasattr(input_image, "convert"):
                        input_image = input_image.convert("RGB")
                    input_list.append(input_image)
            input_list.append(prompt)

            output_list = self.inferencer.interleave_inference(input_list, think=True, **inference_hyper)
            output_text = ""
            result = {"image": None}
            for item in output_list:
                if isinstance(item, str):
                    output_text = item
                elif isinstance(item, dict):
                    # image dict from gen_image
                    if "image" in item and item["image"] is not None:
                        result["image"] = item["image"]
                elif hasattr(item, "save"):
                    # PIL.Image
                    result["image"] = item
        else:
            if use_edit_mode:
                if isinstance(input_image, (list, tuple)):
                    imgs = []
                    for img in input_image:
                        if hasattr(img, "convert"):
                            img = img.convert("RGB")
                        imgs.append(img)
                    if len(imgs) <= 1:
                        result = self.inferencer(image=imgs[0] if imgs else None, text=prompt, think=False, **inference_hyper)
                    else:
                        output_list = self.inferencer.interleave_inference([*imgs, prompt], think=False, **inference_hyper)
                        output_text = ""
                        result = {"image": None}
                        for item in output_list:
                            if isinstance(item, str):
                                output_text = item
                            elif isinstance(item, dict):
                                if "image" in item and item["image"] is not None:
                                    result["image"] = item["image"]
                            elif hasattr(item, "save"):
                                result["image"] = item
                else:
                    if hasattr(input_image, "convert"):
                        input_image = input_image.convert("RGB")
                    result = self.inferencer(image=input_image, text=prompt, think=False, **inference_hyper)
            else:
                result = self.inferencer(text=prompt, think=False, **inference_hyper)

            # Extract text (if present)
            if "output_text" not in locals():
                output_text = result.get("text", "") if isinstance(result, dict) else str(result)

        # Save image to: {output_dir}/{task}/{key}.png
        # Task's process_results handles reorganization to task-specific structure
        output_images = []
        if isinstance(result, dict) and "image" in result and result["image"] is not None:
            image = result["image"]
            # Convert tensor to PIL Image if needed
            if torch.is_tensor(image):
                from PIL import Image as PILImage

                image = (image.permute(1, 2, 0).cpu() * 255).to(torch.uint8).numpy()
                image = PILImage.fromarray(image)

            save_key = key if key else str(doc_id)
            save_dir = os.path.join(self.output_image_dir, task)
            os.makedirs(save_dir, exist_ok=True)

            image_path = os.path.join(save_dir, f"{save_key}.png")
            image.save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved generated image: {image_path}")

            if source_image is not None and hasattr(source_image, "save"):
                src_image_path = os.path.join(save_dir, f"{save_key}_SRCIMG.png")
                source_image.save(src_image_path)
                eval_logger.info(f"Saved source image: {src_image_path}")

        return output_text, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        desc = "Bagel Editing" if self.task_mode == "edit" else "Bagel Generating"
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc=desc)

        for req in requests:
            # Unpack arguments: (ctx, generation_kwargs, doc_to_visual, doc_id, task, split)
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            doc_uuid = self._make_doc_uuid(task, split, doc_id)

            # Check cache
            cached = self._cache_get(doc_uuid)
            if cached:
                res.append(cached)
                pbar.update(1)
                continue

            # Get document (if available) and potential input images
            doc = None
            try:
                doc = self.task_dict[task][split][doc_id]
            except Exception as e:
                eval_logger.debug(f"Could not retrieve document for doc_id {doc_id}: {e}")

            input_image = None
            if callable(doc_to_visual) and doc is not None:
                try:
                    visuals = doc_to_visual(doc)
                    input_image = self._extract_input_image(visuals)
                    if input_image is not None:
                        eval_logger.debug(f"Detected input image for doc_id {doc_id}")
                except Exception as e:
                    eval_logger.warning(f"Failed to get input image for doc_id {doc_id}: {e}")

            key = self._doc_get(doc, "key", str(doc_id))
            source_image = self._doc_get(doc, "input_image") or self._doc_get(doc, "input_image_raw")
            if source_image is not None and hasattr(source_image, "convert"):
                source_image = source_image.convert("RGB")

            # Generate/Edit
            prompt = contexts
            output_text, output_images = self.generate_text_and_image(prompt, str(doc_id), task, input_image=input_image, key=key, source_image=source_image)

            # Format output
            formatted_output = self.format_output(output_text, output_images)
            res.append(formatted_output)

            # Update cache
            self._cache_put(doc_uuid, formatted_output)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError("Bagel is a generation model and does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")
