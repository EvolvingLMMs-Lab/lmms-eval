import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add Bagel repository to Python path
# Expected: lmms-eval/Bagel/ directory at project root
wd = Path(__file__).parent.parent.parent.parent.resolve()
bagel_path = os.path.join(str(wd), "Bagel")
if os.path.exists(bagel_path):
    sys.path.append(bagel_path)
    eval_logger.info(f"Added Bagel path to sys.path: {bagel_path}")
else:
    eval_logger.warning(f"Bagel repository not found at {bagel_path}. " f"Please clone it: cd {wd} && git clone https://github.com/ByteDance-Seed/Bagel.git")


def _check_bagel_modifications(bagel_path: str) -> bool:
    """
    Check if the required modifications have been made to the Bagel repository.

    Returns True if modifications are detected, False otherwise.
    """
    modifications_needed = []

    # Check 1: Bagel/modeling/bagel/bagel.py - forward_cache_update_vae dtype/device fix
    bagel_model_file = os.path.join(bagel_path, "modeling/bagel/bagel.py")
    if os.path.exists(bagel_model_file):
        with open(bagel_model_file, "r") as f:
            content = f.read()
            if "padded_images.to(device=vae_model.encoder.conv_in.weight.device" not in content:
                modifications_needed.append("Bagel/modeling/bagel/bagel.py: forward_cache_update_vae() method needs dtype/device fix for VAE encode")

    # Check 2: Bagel/inferencer.py - decode_image dtype/device fix
    inferencer_file = os.path.join(bagel_path, "inferencer.py")
    if os.path.exists(inferencer_file):
        with open(inferencer_file, "r") as f:
            content = f.read()
            if "latent.to(device=self.vae_model.decoder.conv_in.weight.device" not in content:
                modifications_needed.append("Bagel/inferencer.py: decode_image() method needs dtype/device fix for VAE decode")

    if modifications_needed:
        eval_logger.warning("=" * 80)
        eval_logger.warning("IMPORTANT: Bagel repository requires modifications to work with lmms-eval!")
        eval_logger.warning("=" * 80)
        eval_logger.warning("")
        eval_logger.warning("The following modifications are needed to fix dtype/device mismatch errors:")
        eval_logger.warning("")
        for i, mod in enumerate(modifications_needed, 1):
            eval_logger.warning(f"  {i}. {mod}")
        eval_logger.warning("")
        eval_logger.warning("Required changes:")
        eval_logger.warning("")
        eval_logger.warning("1. In Bagel/modeling/bagel/bagel.py, find forward_cache_update_vae() method and add:")
        eval_logger.warning("   BEFORE: padded_latent = vae_model.encode(padded_images)")
        eval_logger.warning("   AFTER:  padded_images = padded_images.to(device=vae_model.encoder.conv_in.weight.device, dtype=vae_model.encoder.conv_in.weight.dtype)")
        eval_logger.warning("           padded_latent = vae_model.encode(padded_images)")
        eval_logger.warning("")
        eval_logger.warning("2. In Bagel/inferencer.py, find decode_image() method and add:")
        eval_logger.warning("   BEFORE: image = self.vae_model.decode(latent)")
        eval_logger.warning("   AFTER:  latent = latent.to(device=self.vae_model.decoder.conv_in.weight.device, dtype=self.vae_model.decoder.conv_in.weight.dtype)")
        eval_logger.warning("           image = self.vae_model.decode(latent)")
        eval_logger.warning("")
        eval_logger.warning("=" * 80)
        return False
    return True


# Check for required Bagel modifications when module is loaded
if os.path.exists(bagel_path):
    _check_bagel_modifications(bagel_path)


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
        task_mode: str = "generate",  # "generate" for text-to-image, "edit" for image editing
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
            from data.data_utils import add_special_tokens, pil_img2rgb
            from data.transforms import ImageTransform
            from inferencer import InterleaveInferencer
            from modeling.autoencoder import load_ae
            from modeling.bagel import Bagel as BagelModel
            from modeling.bagel import (
                BagelConfig,
                Qwen2Config,
                Qwen2ForCausalLM,
                SiglipVisionConfig,
                SiglipVisionModel,
            )
            from modeling.qwen2 import Qwen2Tokenizer

            self.add_special_tokens = add_special_tokens
            self.pil_img2rgb = pil_img2rgb
            self.ImageTransform = ImageTransform
            self.InterleaveInferencer = InterleaveInferencer
            self.load_ae = load_ae
            self.BagelConfig = BagelConfig
            self.BagelModel = BagelModel
            self.Qwen2Config = Qwen2Config
            self.Qwen2ForCausalLM = Qwen2ForCausalLM
            self.SiglipVisionConfig = SiglipVisionConfig
            self.SiglipVisionModel = SiglipVisionModel
            self.Qwen2Tokenizer = Qwen2Tokenizer

        except Exception as e:
            raise ImportError(
                f"Failed to import Bagel dependencies. "
                f"Please ensure:\n"
                f"  1. Bagel repository is cloned at lmms-eval root: "
                f"git clone https://github.com/ByteDance-Seed/Bagel.git\n"
                f"  2. Model weights are downloaded to Bagel/models/BAGEL-7B-MoT/\n"
                f"Error: {e}"
            )

        self.pretrained = pretrained
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.show_thinking = show_thinking
        self.continual_mode = continual_mode
        self.task_mode = task_mode  # "generate" or "edit"

        # Validate task mode
        if task_mode not in ["generate", "edit"]:
            raise ValueError(f"Invalid task_mode: {task_mode}. Must be 'generate' or 'edit'")

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

        # Check for task-specific output directory from environment variables
        # Priority: output_image_dir param > IMGEDIT_OUTPUT_DIR > GEDIT_BENCH_OUTPUT_DIR > default
        if output_image_dir is not None:
            self.output_image_dir = output_image_dir
        elif os.getenv("IMGEDIT_OUTPUT_DIR"):
            self.output_image_dir = os.getenv("IMGEDIT_OUTPUT_DIR")
            eval_logger.info(f"Using IMGEDIT_OUTPUT_DIR: {self.output_image_dir}")
        elif os.getenv("GEDIT_BENCH_OUTPUT_DIR"):
            self.output_image_dir = os.getenv("GEDIT_BENCH_OUTPUT_DIR")
            eval_logger.info(f"Using GEDIT_BENCH_OUTPUT_DIR: {self.output_image_dir}")
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

    def _load_model(self):
        """Load Bagel model components"""
        model_path = self.pretrained

        # Load configurations
        llm_config = self.Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = self.SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        # Load VAE
        vae_model, vae_config = self.load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        # Create model config
        config = self.BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        # Initialize model with empty weights
        with init_empty_weights():
            language_model = self.Qwen2ForCausalLM(llm_config)
            vit_model = self.SiglipVisionModel(vit_config)
            model = self.BagelModel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Load tokenizer
        tokenizer = self.Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = self.add_special_tokens(tokenizer)

        # Create transforms
        vae_transform = self.ImageTransform(1024, 512, 16)
        vit_transform = self.ImageTransform(980, 224, 14)

        # Load checkpoint based on precision mode
        checkpoint_path = os.path.join(model_path, "ema.safetensors")

        local_rank = self._rank
        if hasattr(self, "accelerator") and self.accelerator is not None:
            local_rank = self.accelerator.local_process_index
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        eval_logger.info(f"Loading model to {device}")

        device_map = {"": device}

        if self.precision_mode == "bf16":
            inference_dtype = torch.bfloat16
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=checkpoint_path,
                device_map=device_map,
                offload_buffers=False,
                dtype=inference_dtype,
                force_hooks=True,
            ).eval()
            eval_logger.info("Loaded model in BFloat16 precision")

        elif self.precision_mode == "4bit":
            # NF4: 4-bit quantization
            try:
                from accelerate.utils import (
                    BnbQuantizationConfig,
                    load_and_quantize_model,
                )

                bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
                model = load_and_quantize_model(
                    model,
                    weights_location=checkpoint_path,
                    bnb_quantization_config=bnb_quantization_config,
                    device_map=device_map,
                    offload_folder="offload",
                ).eval()
                eval_logger.info("Loaded model in 4-bit (NF4) quantization")
            except ImportError:
                raise ImportError("4-bit quantization requires bitsandbytes. " "Install it with: pip install bitsandbytes")
            inference_dtype = torch.bfloat16

        elif self.precision_mode == "8bit":
            # INT8: 8-bit quantization
            try:
                from accelerate.utils import (
                    BnbQuantizationConfig,
                    load_and_quantize_model,
                )

                bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
                model = load_and_quantize_model(
                    model,
                    weights_location=checkpoint_path,
                    bnb_quantization_config=bnb_quantization_config,
                    device_map=device_map,
                    offload_folder="offload",
                ).eval()
                eval_logger.info("Loaded model in 8-bit (INT8) quantization")
            except ImportError:
                raise ImportError("8-bit quantization requires bitsandbytes. " "Install it with: pip install bitsandbytes")
            inference_dtype = torch.float32

        else:
            raise ValueError(f"Unknown precision mode: {self.precision_mode}")

        # Move VAE model to the same device/dtype as main model
        vae_model = vae_model.to(device, dtype=inference_dtype)
        eval_logger.info(f"Moved VAE model to {device} (dtype={inference_dtype})")

        # Create inferencer
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

    def generate_text_and_image(self, prompt: str, doc_id: str, task: str, input_image=None, key: str = None, task_type: str = None, instruction_language: str = None, source_image=None, edit_type: str = None) -> Tuple[str, List[str]]:
        """
        Generate or edit image based on prompt and optional input image

        Args:
            prompt: Input text prompt (for generation) or editing instruction (for edit)
            doc_id: Document ID for file naming
            task: Task name for file naming
            input_image: Optional PIL Image for editing tasks
            key: Unique key for naming (used by GEdit-Bench and ImgEdit)
            task_type: Task type for GEdit-Bench (e.g., "background_change")
            instruction_language: Language for GEdit-Bench ("en" or "cn")
            source_image: Original source image to save as _SRCIMG
            edit_type: Edit type for ImgEdit (e.g., "replace", "add", "adjust")

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        if self.task_mode == "edit":
            # Image editing mode
            if input_image is None:
                eval_logger.warning(f"Edit mode but no input image provided for doc_id {doc_id}")
                return "", []

            # Prepare edit hyperparameters (different from generation)
            inference_hyper = {
                "max_think_token_n": self.max_think_token_n if self.show_thinking else 1024,
                "do_sample": self.do_sample if self.show_thinking else False,
                "text_temperature": self.text_temperature if self.show_thinking else 0.3,
                "cfg_text_scale": self.cfg_text_scale,
                "cfg_img_scale": self.cfg_img_scale,
                "cfg_interval": [0.0, 1.0],  # Edit tasks use [0.0, 1.0]
                "timestep_shift": self.timestep_shift,
                "num_timesteps": self.num_timesteps,
                "cfg_renorm_min": self.cfg_renorm_min,
                "cfg_renorm_type": self.cfg_renorm_type,
            }

            # Ensure input_image is RGB
            if hasattr(input_image, "convert"):
                input_image = input_image.convert("RGB")

            # Generate edited image
            result = self.inferencer(image=input_image, text=prompt, think=self.show_thinking, **inference_hyper)
        else:
            # Text-to-image generation mode
            # Prepare generation hyperparameters
            inference_hyper = {
                "max_think_token_n": self.max_think_token_n if self.show_thinking else 1024,
                "do_sample": self.do_sample if self.show_thinking else False,
                "text_temperature": self.text_temperature if self.show_thinking else 0.3,
                "cfg_text_scale": self.cfg_text_scale,
                "cfg_interval": [self.cfg_interval, 1.0],
                "timestep_shift": self.timestep_shift,
                "num_timesteps": self.num_timesteps,
                "cfg_renorm_min": self.cfg_renorm_min,
                "cfg_renorm_type": self.cfg_renorm_type,
                "image_shapes": self.image_shapes,
            }

            # Generate new image
            result = self.inferencer(text=prompt, think=self.show_thinking, **inference_hyper)

        # Extract text
        output_text = result.get("text", "")

        # Save image based on task type
        output_images = []
        if "image" in result and result["image"] is not None:
            image = result["image"]

            # Check if this is ImgEdit task (has IMGEDIT_MODEL_NAME env var or edit_type)
            imgedit_model_name = os.getenv("IMGEDIT_MODEL_NAME")
            gedit_model_name = os.getenv("GEDIT_BENCH_MODEL_NAME")

            if imgedit_model_name and key and edit_type:
                # ImgEdit style path: {output_dir}/{model_name}/{key}.png
                save_dir = os.path.join(self.output_image_dir, imgedit_model_name)
                os.makedirs(save_dir, exist_ok=True)

                # Save generated image
                image_path = os.path.join(save_dir, f"{key}.png")
                image.save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Saved ImgEdit image: {image_path}")

                # Save source image as _SRCIMG if provided
                if source_image is not None:
                    src_image_path = os.path.join(save_dir, f"{key}_SRCIMG.png")
                    if hasattr(source_image, "save"):
                        source_image.save(src_image_path)
                        eval_logger.info(f"Saved source image: {src_image_path}")

            elif key and task_type and instruction_language:
                # GEdit-Bench style path: {output_dir}/{model_name}/fullset/{task_type}/{instruction_language}/{key}.png
                model_name = gedit_model_name or "bagel"
                save_dir = os.path.join(self.output_image_dir, model_name, "fullset", task_type, instruction_language)
                os.makedirs(save_dir, exist_ok=True)

                # Save generated image
                image_path = os.path.join(save_dir, f"{key}.png")
                image.save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Saved GEdit-Bench image: {image_path}")

                # Save source image as _SRCIMG if provided
                if source_image is not None:
                    src_image_path = os.path.join(save_dir, f"{key}_SRCIMG.png")
                    if hasattr(source_image, "save"):
                        source_image.save(src_image_path)
                        eval_logger.info(f"Saved source image: {src_image_path}")
            else:
                # Fallback to simple naming
                safe_filename = f"{task}_{doc_id}.png"
                image_path = os.path.join(self.output_image_dir, safe_filename)
                image.save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Saved image: {image_path}")

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

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for req in requests:
            # Unpack arguments: (ctx, generation_kwargs, doc_to_visual, doc_id, task, split)
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            # Get input image and doc metadata for edit tasks
            input_image = None
            source_image = None
            key = None
            task_type = None
            instruction_language = None
            edit_type = None  # For ImgEdit
            doc = None

            # Try to get the document from task dataset
            try:
                doc = self.task_dict[task][split][doc_id]
            except Exception as e:
                eval_logger.debug(f"Could not get doc for doc_id {doc_id}: {e}")

            if self.task_mode == "edit":
                # doc_to_visual is a function that returns list of images
                if callable(doc_to_visual) and doc is not None:
                    try:
                        visuals = doc_to_visual(doc)
                        if visuals and len(visuals) > 0:
                            input_image = visuals[0]
                            if hasattr(input_image, "convert"):
                                input_image = input_image.convert("RGB")
                            eval_logger.debug(f"Got input image for doc_id {doc_id}")
                    except Exception as e:
                        eval_logger.warning(f"Failed to get input image for doc_id {doc_id}: {e}")

            # Extract task-specific fields from doc
            if doc is not None:
                key = doc.get("key", str(doc_id))
                # GEdit-Bench specific fields
                task_type = doc.get("task_type", "unknown")
                instruction_language = doc.get("instruction_language", "en")
                # ImgEdit specific fields
                edit_type = doc.get("edit_type")  # e.g., "replace", "add", "adjust"
                # Get source image (original un-resized image) for saving as _SRCIMG
                source_image = doc.get("input_image") or doc.get("input_image_raw")
                if source_image and hasattr(source_image, "convert"):
                    source_image = source_image.convert("RGB")

            # Generate/Edit
            prompt = contexts
            output_text, output_images = self.generate_text_and_image(
                prompt, str(doc_id), task, input_image=input_image, key=key, task_type=task_type, instruction_language=instruction_language, source_image=source_image, edit_type=edit_type
            )

            # Format output
            formatted_output = self.format_output(output_text, output_images)
            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError("Bagel is a generation model and does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")
