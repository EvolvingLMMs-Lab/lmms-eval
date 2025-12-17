import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import (
    Accelerator,
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
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


@register_model("bagel")
class Bagel(lmms):
    """
    Bagel Multimodal Model
    Supports both image understanding and text-to-image generation

    Modes:
        - "understanding": Visual understanding (image + text -> text)
        - "generation": Image generation (text -> image)

    Example usage for understanding:
    accelerate launch -m lmms_eval \
        --model bagel \
        --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=understanding \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/

    Example usage for generation:
    accelerate launch -m lmms_eval \
        --model bagel \
        --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=generation \
        --tasks ueval \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        mode: str = "generation",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        output_image_dir: Optional[str] = None,
        show_thinking: bool = False,
        cfg_text_scale: float = 4.0,
        cfg_interval: float = 0.4,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        max_think_token_n: int = 1024,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        seed: int = 0,
        image_ratio: str = "1:1",
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(f"mode must be 'understanding' or 'generation', got '{mode}'")

        self.mode = mode
        self.max_new_tokens = max_new_tokens

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
        self.cfg_interval = cfg_interval
        self.timestep_shift = timestep_shift
        self.num_timesteps = num_timesteps
        self.cfg_renorm_min = cfg_renorm_min
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

        if output_image_dir is None:
            self.output_image_dir = os.path.join(self.response_persistent_folder, "bagel_generated_images")
        else:
            self.output_image_dir = output_image_dir

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

        # Setup device map for multi-GPU
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        # Ensure certain modules are on the same device
        same_device_modules = ["language_model.model.embed_tokens", "time_embedder", "latent_pos_embed", "vae2llm", "llm2vae", "connector", "vit_pos_embed"]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = first_device if k in device_map else "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Load checkpoint based on precision mode
        checkpoint_path = os.path.join(model_path, "ema.safetensors")

        if self.precision_mode == "bf16":
            # BF16: Full precision
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=checkpoint_path,
                device_map=device_map,
                offload_buffers=True,
                offload_folder="offload",
                dtype=torch.bfloat16,
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

        else:
            raise ValueError(f"Unknown precision mode: {self.precision_mode}")

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

    def understand_image(self, prompt: str, image, doc_id: str) -> str:
        """
        Understand image and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        self.set_seed(self.seed)

        # Call inferencer in understanding mode
        result = self.inferencer(
            image=image,
            text=prompt,
            understanding_output=True,
            think=self.show_thinking,
            max_think_token_n=self.max_new_tokens if self.show_thinking else self.max_new_tokens,
            do_sample=self.do_sample,
            text_temperature=self.text_temperature,
        )

        # Extract text answer
        output_text = result.get("text", "")
        return output_text

    def generate_text_and_image(self, prompt: str, doc_id: str, task: str) -> Tuple[str, List[str]]:
        """
        Generate text and image from prompt

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        # Prepare inference hyperparameters
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

        # Generate
        result = self.inferencer(text=prompt, think=self.show_thinking, **inference_hyper)

        # Extract text
        output_text = result.get("text", "")

        # Save image
        output_images = []
        if "image" in result and result["image"] is not None:
            image = result["image"]
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

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Bagel Generating")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            prompt = contexts

            # Choose mode: understanding or generation
            if self.mode == "understanding":
                # Image understanding mode
                if doc_to_visual is None:
                    eval_logger.warning(f"No image provided for understanding mode, doc_id={doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Get image from doc_to_visual
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)

                if not visuals or len(visuals) == 0:
                    eval_logger.warning(f"No visual data found for doc_id={doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Use first image for understanding
                image = visuals[0]
                output_text = self.understand_image(prompt, image, str(doc_id))
                formatted_output = output_text

            else:
                # Image generation mode
                output_text, output_images = self.generate_text_and_image(prompt, str(doc_id), task)
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
