import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import (
    Accelerator,
    DistributedType,
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Try to import Bagel dependencies
try:
    from bagel.data.data_utils import add_special_tokens, pil_img2rgb
    from bagel.data.transforms import ImageTransform
    from bagel.inferencer import InterleaveInferencer
    from bagel.modeling.autoencoder import load_ae
    from bagel.modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from bagel.modeling.qwen2 import Qwen2Tokenizer

    BAGEL_AVAILABLE = True
except ImportError as e:
    BAGEL_AVAILABLE = False
    BAGEL_IMPORT_ERROR = str(e)


# Mode-specific base parameters as specified by user
BASE_PARAMS: Dict[str, Dict[str, Any]] = {
    "generate": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    ),
    "think_generate": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        think=True,
    ),
    "edit": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    ),
    "think_edit": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        think=True,
    ),
    "understanding": dict(
        max_think_token_n=1000,
        do_sample=False,
        understanding_output=True,
    ),
    "think_understanding": dict(
        max_think_token_n=1000,
        do_sample=False,
        understanding_output=True,
        think=True,
    ),
}


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


@register_model("bagel_umm")
class BagelUMM(lmms):
    """
    Bagel Unified Multimodal Model for lmms-eval.
    
    Supports multiple modes:
    - generate: Text-to-image generation
    - think_generate: Text-to-image with thinking process
    - edit: Image editing
    - think_edit: Image editing with thinking process
    - understanding: Visual question answering (default)
    - think_understanding: VQA with thinking process
    
    Example usage:
        accelerate launch -m lmms_eval \\
            --model bagel_umm \\
            --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=understanding \\
            --tasks mme \\
            --batch_size 1 \\
            --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "",
        mode: str = "understanding",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        torch_dtype: Optional[str] = "bfloat16",
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        # Generation parameters (can override mode defaults)
        max_think_token_n: Optional[int] = None,
        do_sample: Optional[bool] = None,
        text_temperature: float = 0.3,
        cfg_text_scale: Optional[float] = None,
        cfg_img_scale: Optional[float] = None,
        cfg_interval: Optional[List[float]] = None,
        timestep_shift: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        cfg_renorm_min: Optional[float] = None,
        cfg_renorm_type: Optional[str] = None,
        # Image generation settings
        image_shapes: Tuple[int, int] = (1024, 1024),
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if not BAGEL_AVAILABLE:
            raise ImportError(f"Failed to import Bagel dependencies: {BAGEL_IMPORT_ERROR}\n" "Please install the Bagel package by running:\n" "uv pip install git+https://github.com/oscarqjh/Bagel.git")

        # Validate mode
        if mode not in BASE_PARAMS:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(BASE_PARAMS.keys())}")

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")
        self.attn_implementation = attn_implementation

        self.pretrained = pretrained
        self.mode = mode
        self.use_cache = use_cache
        self.text_temperature = text_temperature
        self.image_shapes = image_shapes
        self._output_dir_base = output_dir  # Store for later, create after accelerator setup

        # Build inference parameters from mode defaults + overrides
        self.inference_params = BASE_PARAMS[mode].copy()
        if max_think_token_n is not None:
            self.inference_params["max_think_token_n"] = max_think_token_n
        if do_sample is not None:
            self.inference_params["do_sample"] = do_sample
        if cfg_text_scale is not None:
            self.inference_params["cfg_text_scale"] = cfg_text_scale
        if cfg_img_scale is not None:
            self.inference_params["cfg_img_scale"] = cfg_img_scale
        if cfg_interval is not None:
            self.inference_params["cfg_interval"] = cfg_interval
        if timestep_shift is not None:
            self.inference_params["timestep_shift"] = timestep_shift
        if num_timesteps is not None:
            self.inference_params["num_timesteps"] = num_timesteps
        if cfg_renorm_min is not None:
            self.inference_params["cfg_renorm_min"] = cfg_renorm_min
        if cfg_renorm_type is not None:
            self.inference_params["cfg_renorm_type"] = cfg_renorm_type

        # Set torch dtype
        if torch_dtype == "bfloat16":
            self._torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self._torch_dtype = torch.float16
        else:
            self._torch_dtype = torch.float32

        # Set up accelerator and device
        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Set up output directory for generated images (after accelerator setup)
        # Only main process generates run_id to ensure all processes use same folder
        if self.mode in ["generate", "think_generate", "edit", "think_edit"]:
            if accelerator.is_main_process:
                run_id = generate_run_id()
                if self._output_dir_base is None:
                    self.output_dir = os.path.join("./logs/bagel_images", run_id)
                else:
                    self.output_dir = os.path.join(self._output_dir_base, run_id)
                os.makedirs(self.output_dir, exist_ok=True)
                # Write output_dir to a temp file for other processes
                with open("/tmp/bagel_output_dir.txt", "w") as f:
                    f.write(self.output_dir)
                eval_logger.info(f"Bagel output directory: {self.output_dir}")

            # Wait for main process to create directory
            accelerator.wait_for_everyone()

            # Non-main processes read the output_dir from temp file
            if not accelerator.is_main_process:
                with open("/tmp/bagel_output_dir.txt", "r") as f:
                    self.output_dir = f.read().strip()

        # Load model
        self._load_model()

        self.batch_size_per_gpu = int(batch_size)

        # Set up distributed training
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"Bagel model initialized in '{mode}' mode")

    def _load_model(self):
        """Load Bagel model components."""
        # Handle HuggingFace Hub paths - download if not a local directory
        # Use barrier to ensure only main process downloads to avoid race conditions
        if os.path.isdir(self.pretrained):
            model_path = self.pretrained
        else:
            # First try to load from cache without downloading
            try:
                model_path = snapshot_download(repo_id=self.pretrained, local_files_only=True)
                eval_logger.info(f"Loaded model from cache: {model_path}")
            except Exception:
                # Not in cache, need to download
                # Only main process downloads, others wait
                if self.accelerator.is_main_process:
                    eval_logger.info(f"Downloading model from HuggingFace Hub: {self.pretrained}")
                    # Use user-specific cache to avoid permission issues with shared caches
                    user_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    model_path = snapshot_download(repo_id=self.pretrained, cache_dir=user_cache_dir)
                    eval_logger.info(f"Model downloaded to: {model_path}")

                # Wait for main process to finish downloading
                self.accelerator.wait_for_everyone()

                # Non-main processes get the cached path
                if not self.accelerator.is_main_process:
                    user_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    model_path = snapshot_download(repo_id=self.pretrained, cache_dir=user_cache_dir, local_files_only=True)

        # Load LLM config
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        # Apply attention implementation to LLM config
        if self.attn_implementation is not None:
            llm_config._attn_implementation = self.attn_implementation

        # Load ViT config
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        # Apply attention implementation to ViT config
        if self.attn_implementation is not None:
            vit_config._attn_implementation = self.attn_implementation

        # Load VAE
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        # Create Bagel config
        config = BagelConfig(
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
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Load tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Create transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Infer device map - when using accelerate with data parallelism,
        # each process should only use its assigned GPU
        if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
            # Data parallelism: each process loads full model on its own GPU
            local_gpu = self.accelerator.local_process_index
            device_map = infer_auto_device_map(
                model,
                max_memory={local_gpu: "80GiB"},
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )
        else:
            # Single process or model parallelism: spread across all GPUs
            device_map = infer_auto_device_map(
                model,
                max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )

        # Ensure related modules are on the same device
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]

        # Determine the target device for same_device_modules
        if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
            # Data parallelism: all modules go to local GPU
            target_device = f"cuda:{self.accelerator.local_process_index}"
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = target_device
        elif torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = first_device if k in device_map else "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Find checkpoint file - support multiple formats
        checkpoint_candidates = [
            os.path.join(model_path, "ema.safetensors"),
            os.path.join(model_path, "model.safetensors"),
            os.path.join(model_path, "model.safetensors.index.json"),
        ]

        checkpoint = None
        for candidate in checkpoint_candidates:
            if os.path.exists(candidate):
                # For sharded models, use the directory containing the index
                if candidate.endswith(".index.json"):
                    checkpoint = model_path
                else:
                    checkpoint = candidate
                break

        if checkpoint is None:
            raise FileNotFoundError(f"Could not find checkpoint in {model_path}. " f"Expected one of: {checkpoint_candidates}")

        eval_logger.info(f"Loading checkpoint from: {checkpoint}")

        # Load checkpoint and dispatch
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint,
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=self._torch_dtype,
            force_hooks=True,
        ).eval()

        # Create inferencer
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._vae_transform = vae_transform
        self._vit_transform = vit_transform
        self._new_token_ids = new_token_ids

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
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 4096

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Bagel")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            # Build input list for inferencer
            input_list = []
            processed_visuals = []

            # Process and collect all visual inputs
            if visuals is not None:
                for visual in visuals:
                    if isinstance(visual, Image.Image):
                        # Convert PIL Image to RGB format
                        img = pil_img2rgb(visual)
                        processed_visuals.append(img)
                    elif isinstance(visual, str):
                        # Handle image file paths
                        if os.path.exists(visual):
                            img = Image.open(visual)
                            img = pil_img2rgb(img)
                            processed_visuals.append(img)
                        else:
                            eval_logger.warning(f"Image file not found, skipping: {visual}")

            # Count <image> tokens in context
            num_image_tags = contexts.count("<image>")
            num_visuals = len(processed_visuals)

            # Build input_list based on <image> token presence
            if num_image_tags == 0:
                # Case 1: No <image> tokens in context
                # Simply prepend all images before the text prompt
                if num_visuals > 0:
                    input_list = processed_visuals + [contexts.strip()]
                else:
                    # text-only input (no visuals)
                    input_list = [contexts.strip()]

            elif num_image_tags == num_visuals:
                # Case 2: Number of <image> tokens matches number of visuals
                # Split context by <image> and interleave with images
                context_parts = contexts.split("<image>")

                # Sanity check: splitting by N tags should give N+1 parts
                assert len(context_parts) == num_visuals + 1, f"Split error: expected {num_visuals + 1} parts, got {len(context_parts)}"

                for i, text_part in enumerate(context_parts):
                    # Add text part (may be empty string after strip)
                    text_stripped = text_part.strip()
                    if text_stripped:  # Only add non-empty text
                        input_list.append(text_stripped)

                    # Add corresponding image after this text part (except for last part)
                    if i < num_visuals:
                        input_list.append(processed_visuals[i])

            else:
                # Case 3: Mismatch between <image> tokens and actual visuals
                raise ValueError(f"Mismatch between <image> tokens and visuals: " f"Found {num_image_tags} <image> token(s) in context, " f"but received {num_visuals} visual(s). " f"Context preview: '{contexts[:200]}...'")

            # Final sanity check: input_list should not be empty
            if not input_list:
                raise ValueError(f"Failed to build input_list: no valid inputs. " f"Context: '{contexts[:100]}...', Visuals: {num_visuals}")

            # Prepare inference parameters
            inference_params = self.inference_params.copy()
            inference_params["text_temperature"] = self.text_temperature

            # Add image shapes for generation modes
            if self.mode in ["generate", "think_generate", "edit", "think_edit"]:
                inference_params["image_shapes"] = self.image_shapes

            eval_logger.debug(f"[generate_until] input_list: {input_list}")
            # Run inference
            with torch.autocast(device_type="cuda", enabled=True, dtype=self._torch_dtype):
                output_list = self.inferencer.interleave_inference(input_list, **inference_params)

            # Process outputs
            output_text = ""
            output_images = []

            for output in output_list:
                if isinstance(output, str):
                    output_text = output
                elif isinstance(output, Image.Image):
                    # Save generated image
                    image_filename = f"{task}_{doc_id}_{len(output_images)}.png"
                    image_path = os.path.join(self.output_dir, image_filename)
                    output.save(image_path)
                    output_images.append(image_path)
                    eval_logger.debug(f"Saved generated image: {image_path}")

            # Format output based on mode
            # For image generation/editing modes, return JSON with image paths
            # For understanding modes, return plain text
            if self.mode in ["generate", "think_generate", "edit", "think_edit"] and output_images:
                output_dict = {"text": output_text, "images": output_images}
                formatted_output = json.dumps(output_dict, ensure_ascii=False)
                res.append(formatted_output)
            else:
                res.append(output_text)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Bagel")
