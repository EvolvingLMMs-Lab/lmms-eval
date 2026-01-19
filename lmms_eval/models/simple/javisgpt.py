import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add JavisGPT repository to Python path
# Expected: lmms-eval/../JavisGPT/ directory
wd = Path(__file__).parent.parent.parent.parent.resolve()
javisgpt_path = os.path.join(str(wd.parent), "JavisGPT")
if os.path.exists(javisgpt_path):
    sys.path.append(javisgpt_path)
    eval_logger.info(f"Added JavisGPT path to sys.path: {javisgpt_path}")
else:
    eval_logger.warning(
        f"JavisGPT repository not found at {javisgpt_path}. "
        f"Please clone it: cd {wd.parent} && git clone https://github.com/JavisVerse/JavisGPT.git"
    )


@register_model("javisgpt")
class JavisGPT(lmms):
    """
    JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation

    Supports both audio-visual understanding and generation modes:
        - "understanding": Audio-visual understanding (video/image/audio + text -> text)
        - "generation": Image or video generation (text -> image/video)

    Example usage for understanding:
    python -m lmms_eval \
        --model javisgpt \
        --model_args pretrained=/path/to/JavisGPT-v0.1-7B-Instruct,mode=understanding,base_model=/path/to/Qwen2.5-VL-7B-Instruct,beats_path=/path/to/BEATs.pt \
        --tasks your_task \
        --batch_size 1 \
        --output_path ./logs/

    Example usage for image generation:
    python -m lmms_eval \
        --model javisgpt \
        --model_args pretrained=/path/to/JavisGPT-v0.1-7B-Instruct,mode=generation,output_type=image,base_model=/path/to/Qwen2.5-VL-7B-Instruct,beats_path=/path/to/BEATs.pt,avgen_cfg_path=/path/to/javisdit_v0.1_image.py \
        --tasks your_task \
        --batch_size 1 \
        --output_path ./logs/

    Example usage for video generation:
    python -m lmms_eval \
        --model javisgpt \
        --model_args pretrained=/path/to/JavisGPT-v0.1-7B-Instruct,mode=generation,output_type=video,base_model=/path/to/Qwen2.5-VL-7B-Instruct,beats_path=/path/to/BEATs.pt,avgen_cfg_path=/path/to/javisdit_v0.1.py \
        --tasks your_task \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        base_model: str,
        beats_path: str,
        mode: str = "understanding",
        base_arch: str = "Qwen2_5_VL",
        avgen_cfg_path: Optional[str] = None,
        av_gen_token_num: int = 377,
        output_type: str = "image",  # "image" or "video"
        num_frames: Optional[int] = None,
        image_size: Tuple[int, int] = (512, 512),  # (H, W)
        output_image_dir: Optional[str] = None,
        output_video_dir: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        seed: int = 0,
        use_audio_in_video: bool = True,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        save_results: bool = True,
        results_file: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        # Validate output_type
        if output_type not in ["image", "video"]:
            raise ValueError(
                f"output_type must be 'image' or 'video', got '{output_type}'"
            )

        self.mode = mode
        self.output_type = output_type
        self.pretrained = pretrained
        self.base_model = base_model
        self.beats_path = beats_path
        self.base_arch = base_arch
        self.avgen_cfg_path = avgen_cfg_path
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.seed = seed
        self.use_audio_in_video = use_audio_in_video
        self.continual_mode = continual_mode
        self.image_size = image_size
        self.save_results = save_results

        # Set num_frames based on output_type if not explicitly provided
        if num_frames is None:
            self.num_frames = 1 if output_type == "image" else 102
        else:
            self.num_frames = num_frames

        eval_logger.info(
            f"JavisGPT initialized with mode={mode}, output_type={output_type}, num_frames={self.num_frames}"
        )

        # Set environment variables for JavisGPT
        os.environ["BASE_ARCH"] = base_arch
        os.environ["AV_GEN_TOKEN_NUM"] = str(av_gen_token_num)

        # Import JavisGPT dependencies
        try:
            from javisgpt.model import (
                JavisConfig,
                JavisGPTForConditionalGeneration,
                JavisProcessor,
            )
            from javisgpt.conversation import conv_templates
            from transformers import AutoTokenizer

            self.JavisConfig = JavisConfig
            self.JavisGPTForConditionalGeneration = JavisGPTForConditionalGeneration
            self.JavisProcessor = JavisProcessor
            self.conv_templates = conv_templates
            self.AutoTokenizer = AutoTokenizer

        except Exception as e:
            raise ImportError(
                f"Failed to import JavisGPT dependencies. "
                f"Please ensure:\n"
                f"  1. JavisGPT repository is cloned at lmms-eval/../JavisGPT\n"
                f"  2. Dependencies are installed: cd JavisGPT && pip install -e .\n"
                f"  3. BASE_ARCH environment variable is set\n"
                f"Error: {e}"
            )

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/javisgpt_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        # Setup output directories based on output_type
        if output_type == "image":
            if output_image_dir is None:
                self.output_dir = os.path.join(
                    self.response_persistent_folder, "javisgpt_generated_images"
                )
            else:
                self.output_dir = output_image_dir
        else:  # video
            if output_video_dir is None:
                self.output_dir = os.path.join(
                    self.response_persistent_folder, "javisgpt_generated_videos"
                )
            else:
                self.output_dir = output_video_dir

        os.makedirs(self.output_dir, exist_ok=True)
        eval_logger.info(f"Output directory: {self.output_dir}")

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "javisgpt_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded cache: {len(self.response_cache)} records"
                )

        # Setup results file for saving detailed results
        if self.save_results:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            if results_file is None:
                self.results_file = os.path.join(
                    self.response_persistent_folder, "javisgpt_results.jsonl"
                )
            else:
                self.results_file = results_file
            eval_logger.info(f"Results will be saved to: {self.results_file}")
            # Initialize results list
            self.results_list = []

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Set device
        if device is None:
            self._device = (
                f"cuda:{self._rank}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = device

        # Load model
        eval_logger.info(f"Loading JavisGPT model from {pretrained}")
        self._load_model()

        eval_logger.info("JavisGPT model initialized successfully")

    def _load_model(self):
        """Load JavisGPT model components"""
        # Load tokenizer
        eval_logger.info(f"Loading tokenizer from {self.base_model}")
        tokenizer = self.AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )

        # Load processor
        eval_logger.info(f"Loading processor from {self.base_model}")
        processor = self.JavisProcessor.from_pretrained(
            self.base_model, trust_remote_code=True
        )

        # Load configuration first to ensure beats_cfg is properly set
        eval_logger.info(f"Loading configuration from {self.pretrained}")
        config = self.JavisConfig.from_pretrained(
            self.pretrained,
            trust_remote_code=True,
        )
        eval_logger.info(f"Configuration loaded: beats_cfg has {len(config.beats_cfg)} keys")

        # Check if pretrained is an adapter or full model
        adapter_path = os.path.join(self.pretrained, "adapter_model.safetensors")
        is_adapter = os.path.exists(adapter_path)

        if is_adapter:
            # Load base model first, then adapter
            eval_logger.info(f"Loading base model from {self.base_model}")
            model = self.JavisGPTForConditionalGeneration.from_pretrained(
                self.base_model,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self._device,
                trust_remote_code=True,
            )
            eval_logger.info(f"Loading adapter from {self.pretrained}")
            # Load adapter weights
            from safetensors.torch import load_file
            adapter_weights = load_file(adapter_path)
            model.load_state_dict(adapter_weights, strict=False)
        else:
            # Load full model directly
            eval_logger.info(f"Loading model from {self.pretrained}")
            model = self.JavisGPTForConditionalGeneration.from_pretrained(
                self.pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self._device,
                trust_remote_code=True,
            )

        # Load BEATs weights
        eval_logger.info(f"Loading BEATs weights from {self.beats_path}")
        beats_checkpoint = torch.load(self.beats_path, map_location="cpu")
        model.beats.load_state_dict(beats_checkpoint, strict=False)

        # Load projector weights if exists
        projector_path = os.path.join(self.pretrained, "mm_proj_all.bin")
        if os.path.exists(projector_path):
            eval_logger.info(f"Loading projector weights from {projector_path}")
            projector_weights = torch.load(projector_path, map_location="cpu")
            model.load_state_dict(projector_weights, strict=False)

        # Set model to eval mode
        model.eval()

        # Load generation interface if in generation mode
        if self.mode == "generation":
            if self.avgen_cfg_path is None:
                raise ValueError(
                    "avgen_cfg_path must be provided for generation mode"
                )
            eval_logger.info(
                f"Initializing AV generation interface from {self.avgen_cfg_path}"
            )

            # Initialize av_generator if it doesn't exist
            if not hasattr(model, 'av_generator') or model.av_generator is None:
                eval_logger.info(
                    "av_generator not found in model, initializing dynamically"
                )
                from interface.javisdit_interface import JavisDiTInterface
                from javisgpt.constants import AV_GEN_TOKEN_NUM
                import torch.nn as nn

                # Initialize av_generator
                model.av_generator = JavisDiTInterface(self.avgen_cfg_path)

                # Create avgen_token parameter
                model.avgen_token = nn.Parameter(
                    torch.rand((1, AV_GEN_TOKEN_NUM, model.config.hidden_size)) * 1e-3
                ).to(model.device, dtype=model.dtype)

                # Create condition projector
                model.avgen_cond_proj = model.av_generator.get_cond_projector(
                    model.config.hidden_size
                ).to(model.device, dtype=model.dtype)

                eval_logger.info(
                    f"av_generator initialized with {model.av_generator.cond_embed_num} condition embeddings"
                )

            # Set num_frames and image_size
            if hasattr(model, 'av_generator') and model.av_generator is not None:
                eval_logger.info(
                    f"Setting av_generator num_frames to {self.num_frames}"
                )
                model.av_generator.num_frames = self.num_frames
                model.av_generator.image_size = self.image_size

            # Initialize avgen_token if model has the method
            if hasattr(model, 'init_avgen_token'):
                eval_logger.info("Initializing avgen_token")
                model.init_avgen_token()

        self._model = model
        self._tokenizer = tokenizer
        self._processor = processor

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
    def processor(self):
        return self._processor

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

    def understand_audiovisual(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        doc_id: str = "",
    ) -> str:
        """
        Understand audio-visual content and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand (optional)
            video_path: Path to video file (optional)
            audio_path: Path to audio file (optional)
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        self.set_seed(self.seed)

        # Prepare conversation message for processor
        messages = []

        # Build user message content
        content_parts = []
        if image is not None:
            content_parts.append({"type": "image", "image": image})
        if video_path is not None:
            # Video handling would go here
            pass
        if audio_path is not None:
            # Audio handling would go here
            pass
        content_parts.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content_parts})

        # Generate prompt text using processor's chat template
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Prepare inputs using processor
        inputs = self.processor(
            text=[prompt_text],
            images=[image] if image is not None else None,
            videos=None,
            return_tensors="pt",
        )

        # Replace token IDs with special indices for JavisGPT
        # image_token_id (151655) -> IMAGE_TOKEN_INDEX (-200)
        # video_token_id (151656) -> VIDEO_TOKEN_INDEX (-400)
        # audio_token_id -> AUDIO_TOKEN_INDEX (-300)
        from javisgpt.constants import (
            IMAGE_TOKEN_INDEX,
            VIDEO_TOKEN_INDEX,
            AUDIO_TOKEN_INDEX,
        )

        input_ids = inputs["input_ids"]
        config = self.model.config

        # Replace image tokens
        if image is not None:
            input_ids = input_ids.masked_fill(
                input_ids == config.image_token_id, IMAGE_TOKEN_INDEX
            )

        # Replace video tokens (if applicable)
        if hasattr(config, "video_token_id"):
            input_ids = input_ids.masked_fill(
                input_ids == config.video_token_id, VIDEO_TOKEN_INDEX
            )

        # Replace audio tokens (if applicable)
        if hasattr(config, "audio_token_id"):
            input_ids = input_ids.masked_fill(
                input_ids == config.audio_token_id, AUDIO_TOKEN_INDEX
            )

        inputs["input_ids"] = input_ids

        # Move inputs to device
        inputs = {k: v.to(self.device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype)
                  for k, v in inputs.items()}

        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                use_cache=True,
            )

        # Decode output
        input_token_len = inputs["input_ids"].shape[1]
        output_text = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]

        return output_text.strip()

    def save_image_from_video_tensor(
        self, video_tensor: torch.Tensor, save_path: str
    ) -> str:
        """
        Extract first frame from video tensor and save as image

        Args:
            video_tensor: Video tensor of shape (C, T, H, W) or (T, H, W, C)
            save_path: Path to save the image (without extension)

        Returns:
            Path to saved image
        """
        # Ensure tensor is on CPU and convert to numpy
        video_np = video_tensor.cpu().numpy()

        # Handle different tensor formats
        if video_np.ndim == 4:
            # Check if it's (C, T, H, W) or (T, H, W, C)
            if video_np.shape[1] == 1 or (
                video_np.shape[0] == 3 and video_np.shape[1] > 3
            ):
                # (C, T, H, W) format
                frame = video_np[:, 0, :, :]  # Get first frame: (C, H, W)
                frame = np.transpose(frame, (1, 2, 0))  # Convert to (H, W, C)
            else:
                # (T, H, W, C) format
                frame = video_np[0, :, :, :]  # Get first frame: (H, W, C)
        elif video_np.ndim == 3:
            # (H, W, C) format - single frame
            frame = video_np
        else:
            raise ValueError(f"Unexpected video tensor shape: {video_np.shape}")

        # Normalize to [0, 255] if needed
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        # Handle grayscale or RGB
        if frame.shape[-1] == 1:
            frame = frame.squeeze(-1)
            image = Image.fromarray(frame, mode="L")
        else:
            image = Image.fromarray(frame, mode="RGB")

        # Save image
        image_path = f"{save_path}.png"
        image.save(image_path)
        eval_logger.info(f"Saved image: {image_path}")

        return image_path

    def generate_content(
        self, prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Generate image or video from text prompt

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_content_paths)
        """
        self.set_seed(self.seed)

        # Prepare conversation with AV generation token
        conv = self.conv_templates["qwen_2"].copy()
        # Add special token for AV generation
        prompt_with_token = f"{prompt} <|av_gen|>"
        conv.append_message(conv.roles[0], prompt_with_token)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        # Prepare inputs
        inputs = self.processor(
            text=[prompt_formatted], return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)

        # Generate response and AV content
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        # Decode output text
        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids.sequences
        output_text = self.tokenizer.batch_decode(
            generated_ids[:, input_token_len:], skip_special_tokens=True
        )[0]

        # Generate actual content (image or video)
        output_paths = []
        try:
            if hasattr(self.model, "av_generator") and self.model.av_generator is not None:
                # Extract AV generation embeddings
                from javisgpt.constants import GEN_AUDIO_VIDEO_TOKEN_INDEX

                hidden_states = output_ids.hidden_states[0][-1]
                avgen_embeds = self.model.parse_avgen_embed(
                    hidden_states, inputs["input_ids"], GEN_AUDIO_VIDEO_TOKEN_INDEX
                )

                # Generate audio and video
                audio, video = self.model.generate_audio_video_direct(avgen_embeds)

                # Save based on output_type
                if self.output_type == "image":
                    # Extract and save first frame as image
                    base_path = os.path.join(self.output_dir, f"{task}_{doc_id}")
                    image_path = self.save_image_from_video_tensor(
                        video[0], base_path
                    )
                    output_paths.append(image_path)
                else:  # video
                    # Save as video with audio
                    safe_filename = f"{task}_{doc_id}"
                    save_path_prefix = os.path.join(self.output_dir, safe_filename)
                    video_path = self.model.av_generator.save(
                        audio[0], video[0], save_path=save_path_prefix
                    )
                    output_paths.append(video_path)

        except Exception as e:
            eval_logger.error(
                f"Failed to generate content for doc {doc_id}: {e}"
            )
            eval_logger.exception(e)

        return output_text.strip(), output_paths

    def format_output(self, text: str, content_paths: List[str]) -> str:
        """Format output as JSON string"""
        if self.output_type == "image":
            output_dict = {"text": text, "images": content_paths}
        else:
            output_dict = {"text": text, "videos": content_paths}
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

    def save_result(
        self,
        doc_id: str,
        task: str,
        split: str,
        prompt: str,
        output: str,
        mode: str,
        content_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Save a single result to the results file

        Args:
            doc_id: Document ID
            task: Task name
            split: Dataset split
            prompt: Input prompt
            output: Model output text
            mode: Mode (understanding or generation)
            content_paths: List of generated content paths (for generation mode)
        """
        if not self.save_results:
            return

        result_entry = {
            "doc_id": str(doc_id),
            "task": task,
            "split": split,
            "mode": mode,
            "prompt": prompt,
            "output": output,
        }

        if mode == "generation" and content_paths:
            if self.output_type == "image":
                result_entry["generated_images"] = content_paths
            else:
                result_entry["generated_videos"] = content_paths

        # Append to results list
        self.results_list.append(result_entry)

        # Write to file incrementally (JSONL format)
        try:
            with open(self.results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            eval_logger.error(f"Failed to save result to {self.results_file}: {e}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc=f"JavisGPT {self.mode.capitalize()} ({self.output_type})",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
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
                # Audio-visual understanding mode
                image = None
                if doc_to_visual is not None:
                    visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                    visuals = self.flatten(visuals)
                    if visuals and len(visuals) > 0:
                        image = visuals[0]

                output_text = self.understand_audiovisual(
                    prompt=prompt, image=image, doc_id=str(doc_id)
                )
                formatted_output = output_text

                # Save result
                self.save_result(
                    doc_id=doc_id,
                    task=task,
                    split=split,
                    prompt=prompt,
                    output=output_text,
                    mode="understanding",
                )

            else:
                # Content generation mode (image or video)
                output_text, output_paths = self.generate_content(
                    prompt, str(doc_id), task
                )
                formatted_output = self.format_output(output_text, output_paths)

                # Save result
                self.save_result(
                    doc_id=doc_id,
                    task=task,
                    split=split,
                    prompt=prompt,
                    output=output_text,
                    mode="generation",
                    content_paths=output_paths,
                )

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(
                        self.response_cache, f, ensure_ascii=False, indent=2
                    )

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "JavisGPT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation for JavisGPT"
        )
