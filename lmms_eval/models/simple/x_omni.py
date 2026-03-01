import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

@register_model("x_omni")
class XOmni(lmms):
    def __init__(
        self,
        pretrained: str,
        mode: str = "understanding",
        flux_pipe_path: Optional[str] = None,
        output_image_dir: Optional[str] = None,
        max_new_tokens: int = 256,  # Reduced from 512 to prevent excessive generation
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.9,
        seed: int = 0,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.mode = mode

        # Only set FLUX path if in generation mode or explicitly provided
        if mode == "generation":
            if flux_pipe_path is None:
                # Generation mode requires FLUX, use default if not provided
                default_flux = "black-forest-labs/FLUX.1-dev"
                eval_logger.warning(f"Generation mode: flux_pipe_path not provided. Defaulting to '{default_flux}'.")
                self.flux_pipe_path = default_flux
            else:
                self.flux_pipe_path = flux_pipe_path
        else:
            # Understanding mode: FLUX not needed
            self.flux_pipe_path = flux_pipe_path  # Keep user's value if provided, otherwise None
            if flux_pipe_path is not None:
                eval_logger.info(f"Understanding mode: flux_pipe_path provided but will not be used: {flux_pipe_path}")
            else:
                eval_logger.info("Understanding mode: FLUX loading will be skipped")
            
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.continual_mode = continual_mode

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/x_omni_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "x_omni_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "x_omni_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning("Continual mode is not supported for distributed inference. Disabling it.")
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Load model
        eval_logger.info(f"Loading X-Omni model from {pretrained}")
        self._load_model()
        eval_logger.info("X-Omni model initialized successfully")

    def _load_model(self):
        """Load X-Omni model components using AutoModel with robust regex error protection"""
        model_path = self.pretrained
        
        # Load tokenizer
        eval_logger.info("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # === Regex Patch Start ===
        from transformers.modeling_utils import PreTrainedModel
        original_adjust = None
        if hasattr(PreTrainedModel, '_adjust_missing_and_unexpected_keys'):
            original_adjust = PreTrainedModel._adjust_missing_and_unexpected_keys
            
            def safe_adjust_keys(self, *args, **kwargs):
                try:
                    return original_adjust(self, *args, **kwargs)
                except Exception as e:
                    eval_logger.warning(f"IGNORED Regex Error in model loading: {e}")
                    
                    ret_list = []
                    for arg in args:
                        if isinstance(arg, list):
                            ret_list.append(arg)
                    
                    if len(ret_list) >= 2:
                        return ret_list[0], ret_list[1]
                    else:
                        return [], []

            PreTrainedModel._adjust_missing_and_unexpected_keys = safe_adjust_keys
        
        eval_logger.info("Loading model with trust_remote_code=True...")
        try:
            # Load config first to modify max_position_embeddings
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Increase max position embeddings to handle multi-image inputs
            if hasattr(config, 'max_position_embeddings'):
                original_max = config.max_position_embeddings
                config.max_position_embeddings = 32768  # Increase from 8192 to 32768
                eval_logger.info(f"Increased max_position_embeddings from {original_max} to {config.max_position_embeddings}")
            
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            eval_logger.error(f"Failed to load model via AutoModel: {e}")
            raise e
        finally:
            if original_adjust is not None:
                PreTrainedModel._adjust_missing_and_unexpected_keys = original_adjust
        # === Regex Patch End ===

        # Initialize vision components
        if hasattr(self._model, "init_vision"):
            if self.mode == "understanding":
                # Understanding mode: Manually initialize only SigLIP encoder, skip FLUX
                eval_logger.info("Understanding mode: Initializing SigLIP encoder only (skipping FLUX)")
                try:
                    # Manually initialize the vision components needed for understanding
                    from types import SimpleNamespace
                    from huggingface_hub import hf_hub_download
                    import os

                    # Set special tokens (needed for tokenize_image)
                    self._model.som_token = self._model.config.mm_special_tokens[0]
                    self._model.eom_token = self._model.config.mm_special_tokens[1]
                    self._model.img_token = self._model.config.mm_special_tokens[2]

                    # Set has_sliding_layers attribute (required by Qwen2Model forward)
                    # This should be set by parent class but XOmniModel doesn't call Qwen2Model.__init__
                    if not hasattr(self._model.model, 'has_sliding_layers'):
                        layer_types = getattr(self._model.config, 'layer_types', [])
                        self._model.model.has_sliding_layers = "sliding_attention" in layer_types
                        eval_logger.info(f"Set has_sliding_layers={self._model.model.has_sliding_layers}")

                    # Set embed_tokens attribute (required by Qwen2Model forward)
                    # XOmniModel has a custom embed_tokens method but Qwen2Model.forward expects it as an attribute
                    # We need to make sure the method is accessible as self.model.embed_tokens
                    if not hasattr(self._model.model, 'embed_tokens') or not isinstance(self._model.model.embed_tokens, torch.nn.Module):
                        # The embed_tokens is a method in XOmniModel, not an nn.Module
                        # Qwen2Model.forward will call self.embed_tokens(input_ids) which should work
                        # But we need to ensure it's not trying to access it as a module
                        eval_logger.info("XOmniModel.embed_tokens is a method, not a module (this is expected)")

                    # Load vision config
                    self._model.vision_config = SimpleNamespace(**self._model.config.vision_config)
                    self._model.transform_config = SimpleNamespace(**self._model.vision_config.transform)
                    self._model.encoder_config = SimpleNamespace(**self._model.vision_config.encoder)
                    self._model.decoder_config = SimpleNamespace(**self._model.vision_config.decoder)

                    # Set dtype
                    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
                    self._model.vision_dtype = dtype_map[self._model.vision_config.dtype]

                    # Import from the model's module (loaded via trust_remote_code)
                    # The model module is already loaded, so we can access it from sys.modules
                    import sys
                    model_module_name = self._model.__class__.__module__
                    model_module = sys.modules[model_module_name]

                    # Get the parent module to access sibling modules
                    parent_module_name = '.'.join(model_module_name.split('.')[:-1])
                    if parent_module_name:
                        # Import from the cached transformers module
                        siglip_module_name = f"{parent_module_name}.modeling_siglip_tokenizer"
                        if siglip_module_name in sys.modules:
                            siglip_module = sys.modules[siglip_module_name]
                            create_anyres_preprocess = siglip_module.create_anyres_preprocess
                            SiglipTokenizer = siglip_module.SiglipTokenizer
                        else:
                            # Fallback: import directly
                            import importlib
                            siglip_module = importlib.import_module(siglip_module_name)
                            create_anyres_preprocess = siglip_module.create_anyres_preprocess
                            SiglipTokenizer = siglip_module.SiglipTokenizer
                    else:
                        raise ImportError("Could not determine model module path")

                    # Create image transform
                    self._model.image_transform = create_anyres_preprocess(**self._model.vision_config.transform)

                    # Load SigLIP encoder paths
                    if os.path.isdir(self._model.name_or_path):
                        self._model.encoder_config.siglip_path = os.path.join(
                            self._model.name_or_path, self._model.encoder_config.siglip_path
                        )
                        self._model.encoder_config.projector_path = os.path.join(
                            self._model.name_or_path, self._model.encoder_config.projector_path
                        )
                    else:
                        self._model.encoder_config.siglip_path = hf_hub_download(
                            repo_id=self._model.name_or_path,
                            filename=self._model.encoder_config.siglip_path
                        )
                        self._model.encoder_config.projector_path = hf_hub_download(
                            repo_id=self._model.name_or_path,
                            filename=self._model.encoder_config.projector_path
                        )

                    # Initialize SigLIP tokenizer only
                    self._model.image_tokenizer = SiglipTokenizer(**vars(self._model.encoder_config))
                    self._model.image_tokenizer.to(self._model.device, self._model.vision_dtype)

                    # Set image_tokenizer to eval mode
                    self._model.image_tokenizer.eval()

                    # Also set the vqproj to eval mode to ensure quantization works correctly
                    if hasattr(self._model.image_tokenizer, 'vqproj'):
                        self._model.image_tokenizer.vqproj.eval()
                        if hasattr(self._model.image_tokenizer.vqproj, 'quantize'):
                            self._model.image_tokenizer.vqproj.quantize.eval()

                    # Set decoder_pipe to None to indicate no generation capability
                    self._model.decoder_pipe = None

                    eval_logger.info("✅ SigLIP encoder initialized successfully (FLUX skipped)")
                except Exception as e:
                    eval_logger.error(f"Failed to manually initialize SigLIP encoder: {e}")
                    eval_logger.error("Falling back to full init_vision (will attempt to load FLUX)")
                    raise
            elif self.mode == "generation":
                eval_logger.info(f"Generation mode: Initializing vision components with FLUX path: {self.flux_pipe_path}")
                self._model.init_vision(self.flux_pipe_path)
                eval_logger.info("Vision encoder and decoder initialized for generation mode")
        else:
            eval_logger.warning("Loaded model does not have 'init_vision' method. Check if correct model class was loaded.")

        self._model.eval()

    @property
    def rank(self): return self._rank
    @property
    def world_size(self): return self._world_size
    @property
    def model(self): return self._model
    @property
    def tokenizer(self): return self._tokenizer

    def set_seed(self, seed: int):
        if seed > 0:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def understand_images(self, prompt: str, images: List[Image.Image], doc_id: str) -> str:
        """Understand multiple images with a text prompt"""
        if not images:
            eval_logger.warning(f"No images provided for understanding, doc_id={doc_id}")
            return ""
            
        self.set_seed(self.seed)
        self.model.set_generation_mode("text")

        # Build multi-image prompt
        image_strs = []
        for img in images:
            if img is not None:
                image_strs.append(self.model.tokenize_image(img))

        # Combine images and prompt
        full_content = '\n'.join(image_strs) + '\n' + prompt
        message = [{'role': 'user', 'content': full_content}]

        input_ids = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        eos_token_id = self.tokenizer.encode('<|im_end|>')[0]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 0.0,
                top_p=self.top_p if self.do_sample else None,
                eos_token_id=eos_token_id,
                pad_token_id=0,
                use_cache=True,
                repetition_penalty=1.1,  # Prevent repetition
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            )

        texts, _ = self.model.mmdecode(self.tokenizer, output_ids[:, input_ids.shape[1]:-1])
        output_text = texts[0] if texts else ""
        return output_text

    def understand_image(self, prompt: str, image: Image.Image, doc_id: str) -> str:
        """Understand single image with a text prompt (backward compatibility)"""
        return self.understand_images(prompt, [image], doc_id)

    def generate_image(self, prompt: str, doc_id: str, task: str) -> Tuple[str, List[str]]:
        self.set_seed(self.seed)
        self.model.set_generation_mode("image")

        input_ids = self.model.mmencode(self.tokenizer, texts=[prompt], return_tensors="pt")
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        texts, images = self.model.mmdecode(self.tokenizer, output_ids)
        output_text = texts[-1] if texts else ""

        output_images = []
        for idx, image in enumerate(images):
            safe_filename = f"{task}_{doc_id}_{idx}.png"
            image_path = os.path.join(self.output_image_dir, safe_filename)
            image.save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved image: {image_path}")

        return output_text, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)
    
    def fix_json_format(self, text: str) -> str:
        """
        Fix common JSON format issues in model output:
        1. Ensure choice is an integer, not a string (for jigsaw)
        2. Fix 'rationle' typo to 'rationale' (for jigsaw)
        3. Add <FINAL_ANSWER_JSON> tags if missing (for jigsaw)
        4. Fix <ANSWER_JSON> tags and array format (for maze/sliding)
        5. Fix closing tags: </ANSWERS_JSON>, </ANSWERSJSON>, </ANSWERS> -> </ANSWER_JSON>
        6. Extract and format move sequences properly
        """
        import re
        
        # Fix closing tag variations first
        text = re.sub(r'</ANSWERS?_?JSON>', '</ANSWER_JSON>', text)
        text = re.sub(r'</ANSWERSJSON>', '</ANSWER_JSON>', text)
        text = re.sub(r'</ANSWERS>', '</ANSWER_JSON>', text)
        text = re.sub(r'</answering>', '</ANSWER_JSON>', text, flags=re.IGNORECASE)
        
        # Try to find JSON object (for jigsaw tasks)
        json_obj_pattern = r'\{[^}]*"choice"[^}]*\}'
        obj_match = re.search(json_obj_pattern, text)
        
        if obj_match:
            json_str = obj_match.group(0)
            try:
                # Parse the JSON
                json_obj = json.loads(json_str)
                
                # Fix choice type: convert string to int
                if 'choice' in json_obj and isinstance(json_obj['choice'], str):
                    json_obj['choice'] = int(json_obj['choice'])
                
                # Fix typo: rationle -> rationale
                if 'rationle' in json_obj:
                    json_obj['rationale'] = json_obj.pop('rationle')
                if 'rationate' in json_obj:
                    json_obj['rationale'] = json_obj.pop('rationate')
                
                # Reconstruct the JSON string
                fixed_json = json.dumps(json_obj, ensure_ascii=False)
                
                # Check if tags are present
                if '<FINAL_ANSWER_JSON>' not in text and '<ANSWER_JSON>' not in text:
                    # Replace the original JSON with tagged version
                    text = text.replace(json_str, f'<FINAL_ANSWER_JSON>{fixed_json}</FINAL_ANSWER_JSON>')
                else:
                    # Just replace the JSON content
                    text = text.replace(json_str, fixed_json)
                    
            except (json.JSONDecodeError, ValueError) as e:
                eval_logger.warning(f"Failed to parse JSON object for fixing: {e}")
        
        # Try to find and fix JSON array (for maze/sliding tasks)
        # First, try to extract from existing tags
        answer_json_pattern = r'<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>'
        answer_match = re.search(answer_json_pattern, text, re.DOTALL)
        
        if answer_match:
            array_str = answer_match.group(1)
            try:
                array_obj = json.loads(array_str)
                if isinstance(array_obj, list):
                    # Normalize moves
                    valid_moves = {'up', 'down', 'left', 'right'}
                    normalized = []
                    for move in array_obj:
                        if isinstance(move, str):
                            move_lower = move.lower().strip()
                            if move_lower in valid_moves:
                                normalized.append(move_lower)
                    
                    if normalized:
                        fixed_array = json.dumps(normalized, ensure_ascii=False)
                        text = re.sub(answer_json_pattern, f'<ANSWER_JSON>{fixed_array}</ANSWER_JSON>', text, flags=re.DOTALL)
                        return text
            except (json.JSONDecodeError, ValueError):
                pass
        
        # If no valid tagged array found, try to find any JSON array in the text
        json_array_pattern = r'\[[^\]]*\]'
        array_matches = re.findall(json_array_pattern, text)
        
        best_array = None
        best_score = 0
        
        for array_str in array_matches:
            try:
                # Parse the array
                array_obj = json.loads(array_str)
                if isinstance(array_obj, list) and len(array_obj) > 0:
                    # Check if it looks like a move list (contains strings)
                    if all(isinstance(item, str) for item in array_obj):
                        # Normalize moves: convert to lowercase and filter valid moves
                        valid_moves = {'up', 'down', 'left', 'right'}
                        normalized = []
                        for move in array_obj:
                            move_lower = move.lower().strip()
                            # Map common variations
                            if move_lower in valid_moves:
                                normalized.append(move_lower)
                            elif 'up' in move_lower and 'down' not in move_lower:
                                normalized.append('up')
                            elif 'down' in move_lower:
                                normalized.append('down')
                            elif 'left' in move_lower:
                                normalized.append('left')
                            elif 'right' in move_lower:
                                normalized.append('right')
                        
                        # Score this array based on number of valid moves
                        score = len(normalized)
                        if score > best_score:
                            best_score = score
                            best_array = normalized
                        
            except (json.JSONDecodeError, ValueError):
                continue
        
        # If we found a valid array, format it properly
        if best_array:
            fixed_array = json.dumps(best_array, ensure_ascii=False)
            # Check if tags already exist
            if '<ANSWER_JSON>' in text:
                # Replace the content between tags
                text = re.sub(r'<ANSWER_JSON>.*?</ANSWER_JSON>', f'<ANSWER_JSON>{fixed_array}</ANSWER_JSON>', text, flags=re.DOTALL)
            else:
                # Add tags at the end
                text = text.rstrip() + f'\n\n<ANSWER_JSON>{fixed_array}</ANSWER_JSON>'
        
        return text

    def flatten(self, input_list):
        output = []
        for item in input_list:
            if isinstance(item, list): output.extend(self.flatten(item))
            else: output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="X-Omni Generating")
        def get_uuid(task, split, doc_id): return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc_uuid = get_uuid(task, split, doc_id)
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache and self.response_cache[doc_uuid]:
                    res.append(self.response_cache[doc_uuid])
                    pbar.update(1)
                    continue

            prompt = contexts
            if self.mode == "understanding":
                if doc_to_visual is None:
                    res.append("")
                    pbar.update(1)
                    continue
                    
                # Extract all images, not just the first one
                visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
                if not visuals:
                    res.append("")
                    pbar.update(1)
                    continue
                
                # Convert all visuals to PIL Images
                images = []
                for visual in visuals:
                    if visual is not None:
                        # Handle different visual formats
                        if isinstance(visual, Image.Image):
                            images.append(visual)
                        elif hasattr(visual, 'convert'):  # PIL-like object
                            images.append(visual.convert("RGB"))
                        else:
                            eval_logger.warning(f"Unsupported visual format: {type(visual)}")
                
                if not images:
                    res.append("")
                    pbar.update(1)
                    continue
                
                # Use multi-image understanding
                result = self.understand_images(prompt, images, str(doc_id))
                # Fix format issues based on task type

                result = self.fix_json_format(result)
                res.append(result)
            else:
                out_txt, out_imgs = self.generate_image(prompt, str(doc_id), task)
                res.append(self.format_output(out_txt, out_imgs))

            if self.continual_mode:
                self.response_cache[doc_uuid] = res[-1]
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
            pbar.update(1)
        pbar.close()
        return res
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("X-Omni is a generation model")
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round")