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

"""
X-Omni Visual Chain-of-Thought (CoT) Model

This implementation uses X-Omni's dual capabilities for visual reasoning:
1. Stage 1: Generate a visual diagram to help answer the question (using FLUX generation)
2. Stage 2: Answer the question using both the original image and generated diagram (using SigLIP understanding)

Example usage:
    python -m lmms_eval --model x_omni_visual_cot \
        --model_args pretrained=X-Omni/X-Omni-7B,flux_pipe_path=black-forest-labs/FLUX.1-dev \
        --tasks mme --batch_size 1 --device cuda:0
"""

@register_model("x_omni_visual_cot")
class XOmniVisualCoT(lmms):
    def __init__(
        self,
        pretrained: str,
        flux_pipe_path: Optional[str] = None,
        output_image_dir: Optional[str] = None,
        stage1_max_new_tokens: int = 512,
        stage1_do_sample: bool = False,
        stage1_temperature: float = 0.7,
        stage1_top_p: float = 0.9,
        stage1_num_inference_steps: int = 28,
        stage1_guidance_scale: float = 3.5,
        stage2_max_new_tokens: int = 1024,
        stage2_do_sample: bool = False,
        stage2_temperature: float = 0.0,
        stage2_top_p: float = 0.9,
        seed: int = 0,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        generation_prompt_template: str = "Generate a detailed visual diagram to help answer: {question}",
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        fail_gracefully: bool = True,
        offload_flux_to_cpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.flux_pipe_path = flux_pipe_path or "black-forest-labs/FLUX.1-dev"
        self.seed = seed
        self.continual_mode = continual_mode
        self.generation_prompt_template = generation_prompt_template
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.offload_flux_to_cpu = offload_flux_to_cpu

        # Stage 1 parameters (generation)
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_do_sample = stage1_do_sample
        self.stage1_temperature = stage1_temperature
        self.stage1_top_p = stage1_top_p
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale

        # Stage 2 parameters (understanding)
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_do_sample = stage2_do_sample
        self.stage2_temperature = stage2_temperature
        self.stage2_top_p = stage2_top_p

        # Setup output directories
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/x_omni_visual_cot"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        if intermediate_dir is None:
            self.intermediate_dir = os.path.join(
                self.response_persistent_folder, "intermediate_artifacts"
            )
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved to: {self.intermediate_dir}")

        # Setup response cache
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "x_omni_visual_cot_response.json"
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

        # Load model with full vision capabilities
        eval_logger.info(f"Loading X-Omni model from {pretrained}")
        self._load_model()
        eval_logger.info("X-Omni Visual CoT model initialized successfully")

    def _load_model(self):
        """Load X-Omni model with both generation and understanding capabilities"""
        model_path = self.pretrained

        # Load tokenizer
        eval_logger.info("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Apply regex patch for robust loading (only if method exists)
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
            
            # Load model without device_map to avoid issues with custom layers
            # We'll manually place it on available GPUs
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Manually move model to GPU(s)
            if torch.cuda.is_available():
                # Use the first available GPU
                self._model = self._model.to('cuda:0')
                eval_logger.info("Model loaded on cuda:0")
        except Exception as e:
            eval_logger.error(f"Failed to load model: {e}")
            raise e
        finally:
            if original_adjust is not None:
                PreTrainedModel._adjust_missing_and_unexpected_keys = original_adjust

        # Initialize full vision components (both encoder and decoder)
        if hasattr(self._model, "init_vision"):
            eval_logger.info(f"Initializing full vision components with FLUX path: {self.flux_pipe_path}")
            self._model.init_vision(self.flux_pipe_path)
            eval_logger.info("Vision encoder (SigLIP) and decoder (FLUX) initialized")
        else:
            eval_logger.warning("Model does not have 'init_vision' method")

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

    def _extract_image_from_various_formats(self, img_data) -> Optional[Image.Image]:
        """Extract PIL Image from various formats"""
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
                    return self._extract_image_from_various_formats(img_data["image"])
            else:
                return Image.open(img_data).convert("RGB")
        except Exception as e:
            eval_logger.debug(f"Failed to extract image: {e}")
            return None

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        doc_id: str,
        task: str,
        original_image: Optional[Image.Image] = None
    ) -> Tuple[str, List[str]]:
        """Stage 1: Generate visual diagram using X-Omni's FLUX generation"""
        try:
            self.set_seed(self.seed)
            self.model.set_generation_mode("image")

            # Move FLUX to GPU if it was offloaded
            if self.offload_flux_to_cpu and hasattr(self.model, 'flux_pipe') and self.model.flux_pipe is not None:
                if hasattr(self.model.flux_pipe, 'to'):
                    eval_logger.debug("Moving FLUX to GPU for generation")
                    self.model.flux_pipe.to('cuda:0')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Build prompt with optional image context
            if original_image is not None:
                image_str = self.model.tokenize_image(original_image)
                full_prompt = image_str + '\\n' + generation_prompt
            else:
                full_prompt = generation_prompt

            input_ids = self.model.mmencode(
                self.tokenizer,
                texts=[full_prompt],
                return_tensors="pt"
            )

            # For distributed models, get the device of the embedding layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_embed_tokens'):
                input_device = self.model.model.lm_embed_tokens.weight.device
            else:
                input_device = self.model.device

            input_ids = input_ids.to(input_device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.stage1_max_new_tokens,
                    do_sample=self.stage1_do_sample,
                    temperature=self.stage1_temperature if self.stage1_do_sample else 1.0,
                    top_p=self.stage1_top_p if self.stage1_do_sample else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            texts, images = self.model.mmdecode(self.tokenizer, output_ids)
            output_text = texts[-1] if texts else ""

            # Save generated images
            output_images = []
            for idx, image in enumerate(images):
                task_dir = os.path.join(self.output_image_dir, task)
                os.makedirs(task_dir, exist_ok=True)
                safe_filename = f"{doc_id}_gen_{idx}.png"
                image_path = os.path.join(task_dir, safe_filename)
                image.save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Generated image saved: {image_path}")

            # Clear GPU cache after generation
            del output_ids, input_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Offload FLUX to CPU to save GPU memory
            if self.offload_flux_to_cpu and hasattr(self.model, 'flux_pipe') and self.model.flux_pipe is not None:
                if hasattr(self.model.flux_pipe, 'to'):
                    eval_logger.debug("Offloading FLUX to CPU to save GPU memory")
                    self.model.flux_pipe.to('cpu')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return output_text, output_images

        except Exception as e:
            eval_logger.error(f"Stage 1 generation error: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return "", []
            raise e

    def _stage2_understand_with_images(
        self,
        question: str,
        original_image: Optional[Image.Image],
        generated_image_path: str
    ) -> str:
        """Stage 2: Answer question using both original and generated images"""
        try:
            self.set_seed(self.seed)
            self.model.set_generation_mode("text")

            # Load generated image
            gen_img = Image.open(generated_image_path).convert("RGB")

            # Build multi-image prompt
            image_strs = []
            if original_image is not None:
                image_strs.append(self.model.tokenize_image(original_image))
            image_strs.append(self.model.tokenize_image(gen_img))

            # Combine images and question
            full_prompt = '\\n'.join(image_strs) + '\\n' + question
            message = [{'role': 'user', 'content': full_prompt}]

            input_ids = self.tokenizer.apply_chat_template(
                message, add_generation_prompt=True, return_tensors="pt"
            )

            # For distributed models, get the device of the embedding layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_embed_tokens'):
                input_device = self.model.model.lm_embed_tokens.weight.device
            else:
                input_device = self.model.device

            input_ids = input_ids.to(input_device)
            attention_mask = torch.ones_like(input_ids)
            eos_token_id = self.tokenizer.encode('<|im_end|>')[0]

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.stage2_max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature if self.stage2_do_sample else 0.0,
                    top_p=self.stage2_top_p if self.stage2_do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=0,
                    use_cache=True,
                    repetition_penalty=1.1,  # Prevent repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                )

            texts, _ = self.model.mmdecode(self.tokenizer, output_ids[:, input_ids.shape[1]:-1])
            output_text = texts[0] if texts else ""

            # Clear GPU cache after generation
            del output_ids, input_ids, attention_mask, gen_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_text

        except Exception as e:
            eval_logger.error(f"Stage 2 understanding error: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return ""
            raise e

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
        """Save intermediate artifacts for debugging"""
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

        eval_logger.debug(f"Saved intermediate artifacts: {metadata_path}")

    def flatten(self, input_list):
        """Flatten nested lists"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

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
        Uni-MMMU interleaved generation for X-Omni Visual CoT.
        
        Aligned with Bagel's approach:
        - Jigsaw: gen_image(cand0) → gen_image(cand1) → gen_text(answer)
        - Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)

        Args:
            input_images: List of input images
            prompt: Base prompt text
            doc_id: Document ID for file naming
            task: Task name for file naming
            interleaved_config: Configuration dict from yaml
            doc: Document data for dynamic num_images extraction

        Returns:
            Tuple of (final_text_answer, list_of_generated_image_paths)
        """
        import json as json_module

        task_type = interleaved_config.get("task_type", "jigsaw")

        # Get num_images dynamically from doc if available
        num_images = interleaved_config.get("num_images", 2)
        if doc is not None:
            if task_type == "maze":
                steps_str = doc.get("steps", "[]")
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                steps_str = doc.get("steps_words", "[]")
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)

        # Extract original image from input_images
        original_image = None
        if input_images and len(input_images) > 0:
            original_image = self._extract_image_from_various_formats(input_images[0])

        generated_images = []

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            # Image 1: Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            gen_prompt1 = prompt + "\\n\\n" + suffix1

            _, img_paths_0 = self._stage1_generate_image(
                generation_prompt=gen_prompt1,
                doc_id=f"{doc_id}_cand0",
                task=task,
                original_image=original_image,
            )
            if img_paths_0:
                generated_images.extend(img_paths_0)
                eval_logger.info(f"Saved jigsaw image 0: {img_paths_0[0]}")

            # Image 2: Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            gen_prompt2 = prompt + "\\n\\n" + suffix2

            _, img_paths_1 = self._stage1_generate_image(
                generation_prompt=gen_prompt2,
                doc_id=f"{doc_id}_cand1",
                task=task,
                original_image=original_image,
            )
            if img_paths_1:
                generated_images.extend(img_paths_1)
                eval_logger.info(f"Saved jigsaw image 1: {img_paths_1[0]}")

            # Final answer using all images (original + generated)
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\\n'
                "Do not output any additional images."
            )
            final_question = prompt + "\\n\\n" + final_suffix

            # Build complete image context: original images + generated images
            all_images = []
            # Add original input images (reference + candidates)
            for img in input_images:
                if img is not None:
                    processed_img = self._extract_image_from_various_formats(img)
                    if processed_img is not None:
                        all_images.append(processed_img)
            
            # Add generated completion images
            for img_path in generated_images:
                try:
                    gen_img = Image.open(img_path).convert("RGB")
                    all_images.append(gen_img)
                except Exception as e:
                    eval_logger.warning(f"Failed to load generated image {img_path}: {e}")

            # Generate final answer with all images
            final_text = self._generate_text_with_multiple_images(
                images=all_images,
                question=final_question,
                doc_id=doc_id
            )

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            # Following Bagel's approach: generate planning text first, then image
            
            step_texts = []  # Store all planning texts
            step_images = []  # Store all generated step images
            current_context_images = [original_image] if original_image else []
            
            for i in range(1, num_images + 1):
                # Step 1: Generate planning text (like Bagel)
                if task_type == "maze":
                    plan_suffix = f'Now planning for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right."'
                else:  # sliding
                    plan_suffix = f'Now planning for step {i}, Please output a sentence describing which tile to move and in which direction.'

                plan_prompt = prompt + "\\n\\n" + plan_suffix
                
                # Generate planning text using current context
                plan_text = self._generate_text_with_multiple_images(
                    images=current_context_images,
                    question=plan_prompt,
                    doc_id=f"{doc_id}_plan_{i}",
                    max_tokens=128
                )
                
                eval_logger.info(f"Step {i} plan: {plan_text}")
                step_texts.append(plan_text)

                # Step 2: Generate step image based on plan
                img_prompt = prompt + "\\n\\n" + plan_text + f"\\n\\nNow, generate the image for step {i}."
                
                # Use the most recent image as context for generation
                context_image = current_context_images[-1] if current_context_images else None
                
                _, img_paths = self._stage1_generate_image(
                    generation_prompt=img_prompt,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task,
                    original_image=context_image,
                )

                if img_paths:
                    generated_images.extend(img_paths)
                    eval_logger.info(f"Saved step {i} image: {img_paths[0]}")
                    
                    # Load and add to context for next step
                    try:
                        step_img = Image.open(img_paths[0]).convert("RGB")
                        step_images.append(step_img)
                        current_context_images.append(step_img)
                        eval_logger.debug(f"Added step {i} image to context")
                    except Exception as e:
                        eval_logger.warning(f"Failed to load generated image for step {i}: {e}")

            # Final answer generation with complete context
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            
            # Build complete context: original + all step texts and images
            context_parts = [prompt]
            for i, (plan_text, step_img) in enumerate(zip(step_texts, step_images), 1):
                context_parts.append(f"Step {i} plan: {plan_text}")
                context_parts.append(f"Step {i} completed.")
            context_parts.append(final_suffix)
            
            final_question = "\\n\\n".join(context_parts)

            # Generate final answer with all context images
            final_text = self._generate_text_with_multiple_images(
                images=current_context_images,
                question=final_question,
                doc_id=doc_id
            )

        return final_text, generated_images

    def _generate_text_with_multiple_images(
        self,
        images: List[Image.Image],
        question: str,
        doc_id: str,
        max_tokens: int = None
    ) -> str:
        """
        Generate text response using multiple images as context.
        
        Args:
            images: List of PIL Images to use as context
            question: Text question/prompt
            doc_id: Document ID for logging
            max_tokens: Maximum tokens to generate (defaults to stage2_max_new_tokens)
            
        Returns:
            Generated text response
        """
        if not images:
            eval_logger.warning(f"No images provided for text generation, doc_id={doc_id}")
            return ""
            
        try:
            self.set_seed(self.seed)
            self.model.set_generation_mode("text")

            # Build multi-image prompt
            image_strs = []
            for img in images:
                if img is not None:
                    image_strs.append(self.model.tokenize_image(img))

            # Combine images and question
            full_prompt = '\\n'.join(image_strs) + '\\n' + question
            message = [{'role': 'user', 'content': full_prompt}]

            input_ids = self.tokenizer.apply_chat_template(
                message, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = input_ids.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
            eos_token_id = self.tokenizer.encode('<|im_end|>')[0]

            max_new_tokens = max_tokens if max_tokens is not None else self.stage2_max_new_tokens

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature if self.stage2_do_sample else 0.0,
                    top_p=self.stage2_top_p if self.stage2_do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=0,
                    use_cache=True,
                    repetition_penalty=1.1,  # Prevent repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                )

            texts, _ = self.model.mmdecode(self.tokenizer, output_ids[:, input_ids.shape[1]:-1])
            result_text = texts[0] if texts else ""

            # Clear GPU cache
            del output_ids, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result_text

        except Exception as e:
            eval_logger.error(f"Multi-image text generation failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            else:
                raise

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="X-Omni Visual CoT Generating"
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache and self.response_cache[doc_uuid]:
                    res.append(self.response_cache[doc_uuid])
                    pbar.update(1)
                    continue

            # Initialize original_image to avoid UnboundLocalError
            original_image = None

            # Check if this is Uni-MMMU interleaved generation mode
            bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if bagel_interleaved is not None:
                # Uni-MMMU interleaved generation mode
                eval_logger.info(f"Uni-MMMU interleaved mode for doc {doc_id}")

                # Get input images and doc data
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual:
                    visuals = doc_to_visual(doc)
                    if visuals:
                        input_images = visuals if isinstance(visuals, list) else [visuals]

                # Generate using interleaved mode
                final_ans, gen_paths = self.generate_uni_mmmu_interleaved(
                    input_images, contexts, str(doc_id), task, bagel_interleaved, doc
                )

                # Fix format issues based on task type
                final_ans = self.fix_json_format(final_ans)

                # Save intermediate artifacts if enabled
                self._save_intermediate_artifacts(
                    doc_id=str(doc_id),
                    task=task,
                    generation_prompt=f"Interleaved generation: {bagel_interleaved.get('task_type', 'unknown')}",
                    stage1_text="",
                    generated_images=gen_paths,
                    question=contexts,
                    stage2_answer=final_ans,
                )

                res.append(final_ans)
            else:
                # Standard multi-image generation mode
                # Get all original images
                original_images = []
                if doc_to_visual is not None:
                    visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
                    for visual in visuals:
                        if visual is not None:
                            img = self._extract_image_from_various_formats(visual)
                            if img is not None:
                                original_images.append(img)

                # Use first image for generation context, but keep all for final answer
                primary_image = original_images[0] if original_images else None

                # Stage 1: Generate visual diagram
                generation_prompt = self.generation_prompt_template.format(question=contexts)
                eval_logger.info(f"Stage 1: Generating visual diagram for doc {doc_id}")
                stage1_text, gen_paths = self._stage1_generate_image(
                    generation_prompt,
                    str(doc_id),
                    task,
                    primary_image
                )

                # Stage 2: Answer with all images (original + generated)
                if gen_paths:
                    eval_logger.info(f"Stage 2: Answering with all images for doc {doc_id}")
                    
                    # Load generated image
                    generated_images = []
                    for gen_path in gen_paths:
                        try:
                            gen_img = Image.open(gen_path).convert("RGB")
                            generated_images.append(gen_img)
                        except Exception as e:
                            eval_logger.warning(f"Failed to load generated image {gen_path}: {e}")
                    
                    # Combine all images for final answer
                    all_images = original_images + generated_images
                    
                    final_answer = self._generate_text_with_multiple_images(
                        images=all_images,
                        question=contexts,
                        doc_id=str(doc_id)
                    )
                else:
                    eval_logger.warning(f"No image generated for doc {doc_id}, using original images only")
                    if original_images:
                        final_answer = self._generate_text_with_multiple_images(
                            images=original_images,
                            question=contexts,
                            doc_id=str(doc_id)
                        )
                    else:
                        final_answer = ""

                # Fix format issues based on task type
                final_answer = self.fix_json_format(final_answer)

                # Save intermediate artifacts
                self._save_intermediate_artifacts(
                    doc_id=str(doc_id),
                    task=task,
                    generation_prompt=generation_prompt,
                    stage1_text=stage1_text,
                    generated_images=gen_paths,
                    question=contexts,
                    stage2_answer=final_answer,
                )

                res.append(final_answer)

            # Clear memory after each request
            if 'original_images' in locals() and original_images:
                for img in original_images:
                    if img is not None:
                        del img
            if 'generated_images' in locals() and generated_images:
                for img in generated_images:
                    if img is not None:
                        del img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = res[-1]
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("X-Omni Visual CoT is a generation model")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round")
