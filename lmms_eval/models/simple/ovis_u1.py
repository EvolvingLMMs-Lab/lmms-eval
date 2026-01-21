from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.models.auto import configuration_auto

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@contextmanager
def patch_autoconfig_register():
    """Patch AutoConfig.register to allow re-registering existing configs.

    This is needed because some model repos (like Ovis-U1) try to register
    configs (e.g., 'aimv2') that are already built into newer transformers versions.
    """
    original_register = configuration_auto.CONFIG_MAPPING.register

    def patched_register(key, value, exist_ok=False):
        try:
            original_register(key, value, exist_ok=True)
        except TypeError:
            # Older transformers versions don't support exist_ok parameter
            try:
                original_register(key, value)
            except ValueError:
                # Config already exists, ignore
                pass

    configuration_auto.CONFIG_MAPPING.register = patched_register
    try:
        yield
    finally:
        configuration_auto.CONFIG_MAPPING.register = original_register


@register_model("ovis_u1")
class OvisU1(lmms):
    """
    Ovis-U1 Model
    https://huggingface.co/AIDC-AI/Ovis-U1-3B

    Ovis-U1 is a unified multimodal model that integrates multimodal understanding,
    text-to-image generation, and image editing within a single framework.

    Example usage:
        python -m lmms_eval --model ovis_u1 \
            --model_args pretrained=AIDC-AI/Ovis-U1-3B \
            --tasks mme,mmmu_val --batch_size 1 --device cuda:0
    """

    def __init__(
        self,
        pretrained: str = "AIDC-AI/Ovis-U1-3B",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        save_intermediate: bool = False,  # Ignored for ovis_u1, used in ovis_u1_visual_cot
        **kwargs,
    ) -> None:
        super().__init__()
        # Ignore save_intermediate parameter (used in visual_cot variant)
        # Check for unexpected kwargs
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        # Prepare model loading arguments
        model_kwargs = {
            "device_map": device,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype if dtype != "auto" else torch.bfloat16,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load model with patched AutoConfig.register to handle duplicate config names
        eval_logger.info(f"Loading Ovis-U1 model from {pretrained}")
        with patch_autoconfig_register():
            self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs)

        # Ovis-U1 uses model.text_tokenizer
        self._tokenizer = self._model.text_tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for Ovis-U1"
        self.use_cache = use_cache
        self.pretrained = pretrained

        # Setup distributed training
        if accelerator.num_processes > 1:
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert accelerator.distributed_type in distributed_type_list, (
                "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed supported"
            )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        """Return the model config."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def model(self):
        """Return the model, unwrapping it if using Accelerate."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        """Return the end of text token id."""
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        """Return the batch size."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return the device."""
        return self._device

    @property
    def rank(self):
        """Return the process rank."""
        return self._rank

    @property
    def world_size(self):
        """Return the world size."""
        return self._world_size

    def flatten(self, input_list):
        """Flatten a nested list."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood (not implemented for Ovis-U1)."""
        raise NotImplementedError("Loglikelihood not implemented for Ovis-U1")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until stopping criteria are met."""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        # Group requests by generation kwargs
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            # Get generation kwargs
            gen_kwargs = all_gen_kwargs[0]

            # Set default values
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Process each context
            assert len(contexts) == 1, "Batch size must be 1"
            context = contexts[0]

            # Prepare images
            images = []
            eval_logger.info(f"Processing visuals: {len(visuals)} items")
            for idx, visual in enumerate(visuals):
                eval_logger.debug(f"Visual {idx}: type={type(visual)}")
                
                if isinstance(visual, str):
                    # If visual is a path, load it
                    visual = Image.open(visual).convert("RGB")
                    eval_logger.debug(f"  -> Loaded image from path: {visual.size}")
                elif isinstance(visual, Image.Image):
                    # Already a PIL Image
                    visual = visual.convert("RGB")
                    eval_logger.debug(f"  -> Already PIL Image: {visual.size}")
                elif isinstance(visual, dict):
                    # Handle dict format - common in HuggingFace datasets
                    if "bytes" in visual:
                        # Image bytes
                        from io import BytesIO
                        visual = Image.open(BytesIO(visual["bytes"])).convert("RGB")
                        eval_logger.debug(f"  -> Loaded image from bytes: {visual.size}")
                    elif "path" in visual:
                        # Image path
                        visual = Image.open(visual["path"]).convert("RGB")
                        eval_logger.debug(f"  -> Loaded image from dict path: {visual.size}")
                    elif "image" in visual:
                        # Nested image
                        img = visual["image"]
                        if isinstance(img, str):
                            visual = Image.open(img).convert("RGB")
                        elif isinstance(img, Image.Image):
                            visual = img.convert("RGB")
                        else:
                            eval_logger.warning(f"  -> Skipping dict with unsupported image type: {type(img)}")
                            continue
                        eval_logger.debug(f"  -> Loaded image from dict: {visual.size}")
                    else:
                        eval_logger.warning(f"  -> Skipping dict without recognized keys: {visual.keys()}")
                        continue
                else:
                    # Skip non-image types
                    eval_logger.warning(f"  -> Skipping non-image type: {type(visual)}")
                    continue
                images.append(visual)
            
            eval_logger.info(f"Final images list: {len(images)} images")

            # Build query with <image> tokens for Ovis-U1
            # Add one <image> token per image
            if len(images) > 0:
                image_placeholder = "<image>" * len(images)
                query = image_placeholder + context
                eval_logger.info(f"Processing {len(images)} images")
                eval_logger.info(f"Query starts with: {query[:100]}")
            else:
                query = context
                eval_logger.info("No images to process")

            # Preprocess inputs using model's built-in method
            # Ovis-U1's preprocess_inputs returns 4 values
            try:
                eval_logger.info(f"Calling preprocess_inputs with:")
                eval_logger.info(f"  - query length: {len(query)}")
                eval_logger.info(f"  - images: {len(images) if images else 0} images")
                eval_logger.info(f"  - image types: {[type(img) for img in images] if images else []}")
                
                result = self.model.preprocess_inputs(
                    text_or_conversations=query,
                    images=images if len(images) > 0 else None,
                )
                
                # Debug: Check what we got
                eval_logger.info(f"preprocess_inputs returned {len(result)} values")
                eval_logger.info(f"Result types: {[type(r) for r in result]}")
                
                _, input_ids, pixel_values, grid_thws = result
                
                eval_logger.info(f"input_ids: {input_ids.shape if input_ids is not None else None}")
                eval_logger.info(f"pixel_values: {pixel_values.shape if pixel_values is not None else None}")
                eval_logger.info(f"grid_thws: {grid_thws.shape if grid_thws is not None else None}")
                
                if len(images) > 0 and (pixel_values is None or grid_thws is None):
                    eval_logger.error(f"ERROR: We have {len(images)} images but pixel_values or grid_thws is None!")
                    eval_logger.error(f"This means preprocess_inputs didn't process the images correctly")
                    
            except Exception as e:
                eval_logger.error(f"Error in preprocess_inputs: {e}")
                eval_logger.error(f"Query: {query[:200]}")
                eval_logger.error(f"Number of images: {len(images)}")
                raise

            # Move to device
            input_ids = input_ids.unsqueeze(0).to(self._device)
            attention_mask = torch.ones_like(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device=self._device, dtype=self.model.dtype)
            if grid_thws is not None:
                grid_thws = grid_thws.to(self._device)

            # Set generation parameters
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # Prepare generation kwargs
            generate_kwargs = {
                "inputs": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
            }

            # Add sampling parameters if temperature > 0
            if gen_kwargs["temperature"] > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = gen_kwargs["temperature"]
                if gen_kwargs["top_p"] is not None:
                    generate_kwargs["top_p"] = gen_kwargs["top_p"]
            else:
                generate_kwargs["do_sample"] = False

            if gen_kwargs["num_beams"] > 1:
                generate_kwargs["num_beams"] = gen_kwargs["num_beams"]

            # Generate response
            outputs = self.model.generate(**generate_kwargs)

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up to free memory
            del outputs, input_ids, pixel_values, grid_thws
            torch.cuda.empty_cache()

            res.append(response)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), response
            )
            pbar.update(1)

        # Reorder results to original order
        res = re_ords.get_original(res)
        pbar.close()
        return res

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
        Uni-MMMU interleaved generation for Ovis-U1.

        This implements interleaved text and image generation for Uni-MMMU tasks:
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
        import os

        task_type = interleaved_config.get("task_type", "jigsaw")
        num_images = interleaved_config.get("num_images", 2)
        output_dir = interleaved_config.get("output_dir", "./logs/ovis_u1_generated_images")
        os.makedirs(output_dir, exist_ok=True)

        # Get num_images dynamically from doc if available
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

        generated_images = []
        conversation_history = []

        # Add input images to conversation
        for img in input_images:
            if img is not None:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif isinstance(img, dict):
                    if "bytes" in img:
                        from io import BytesIO
                        img = Image.open(BytesIO(img["bytes"])).convert("RGB")
                    elif "path" in img:
                        img = Image.open(img["path"]).convert("RGB")
                    elif "image" in img:
                        img_data = img["image"]
                        if isinstance(img_data, str):
                            img = Image.open(img_data).convert("RGB")
                        else:
                            img = img_data.convert("RGB")
                conversation_history.append(img)

        # Add initial prompt
        conversation_history.append(prompt)

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            # Image 1: Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            conversation_history.append(suffix1)
            
            # Generate image using Ovis-U1's generation capability
            # Note: This requires Ovis-U1 to have image generation support
            try:
                img0 = self.model.generate_image(conversation_history)
                img0_path = os.path.join(output_dir, f"{task}_{doc_id}_cand0.png")
                img0.save(img0_path)
                generated_images.append(img0_path)
                eval_logger.info(f"Saved jigsaw image 0: {img0_path}")
                conversation_history.append(img0)
                conversation_history.append("COMPLETED WITH CANDIDATE 0:")
            except Exception as e:
                eval_logger.error(f"Failed to generate image 0: {e}")

            # Image 2: Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            conversation_history.append(suffix2)
            
            try:
                img1 = self.model.generate_image(conversation_history)
                img1_path = os.path.join(output_dir, f"{task}_{doc_id}_cand1.png")
                img1.save(img1_path)
                generated_images.append(img1_path)
                eval_logger.info(f"Saved jigsaw image 1: {img1_path}")
                conversation_history.append(img1)
                conversation_history.append("COMPLETED WITH CANDIDATE 1:")
            except Exception as e:
                eval_logger.error(f"Failed to generate image 1: {e}")

            # Final answer
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            conversation_history.append(final_suffix)

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            for i in range(1, num_images + 1):
                # Generate planning text
                if task_type == "maze":
                    plan_suffix = f'Now planning for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right."'
                else:  # sliding
                    plan_suffix = f'Now planning for step {i}, Please output a sentence describing which tile to move and in which direction.'

                conversation_history.append(plan_suffix)
                
                try:
                    plan_text = self.model.generate_text(conversation_history, max_new_tokens=128)
                    eval_logger.info(f"Step {i} plan: {plan_text}")
                    conversation_history.append(plan_text)
                except Exception as e:
                    eval_logger.error(f"Failed to generate plan text for step {i}: {e}")
                    conversation_history.append(f"Step {i}")

                # Generate step image
                img_suffix = f"Now, generate the image for step {i}."
                conversation_history.append(img_suffix)
                
                try:
                    img = self.model.generate_image(conversation_history)
                    img_path = os.path.join(output_dir, f"{task}_{doc_id}_step_{i:04d}.png")
                    img.save(img_path)
                    generated_images.append(img_path)
                    eval_logger.info(f"Saved step {i} image: {img_path}")
                    conversation_history.append(img)
                except Exception as e:
                    eval_logger.error(f"Failed to generate image for step {i}: {e}")

            # Final answer
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            conversation_history.append(final_suffix)

        # Generate final text answer
        try:
            final_text = self.model.generate_text(conversation_history, max_new_tokens=512)
        except Exception as e:
            eval_logger.error(f"Failed to generate final text: {e}")
            final_text = ""

        return final_text, generated_images

    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate for multi-round conversations (not implemented)."""
        raise NotImplementedError("Multi-round generation not yet implemented")

