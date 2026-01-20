import base64
import os
from io import BytesIO
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from diffusers import DiffusionPipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import diffusers or transformers; Please install via `pip install diffusers transformers qwen-vl-utils`")


@register_model("qwen_image_edit")
class QwenImageEdit(lmms):
    """
    Qwen-Image-Edit 20B Model
    https://huggingface.co/Qwen/Qwen-Image-Edit
    
    A 20B parameter MMDiT model that supports BOTH:
    1. Image Understanding (VQA mode): Uses Qwen2.5-VL encoder for image-to-text tasks
    2. Image Editing: Text-guided image editing in Chinese and English
    
    The model internally uses Qwen2.5-VL as semantic encoder, enabling visual understanding.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",  # Qwen2.5-VL for understanding
        mode: str = "understanding",  # "understanding" or "editing"
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        torch_dtype: str = "bfloat16",
        # Understanding mode parameters
        max_pixels: int = 1605632,
        min_pixels: int = 256 * 28 * 28,
        system_prompt: str = "You are a helpful assistant.",
        # Editing mode parameters  
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_type: str = "pil",
        save_generated_images: bool = True,
        generated_image_dir: str = "./qwen_edit_generated_images",
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Validate mode
        if mode not in ["understanding", "editing"]:
            raise ValueError(f"mode must be 'understanding' or 'editing', got {mode}")
        self.mode = mode
        
        # Validate torch dtype
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if torch_dtype not in dtype_mapping:
            raise ValueError(f"torch_dtype must be one of {list(dtype_mapping.keys())}, got {torch_dtype}")
        self.torch_dtype = dtype_mapping[torch_dtype]

        accelerator = Accelerator()
        self.accelerator = accelerator
        
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        eval_logger.info(f"Loading Qwen model in {mode} mode from {pretrained}")
        
        if mode == "understanding":
            # Load Qwen2.5-VL for image understanding (VQA)
            try:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                ).eval()
                self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
                self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
                self.system_prompt = system_prompt
                self.max_pixels = max_pixels
                self.min_pixels = min_pixels
                eval_logger.info(f"Successfully loaded Qwen2.5-VL for understanding mode")
            except Exception as e:
                eval_logger.error(f"Failed to load Qwen2.5-VL: {e}")
                raise
        else:
            # Load diffusion pipeline for image editing
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    pretrained,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    use_safetensors=True,
                )
                eval_logger.info(f"Successfully loaded Qwen-Image-Edit pipeline for editing mode")
            except Exception as e:
                eval_logger.error(f"Failed to load Qwen-Image-Edit: {e}")
                raise

        # Parameters
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.output_type = output_type
        self.save_generated_images = save_generated_images
        self.generated_image_dir = generated_image_dir
        if self.save_generated_images and mode == "editing":
            os.makedirs(self.generated_image_dir, exist_ok=True)
            eval_logger.info(f"Generated images will be saved to: {self.generated_image_dir}")

        self.batch_size_per_gpu = int(batch_size)
        self._config = self._model.config if mode == "understanding" else None
        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if mode == "understanding":
                if accelerator.distributed_type == DistributedType.FSDP:
                    self._model = accelerator.prepare(self._model)
                else:
                    self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer if self.mode == "understanding" else None

    @property
    def model(self):
        if self.mode == "understanding":
            if hasattr(self, "accelerator"):
                return self.accelerator.unwrap_model(self._model)
            return self._model
        return None

    @property
    def eot_token_id(self):
        if self.mode == "understanding":
            return self.tokenizer.eos_token_id
        return None

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def flatten(self, input):
        """Flatten a list of lists into a single list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or return PIL Image directly."""
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def save_image(self, image: Image.Image, doc_id: str, task: str = "edit") -> str:
        """Save generated image and return the path."""
        if not self.save_generated_images:
            return ""
        
        filename = f"{task}_{doc_id}.png"
        filepath = os.path.join(self.generated_image_dir, filename)
        image.save(filepath)
        eval_logger.debug(f"Saved generated image to: {filepath}")
        return filepath

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
        Uni-MMMU interleaved generation - Fully aligned with Bagel implementation.
        
        For Qwen-Image-Edit, this requires both editing and understanding capabilities:
        - Image generation: Uses diffusion pipeline (editing mode)
        - Text generation: Uses Qwen2.5-VL (understanding mode)
        
        This method temporarily loads both models if needed.
        
        Args:
            input_images: List of input images
            prompt: Base prompt text
            doc_id: Document ID for file naming
            task: Task name
            interleaved_config: Configuration from YAML
            doc: Document data for dynamic num_images
            
        Returns:
            Tuple of (final_text_answer, list_of_generated_image_paths)
        """
        import json as json_module
        
        task_type = interleaved_config.get("task_type", "jigsaw")
        
        # Get num_images dynamically from doc
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
        
        # Get generation parameters
        cfg_text_scale = interleaved_config.get("cfg_text_scale", 4.0)
        cfg_interval = interleaved_config.get("cfg_interval", 0.4)
        timestep_shift = interleaved_config.get("timestep_shift", 3.0)
        num_timesteps = interleaved_config.get("num_timesteps", 50)
        
        generated_images = []
        
        # Ensure we have both editing and understanding capabilities
        if self.mode != "editing":
            raise ValueError(
                "Uni-MMMU interleaved mode requires editing mode. "
                "Use: --model_args mode=editing"
            )
        
        # We need understanding model for text generation
        # Load it temporarily if not already loaded
        if not hasattr(self, '_understanding_model'):
            eval_logger.info("Loading understanding model for text generation...")
            from lmms_eval.models.simple.qwen_image_edit import QwenImageEdit
            self._understanding_model = QwenImageEdit(
                pretrained=self.pretrained if "VL" in self.pretrained else "Qwen/Qwen2-VL-7B-Instruct",
                mode="understanding",
            )
        
        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            eval_logger.info(f"Jigsaw mode: Generating 2 completion images for doc {doc_id}")
            
            # Image 1: Candidate 0 completion
            edit_prompt_0 = f"{prompt}\n\nOutput ONLY a single image with Candidate 0 placed in the bottom-right cell."
            
            class MockEditRequest:
                def __init__(self, doc, args):
                    self.doc = doc
                    self.arguments = args
            
            mock_doc_0 = {
                "image": input_images[0] if input_images else None,
                "prompt": edit_prompt_0,
                "task": task,
                "doc_id": f"{doc_id}_cand0",
                "num_inference_steps": num_timesteps,
                "guidance_scale": cfg_text_scale,
            }
            
            request_0 = MockEditRequest(mock_doc_0, [edit_prompt_0])
            result_0 = self._generate_editing([request_0])
            
            img0_path = os.path.join(self.generated_image_dir, f"{task}_{doc_id}_cand0.png")
            if os.path.exists(img0_path):
                generated_images.append(img0_path)
                eval_logger.info(f"Saved jigsaw image 0: {img0_path}")
            
            # Image 2: Candidate 1 completion  
            edit_prompt_1 = f"{prompt}\n\nOutput ONLY a single image with Candidate 1 placed in the bottom-right cell."
            
            mock_doc_1 = {
                "image": input_images[0] if input_images else None,
                "prompt": edit_prompt_1,
                "task": task,
                "doc_id": f"{doc_id}_cand1",
                "num_inference_steps": num_timesteps,
                "guidance_scale": cfg_text_scale,
            }
            
            request_1 = MockEditRequest(mock_doc_1, [edit_prompt_1])
            result_1 = self._generate_editing([request_1])
            
            img1_path = os.path.join(self.generated_image_dir, f"{task}_{doc_id}_cand1.png")
            if os.path.exists(img1_path):
                generated_images.append(img1_path)
                eval_logger.info(f"Saved jigsaw image 1: {img1_path}")
            
            # Generate final answer using understanding model
            final_prompt = (
                f"{prompt}\n\n"
                f"Two completion images have been generated. "
                f'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{{"choice": 0 or 1, "rationale": "≤30 words"}}</FINAL_ANSWER_JSON>'
            )
            
            # Load generated images
            completion_images = []
            for img_path in generated_images:
                if os.path.exists(img_path):
                    completion_images.append(Image.open(img_path).convert("RGB"))
            
            # Use understanding model to generate final answer
            class MockUnderstandRequest:
                def __init__(self, doc_id, task, split, prompt, images):
                    self.args = (
                        prompt,
                        {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False},
                        lambda doc: images,
                        doc_id,
                        task,
                        split,
                    )
            
            if not hasattr(self._understanding_model, 'task_dict'):
                self._understanding_model.task_dict = {}
            if task not in self._understanding_model.task_dict:
                self._understanding_model.task_dict[task] = {}
            if 'test' not in self._understanding_model.task_dict[task]:
                self._understanding_model.task_dict[task]['test'] = {}
            self._understanding_model.task_dict[task]['test'][doc_id] = {}
            
            all_images = (input_images if input_images else []) + completion_images
            mock_request = MockUnderstandRequest(doc_id, task, 'test', final_prompt, all_images)
            final_text = self._understanding_model._generate_understanding([mock_request])[0]
            
        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            eval_logger.info(f"{task_type.capitalize()} mode: Generating {num_images} step images for doc {doc_id}")
            
            accumulated_text = prompt
            current_image = input_images[0] if input_images else None
            
            for i in range(1, num_images + 1):
                # Generate planning text
                if task_type == "maze":
                    plan_prompt = f'{accumulated_text}\n\nNow planning for step {i}, output: "Next, move one step up/down/left/right."'
                else:  # sliding
                    plan_prompt = f'{accumulated_text}\n\nNow planning for step {i}, describe which tile to move and in which direction.'
                
                class MockUnderstandRequest:
                    def __init__(self, doc_id, task, split, prompt, images):
                        self.args = (
                            prompt,
                            {"max_new_tokens": 128, "temperature": 0.0, "do_sample": False},
                            lambda doc: images if images else [],
                            doc_id,
                            task,
                            split,
                        )
                
                if not hasattr(self._understanding_model, 'task_dict'):
                    self._understanding_model.task_dict = {}
                if task not in self._understanding_model.task_dict:
                    self._understanding_model.task_dict[task] = {}
                if 'test' not in self._understanding_model.task_dict[task]:
                    self._understanding_model.task_dict[task]['test'] = {}
                self._understanding_model.task_dict[task]['test'][doc_id] = {}
                
                mock_request = MockUnderstandRequest(doc_id, task, 'test', plan_prompt, [current_image] if current_image else [])
                plan_text = self._understanding_model._generate_understanding([mock_request])[0]
                
                eval_logger.info(f"Step {i} plan: {plan_text}")
                accumulated_text += f"\n{plan_text}"
                
                # Generate step image
                img_prompt = f"{accumulated_text}\n\nNow, generate the image for step {i}."
                
                class MockEditRequest:
                    def __init__(self, doc, args):
                        self.doc = doc
                        self.arguments = args
                
                mock_doc = {
                    "image": current_image,
                    "prompt": img_prompt,
                    "task": task,
                    "doc_id": f"{doc_id}_step_{i:04d}",
                    "num_inference_steps": num_timesteps,
                    "guidance_scale": cfg_text_scale,
                }
                
                request = MockEditRequest(mock_doc, [img_prompt])
                self._generate_editing([request])
                
                img_path = os.path.join(self.generated_image_dir, f"{task}_{doc_id}_step_{i:04d}.png")
                if os.path.exists(img_path):
                    generated_images.append(img_path)
                    current_image = Image.open(img_path).convert("RGB")
                    eval_logger.info(f"Saved step {i} image: {img_path}")
            
            # Generate final answer
            final_prompt = (
                f"{accumulated_text}\n\n"
                f"After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                f"as <ANSWER_JSON>[...]</ANSWER_JSON>."
            )
            
            all_step_images = [Image.open(p).convert("RGB") for p in generated_images if os.path.exists(p)]
            all_images = (input_images if input_images else []) + all_step_images
            
            mock_request = MockUnderstandRequest(doc_id, task, 'test', final_prompt, all_images)
            final_text = self._understanding_model._generate_understanding([mock_request])[0]
        
        return final_text, generated_images

    def generate_until(self, requests) -> List[str]:
        """
        Generate outputs based on mode:
        - Understanding mode: Image + question → text answer (VQA)
        - Editing mode: Image + edit instruction → edited image
        - Uni-MMMU interleaved mode: Detected via bagel_interleaved config
        """
        # Check for Uni-MMMU interleaved mode (aligned with Bagel)
        has_interleaved = any(
            req.args[1].get("bagel_interleaved", None) is not None
            for req in requests
        )
        
        if has_interleaved:
            # Uni-MMMU interleaved generation
            import json
            res = []
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Uni-MMMU Generating")
            
            for request in requests:
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
                bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)
                
                if bagel_interleaved is not None:
                    eval_logger.info(f"Uni-MMMU interleaved mode for doc {doc_id}")
                    
                    # Get document and input images
                    doc = self.task_dict[task][split][doc_id]
                    input_images = []
                    if doc_to_visual is not None:
                        visuals = doc_to_visual(doc)
                        if isinstance(visuals, list):
                            input_images = visuals
                        else:
                            input_images = [visuals]
                    
                    # Generate using interleaved mode
                    output_text, output_images = self.generate_uni_mmmu_interleaved(
                        input_images, contexts, str(doc_id), task, bagel_interleaved, doc
                    )
                    
                    # Format output as JSON (aligned with Bagel)
                    formatted_output = json.dumps(
                        {"text": output_text, "images": output_images},
                        ensure_ascii=False
                    )
                    res.append(formatted_output)
                else:
                    res.append("")
                
                pbar.update(1)
            
            pbar.close()
            return res
        
        # Regular generation modes
        if self.mode == "understanding":
            return self._generate_understanding(requests)
        else:
            return self._generate_editing(requests)
    
    def _generate_understanding(self, requests) -> List[str]:
        """Image understanding mode: VQA using Qwen2.5-VL"""
        res = []
        
        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (Understanding)")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": self.system_prompt}]
                
                if visual_list[i] is not None:
                    content = []
                    for visual in visual_list[i]:
                        if isinstance(visual, Image.Image):
                            content.append({"type": "image", "image": visual})
                    content.append({"type": "text", "text": context})
                    message.append({"role": "user", "content": content})
                else:
                    message.append({"role": "user", "content": context})
                
                batched_messages.append(message)

            # Process with Qwen2.5-VL
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate
            generation_kwargs = {
                "max_new_tokens": gen_kwargs.get("max_new_tokens", 512),
                "temperature": gen_kwargs.get("temperature", 0.0),
                "do_sample": gen_kwargs.get("do_sample", False),
            }
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            for output_text in output_texts:
                res.append(output_text)
            pbar.update(len(chunk))

        pbar.close()
        return res
    
    def _generate_editing(self, requests) -> List[str]:
        """Image editing mode: Generate edited images"""
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (Editing)")

        for request in requests:
            doc = request.doc
            prompt = request.arguments[0] if request.arguments else doc.get("prompt", "")
            
            task_name = doc.get("task", "image_edit")
            doc_id = doc.get("doc_id", f"doc_{len(res)}")
            
            # Load input image
            if "image" in doc:
                input_image = self.load_image(doc["image"])
            elif "images" in doc and len(doc["images"]) > 0:
                input_image = self.load_image(doc["images"][0])
            else:
                eval_logger.warning(f"No input image found for doc_id {doc_id}, skipping")
                res.append("")
                pbar.update(1)
                continue

            num_steps = doc.get("num_inference_steps", self.num_inference_steps)
            cfg_scale = doc.get("guidance_scale", self.guidance_scale)
            
            try:
                output = self.pipeline(
                    prompt=prompt,
                    image=input_image,
                    num_inference_steps=num_steps,
                    guidance_scale=cfg_scale,
                    output_type=self.output_type,
                )
                
                generated_image = output.images[0]
                image_path = self.save_image(generated_image, doc_id, task_name)
                response = f"Generated image saved to: {image_path}" if image_path else "Image generated"
                res.append(response)
                
            except Exception as e:
                eval_logger.error(f"Error generating image for doc_id {doc_id}: {e}")
                res.append("")
            
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for image editing tasks.
        
        Note: This is not typically applicable for diffusion-based image editing models.
        Returns dummy values.
        """
        eval_logger.warning(
            "loglikelihood is not supported for Qwen-Image-Edit (diffusion model). "
            "Returning dummy values."
        )
        return [(0.0, False) for _ in requests]

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round generation"""
        if self.mode == "understanding":
            eval_logger.warning("Multi-round generation not fully implemented. Using single-round.")
        return self.generate_until(requests)

