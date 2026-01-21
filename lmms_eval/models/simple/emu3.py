"""
Emu3 Multimodal Model Integration

Emu3 is a multimodal LLM that uses vector quantization to tokenize images into discrete tokens.
It supports both text generation from images and image generation from text using next-token prediction.

Paper: https://huggingface.co/papers/2409.18869
Model: https://huggingface.co/BAAI/Emu3-Chat-hf

Usage for understanding:
    python -m lmms_eval \
        --model emu3 \
        --model_args pretrained=BAAI/Emu3-Chat-hf \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/

Usage for generation:
    python -m lmms_eval \
        --model emu3 \
        --model_args pretrained=BAAI/Emu3-Gen-hf,mode=generation \
        --tasks ueval \
        --batch_size 1 \
        --output_path ./logs/
"""

import json
import os
from typing import List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("emu3")
class Emu3(lmms):
    """
    Emu3: Next-Token Prediction Multimodal Model

    Supports:
    - Text generation from images (understanding mode)
    - Image generation from text (generation mode)
    """

    def __init__(
        self,
        pretrained: str = "BAAI/Emu3-Chat-hf",
        mode: str = "understanding",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        # Text generation parameters
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        # Image generation parameters
        image_generation_steps: int = 50000,
        image_height: int = 1024,
        image_width: int = 1024,
        cfg_scale: float = 4.0,
        negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.",
        # Model loading
        use_flash_attention_2: bool = False,
        # Output
        output_image_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(f"mode must be 'understanding' or 'generation', got '{mode}'")

        self.pretrained = pretrained
        self.mode = mode
        self.device_str = device
        self.device_map = device_map
        self.batch_size_per_gpu = batch_size

        # Text generation parameters
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        # Image generation parameters
        self.image_generation_steps = image_generation_steps
        self.image_height = image_height
        self.image_width = image_width
        self.cfg_scale = cfg_scale
        self.negative_prompt = negative_prompt

        self.use_flash_attention_2 = use_flash_attention_2

        # Setup output directory for generated images
        if output_image_dir is None:
            self.output_image_dir = "./logs/emu3_generated_images"
        else:
            self.output_image_dir = output_image_dir

        if self.mode == "generation":
            os.makedirs(self.output_image_dir, exist_ok=True)
            eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Load model and processor
        eval_logger.info(f"Loading Emu3 model from {pretrained} in {mode} mode")
        self._load_model()
        eval_logger.info("Emu3 initialized successfully")

    def _load_model(self):
        """Load Emu3 model and processor"""
        from transformers import Emu3ForConditionalGeneration, Emu3Processor

        # Determine dtype
        dtype = torch.bfloat16

        # Setup attention implementation
        attn_implementation = "flash_attention_2" if self.use_flash_attention_2 else "eager"
        if self.use_flash_attention_2:
            try:
                import flash_attn
                eval_logger.info("Using Flash Attention 2")
            except ImportError:
                eval_logger.warning("flash_attn not installed, falling back to eager attention")
                attn_implementation = "eager"

        # Load processor
        self.processor = Emu3Processor.from_pretrained(self.pretrained)

        # Set padding side to left for batched generation (recommended by Emu3)
        self.processor.tokenizer.padding_side = "left"

        # Load model
        if self.device_map == "auto":
            # Auto device map for multi-GPU
            self.model = Emu3ForConditionalGeneration.from_pretrained(
                self.pretrained,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            ).eval()
            eval_logger.info(f"Model loaded with automatic device mapping")
        else:
            # Single device
            device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
            self.model = Emu3ForConditionalGeneration.from_pretrained(
                self.pretrained,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            ).to(device).eval()
            eval_logger.info(f"Model loaded on {device}")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, input_list):
        """Flatten nested lists"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for a list of requests"""
        res = []

        pbar = tqdm(
            total=len(requests),
            desc=f"Generating with Emu3 ({self.mode})",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Get images if available
            images = []
            if doc_to_visual is not None:
                doc = self.task_dict[task][split][doc_id]
                visuals = [doc_to_visual(doc)]
                visuals = self.flatten(visuals)

                for visual in visuals:
                    if isinstance(visual, str):
                        # Path to image
                        img = Image.open(visual).convert("RGB")
                    elif isinstance(visual, Image.Image):
                        img = visual.convert("RGB")
                    else:
                        eval_logger.warning(f"Unsupported visual type: {type(visual)}")
                        continue
                    images.append(img)

            # Generate response based on mode
            if self.mode == "understanding":
                # Understanding mode: image + text -> text
                if len(images) > 0:
                    response = self._generate_multimodal(contexts, images, gen_kwargs)
                else:
                    response = self._generate_text_only(contexts, gen_kwargs)
                formatted_output = response
            else:
                # Generation mode: text (+ optional image) -> image + text
                output_text, output_images = self._generate_image(
                    contexts, images if len(images) > 0 else None, doc_id, task, gen_kwargs
                )
                formatted_output = self.format_output(output_text, output_images)

            res.append(formatted_output)
            pbar.update(1)

        pbar.close()
        return res

    def _generate_text_only(self, context: str, gen_kwargs: dict) -> str:
        """Generate text-only response"""
        # Prepare inputs
        inputs = self.processor(
            text=context,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        # Get generation parameters
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        do_sample = gen_kwargs.get("do_sample", self.do_sample)
        temperature = gen_kwargs.get("temperature", self.temperature)
        top_p = gen_kwargs.get("top_p", self.top_p)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )

        # Decode
        generated_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return generated_text

    def _generate_multimodal(self, context: str, images: List[Image.Image], gen_kwargs: dict) -> str:
        """Generate multimodal response (image understanding)"""
        # For single image, we can pass it directly
        # For multiple images, use the first one
        if len(images) == 1:
            image = images[0]
        else:
            eval_logger.warning(f"Multiple images provided ({len(images)}), using first image only")
            image = images[0]

        # Debug: log image info
        eval_logger.debug(f"Image type: {type(image)}, mode: {getattr(image, 'mode', 'N/A')}, size: {getattr(image, 'size', 'N/A')}")

        # Ensure image is RGB
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')

        # Build conversation using chat template (required by Emu3)
        # Format from official docs: https://huggingface.co/docs/transformers/en/model_doc/emu3
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": context},
                ],
            },
        ]

        # Apply chat template to get properly formatted prompt
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        eval_logger.debug(f"Prompt (first 100 chars): {prompt[:100]}")

        try:
            # Prepare inputs - use list format as per official example
            inputs = self.processor(
                images=[image],
                text=[prompt],
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)
        except Exception as e:
            eval_logger.error(f"Processor error: {e}")
            eval_logger.error(f"Image info: type={type(image)}, mode={getattr(image, 'mode', 'N/A')}, size={getattr(image, 'size', 'N/A')}")
            raise

        # Get generation parameters
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        do_sample = gen_kwargs.get("do_sample", self.do_sample)
        temperature = gen_kwargs.get("temperature", self.temperature)
        top_p = gen_kwargs.get("top_p", self.top_p)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )

        # Decode - skip the input tokens
        generated_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return generated_text

    def _generate_image(
        self,
        prompt: str,
        input_images: Optional[List[Image.Image]],
        doc_id: str,
        task: str,
        gen_kwargs: dict
    ) -> Tuple[str, List[str]]:
        """Generate image from text prompt (optionally conditioned on input image)"""
        # Prepare inputs for image generation
        inputs = self.processor(
            text=prompt,
            padding=True,
            return_tensors="pt",
            return_for_image_generation=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        # Prepare negative prompt
        neg_inputs = self.processor(
            text=self.negative_prompt,
            return_tensors="pt"
        ).to(self.model.device)

        # Get image size
        image_sizes = inputs.pop("image_sizes")
        HEIGHT, WIDTH = self.image_height, self.image_width
        VISUAL_TOKENS = self.model.vocabulary_mapping.image_tokens

        # Define prefix constraint function for image generation
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            height, width = HEIGHT, WIDTH
            visual_tokens = VISUAL_TOKENS
            image_wrapper_token_id = torch.tensor([self.processor.tokenizer.image_wrapper_token_id], device=self.model.device)
            eoi_token_id = torch.tensor([self.processor.tokenizer.eoi_token_id], device=self.model.device)
            eos_token_id = torch.tensor([self.processor.tokenizer.eos_token_id], device=self.model.device)
            pad_token_id = torch.tensor([self.processor.tokenizer.pad_token_id], device=self.model.device)
            eof_token_id = torch.tensor([self.processor.tokenizer.eof_token_id], device=self.model.device)
            eol_token_id = self.processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

            position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
            offset = input_ids.shape[0] - position
            if offset % (width + 1) == 0:
                return (eol_token_id, )
            elif offset == (width + 1) * height + 1:
                return (eof_token_id, )
            elif offset == (width + 1) * height + 2:
                return (eoi_token_id, )
            elif offset == (width + 1) * height + 3:
                return (eos_token_id, )
            elif offset > (width + 1) * height + 3:
                return (pad_token_id, )
            else:
                return visual_tokens

        # Generate
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.image_generation_steps,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                return_dict_in_generate=True,
                negative_prompt_ids=neg_inputs.input_ids,
                negative_prompt_attention_mask=neg_inputs.attention_mask,
            )

        # Decode image tokens
        image = self.model.decode_image_tokens(
            out.sequences[:, inputs.input_ids.shape[1]:],
            height=HEIGHT,
            width=WIDTH
        )
        images = self.processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image")

        # Save generated images
        output_images = []
        for i, img in enumerate(images['pixel_values']):
            safe_filename = f"{task}_{doc_id}_{i}.png"
            image_path = os.path.join(self.output_image_dir, safe_filename)
            img.save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved generated image: {image_path}")

        # Return empty text and image paths
        return "", output_images

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Calculate log-likelihood for requests"""
        eval_logger.warning("loglikelihood not implemented for Emu3, returning dummy values")
        return [(0.0, False) for _ in requests]

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Generate responses for multi-round conversations"""
        eval_logger.warning("generate_until_multi_round not fully implemented for Emu3, using generate_until")
        return self.generate_until(requests)
