#!/usr/bin/env python3
"""
Google Nano Banana Visual Chain-of-Thought Model for lmms-eval

Two-stage inference using Google's Gemini API or DMXAPI proxy:
1. Stage 1: Generate visualization image using Nano Banana
2. Stage 2: Answer question using Nano Banana with original + generated images

Environment:
  # Option 1: Direct Google API
  GOOGLE_API_KEY           Your Google AI Studio API key

  # Option 2: DMXAPI Proxy (OpenAI-compatible)
  DMXAPI_API_KEY           Your DMXAPI API key
  DMXAPI_BASE_URL          e.g., "https://www.dmxapi.com/v1" (default)

  # Model selection
  NANO_BANANA_MODEL        e.g., "gemini-2.5-flash-image" or "nano-banana-2"

Usage:
    # Using DMXAPI proxy
    export DMXAPI_API_KEY="your-dmxapi-key"
    python -m lmms_eval \
        --model nano_banana_visual_cot \
        --model_args save_intermediate=true \
        --tasks geometry3k_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import base64
import json
import os
import re
import time
from io import BytesIO
from typing import List, Optional, Tuple

import requests as http_requests
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5


def build_client():
    """Build API client - supports both Google direct and DMXAPI proxy"""
    # Check for DMXAPI first (OpenAI-compatible proxy)
    dmxapi_key = os.getenv(
        "DMXAPI_API_KEY", "sk-bdjuIHa6i6QYO8xAepAtiFgYSYthiL58uHKU86QOOfWjZb4J"
    )
    if dmxapi_key:
        from openai import OpenAI

        base_url = os.getenv("DMXAPI_BASE_URL", "https://www.dmxapi.com/v1")
        client = OpenAI(api_key=dmxapi_key, base_url=base_url)
        return client, "dmxapi"

    # Fall back to direct Google API
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not found. Install with: pip install google-genai"
            )
        client = genai.Client(api_key=google_api_key)
        return client, "google"

    raise ValueError(
        "No API key found. Set either DMXAPI_API_KEY or GOOGLE_API_KEY environment variable."
    )


@register_model("nano_banana_visual_cot")
class NanoBananaVisualCoT(lmms):
    """
    Google Nano Banana Visual Chain-of-Thought Model

    Two-stage inference:
    1. Generate auxiliary image using Nano Banana
    2. Answer question using Nano Banana with both images

    Supports both direct Google API and DMXAPI proxy
    """

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        # Model name
        model_name: str = None,
        image_model_name: str = None,
        # Output
        output_dir: str = None,
        save_intermediate: bool = False,
        # Caching
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.timeout = timeout
        self.max_retries = max_retries

        # Build client
        eval_logger.info("Building Nano Banana client...")
        self.client, self.api_type = build_client()

        # Model names - different defaults for different API types
        if self.api_type == "dmxapi":
            self.model_name = model_name or os.getenv(
                "NANO_BANANA_MODEL", "gemini-2.5-flash-image"
            )
            self.image_model_name = image_model_name or os.getenv(
                "NANO_BANANA_IMAGE_MODEL", "nano-banana-2"
            )
        else:
            self.model_name = model_name or os.getenv(
                "NANO_BANANA_MODEL", "gemini-2.5-flash-preview-05-20"
            )
            self.image_model_name = self.model_name  # Google uses same model

        eval_logger.info(f"Using {self.api_type} API")
        eval_logger.info(f"Chat model: {self.model_name}")
        eval_logger.info(f"Image model: {self.image_model_name}")

        # Output directories
        self.output_dir = output_dir or "./logs/nano_banana_visual_cot"
        self.save_intermediate = save_intermediate
        eval_logger.info(
            f"save_intermediate: {save_intermediate}, output_dir: {self.output_dir}"
        )
        if save_intermediate:
            self.intermediate_dir = self.output_dir
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved under: {self.intermediate_dir}"
            )

        # Caching
        self.continual_mode = continual_mode
        if continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = os.path.join(self.output_dir, "cache")
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                response_persistent_folder, "visual_cot_response.json"
            )
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded {len(self.response_cache)} cached responses")
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        self._rank = 0
        self._world_size = 1

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")

    def _stage1_generate_image(
        self,
        prompt: str,
        original_image: Optional[Image.Image],
        doc_id: str,
        task: str,
    ) -> Optional[Image.Image]:
        """
        Stage 1: Generate auxiliary image using Nano Banana

        Args:
            prompt: Generation prompt
            original_image: Original image (for context/editing)
            doc_id: Document ID for naming
            task: Task name

        Returns:
            Generated PIL Image or None if failed
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")

        for attempt in range(self.max_retries):
            try:
                generated_image = None

                if self.api_type == "dmxapi":
                    # DMXAPI uses OpenAI-compatible images.generate endpoint
                    response = self.client.images.generate(
                        model=self.image_model_name,
                        prompt=prompt,
                        size="1024x1024",
                        n=1,
                    )
                    # Get image from response
                    image_data = response.data[0]
                    if hasattr(image_data, "url") and image_data.url:
                        img_response = http_requests.get(image_data.url, timeout=60)
                        img_response.raise_for_status()
                        generated_image = Image.open(
                            BytesIO(img_response.content)
                        ).convert("RGB")
                    elif hasattr(image_data, "b64_json") and image_data.b64_json:
                        image_bytes = base64.b64decode(image_data.b64_json)
                        generated_image = Image.open(BytesIO(image_bytes)).convert(
                            "RGB"
                        )
                else:
                    # Google API - use generate_content with image modality
                    from google.genai import types

                    content = []
                    if original_image is not None:
                        content.append(original_image)
                    content.append(prompt)

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=content,
                        config=types.GenerateContentConfig(
                            response_modalities=["TEXT", "IMAGE"],
                        ),
                    )

                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data is not None:
                            image_bytes = part.inline_data.data
                            generated_image = Image.open(BytesIO(image_bytes)).convert(
                                "RGB"
                            )
                            break

                if generated_image is None:
                    eval_logger.warning(
                        f"No image generated for doc {doc_id}, attempt {attempt + 1}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    continue

                # Save if enabled
                if self.save_intermediate:
                    task_dir = os.path.join(self.intermediate_dir, task)
                    os.makedirs(task_dir, exist_ok=True)
                    save_path = os.path.join(task_dir, f"{doc_id}_stage1.png")
                    generated_image.save(save_path)
                    eval_logger.debug(f"Saved generated image to {save_path}")

                return generated_image

            except Exception as e:
                eval_logger.warning(
                    f"Stage 1 attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP)

        eval_logger.error(f"Stage 1 failed for doc {doc_id}")
        return None

    def _stage2_answer(
        self,
        question: str,
        original_image: Optional[Image.Image],
        auxiliary_image: Optional[Image.Image],
        doc_id: str,
    ) -> str:
        """
        Stage 2: Answer question using Nano Banana with both images

        Args:
            question: Question text
            original_image: Original image
            auxiliary_image: Generated auxiliary image
            doc_id: Document ID

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering for doc {doc_id}")

        for attempt in range(self.max_retries):
            try:
                if self.api_type == "dmxapi":
                    # DMXAPI uses OpenAI-compatible chat.completions endpoint
                    messages = [{"role": "user", "content": []}]

                    # Add original image
                    if original_image is not None:
                        base64_img = self._encode_image_base64(original_image)
                        messages[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_img}"
                                },
                            }
                        )

                    # Add auxiliary image
                    if auxiliary_image is not None:
                        base64_aux = self._encode_image_base64(auxiliary_image)
                        messages[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_aux}"
                                },
                            }
                        )

                    # Add question text
                    messages[0]["content"].append({"type": "text", "text": question})

                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                    )
                    return response.choices[0].message.content
                else:
                    # Google API - use generate_content
                    content = []
                    if original_image is not None:
                        content.append(original_image)
                    if auxiliary_image is not None:
                        content.append(auxiliary_image)
                    content.append(question)

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=content,
                    )
                    response_text = ""
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            response_text += part.text
                    return response_text

            except Exception as e:
                eval_logger.warning(
                    f"Stage 2 attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP)

        eval_logger.error(f"Stage 2 failed for doc {doc_id}")
        return ""

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method implementing two-stage visual CoT"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Nano Banana Visual CoT",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Check cache
            doc_uuid = f"{task}___{split}___{doc_id}"
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    res.append(self.response_cache[doc_uuid])
                    pbar.update(1)
                    continue

            # Get original image
            original_image = None
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if visuals and len(visuals) > 0:
                original_image = visuals[0]

            # Parse prompt to extract generation prompt and question
            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
            )

            if gen_prompt_match and question_match:
                generation_prompt = gen_prompt_match.group(1).strip()
                question = question_match.group(1).strip()
            else:
                # Fallback: use entire context as question, generate default prompt
                question = contexts
                generation_prompt = (
                    f"Generate a visualization to help answer: {contexts[:200]}"
                )

            eval_logger.info(f"Processing doc {doc_id} from task {task}")

            # Stage 1: Generate auxiliary image
            auxiliary_image = self._stage1_generate_image(
                prompt=generation_prompt,
                original_image=original_image,
                doc_id=doc_id,
                task=task,
            )

            # Stage 2: Answer with both images
            answer = self._stage2_answer(
                question=question,
                original_image=original_image,
                auxiliary_image=auxiliary_image,
                doc_id=doc_id,
            )

            res.append(answer)
            pbar.update(1)

            # Save cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = answer
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, indent=2)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Nano Banana Visual CoT does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round dialogue not implemented")
