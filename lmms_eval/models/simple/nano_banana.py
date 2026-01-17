#!/usr/bin/env python3
"""
Google Nano Banana (Gemini 2.5 Flash Image) Model for lmms-eval

Multimodal understanding model using Google's Gemini API or DMXAPI proxy.

Environment:
  # Option 1: Direct Google API
  GOOGLE_API_KEY           Your Google AI Studio API key

  # Option 2: DMXAPI Proxy (OpenAI-compatible)
  DMXAPI_API_KEY           Your DMXAPI API key
  DMXAPI_BASE_URL          e.g., "https://www.dmxapi.com/v1" (default)

  # Model selection
  NANO_BANANA_MODEL        e.g., "gemini-2.5-flash-preview-05-20" or "nano-banana-2"

Usage:
    # Using DMXAPI proxy
    export DMXAPI_API_KEY="your-dmxapi-key"
    python -m lmms_eval \
        --model nano_banana \
        --tasks mme \
        --batch_size 1 \
        --output_path ./logs/
"""

import base64
import json
import os
import time
from io import BytesIO
from typing import List, Tuple

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


@register_model("nano_banana")
class NanoBanana(lmms):
    """
    Google Nano Banana (Gemini 2.5 Flash Image) Model

    Multimodal understanding: image + text -> text
    Supports both direct Google API and DMXAPI proxy
    """

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        model_name: str = None,
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

        # Model name - different defaults for different API types
        if model_name:
            self.model_name = model_name
        elif self.api_type == "dmxapi":
            self.model_name = os.getenv("NANO_BANANA_MODEL", "gemini-2.5-flash-image")
        else:
            self.model_name = os.getenv(
                "NANO_BANANA_MODEL", "gemini-2.5-flash-preview-05-20"
            )

        eval_logger.info(f"Using {self.api_type} API with model: {self.model_name}")

        # Continual mode (response caching)
        self.continual_mode = continual_mode
        if continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/nano_banana_cache"

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                response_persistent_folder, f"{self.model_name}_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded {len(self.response_cache)} cached responses"
                )
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        # Single process for API calls
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
        image.save(output_buffer, format="JPEG")
        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")

    def _call_dmxapi(self, content: list, images: list) -> str:
        """Call DMXAPI (OpenAI-compatible) for chat completion"""
        messages = [{"role": "user", "content": []}]

        # Add images
        for img in images:
            base64_img = self._encode_image_base64(img)
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                }
            )

        # Add text
        for item in content:
            if isinstance(item, str):
                messages[0]["content"].append({"type": "text", "text": item})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content

    def _call_google(self, content: list) -> str:
        """Call Google Gemini API directly"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=content,
        )
        response_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text
        return response_text

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc=f"Nano Banana ({self.model_name})",
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

            # Get images
            images = []
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if visuals:
                for visual in visuals:
                    if isinstance(visual, Image.Image):
                        images.append(visual)

            # Call API with retry
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    if self.api_type == "dmxapi":
                        response_text = self._call_dmxapi([contexts], images)
                    else:
                        # Google API - build content list
                        content = images + [contexts]
                        response_text = self._call_google(content)
                    break
                except Exception as e:
                    eval_logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(f"All retries failed for doc {doc_id}")
                        response_text = ""

            res.append(response_text)
            pbar.update(1)

            # Save cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, indent=2)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for API models"""
        raise NotImplementedError(
            "Nano Banana does not support loglikelihood computation"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for Nano Banana"
        )
