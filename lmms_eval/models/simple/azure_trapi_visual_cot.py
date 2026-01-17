#!/usr/bin/env python3
"""
Azure OpenAI Visual Chain-of-Thought Model

Two-stage inference using Azure OpenAI:
1. Stage 1: Generate visualization image using gpt-image-1 (DALL-E)
2. Stage 2: Answer question using gpt-4o with original + generated images

Environment:
  AZURE_OPENAI_ENDPOINT    e.g., "https://mcg-vision-flow-oai-eus2.openai.azure.com/"
  AZURE_OPENAI_API_KEY     Your API key (or use Azure CLI credential)
  AZURE_CHAT_DEPLOYMENT    e.g., "gpt-4o" (default) - for Stage 2
  AZURE_IMAGE_DEPLOYMENT   e.g., "gpt-image-1" (default) - for Stage 1

Usage:
    python -m lmms_eval \
        --model azure_trapi_visual_cot \
        --tasks illusionbench_arshia_icon_visual_cot \
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

import requests
from azure.identity import AzureCliCredential, get_bearer_token_provider
from loguru import logger as eval_logger
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5


def build_client():
    """Build Azure OpenAI client"""
    endpoint = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://mcg-vision-flow-oai-eus2.openai.azure.com/",
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Try API key first, fall back to Azure CLI credential
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    else:
        # Use Azure CLI credential
        scope = os.getenv(
            "AZURE_OPENAI_SCOPE", "https://cognitiveservices.azure.com/.default"
        )
        token_provider = get_bearer_token_provider(AzureCliCredential(), scope)
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
    return client


@register_model("azure_trapi_visual_cot")
class AzureTRAPIVisualCoT(lmms):
    """
    Azure OpenAI Visual Chain-of-Thought Model

    Two-stage inference:
    1. Generate auxiliary image using gpt-image-1
    2. Answer question using gpt-4o with both images
    """

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        # Deployment names
        chat_deployment: str = None,
        image_deployment: str = None,
        # Stage 1 parameters
        stage1_image_size: str = "1024x1024",
        stage1_quality: str = "auto",
        # Stage 2 parameters
        stage2_max_tokens: int = 512,
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

        # Deployment names
        self.chat_deployment = chat_deployment or os.getenv(
            "AZURE_CHAT_DEPLOYMENT", "gpt-4o"
        )
        self.image_deployment = image_deployment or os.getenv(
            "AZURE_IMAGE_DEPLOYMENT", "gpt-image-1"
        )

        # Stage 1 parameters
        self.stage1_image_size = stage1_image_size
        self.stage1_quality = stage1_quality

        # Stage 2 parameters
        self.stage2_max_tokens = stage2_max_tokens

        # Output directories
        self.output_dir = output_dir or "./logs/azure_trapi_visual_cot"
        self.save_intermediate = save_intermediate
        eval_logger.info(f"save_intermediate: {save_intermediate}, output_dir: {self.output_dir}")
        if save_intermediate:
            # Structure: {output_dir}/{task_name}/
            # No need to add model name again since output_dir already contains it
            self.intermediate_dir = self.output_dir
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved under: {self.intermediate_dir}")

        # Build client
        eval_logger.info("Building Azure OpenAI client...")
        self.client = build_client()
        eval_logger.info(f"Chat deployment: {self.chat_deployment}")
        eval_logger.info(f"Image deployment: {self.image_deployment}")

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

    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        # Convert to RGB if needed (JPEG only supports RGB and L modes)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG")
        byte_data = output_buffer.getvalue()
        return base64.b64encode(byte_data).decode("utf-8")

    def _stage1_generate_image(
        self, prompt: str, original_image: Optional[Image.Image], doc_id: str, task: str
    ) -> Optional[Image.Image]:
        """
        Stage 1: Generate auxiliary image using gpt-image-1

        Args:
            prompt: Generation prompt
            original_image: Original image (for context, if model supports it)
            doc_id: Document ID for naming
            task: Task name

        Returns:
            Generated PIL Image or None if failed
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")

        for attempt in range(self.max_retries):
            try:
                # Call image generation API
                response = self.client.images.generate(
                    model=self.image_deployment,
                    prompt=prompt,
                    size=self.stage1_image_size,
                    quality=self.stage1_quality,
                    n=1,
                )

                # Get image - try URL first, then base64
                image_data = response.data[0]
                if image_data.url:
                    image_response = requests.get(image_data.url, timeout=60)
                    image_response.raise_for_status()
                    generated_image = Image.open(BytesIO(image_response.content)).convert("RGB")
                elif image_data.b64_json:
                    image_bytes = base64.b64decode(image_data.b64_json)
                    generated_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                else:
                    raise ValueError("No image URL or base64 data in response")

                # Save if enabled
                if self.save_intermediate:
                    # Create task-specific directory
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
        Stage 2: Answer question using gpt-4o with both images

        Args:
            question: Question text
            original_image: Original image
            auxiliary_image: Generated auxiliary image
            doc_id: Document ID

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering for doc {doc_id}")

        messages = [{"role": "user", "content": []}]

        # Add original image first
        if original_image is not None:
            base64_img = self.encode_image(original_image)
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                }
            )

        # Add auxiliary image
        if auxiliary_image is not None:
            base64_aux = self.encode_image(auxiliary_image)
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_aux}"},
                }
            )

        # Add question text
        messages[0]["content"].append({"type": "text", "text": question})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_deployment,
                    messages=messages,
                    max_tokens=self.stage2_max_tokens,
                )
                return response.choices[0].message.content
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
            desc="Azure OpenAI Visual CoT",
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
                generation_prompt = f"Generate a visualization to help answer: {contexts[:200]}"

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
            "Azure TRAPI Visual CoT does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round dialogue not implemented")
