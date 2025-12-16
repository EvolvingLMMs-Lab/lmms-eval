#!/usr/bin/env python3
"""
Azure OpenAI model with Azure AD authentication for lmms-eval

Directly adapted from api.py - uses same authentication and client setup.

Environment (optional):
  TRAPI_INSTANCE           e.g., "gcr/shared" (default)
  TRAPI_DEPLOYMENT         e.g., "gpt-4o_2024-11-20" (default)
  TRAPI_API_VERSION        e.g., "2024-10-21" (default)
  TRAPI_SCOPE              e.g., "api://trapi/.default" (default)

Usage:
    python -m lmms_eval \
        --model azure_trapi \
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

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from loguru import logger as eval_logger
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5


def build_client():
    """
    Build Azure OpenAI client - EXACTLY as in api.py

    Returns:
        tuple: (client, deployment_name)
    """
    # Config with env overrides
    scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
    api_version = os.getenv("TRAPI_API_VERSION", "2024-10-21")
    deployment_name = os.getenv("TRAPI_DEPLOYMENT", "gpt-4o_2024-11-20")
    instance = os.getenv("TRAPI_INSTANCE", "gcr/shared")
    endpoint = f"https://trapi.research.microsoft.com/{instance}"

    # Prepare chained credential: Azure CLI -> Managed Identity
    chained = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    )
    credential_provider = get_bearer_token_provider(chained, scope)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential_provider,
        api_version=api_version,
    )
    return client, deployment_name


def chat_once(client: AzureOpenAI, model: str, messages: List[dict]) -> str:
    """
    Single chat completion - adapted from api.py

    Args:
        client: AzureOpenAI client
        model: Deployment name
        messages: Chat messages in OpenAI format

    Returns:
        str: Response text
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content


@register_model("azure_trapi")
class AzureTRAPI(lmms):
    """
    Azure OpenAI with Azure AD authentication

    Uses the same build_client() and chat_once() logic as api.py
    """

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.timeout = timeout
        self.max_retries = max_retries

        # Build client using the exact same function as api.py
        eval_logger.info("Building Azure TRAPI client...")
        self.client, self.deployment_name = build_client()
        eval_logger.info(f"Connected to deployment: {self.deployment_name}")

        # Continual mode (response caching)
        self.continual_mode = continual_mode
        if continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/azure_trapi_cache"

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                response_persistent_folder, f"{self.deployment_name}_response.json"
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

    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc=f"Azure TRAPI ({self.deployment_name})",
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

            # Prepare messages in OpenAI format
            messages = [{"role": "user", "content": []}]

            # Add images if available
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if visuals:
                for visual in visuals:
                    if isinstance(visual, Image.Image):
                        base64_image = self.encode_image(visual)
                        messages[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        )

            # Add text
            messages[0]["content"].append({"type": "text", "text": contexts})

            # Call API with retry - using chat_once from api.py
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    response_text = chat_once(
                        self.client, self.deployment_name, messages
                    )
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
            "Azure TRAPI does not support loglikelihood computation"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for Azure TRAPI"
        )
