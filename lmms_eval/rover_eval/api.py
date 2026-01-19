"""
Azure OpenAI API Client for ROVER Evaluation

Uses Azure AD (chained) credentials to call GPT-4o for evaluation.
"""

import os
import base64
import time
import logging
from io import BytesIO
from typing import List, Optional, Tuple, Union

from PIL import Image
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI


# Global client cache
_CLIENT = None
_DEPLOYMENT = None


def get_gpt4o_client() -> Tuple[AzureOpenAI, str]:
    """
    Get or create Azure OpenAI client for GPT-4o.

    Returns:
        Tuple of (client, deployment_name)
    """
    global _CLIENT, _DEPLOYMENT

    if _CLIENT is None:
        scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
        api_version = os.getenv("TRAPI_API_VERSION", "2024-10-21")
        _DEPLOYMENT = os.getenv("TRAPI_DEPLOYMENT", "gpt-4o_2024-11-20")
        instance = os.getenv("TRAPI_INSTANCE", "gcr/shared")
        endpoint = f"https://trapi.research.microsoft.com/{instance}"

        chained = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        credential_provider = get_bearer_token_provider(chained, scope)

        _CLIENT = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential_provider,
            api_version=api_version,
        )

    return _CLIENT, _DEPLOYMENT


def encode_image_to_base64(image_input: Union[str, Image.Image]) -> Optional[str]:
    """
    Encode image to base64 string.

    Args:
        image_input: Can be a file path (string) or PIL Image object

    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        if isinstance(image_input, str):
            # It's a file path
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            # It's a PIL Image object
            buffer = BytesIO()
            # Convert to RGB if necessary (handles RGBA, L, P, etc.)
            if image_input.mode != "RGB":
                image_input = image_input.convert("RGB")
            image_input.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image {image_input}: {e}")
        return None


def call_gpt4o_with_images(
    prompt: str,
    images: List[Union[str, Image.Image]] = None,
    max_tokens: int = 3000,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    """
    Call GPT-4o API with text and optional images.

    Args:
        prompt: Text prompt
        images: List of images (file paths or PIL Image objects)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Number of retry attempts

    Returns:
        Response content string
    """
    client, deployment = get_gpt4o_client()

    # Build message content
    content = [{"type": "text", "text": prompt}]

    if images:
        for img in images:
            img_b64 = encode_image_to_base64(img)
            if img_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })

    messages = [{"role": "user", "content": content}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content
            if content and content.strip():
                time.sleep(1)  # Rate limiting
                return content.strip()
            else:
                logging.warning(f"Attempt {attempt + 1}: Empty response, retrying...")

            time.sleep(2 ** min(attempt, 3))  # Exponential backoff

        except Exception as e:
            logging.warning(f"GPT-4o evaluation attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** min(attempt, 3))

    logging.error(f"GPT-4o evaluation failed after {max_retries} attempts.")
    return ""
