"""AWS Bedrock provider for the llm_judge framework.

Supports both standard IAM credentials and bearer token auth.

Environment variables:
    AWS_REGION                  - AWS region (default: us-west-2)
    AWS_BEARER_TOKEN_BEDROCK    - Bearer token for Bedrock auth (optional)
"""

import os
import time
from typing import Dict, List, Optional, Union

from loguru import logger as eval_logger

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class BedrockProvider(ServerInterface):
    """AWS Bedrock implementation of the Judge interface."""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            region = os.getenv("AWS_REGION", "us-west-2")
            bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

            if bearer_token:
                session = boto3.Session()
                self._client = session.client(
                    "bedrock-runtime",
                    region_name=region,
                    config=Config(signature_version="bearer"),
                    aws_access_key_id="unused",
                    aws_secret_access_key="unused",
                    aws_session_token=bearer_token,
                )
            else:
                self._client = boto3.client("bedrock-runtime", region_name=region)
        return self._client

    def is_available(self) -> bool:
        try:
            self.client
            return True
        except Exception:
            return False

    def evaluate(self, request: Request) -> Response:
        config = request.config or self.config
        messages = self.prepare_messages(request)

        bedrock_messages = []
        for m in messages:
            content_blocks = []
            if isinstance(m["content"], str):
                content_blocks.append({"text": m["content"]})
            elif isinstance(m["content"], list):
                for part in m["content"]:
                    if part.get("type") == "text":
                        content_blocks.append({"text": part["text"]})
                    elif part.get("type") == "image_url":
                        # Bedrock expects base64 image in a different format
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            media_type, b64_data = url.split(";base64,", 1)
                            media_type = media_type.replace("data:", "")
                            import base64
                            content_blocks.append({
                                "image": {
                                    "format": media_type.split("/")[-1],
                                    "source": {"bytes": base64.b64decode(b64_data)},
                                }
                            })
            bedrock_messages.append({"role": m["role"], "content": content_blocks})

        inference_config = {
            "maxTokens": config.max_tokens,
            "temperature": config.temperature,
        }
        if config.top_p is not None:
            inference_config["topP"] = config.top_p

        for attempt in range(config.num_retries):
            try:
                response = self.client.converse(
                    modelId=config.model_name,
                    messages=bedrock_messages,
                    inferenceConfig=inference_config,
                )

                content = response["output"]["message"]["content"][0]["text"]
                usage = response.get("usage", {})

                return Response(
                    content=content.strip(),
                    model_used=config.model_name,
                    usage={
                        "prompt_tokens": usage.get("inputTokens", 0),
                        "completion_tokens": usage.get("outputTokens", 0),
                    },
                    raw_response=response,
                )

            except Exception as e:
                eval_logger.warning(f"Bedrock attempt {attempt + 1}/{config.num_retries} failed: {e}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} Bedrock attempts failed")
                    raise
