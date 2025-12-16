#!/usr/bin/env python3
"""
Azure OpenAI chat completion demo using Azure AD (chained) credentials.

- Attempts Azure CLI login first, then Managed Identity if available
- Targets TRAPI endpoint on research.microsoft.com
- Reads optional environment overrides for flexibility

Environment (optional):
  TRAPI_INSTANCE           e.g., "gcr/shared" (default)
  TRAPI_DEPLOYMENT         e.g., "gpt-4o_2024-11-20" (default)
  TRAPI_API_VERSION        e.g., "2024-10-21" (default)
  TRAPI_SCOPE              e.g., "api://trapi/.default" (default)

Usage:
  python azure_openai_demo.py "What is the capital of France?"
"""

import os
import sys
from typing import List

from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI


def build_client():
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


def chat_once(client: AzureOpenAI, model: str, content: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


def main(argv: List[str]) -> int:
    prompt = argv[1] if len(argv) > 1 else "Give a one word answer, what is the capital of France?"
    client, default_model = build_client()

    # First request with default deployment
    out = chat_once(client, default_model, prompt)
    print(out)

    # Second request with a newer model if available
    second_model = os.getenv("TRAPI_SECOND_MODEL", "gpt-5_2025-08-07")
    try:
        out2 = chat_once(client, second_model, prompt)
        print(out2)
    except Exception as e:
        # Do not hard fail if the second model/deployment is not present
        print(f"[INFO] Second request failed for model '{second_model}': {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

