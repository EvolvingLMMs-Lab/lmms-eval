import asyncio
import base64
import json
import os
import time
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import find_dotenv, load_dotenv
from loguru import logger as eval_logger
from openai import AsyncOpenAI, AzureOpenAI, OpenAI
from PIL import Image

from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.openai_compatible import (
    OpenAICompatible as OpenAICompatibleSimple,
)
from lmms_eval.protocol import ChatMessages

load_dotenv(verbose=True)


@register_model("openai_compatible_chat")
class OpenAICompatible(OpenAICompatibleSimple):
    is_simple = False

    def generate_until(self, requests) -> List[str]:
        res = []
        
        def _collate(x):
            # Simple collation for batching
            return -len(str(x[0])), x[0]

        # Group requests by their generation_kwargs for batching
        from lmms_eval import utils
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        e2e_latency = 0
        total_tokens = 0
        
        for chunk in chunks:
            contexts, all_doc_to_messages, all_gen_kwargs, doc_ids, tasks, splits = zip(*chunk)
            # Assume all gen kwargs in the batch are the same
            gen_kwargs = all_gen_kwargs[0]
            task = tasks[0]
            split = splits[0]
            
            batch_payloads = []
            batch_doc_uuids = []
            batch_responses = []
            
            # Process each request in the batch
            for i, (ctx, doc_to_messages, doc_id) in enumerate(zip(contexts, all_doc_to_messages, doc_ids)):
                doc_uuid = f"{task}___{split}___{doc_id}"
                batch_doc_uuids.append(doc_uuid)
                
                # Check cache first if in continual mode
                if self.continual_mode is True and self.cache_mode == "resume":
                    if doc_uuid in self.response_cache:
                        response_text = self.response_cache[doc_uuid]
                        if response_text:
                            batch_responses.append(response_text)
                            continue

                chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages_raw})

                payload = {"messages": chat_messages.to_openai_messages()}
                payload["model"] = self.model_version

                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if gen_kwargs["max_new_tokens"] > 4096:
                    gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                payload["max_tokens"] = gen_kwargs["max_new_tokens"]
                payload["temperature"] = gen_kwargs["temperature"]

                if "o1" in self.model_version or "o3" in self.model_version or "o4" in self.model_version:
                    del payload["temperature"]
                    payload.pop("max_tokens")
                    payload["reasoning_effort"] = "medium"
                    payload["response_format"] = {"type": "text"}
                    payload["max_completion_tokens"] = gen_kwargs["max_new_tokens"]

                batch_payloads.append(payload)
                batch_responses.append(None)  # Placeholder for response

            # Process all payloads concurrently using asyncio.gather
            async def process_single_request(payload, i):
                if batch_responses[i] is not None:  # Skip cached responses
                    return batch_responses[i], i, 0, 0  # response, index, latency, tokens
                    
                for attempt in range(self.max_retries):
                    try:
                        start_time = time.time()
                        # Convert sync client to async if needed
                        if not isinstance(self.client, AsyncOpenAI):
                            # Use sync client in thread
                            response = self.client.chat.completions.create(**payload)
                        else:
                            response = await self.client.chat.completions.create(**payload)
                        end_time = time.time()

                        response_text = response.choices[0].message.content
                        latency = end_time - start_time
                        
                        # Get token counts from response if available
                        tokens = 0
                        if hasattr(response, "usage"):
                            tokens = response.usage.completion_tokens
                        else:
                            tokens = len(response_text.split())

                        return response_text, i, latency, tokens

                    except Exception as e:
                        error_msg = str(e)
                        eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")

                        if attempt == self.max_retries - 1:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                            return "", i, 0, 0
                        else:
                            await asyncio.sleep(self.timeout)
                
                return "", i, 0, 0  # Fallback

            # Create tasks for all non-cached requests and run them concurrently
            tasks = [
                process_single_request(payload, i)
                for i, payload in enumerate(batch_payloads)
                if batch_responses[i] is None
            ]
            
            if tasks:  # Only if there are requests to process
                async def run_batch():
                    return await asyncio.gather(*tasks)
                
                results = asyncio.run(run_batch())
                for response_text, i, latency, tokens in results:
                    batch_responses[i] = response_text
                    e2e_latency += latency
                    total_tokens += tokens

            # Cache responses if in continual mode
            if self.continual_mode is True:
                for doc_uuid, response_text in zip(batch_doc_uuids, batch_responses):
                    if response_text is not None:
                        self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

            # Add all batch responses to results
            res.extend([r for r in batch_responses if r is not None])
            pbar.update(1)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
