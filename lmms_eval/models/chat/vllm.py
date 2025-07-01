import asyncio
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.vllm import VLLM as VLLMSimple
from lmms_eval.protocol import ChatMessages

NUM_SECONDS_TO_SLEEP = 5

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None


@register_model("vllm_chat")
class VLLM(VLLMSimple):
    is_simple = False

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            batched_messages = []
            for idx in range(len(batch_requests)):
                ctx, doc_to_messages, gen_kwargs, doc_id, task, split = batch_requests[idx].arguments
                chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
                chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95

                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                }
                sampling_params = SamplingParams(**params)
                messages = chat_messages.to_openai_messages()

                batched_messages.append(messages)

            sampling_params = SamplingParams(**params)

            start_time = time.time()
            if self.chat_template is not None:
                with open(self.chat_template, "r") as f:
                    chat_template = f.read()
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=chat_template)
            else:
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages)
            end_time = time.time()

            response_text = [o.outputs[0].text for o in response]

            # Calculate timing metrics for batch
            e2e_latency = end_time - start_time
            total_tokens = 0

            for output_idx, output in enumerate(response):
                if hasattr(output, "metrics") and hasattr(output.metrics, "time_to_first_token"):
                    ttft = output.metrics.time_to_first_token
                else:
                    # Estimate TTFT as a fraction of total time
                    ttft = e2e_latency * 0.1 / len(response)

                output_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], "token_ids") else len(output.outputs[0].text.split())
                total_tokens += output_tokens

                if output_tokens > 1:
                    tpot = (e2e_latency / len(response) - ttft) / (output_tokens - 1)
                    inference_speed = 1 / tpot if tpot > 0 else 0
                else:
                    tpot = e2e_latency / len(response)
                    inference_speed = 0

                eval_logger.info(f"Output {output_idx} - E2E: {e2e_latency/len(response):.3f}s, TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s, Speed: {inference_speed:.1f} tokens/s, Output tokens: {output_tokens}")

            if len(response) > 1:
                avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
                eval_logger.info(f"Batch summary - Total time: {e2e_latency:.3f}s, Total tokens: {total_tokens}, Avg speed: {avg_speed:.1f} tokens/s")

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
