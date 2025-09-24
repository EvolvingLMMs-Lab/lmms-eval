import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.vllm import VLLM as VLLMSimple
from lmms_eval.protocol import ChatMessages

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

WORKERS = int(os.getenv("WORKERS", "32"))


@register_model("vllm_chat")
class VLLM(VLLMSimple):
    is_simple = False

    def __init__(
        self,
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.8,
        batch_size=1,
        max_frame_num=32,
        trust_remote_code=True,
        chat_template=None,
        max_pixels: int = 1605632,
        min_image_pixels=28,
        fps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, tensor_parallel_size, data_parallel_size, gpu_memory_utilization, batch_size, max_frame_num, trust_remote_code, chat_template, min_image_pixels, **kwargs)
        self.fps = fps
        self.max_pixels = max_pixels

    def make_one_request(self, request: Instance) -> Tuple[list[dict], dict]:
        """
        Build OpenAI-style messages and per-request sampling params from an Instance.
        Returns (messages, params_dict). Does not mutate input.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        # Copy to avoid side-effects across threads
        _gen = dict(gen_kwargs or {})
        _gen.setdefault("max_new_tokens", 4096)
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)

        params = {
            "temperature": _gen["temperature"],
            "max_tokens": _gen["max_new_tokens"],
            "top_p": _gen["top_p"],
        }

        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.max_frame_num
        messages = chat_messages.to_openai_messages(video_kwargs=video_kwargs)
        return messages, params

    def generate_until(self, requests) -> List[str]:
        res = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        e2e_latency = 0
        for batch_requests in batched_requests:
            batched_messages = []
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(self.make_one_request, request) for request in batch_requests]
                for future in futures:
                    messages, sampling_params = future.result()
                    batched_messages.append(messages)

            sampling_params = SamplingParams(**sampling_params)
            start_time = time.time()
            if self.chat_template is not None:
                with open(self.chat_template, "r") as f:
                    chat_template = f.read()
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=chat_template)
            else:
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages)
            end_time = time.time()

            response_text = [o.outputs[0].text for o in response]
            for req, text in zip(batch_requests, response_text):
                self.add_request_response_to_cache(req, text)

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        if not self.disable_log_stats:
            metrics = self.get_format_metrics()
            total_tokens = metrics["generation_tokens"]
            avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
            metric_dict = {
                "total_tokens": total_tokens,
                "e2e_latency": e2e_latency,
                "avg_speed": avg_speed,
                "additional_metrics": {
                    "ttft": metrics["ttft"],
                    "tpot": metrics["tpot"],
                    "rank": self.rank,
                },
            }
            log_metrics(**metric_dict)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    def get_format_metrics(self):
        metrics = self.client.get_metrics()
        ttft = 0
        tpot = 0
        generation_tokens = 0
        for metric in metrics:
            name = metric.name
            if "time_to_first_token" in name:
                ttft = metric.sum / metric.count
            if "time_per_output_token_seconds" in name:
                tpot = metric.sum / metric.count
            if name == "vllm:generation_tokens":
                generation_tokens = metric.value

        metrics = {
            "ttft": ttft,
            "tpot": tpot,
            "generation_tokens": generation_tokens,
        }

        return metrics
