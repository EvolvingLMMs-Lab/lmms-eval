import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.vllm import VLLM as VLLMSimple
from lmms_eval.protocol import ChatMessages

LLM, _ = optional_import("vllm", "LLM")
SamplingParams, _ = optional_import("vllm", "SamplingParams")

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
        max_frame_num=768,
        trust_remote_code=True,
        chat_template=None,
        max_pixels: int = 1605632,
        min_image_pixels=28,
        fps: Optional[int] = None,
        nframes: Optional[int] = 32,
        max_new_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(
            model,
            tensor_parallel_size,
            data_parallel_size,
            gpu_memory_utilization,
            batch_size,
            max_frame_num,
            trust_remote_code,
            chat_template,
            min_image_pixels,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        self.fps = fps
        self.max_pixels = max_pixels
        self.nframes = nframes

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
        _gen["max_new_tokens"] = self._select_max_new_tokens(_gen.get("max_new_tokens"))
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)
        params = self._build_sampling_params_dict(_gen)

        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": self.max_frame_num,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        messages = chat_messages.to_openai_messages(video_kwargs=video_kwargs)
        return messages, params

    def generate_until(self, requests) -> List[GenerationResult]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        total_elapsed_time = 0
        sample_token_counts: Optional[TokenCounts] = None
        for batch_requests in batched_requests:
            batched_messages = []
            batched_sampling_params = []
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(self.make_one_request, request) for request in batch_requests]
                for future in futures:
                    messages, sampling_params = future.result()
                    batched_messages.append(messages)
                    batched_sampling_params.append(sampling_params)

            start_time = time.time()

            def _run_chat(request_items: list[tuple[list[dict], dict]]) -> list[str]:
                inputs = [messages for messages, _ in request_items]
                sampling_params = [SamplingParams(**params) for _, params in request_items]
                response = self.client.chat(
                    sampling_params=sampling_params,
                    messages=inputs,
                    chat_template=self.chat_template,
                )
                return [o.outputs[0].text for o in response]

            response_text = self._run_tp_synced(list(zip(batched_messages, batched_sampling_params)), _run_chat)
            end_time = time.time()

            # Calculate timing metrics for batch
            total_elapsed_time += end_time - start_time

            assert len(response_text) == len(batch_requests)
            res.extend([GenerationResult(text=resp_text, token_counts=sample_token_counts) for resp_text in response_text])
            pbar.update(len(batch_requests))

        if not self.disable_log_stats:
            metrics = self.get_format_metrics()
            total_tokens = metrics["generation_tokens"]
            avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
            metric_dict = {
                "total_gen_tokens": total_tokens,
                "total_elapsed_time": total_elapsed_time,
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

    def _resolve_sampling_params(self, gen_kwargs: Optional[dict]) -> dict:
        gen = dict(gen_kwargs or {})
        gen.pop("until", None)
        gen["max_new_tokens"] = self._select_max_new_tokens(gen.get("max_new_tokens"))
        gen.setdefault("temperature", 0)
        gen.setdefault("top_p", 0.95)
        return self._build_sampling_params_dict(gen)

    def _to_openai_messages(self, raw_messages: list[dict]) -> list[dict]:
        chat_messages = ChatMessages(messages=raw_messages)
        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": self.max_frame_num,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        return chat_messages.to_openai_messages(video_kwargs=video_kwargs)

    def _chat_once(self, messages: list[dict], params: dict) -> str:
        response = self.client.chat(
            sampling_params=[SamplingParams(**params)],
            messages=[messages],
            chat_template=self.chat_template,
        )
        return response[0].outputs[0].text

    def generate_until_multi_round(self, requests) -> List[List[str]]:
        """Model-driven multi-round loop.

        Each round we ask the task's ``doc_to_messages`` (with ``round_idx`` /
        ``previous_output``) for the messages of that round. The auto
        implementation in ``ConfigurableMessagesTask`` already handles the
        ``doc_to_text`` -> messages translation, so tasks designed against the
        ``generate_until_multi_round`` protocol work without any chat-specific
        plumbing here. Returns ``List[List[str]]`` (per-sample, per-round).
        """
        results: List[List[str]] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Multi-round Responding")

        for request in requests:
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
            doc = self.task_dict[task][split][doc_id]
            params = self._resolve_sampling_params(gen_kwargs)

            round_outputs: List[str] = []
            previous_round_info = None
            round_idx = 0
            while True:
                if round_idx == 0:
                    raw_messages = doc_to_messages(doc)
                else:
                    payload = doc_to_messages(
                        doc,
                        round_idx=round_idx,
                        previous_output=list(round_outputs),
                        previous_round_info=previous_round_info,
                    )
                    if not (isinstance(payload, tuple) and len(payload) == 4):
                        break
                    raw_messages, terminal, _prev_out, previous_round_info = payload
                    if terminal:
                        break

                messages = self._to_openai_messages(raw_messages)
                round_outputs.append(self._chat_once(messages, params))
                round_idx += 1

            results.append(round_outputs)
            pbar.update(1)

        pbar.close()
        return results

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
