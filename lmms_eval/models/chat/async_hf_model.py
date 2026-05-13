import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


@dataclass
class _WorkerResources:
    model: Union[AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText]
    processor: AutoProcessor
    tokenizer: AutoTokenizer
    device: torch.device


@register_model("async_hf_model")
class AsyncHFModel(lmms):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        worker_gpus: Optional[str] = None,
        worker_count: Optional[int] = None,
        max_num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("async_hf_model manages multi-GPU dispatch internally. " "Please run without accelerate/torchrun multi-process launch.")

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        if kwargs:
            eval_logger.warning(f"Ignoring unsupported kwargs for async_hf_model: {sorted(kwargs.keys())}")

        self.pretrained = pretrained
        self.use_cache = use_cache
        self.max_num_frames = max_num_frames
        self.batch_size_per_gpu = int(batch_size)
        self._rank = 0
        self._world_size = 1

        if self.batch_size_per_gpu != 1:
            eval_logger.warning("async_hf_model currently executes one sample per worker at a time. Overriding batch_size to 1.")
            self.batch_size_per_gpu = 1

        worker_devices = self._resolve_worker_devices(device=device, worker_gpus=worker_gpus, worker_count=worker_count)
        self._workers: List[_WorkerResources] = [self._load_worker(device_id, attn_implementation) for device_id in worker_devices]
        self._device = self._workers[0].device
        self._config = self._workers[0].model.config
        self._max_length = 2048

        eval_logger.info(f"Loaded async_hf_model with {len(self._workers)} worker(s) on devices: {worker_devices}")

    def _resolve_worker_devices(
        self,
        device: Optional[str],
        worker_gpus: Optional[str],
        worker_count: Optional[int],
    ) -> List[str]:
        if device == "cpu":
            return ["cpu"]

        if worker_gpus:
            selected = [gpu.strip() for gpu in worker_gpus.split(",") if gpu.strip()]
            if not selected:
                raise ValueError("worker_gpus was provided but no valid gpu ids were found")
            return [f"cuda:{gpu}" if not gpu.startswith("cuda:") else gpu for gpu in selected]

        if not torch.cuda.is_available():
            eval_logger.warning("CUDA is not available. Falling back to CPU worker.")
            return ["cpu"]

        available = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if worker_count is None:
            return available

        if worker_count <= 0:
            raise ValueError(f"worker_count must be > 0, got {worker_count}")

        return available[: min(worker_count, len(available))]

    def _load_worker(self, device_name: str, attn_implementation: Optional[str]) -> _WorkerResources:
        model_kwargs: Dict[str, object] = {
            "torch_dtype": "bfloat16",
            "device_map": device_name,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        config = AutoConfig.from_pretrained(self.pretrained)
        if config.model_type in AutoModelForCausalLM._model_mapping.keys():
            model_cls = AutoModelForCausalLM
        elif config.model_type in AutoModelForImageTextToText._model_mapping.keys():
            model_cls = AutoModelForImageTextToText
        else:
            model_cls = AutoModel

        model = model_cls.from_pretrained(self.pretrained, **model_kwargs).eval()
        processor = AutoProcessor.from_pretrained(self.pretrained)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        return _WorkerResources(model=model, processor=processor, tokenizer=tokenizer, device=torch.device(device_name))

    @property
    def config(self):
        return self._config

    @property
    def eot_token_id(self):
        return self._workers[0].tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for async_hf_model")

    def _run_single_request(
        self,
        worker: _WorkerResources,
        request: Instance,
    ) -> Tuple[GenerationResult, str, Dict[str, object]]:
        context, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=chat_messages)

        batched_messages = [chat_messages.to_hf_messages()]
        text = worker.processor.apply_chat_template(batched_messages[0], tokenize=False, add_generation_prompt=True)

        images, videos, audios = chat_messages.extract_media()
        inputs = worker.processor(
            text=[text],
            images=images,
            videos=videos,
            audios=audios,
            padding=True,
            return_tensors="pt",
        ).to(worker.device)

        default_gen_kwargs: Dict[str, object] = {
            "max_new_tokens": 4096,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
        }
        current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
        if float(current_gen_kwargs["temperature"]) > 0:
            do_sample = True
            temperature = current_gen_kwargs["temperature"]
            top_p = current_gen_kwargs["top_p"]
        else:
            do_sample = False
            temperature = None
            top_p = None

        pad_token_id = worker.tokenizer.pad_token_id
        start_time = time.time()
        with torch.inference_mode():
            outputs = worker.model.generate(
                **inputs,
                eos_token_id=worker.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_beams=int(current_gen_kwargs["num_beams"]),
                max_new_tokens=int(current_gen_kwargs["max_new_tokens"]),
                use_cache=self.use_cache,
            )
        elapsed = time.time() - start_time

        generated_ids = outputs[0][len(inputs.input_ids[0]) :]
        answer = worker.processor.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        token_counts = TokenCounts(
            input_tokens=int(inputs.input_ids.shape[-1]),
            output_tokens=int(generated_ids.shape[-1]),
        )
        generation_result = GenerationResult(text=answer, token_counts=token_counts)
        return generation_result, context, current_gen_kwargs, elapsed

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        results: List[Optional[GenerationResult]] = [None] * len(requests)
        job_queue: "queue.Queue[Tuple[int, Instance]]" = queue.Queue()
        for idx, request in enumerate(requests):
            job_queue.put((idx, request))

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        lock = threading.Lock()
        errors: List[Exception] = []
        elapsed_times: List[float] = []
        output_tokens = 0

        def worker_loop(worker: _WorkerResources) -> None:
            nonlocal output_tokens
            while True:
                if errors:
                    return
                try:
                    idx, request = job_queue.get_nowait()
                except queue.Empty:
                    return

                try:
                    generation_result, context, gen_kwargs, elapsed = self._run_single_request(worker, request)
                    with lock:
                        results[idx] = generation_result
                        elapsed_times.append(elapsed)
                        output_tokens += generation_result.token_counts.output_tokens
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generation_result.text)
                        pbar.update(1)
                except Exception as exc:
                    with lock:
                        errors.append(exc)
                    return

        threads = [threading.Thread(target=worker_loop, args=(worker,), daemon=True) for worker in self._workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        pbar.close()

        if errors:
            raise errors[0]

        finalized_results = [result for result in results if result is not None]
        if len(finalized_results) != len(requests):
            raise RuntimeError(f"async_hf_model completed {len(finalized_results)} / {len(requests)} requests")

        total_elapsed = sum(elapsed_times)
        avg_speed = output_tokens / total_elapsed if total_elapsed > 0 else 0.0
        log_metrics(
            total_gen_tokens=output_tokens,
            total_elapsed_time=total_elapsed,
            avg_speed=avg_speed,
        )
        return finalized_results

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for async_hf_model")
