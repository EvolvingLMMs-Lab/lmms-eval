"""NanoVLM evaluation model for lmms-eval.

NanoVLM (SigLIP2 + MLP projector + Qwen3-0.6B) is a lightweight VLM
trained with lmms-engine. This wrapper supports async multi-GPU inference:
it loads model replicas on N GPUs and dispatches work via a job queue so
workers run independently without synchronization overhead.

Single-GPU fallback is automatic when only one device is visible.
Use ``worker_gpus`` or ``worker_count`` model_args to control GPU selection.
"""

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Register NanoVLM with transformers Auto classes
import lmms_engine.models.nanovlm  # noqa: F401
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages


@dataclass
class _NanoVLMWorker:
    model: AutoModelForImageTextToText
    tokenizer: AutoTokenizer
    image_processor: AutoImageProcessor
    device: torch.device
    image_token_count: int
    image_token_id: int


@register_model("nanovlm")
class NanoVLM(lmms):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "LMMs-Lab-Speedrun/NanoVLM_Init",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        use_cache: bool = False,
        worker_gpus: Optional[str] = None,
        worker_count: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("NanoVLM manages multi-GPU dispatch internally. Please run without accelerate/torchrun multi-process launch.")

        if kwargs:
            eval_logger.warning(f"Ignoring unsupported kwargs for nanovlm: {sorted(kwargs.keys())}")

        self.pretrained = pretrained
        self.system_prompt = system_prompt
        self.use_cache = use_cache
        self._attn_implementation = attn_implementation

        worker_devices = self._resolve_worker_devices(device=device, worker_gpus=worker_gpus, worker_count=worker_count)
        self._workers: List[_NanoVLMWorker] = [self._load_worker(dev) for dev in worker_devices]

        # Public attributes expected by the lmms-eval framework
        self.model = self._workers[0].model
        self.tokenizer = self._workers[0].tokenizer
        self.config = self._workers[0].model.config
        self.device = self._workers[0].device
        self.batch_size = int(batch_size)
        self.max_length = 4096
        self.eot_token_id = self._workers[0].tokenizer.eos_token_id

        eval_logger.info(f"NanoVLM loaded: {len(self._workers)} worker(s) on {worker_devices}, " f"image_token_count={self._workers[0].image_token_count}, use_cache={self.use_cache}")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _resolve_worker_devices(self, device: Optional[str], worker_gpus: Optional[str], worker_count: Optional[int]) -> List[str]:
        if device == "cpu":
            return ["cpu"]
        if worker_gpus:
            selected = [gpu.strip() for gpu in worker_gpus.split(",") if gpu.strip()]
            return [f"cuda:{gpu}" if not gpu.startswith("cuda:") else gpu for gpu in selected]
        if not torch.cuda.is_available():
            return ["cpu"]
        available = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if worker_count is None:
            return available
        return available[: min(worker_count, len(available))]

    def _load_worker(self, device_name: str) -> _NanoVLMWorker:
        model_kwargs: Dict[str, object] = {"torch_dtype": torch.bfloat16, "device_map": device_name}
        if self._attn_implementation:
            model_kwargs["attn_implementation"] = self._attn_implementation

        eval_logger.info(f"Loading NanoVLM worker on {device_name}")
        model = AutoModelForImageTextToText.from_pretrained(self.pretrained, **model_kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        image_processor = AutoImageProcessor.from_pretrained(self.pretrained)

        config = model.config
        image_token_count = getattr(config, "image_token_count", 256)
        image_token_id = getattr(config, "image_token_id", tokenizer.convert_tokens_to_ids("<|image_pad|>"))

        return _NanoVLMWorker(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=torch.device(device_name),
            image_token_count=image_token_count,
            image_token_id=image_token_id,
        )

    # ------------------------------------------------------------------
    # Inference internals
    # ------------------------------------------------------------------

    def _expand_image_tokens(self, input_ids: List[int], image_token_id: int, image_token_count: int) -> List[int]:
        """Expand each single image_token_id to image_token_count copies."""
        expanded = []
        for token_id in input_ids:
            if token_id == image_token_id:
                expanded.extend([image_token_id] * image_token_count)
            else:
                expanded.append(token_id)
        return expanded

    def _process_single(self, worker: _NanoVLMWorker, hf_messages: List[dict], images: List) -> Tuple[torch.Tensor, dict]:
        """Tokenize with chat template, expand image tokens, and process images."""
        token_ids = worker.tokenizer.apply_chat_template(hf_messages, tokenize=True, add_generation_prompt=True)
        token_ids = self._expand_image_tokens(token_ids, worker.image_token_id, worker.image_token_count)
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        image_inputs = {}
        if images:
            pil_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGB"))
                elif isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                else:
                    pil_images.append(img)
            processed = worker.image_processor(images=pil_images, return_tensors="pt")
            for k, v in processed.items():
                image_inputs[k] = v

        return input_ids, image_inputs

    def _run_single_request(self, worker: _NanoVLMWorker, request: Instance) -> Tuple[str, float, int]:
        """Run inference for a single request on a specific worker. Returns (answer, elapsed, n_tokens)."""
        context, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=chat_messages)

        images, videos, audios = chat_messages.extract_media()
        hf_messages = chat_messages.to_hf_messages()

        if not hf_messages or hf_messages[0]["role"] != "system":
            hf_messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})

        input_ids, image_inputs = self._process_single(worker, hf_messages, images)

        input_ids = input_ids.to(worker.device)
        for k, v in image_inputs.items():
            if isinstance(v, torch.Tensor):
                image_inputs[k] = v.to(worker.device)

        max_new_tokens = gen_kwargs.get("max_new_tokens", 16)
        temperature = gen_kwargs.get("temperature", 0)
        do_sample = temperature > 0

        gen_kwargs_call = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": worker.tokenizer.eos_token_id,
            "pad_token_id": worker.tokenizer.pad_token_id or worker.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
        }
        if do_sample:
            gen_kwargs_call["temperature"] = temperature
            gen_kwargs_call["top_p"] = gen_kwargs.get("top_p", 1.0)

        start_time = time.time()
        with torch.inference_mode():
            output_ids = worker.model.generate(input_ids=input_ids, **image_inputs, **gen_kwargs_call)
        elapsed = time.time() - start_time

        generated_ids = output_ids[0][input_ids.shape[1] :]
        answer = worker.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer, elapsed, len(generated_ids)

    # ------------------------------------------------------------------
    # lmms-eval interface (abstract method implementations)
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Required by abc.abstractmethod in base class; NanoVLM is generate-only.
        raise NotImplementedError("NanoVLM does not support loglikelihood scoring")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate answers for all requests using async multi-GPU dispatch.

        Each worker (one per GPU) pulls jobs from a shared queue and runs
        inference independently.  With a single GPU this reduces to standard
        sequential processing.
        """
        results: List[Optional[str]] = [None] * len(requests)
        job_queue: "queue.Queue[Tuple[int, Instance]]" = queue.Queue()
        for idx, request in enumerate(requests):
            job_queue.put((idx, request))

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="NanoVLM Responding")
        lock = threading.Lock()
        errors: List[Exception] = []
        total_elapsed = 0.0
        total_tokens = 0

        def worker_loop(worker: _NanoVLMWorker) -> None:
            nonlocal total_elapsed, total_tokens
            while True:
                if errors:
                    return
                try:
                    idx, request = job_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    answer, elapsed, n_tokens = self._run_single_request(worker, request)
                    with lock:
                        results[idx] = answer
                        total_elapsed += elapsed
                        total_tokens += n_tokens
                        pbar.update(1)
                except Exception as exc:
                    eval_logger.error(f"Worker on {worker.device} failed: {exc}")
                    with lock:
                        errors.append(exc)
                    return

        threads = [threading.Thread(target=worker_loop, args=(w,), daemon=True) for w in self._workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pbar.close()

        if errors:
            raise errors[0]

        if any(r is None for r in results):
            raise RuntimeError(f"NanoVLM completed {sum(1 for r in results if r is not None)} / {len(requests)} requests")

        if total_elapsed > 0:
            eval_logger.info(f"NanoVLM inference: {total_tokens} tokens in {total_elapsed:.1f}s ({total_tokens / total_elapsed:.1f} tok/s)")

        return results

    def generate_until_multi_round(self, requests) -> List[str]:
        # Required by abc.abstractmethod in base class; not needed for current benchmarks.
        raise NotImplementedError("NanoVLM does not support multi-round generation")
