import time
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.llava_onevision1_5 import (
    Llava_OneVision1_5 as LlavaOneVisionSimple,
)
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


@register_model("llava_onevision1_5_chat")
class Llava_OneVision1_5(LlavaOneVisionSimple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        assert process_vision_info is not None, "qwen_vl_utils is required. Please install it via `pip install qwen-vl-utils`"

        res = []

        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        e2e_latency = 0.0
        total_tokens = 0

        if self.batch_size > 1:
            self.processor.tokenizer.padding_side = "left"

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            raw_messages_list = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids in doc_id]
            chat_messages_list: List[ChatMessages] = [ChatMessages(**{"messages": m}) for m in raw_messages_list]

            # Prepare video processing kwargs
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
                "max_frames": self.max_num_frames,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames

            # Build HF messages and apply chat template
            hf_messages_list = [cm.to_hf_messages(video_kwargs=video_kwargs) for cm in chat_messages_list]

            texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in hf_messages_list]

            if self.rank == 0 and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{texts[0]}\n")

            # Extract image/video inputs consistent with OneVision processing
            image_inputs, video_inputs = process_vision_info(hf_messages_list)

            # Sample video frames to max_num_frames (simple implementation handles only first element)
            if video_inputs is not None and len(video_inputs) > 0 and isinstance(video_inputs[0], torch.Tensor):
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            # Device placement
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Generation kwargs
            gen_kwargs = dict(all_gen_kwargs[0] or {})
            gen_kwargs.setdefault("max_new_tokens", 128)
            gen_kwargs.setdefault("temperature", 0.0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)

            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            do_sample = bool(gen_kwargs.get("temperature", 0) and gen_kwargs["temperature"] > 0)

            gen_args = {
                **inputs,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "num_beams": gen_kwargs["num_beams"],
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
            }
            if do_sample:
                gen_args.update(
                    do_sample=True,
                    temperature=float(gen_kwargs.get("temperature", 1.0)),
                    top_p=float(gen_kwargs.get("top_p", 1.0)) if gen_kwargs.get("top_p") is not None else None,
                )

            try:
                start_time = time.time()
                with torch.inference_mode():
                    cont = self.model.generate(**gen_args)
                end_time = time.time()
                e2e_latency += end_time - start_time

                # Remove prompt tokens
                cont = cont[:, inputs["input_ids"].shape[-1] :]
                total_tokens += cont.shape[-1] if cont.ndim > 1 else int(cont.shape[-1])
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = torch.zeros((1, 0), dtype=torch.long, device=self.device)

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            for text_output in text_outputs:
                res.append(text_output)
                self.cache_hook.add_partial("generate_until", (texts[0], gen_kwargs), text_output)
            pbar.update(1)

        res = re_ords.get_original(res)

        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": total_tokens / e2e_latency if e2e_latency > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
