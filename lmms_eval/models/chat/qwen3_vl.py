import time
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

import torch
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    def get_num_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        start = (inputs.input_ids == vision_start_token_id)        # [B, L]
        end   = (inputs.input_ids == vision_end_token_id)          # [B, L]
        level = start.cumsum(dim=1) - end.cumsum(dim=1)
        in_vision_span = level > 0
        in_vision_span = in_vision_span | start | end
        num_vision_tokens = in_vision_span.sum(dim=1)
        num_tokens = inputs.attention_mask.sum(dim=1)
        return num_tokens, num_vision_tokens

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([(idx, reg.args) for idx, reg in enumerate(requests)], _collate, group_fn=lambda x: x[1][2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        chunk_offset = 0
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        vision_processing_latency = 0
        total_num_input_vision_tokens = 0
        total_num_input_tokens = 0
        for chunk in chunks:
            # Vision Info
            start_time = time.time()
            video_metadata_seq = []
            chunk_request_indices, chunk = zip(*chunk)
            chunk_requests = [requests[idx] for idx in chunk_request_indices]
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            if self.video_sampler.will_process_messages:
                chat_messages, video_metadata_seq = zip(*[self.video_sampler.process_messages(chat_message, eval_logger) for chat_message in chat_messages])
                assert len(chunk_requests) == len(video_metadata_seq)
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            if self.resized_height is not None and self.resized_width is not None:
                video_kwargs = {
                    "resized_height": self.resized_height,
                    "resized_width": self.resized_width
                }
            elif self.max_pixels is not None and self.min_pixels is not None:
                video_kwargs = {
                    "max_pixels": self.max_pixels,
                    "min_pixels": self.min_pixels,
                }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            video_kwargs["video_sampler"] = self.video_sampler
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(batched_messages, return_video_metadata=True)
            if video_inputs:
                frames, video_metadata_seq = zip(*video_inputs)
                video_inputs = list(frames)
                video_metadata_seq = list(video_metadata_seq)
                assert len(chunk_requests) == len(video_metadata_seq)
            else:
                video_inputs = None
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, padding_side=padding_side, return_tensors="pt")
            end_time = time.time()
            vision_info_time = end_time - start_time

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()
            generation_latency = end_time - start_time

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Calculate timing metrics for batch
            vision_processing_latency += vision_info_time
            e2e_latency += generation_latency
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)
            num_input_tokens, num_input_vision_tokens = self.get_num_tokens(inputs)
            total_num_input_tokens += num_input_tokens.sum()
            total_num_input_vision_tokens += num_input_vision_tokens.sum()

            for k, (inst, meta) in enumerate(zip(chunk_requests, video_metadata_seq)):
                inst.video_metadata = meta
                inst.num_input_tokens = num_input_tokens[k].cpu().item()
                inst.num_input_vision_tokens = num_input_vision_tokens[k].cpu().item()

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
                "vision_processing_latency": vision_processing_latency,
                "total_num_input_tokens": total_num_input_tokens,
                "total_num_input_vision_tokens": total_num_input_vision_tokens,
                "num_requests": len(requests),
            },
        }


        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, metric_dict)

            if self.rank == 0:
                total_tokens = sum(m["total_tokens"] for m in gathered)
                total_requests = sum(m["additional_metrics"]["num_requests"] for m in gathered)
                total_vision_processing_latency = sum(m["additional_metrics"]["vision_processing_latency"] for m in gathered)
                total_e2e_latency = sum(m["e2e_latency"] for m in gathered)
                total_num_input_tokens = sum(m["additional_metrics"]["total_num_input_tokens"].cpu().item() for m in gathered)
                total_num_input_vision_tokens = sum(m["additional_metrics"]["total_num_input_vision_tokens"].cpu().item() for m in gathered)
                
                throughput = total_tokens / total_e2e_latency if total_e2e_latency > 0 else 0.0
                avg_latency_per_req = total_e2e_latency / total_requests if total_requests else 0.0
                avg_vision_processing_latency = total_vision_processing_latency / total_requests if total_vision_processing_latency > 0 else 0.0
                avg_num_input_tokens = total_num_input_tokens / total_requests if total_num_input_tokens > 0 else 0.0
                avg_num_input_vision_tokens = total_num_input_vision_tokens / total_requests if total_num_input_vision_tokens > 0 else 0.0
                avg_total_tokens = total_tokens / total_requests if total_tokens > 0 else 0.0

                metric_dict = {
                    "total_tokens": total_tokens,
                    "e2e_latency": total_e2e_latency,
                    "avg_speed": throughput,
                    "additional_metrics": {
                        "rank": self.rank,
                        "vision_processing_latency": total_vision_processing_latency,
                        "total_num_input_tokens": total_num_input_tokens,
                        "total_num_input_vision_tokens": total_num_input_vision_tokens,
                        "num_requests": total_requests,
                        "avg_num_output_tokens": avg_total_tokens,
                        "avg_num_input_tokens": avg_num_input_tokens,
                        "avg_num_input_vision_tokens": avg_num_input_vision_tokens,
                        "avg_vision_processing_latency": avg_vision_processing_latency,
                        "avg_e2e_latency": avg_latency_per_req,
                        "per_worker": gathered
                    },
                }
                log_metrics(**metric_dict)
        else:
            metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
                "vision_processing_latency": vision_processing_latency,
                "total_num_input_tokens": total_num_input_tokens,
                "total_num_input_vision_tokens": total_num_input_vision_tokens,
                "num_requests": len(requests),
                "avg_num_output_tokens": total_tokens / len(requests),
                "avg_num_input_tokens": total_num_input_tokens / len(requests),
                "avg_num_input_vision_tokens": total_num_input_vision_tokens / len(requests),
                "avg_vision_processing_latency": vision_processing_latency / len(requests),
                "avg_e2e_latency": e2e_latency / len(requests),
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
