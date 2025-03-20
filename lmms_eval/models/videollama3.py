import base64
import uuid
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("videollama3")
class VideoLLaMA3(lmms):
    """ 
    VideoLLaMA3 Model
    Video checkpoint from Hugging Face: DAMO-NLP-SG/VideoLLaMA3-7B
    Image checkpoint from Hugging Face: DAMO-NLP-SG/VideoLLaMA3-7B-Image
    
    Example usage:
    
    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model videollama3 \
        --model_args pretrained=DAMO-NLP-SG/VideoLLaMA3-7B \
            --tasks mvbench \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix debug \
            --output_path ./logs/
    
    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model videollama3 \
        --model_args pretrained=DAMO-NLP-SG/VideoLLaMA3-7B-Image \
            --tasks docvqa_test \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix debug \
            --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attention_2: Optional[bool] = True,
        max_num_frames: int = 180,
        use_custom_video_loader=True,  # True for video-mmmu
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=True,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=True,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
            )
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.max_num_frames = max_num_frames
        self.batch_size_per_gpu = int(batch_size)
        self.use_custom_video_loader = use_custom_video_loader

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

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
        raise NotImplementedError("Loglikelihood is not implemented for VideoLLaMA3")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            message = []

            processed_visuals = []
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        if self.use_custom_video_loader:
                            frames, timestamps = read_video_custom(visual)
                            message.append({"role": "user", "content": [{"type": "video", "video": frames, "timestamps": timestamps, "num_frames": len(timestamps)}, {"type": "text", "text": context}]})
                        else:
                            message.append({"role": "user", "content": [{"type": "video", "video": {"video_path": visual, "fps": 1, "max_frames": self.max_num_frames}}, {"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        for v in visual:
                            image_content.append({"type": "image", "image": v})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
            inputs = self.processor(conversation=message, return_tensors="pt", add_generation_prompt=True)

            do_sample = gen_kwargs.get("do_sample", False)
            temperature = gen_kwargs.get("temperature", 0.2 if do_sample else 1.0)
            top_p = gen_kwargs.get("top_p", 0.9 if do_sample else 1.0)
            top_k = gen_kwargs.get("top_k", 20 if do_sample else 50)
            max_new_tokens = gen_kwargs.get("max_new_tokens", 2048)

            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            answers = self.processor.batch_decode(output_ids, skip_special_tokens=True)

            for i, ans in enumerate(answers):
                answers[i] = ans.strip()

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


def read_video_custom(video_path, fps=1, max_frames_num=180, force_include_last_frame=True):
    vr = VideoReader(video_path, ctx=cpu(0))
    duration = len(vr)
    vid_fps = vr.get_avg_fps()
    fps_list = []

    if fps is not None and duration / vid_fps < max_frames_num:
        segment_len = min(vid_fps // fps, duration)
        frame_ids = np.arange(segment_len // 2, duration, segment_len, dtype=int)
        if force_include_last_frame:
            last_frame_id = duration - 1
            if last_frame_id not in frame_ids:
                frame_ids = frame_ids.tolist()
                frame_ids.append(last_frame_id)
    else:
        if duration <= max_frames_num:
            frame_ids = np.arange(duration).astype(int).tolist()
        else:
            frame_ids = np.linspace(0, duration - 1, max_frames_num, dtype=int)
            if force_include_last_frame:
                last_frame_id = duration - 1
                if last_frame_id not in frame_ids:
                    uniform_sampled_frames = np.linspace(0, duration - 1, max_frames_num - 1, dtype=int)
                    frame_ids = uniform_sampled_frames.tolist()
                    frame_ids.append(last_frame_id)

    for frame_id in frame_ids:
        fps_list.append(frame_id / vid_fps)

    frames = vr.get_batch(frame_ids).asnumpy()
    # print(fps_list)
    return frames, fps_list
