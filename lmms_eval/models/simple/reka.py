import os
import time
from typing import List, Tuple

import numpy as np
from accelerate import Accelerator, DistributedType
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64
from lmms_eval.models.model_utils.usage_metrics import is_budget_exceeded, log_usage

NUM_SECONDS_TO_SLEEP = 30

from loguru import logger

eval_logger = logger

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")
ChatMessage, _has_reka = optional_import("reka", "ChatMessage")
RekaClient, _ = optional_import("reka.client", "Reka")
if not _has_reka:
    eval_logger.warning("reka is not installed. Please install reka to use this model.")


@register_model("reka")
class Reka(lmms):
    def __init__(
        self,
        model_version: str = "reka-edge",
        modality: str = "image",
        max_frames_num: int = 5,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.timeout = timeout
        self.reka = RekaClient(api_key=os.getenv("REKA_API_KEY", "YOUR_API_KEY"))

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    def encode_image(self, image):
        if isinstance(image, list):
            media_urls = []
            for img in image:
                base64_str = encode_image_to_base64(
                    img,
                    image_format="PNG",
                    convert_rgb=False,
                    quality=None,
                )
                media_urls.append(f"data:image/jpeg;base64,{base64_str}")
            return media_urls

        base64_str = encode_image_to_base64(
            image,
            image_format="PNG",
            convert_rgb=False,
            quality=None,
        )
        return f"data:image/jpeg;base64,{base64_str}"

    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            base64_str = encode_image_to_base64(
                img,
                image_format="PNG",
                convert_rgb=False,
                quality=None,
            )
            base64_frames.append(f"data:image/jpeg;base64,{base64_str}")

        return base64_frames

    def generate_until(self, requests) -> List[GenerationResult]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if is_budget_exceeded():
                res.append(GenerationResult(text="", token_counts=None))
                pbar.update(1)
                continue

            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            message_content = []

            if self.modality == "image":
                media_urls = self.encode_image(visual)
                message_content.append({"type": "text", "text": context})
                for media_url in media_urls:
                    message_content.append({"type": "image_url", "image_url": media_url})
            elif self.modality == "video":
                message_content.append({"type": "text", "text": context})
                assert len(visual) == 1, "Reka only supports one video per request"
                media_urls = self.encode_video(visual[0])
                assert len(media_urls) == self.max_frames_num, f"Reka only supports {self.max_frames_num} frames per request"
                for media_url in media_urls:
                    message_content.append({"type": "image_url", "image_url": media_url})

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            token_counts = None
            for attempt in range(5):
                try:
                    response = self.reka.chat.create(
                        messages=[
                            ChatMessage(
                                role="user",
                                content=message_content,
                            )
                        ],
                        model=self.model_version,
                    )
                    if hasattr(response, "usage") and response.usage:
                        log_usage(
                            model_name=self.model_version,
                            task_name=task,
                            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                            reasoning_tokens=0,
                            source="model",
                        )
                        token_counts = TokenCounts(
                            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                        )
                    response_text = response.responses[0].message.content.strip()
                    break  # If successful, break out of the loop

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        response_text = ""

            res.append(GenerationResult(text=response_text, token_counts=token_counts))
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Reka not support loglikelihood"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
