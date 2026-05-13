import os
import time
from copy import deepcopy
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.media_encoder import (
    encode_image_to_base64,
    encode_image_to_bytes,
)
from lmms_eval.models.model_utils.usage_metrics import is_budget_exceeded, log_usage

NUM_SECONDS_TO_SLEEP = 5

from loguru import logger

eval_logger = logger

try:
    import anthropic
    import numpy as np
    from decord import VideoReader, cpu
except Exception as e:
    eval_logger.warning(f"Error importing claude: {e}")

API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/complete")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY")


@register_model("claude")
class Claude(lmms):
    def __init__(
        self,
        model_version: str = "claude-3-opus-20240229",
        image_token: str = "<image>",  # Use to separate interleaved image and text
        system_prompt: str = "",  # Whether you want some special system prompt here
        modality: str = "image",
        max_frames_num: int = 10,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.image_token = image_token
        self.system_prompt = system_prompt
        self.modality = modality
        self.max_frames_num = max_frames_num
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
        return encode_image_to_base64(
            image,
            image_format="JPEG",
            convert_rgb=True,
            quality=85,
            copy_if_pil=False,
        )

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        return len(encode_image_to_bytes(image, image_format="PNG"))

    # The max file size is 5MB for claude
    def shrink_image_to_file_size(self, img: Image, max_file_size=4838990) -> Image:
        # Get the current size of the image
        original_size = self.get_image_size(img)

        # If the image size is already smaller than the desired size, return
        if original_size <= max_file_size:
            return img

        # Calculate the ratio to shrink the image
        # Somehow I found out sqrt ratio is not enough to shrink the image
        # below threshold, so I guess we do more
        shrink_ratio = min(0.9, max_file_size / original_size)

        # Resize the image with the calculated ratio
        new_width = int(img.width * shrink_ratio)
        new_height = int(img.height * shrink_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)

        return self.shrink_image_to_file_size(img, max_file_size)

    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            base64_frames.append(
                encode_image_to_base64(
                    img,
                    image_format="JPEG",
                    convert_rgb=True,
                    quality=85,
                    copy_if_pil=False,
                )
            )

        return base64_frames

    def generate_until(self, requests) -> List[GenerationResult]:
        client = anthropic.Anthropic()

        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        empty_image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
            },
        }
        empty_text_block = {"type": "text"}
        empty_messages = [
            {
                "role": "user",
                "content": [],
            }
        ]

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if is_budget_exceeded():
                res.append(GenerationResult(text="", token_counts=None))
                pbar.update(1)
                continue
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []
            for visual in visuals:
                if isinstance(visual, str) and os.path.exists(visual):  # Assuming visual is a path to a video
                    visual = self.encode_video(visual)
                    for img in visual:
                        imgs.append(img)
                else:
                    visual = self.shrink_image_to_file_size(visual)
                    img = self.encode_image(visual)
                    imgs.append(img)

            messages = deepcopy(empty_messages)

            if self.image_token not in contexts:
                for img in imgs:
                    image_block = deepcopy(empty_image_block)
                    image_block["source"]["data"] = img
                    messages[0]["content"].append(image_block)
                text_block = deepcopy(empty_text_block)
                text_block["text"] = contexts
                messages[0]["content"].append(text_block)
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    text_block = deepcopy(empty_text_block)
                    image_block = deepcopy(empty_image_block)
                    text_block["text"] = contexts
                    messages[0]["content"].append(text_block)
                    image_block["source"]["data"] = img
                    messages[0]["content"].append(image_block)

                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the messages
                text_block = deepcopy(empty_text_block)
                text_block["text"] = contexts
                messages["content"].append(text_block)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs or gen_kwargs["top_p"] is None:
                gen_kwargs["top_p"] = 1
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            for attempt in range(5):
                retry_flag = True
                try:
                    message = client.messages.create(model=self.model_version, max_tokens=gen_kwargs["max_new_tokens"], system=self.system_prompt, temperature=gen_kwargs["temperature"], top_p=gen_kwargs["top_p"], messages=messages)
                    retry_flag = False
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        res.append(GenerationResult(text="", token_counts=None))
                        pbar.update(1)
                        continue
                if not retry_flag:
                    break
                eval_logger.info("Retrying...")

            token_counts = None
            if hasattr(message, "usage") and message.usage:
                log_usage(
                    model_name=self.model_version,
                    task_name=task,
                    input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
                    reasoning_tokens=0,
                    source="model",
                )
                token_counts = TokenCounts(
                    input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
                )
            response_text = message.content[0].text
            res.append(GenerationResult(text=response_text, token_counts=token_counts))
            pbar.update(1)

        pbar.close()

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not supported for claude"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Claude")
