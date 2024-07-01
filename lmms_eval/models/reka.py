from PIL import Image
from io import BytesIO
from copy import deepcopy
import numpy as np
import os
import base64
from typing import List, Tuple
from tqdm import tqdm
import requests as url_requests
import time

import json

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType

NUM_SECONDS_TO_SLEEP = 30

from loguru import logger

eval_logger = logger

try:
    from reka.client import Reka as RekaClient
    from reka import ChatMessage
    from decord import VideoReader, cpu
except Exception as e:
    eval_logger.warning(f"Error importing reka: {e}")


@register_model("reka")
class Reka(lmms):
    def __init__(
        self,
        model_version: str = "reka-edge",
        modality: str = "image",
        max_frames_for_video: int = 10,
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,  # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.modality = modality
        self.max_frames_for_video = max_frames_for_video
        self.timeout = timeout
        self.continual_mode = continual_mode
        if self.continual_mode and response_persistent_folder is None:
            raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")
        self.response_persistent_folder = response_persistent_folder
        self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

        if os.path.exists(self.response_persistent_file):
            with open(self.response_persistent_file, "r") as f:
                self.response_cache = json.load(f)
            self.cache_mode = "resume"
        else:
            self.response_cache = {}
            self.cache_mode = "start"

        self.reka = RekaClient(api_key=os.getenv("REKA_API_KEY", "YOUR_API_KEY"))

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
        if type(image) == list:
            media_urls = []
            for img in image:
                output_buffer = BytesIO()
                img.save(output_buffer, format="PNG")
                byte_data = output_buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")
                media_urls.append(f"data:image/jpeg;base64,{base64_str}")
            return media_urls
        else:
            output_buffer = BytesIO()
            image.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")

            return f"data:image/jpeg;base64,{base64_str}"

    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_for_video, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(f"data:image/jpeg;base64,{base64_str}")

        return base64_frames

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
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
                assert len(media_urls) == self.max_frames_for_video, f"Reka only supports {self.max_frames_for_video} frames per request"
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
                    response_text = response.responses[0].message.content.strip()
                    break  # If successful, break out of the loop

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        response_text = ""

            res.append(response_text)
            pbar.update(1)
            if self.continual_mode is True:  # Cache the response
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Reka not support loglikelihood"
