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
import logging

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

NUM_SECONDS_TO_SLEEP = 30
eval_logger = logging.getLogger("lmms-eval")

try:
    import reka
    from decord import VideoReader, cpu

    reka.API_KEY = os.getenv("REKA_API_KEY", "YOUR_API_KEY")
except Exception as e:
    eval_logger.error(f"{e}")
    pass


@register_model("reka")
class Reka(lmms):
    def __init__(
        self,
        model_version: str = "reka-edge",
        modality: str = "image",
        max_frames_for_video: int = 10,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.modality = modality
        self.max_frames_for_video = max_frames_for_video
        self.timeout = timeout

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

    def encode_media(self, media_path):
        img = Image.open(media_path)
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")

        return f"data:image/jpeg;base64,{base64_str}"

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]

            conversations_history = []
            media_urls = []
            for visual in visuals:
                if self.modality == "image":
                    media_url = self.encode_media(visual)
                else:
                    raise NotImplementedError

                conversations_history.append({"type": "human", "text": contexts, "media_url": media_url})

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
                    response = reka.chat(
                        conversations_history=conversations_history,
                        model=self.model_version,
                        request_output_len=gen_kwargs["max_new_tokens"],
                        temperature=gen_kwargs["temperature"],
                    )
                    content = response["text"].strip()
                    break  # If successful, break out of the loop

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        eval_logger.error(f"Response: {response}")
                        content = ""

            res.append(content)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Reka not support loglikelihood"
