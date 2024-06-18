from io import BytesIO
from copy import deepcopy
import os
import base64
import json
from typing import List, Tuple, Union
from tqdm import tqdm
import requests as url_requests
import time
import logging

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

from accelerate import Accelerator, DistributedType

from PIL import Image

NUM_SECONDS_TO_SLEEP = 5
eval_logger = logging.getLogger("lmms-eval")

try:
    import anthropic
    from decord import VideoReader, cpu
    import numpy as np
except Exception as e:
    eval_logger.error(f"Error importing claude: {e}")

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
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.image_token = image_token
        self.system_prompt = system_prompt
        self.modality = modality

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
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

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
        client = anthropic.Anthropic()

        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        empty_image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
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
            ###################### CONTINUAL MODE ######################
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
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
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            for attempt in range(5):
                try:
                    message = client.messages.create(model=self.model_version, max_tokens=gen_kwargs["max_new_tokens"], system=self.system_prompt, temperature=gen_kwargs["temperature"], top_p=gen_kwargs["top_p"], messages=messages)
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        res.append("")
                        pbar.update(1)
                        continue

            res.append(message.content[0].text)
            pbar.update(1)

            ###################### CONTINUAL MODE ######################
            if self.continual_mode is True:  # Cache the response
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not supported for claude"
