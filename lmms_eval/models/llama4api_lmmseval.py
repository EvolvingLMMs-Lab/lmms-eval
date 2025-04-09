import base64
import json
import os
import time
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests
import torch
from accelerate import Accelerator, DistributedType
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from loguru import logger as eval_logger
from PIL import Image


@register_model("llama4api_trial")
class Llama4ApiTrial(lmms):
    def __init__(
        self,
        pretrained: str = "meta-llama/llama-4-maverick:free",
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 2,
        max_frames_num: int = 8,
        fps: int = None,
        max_image_size: int = None,
        device_map: str = "auto",  # Added for compatibility but not used
        attn_implementation: str = None,  # Added for compatibility but not used
        site_url: str = "https://lmms-eval.org",
        site_name: str = "LMMS-Eval",
        continual_mode: bool = False,
        response_persistent_folder: str = "./logs/llama4api_persistent_folder",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = pretrained
        self.timeout = timeout
        self.max_retries = max_retries
        self.fps = fps
        self.max_frames_num = max_frames_num
        self.max_image_size = max_image_size
        self.site_url = site_url
        self.site_name = site_name
        self.continual_mode = continual_mode
        self.response_persistent_file = ""

        # Set up response persistence if continual mode is enabled
        if self.continual_mode:
            self.response_persistent_folder = response_persistent_folder
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version.replace('/', '_')}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert not self.continual_mode, "Continual mode is not supported with distributed inference."
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

    def encode_image(self, image: Union[Image.Image, str]):
        """Encode image to base64 string"""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def load_video(self, video_path, max_frames_num):
        """Load video frames using Decord, consistent with llama_vision.py"""
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, request_instances) -> List[str]:
        res = []
        pbar = tqdm(total=len(request_instances), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in request_instances]:
            # Check if we have a cached response for this document
            if self.continual_mode and hasattr(self, "cache_mode") and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if None in visuals:
                visuals = []
                imgs = []
            else:
                visuals = self.flatten(visuals)
                imgs = []  # multiple images or frames for video
                for visual in visuals:
                    if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                        # Use the more consistent load_video method
                        frames = self.load_video(visual, self.max_frames_num)
                        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
                        frames_pil = [to_pil_image(frame) for frame in frames]
                        # Then encode the PIL images for API usage
                        for frame in frames_pil:
                            img_base64 = self.encode_image(frame)
                            imgs.append(img_base64)
                    elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                        img = self.encode_image(visual)
                        imgs.append(img)
                    elif isinstance(visual, Image.Image):
                        img = self.encode_image(visual)
                        imgs.append(img)

            messages = [{"role": "user", "content": []}]
            # Add the text prompt
            messages[0]["content"].append({"type": "text", "text": contexts})
            # Add images to the message
            for img in imgs:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

            # Set generation parameters
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 1.0

            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            }

            data = {
                "model": self.model_version,
                "messages": messages,
                "max_tokens": gen_kwargs["max_new_tokens"],
                "temperature": gen_kwargs["temperature"],
                "top_p": gen_kwargs["top_p"],
            }

            # Make API call with retries
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(url="https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data), timeout=self.timeout)

                    if response.status_code == 200:
                        response_json = response.json()
                        # Log the full response for debugging
                        if self.rank == 0:
                            eval_logger.debug(f"API Response: {json.dumps(response_json, indent=2)}")

                        # Check if 'choices' exists in the response
                        if "choices" not in response_json:
                            eval_logger.error(f"API response missing 'choices' field: {response_json}")
                            if attempt == self.max_retries - 1:
                                response_text = ""
                            else:
                                time.sleep(self.timeout / 2)
                                continue

                        response_text = response_json["choices"][0]["message"]["content"]
                        break
                    else:
                        eval_logger.error(f"API error: {response.status_code}, {response.text}")
                        if attempt == self.max_retries - 1:
                            response_text = ""
                        else:
                            time.sleep(self.timeout / 2)  # Wait half the timeout before retrying

                except Exception as e:
                    error_msg = str(e)
                    eval_logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")

                    # Add more detailed error information
                    if "response" in locals():
                        try:
                            eval_logger.error(f"Response status: {response.status_code}")
                            eval_logger.error(f"Response content: {response.text}")
                        except:
                            pass

                    # On last attempt, log error and set empty response
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        response_text = ""
                    else:
                        time.sleep(self.timeout / 2)

            res.append(response_text)
            pbar.update(1)

            # Cache the response if in continual mode
            if self.continual_mode:
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, request_instances) -> List[str]:
        if self.continual_mode:
            eval_logger.warning("Continual mode not implemented for multi-round generation. Ignoring continual mode.")
        raise NotImplementedError("Multi-round generation not implemented for Llama4ApiTrial")

    def loglikelihood(self, request_instances: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood computation not implemented for Llama4ApiTrial")

    def test_connection(self):
        """Test API connection with a simple request"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }

        data = {
            "model": self.model_version,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "temperature": 0,
        }

        try:
            eval_logger.info(f"Testing connection to OpenRouter API with model {self.model_version}")
            response = requests.post(url="https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data), timeout=self.timeout)

            eval_logger.info(f"Response status: {response.status_code}")
            eval_logger.info(f"Response content: {response.text}")

            if response.status_code == 200:
                eval_logger.info("Connection test successful")
                return True
            else:
                eval_logger.error(f"Connection test failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            eval_logger.error(f"Connection test failed with exception: {str(e)}")
            return False

    @property
    def rank(self):
        return self._rank
