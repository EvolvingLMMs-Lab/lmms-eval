# Standard library imports
from copy import deepcopy
from io import BytesIO
import base64

import os
import time
import json

# Related third-party imports
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
import numpy as np
from PIL import Image
import requests as url_requests
from tqdm import tqdm
from openai import OpenAI

# Local application/library specific imports
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger

# Conditional imports
try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("Decord is not installed. Video input will not be supported.")

# Constants and global configurations
API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 5

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"


@register_model("batch_gpt4")
class BatchGPT4(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        api_key: str = API_KEY,
        api_url: str = API_URL,
        modality: str = "image",
        max_frames_for_video: int = 10,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_for_video = max_frames_for_video
        self.image_token = "<image>"
        self.timeout = timeout

        self.api_key = api_key
        self.api_url = api_url
        self.client = OpenAI(api_key=api_key)

        accelerator = Accelerator()
        assert accelerator.state.local_process_index == 0, "BatchGPT4 does not support distributed inference."
        assert accelerator.state.num_processes == 1, "BatchGPT4 does not support distributed inference."

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests):
        # Prepare the batch requests data
        requests_data = {}
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Batch Preparing")
        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate([reg.args for reg in requests]):
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    frames = self.encode_video(visual, self.max_frames_for_video)
                    imgs.extend(frames)

            messages = []
            if self.image_token not in contexts:
                messages.append({"role": "user", "content": contexts})
                for img in imgs:
                    messages.append({"role": "user", "content": f"data:image/jpeg;base64,{img}"})
            else:
                contexts_split = contexts.split(self.image_token)
                for idx, context in enumerate(contexts_split):
                    if idx < len(imgs):
                        messages.append({"role": "user", "content": context})
                        messages.append({"role": "user", "content": f"data:image/jpeg;base64,{imgs[idx]}"})
                if len(contexts_split) > len(imgs):
                    messages.append({"role": "user", "content": contexts_split[-1]})

            requests_data[f"request-{idx}"] = {"model": self.model_version, "messages": messages, "max_tokens": gen_kwargs.get("max_new_tokens", 1024)}
            pbar.update(1)

        file_path = os.getenv("HF_HOME", "~/.cache/huggingface") + f"/batchinput_{len(requests_data)}.jsonl"
        file_path = self.create_batch_input_file(requests_data, file_path)
        file_id = self.upload_input_file(file_path)

        batch_response = self.create_batch(file_id, metadata={"description": "Batch Processing for GPT-4"})
        batch_status = self.check_batch_status(batch_response.id)
        while True:
            batch_status = self.check_batch_status(batch_response.id)
            if batch_status.status == "completed":
                eval_logger.info("Batch processing completed.")
                batch_results = self.retrieve_batch_results(batch_status.output_file_id)
                res = [result["response"]["choices"][0]["message"]["content"] for result in json.loads(batch_results)]
                return res
            elif batch_status.status == "failed":
                eval_logger.info("Batch processing failed.")
                res = ["Batch failed"] * len(requests)
                return res
            else:
                eval_logger.info(f"Batch status: {batch_status.status}. Retrying in {NUM_SECONDS_TO_SLEEP} seconds.")
                time.sleep(NUM_SECONDS_TO_SLEEP)

    def loglikelihood(self, requests):
        # TODO
        assert False, "GPT4V not support"

    def create_batch_input_file(self, requests_data, file_path="batchinput.jsonl"):
        with open(file_path, "w") as file:
            for request_id, data in requests_data.items():
                json_record = json.dumps({"custom_id": request_id, "method": "POST", "url": "/v1/chat/completions", "body": data})
                file.write(json_record + "\n")
        return file_path

    def upload_input_file(self, file_path):
        with open(file_path, "rb") as file:
            response = self.client.files.create(file=file, purpose="batch")
        return response.id

    def create_batch(self, file_id, metadata=None):
        if metadata is None:
            metadata = {}
        response = self.client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h", metadata=metadata)
        return response

    def check_batch_status(self, batch_id):
        return self.client.batches.retrieve(batch_id)

    def retrieve_batch_results(self, file_id):
        return self.client.files.content(file_id)

    def cancel_batch(self, batch_id):
        return self.client.batches.cancel(batch_id)

    def list_batches(self, limit=10):
        return self.client.batches.list(limit=limit)
