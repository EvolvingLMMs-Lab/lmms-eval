import base64
import json
import os
import time
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import find_dotenv, load_dotenv
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image

load_dotenv(verbose=True)


@register_model("openai_compatible")
class OpenAICompatible(lmms):
    def __init__(
        self,
        model_version: str = "grok-2-latest",
        timeout: int = 10,
        max_retries: int = 100,
        max_size_in_mb: int = 20,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        azure_openai: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb  # some models have a limit on the size of the image
        self.continual_mode = continual_mode
        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        self.client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
            if not azure_openai
            else AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
        )

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
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
        self.batch_size_per_gpu = int(batch_size)

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        max_size = self.max_size_in_mb * 1024 * 1024  # 20MB in bytes
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        # If image is too large, resize it while maintaining aspect ratio
        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

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

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)  # Pre-allocate result list
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        batch_size = self.batch_size_per_gpu
        
        # Filter out requests that can be served from cache
        uncached_requests = []
        uncached_indices = []
        
        for i, req in enumerate(requests):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            
            # Check cache first
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res[i] = response_text
                        pbar.update(1)
                        continue
            
            # If not in cache, add to uncached requests
            uncached_requests.append(req)
            uncached_indices.append(i)
        
        if uncached_requests:
            # Process uncached requests in batches
            batched_requests = [uncached_requests[i:i + batch_size] for i in range(0, len(uncached_requests), batch_size)]
            batched_indices = [uncached_indices[i:i + batch_size] for i in range(0, len(uncached_indices), batch_size)]
            
            for batch_idx, (batch_requests, batch_orig_indices) in enumerate(zip(batched_requests, batched_indices)):
                # Prepare batch payloads
                batch_payloads = []
                request_uuid_map = {}  # Map custom_id to original request info
                
                for i, req in enumerate(batch_requests):
                    contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
                    
                    # Generate a unique ID for this request
                    doc_uuid = f"{task}___{split}___{doc_id}"
                    custom_id = f"batch_{batch_idx}_req_{i}_{doc_uuid}"
                    request_uuid_map[custom_id] = (batch_orig_indices[i], doc_uuid)
                    
                    # Process visuals
                    visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                    if None in visuals:
                        visuals = []
                        imgs = []
                    else:
                        visuals = self.flatten(visuals)
                        imgs = []  # multiple images or frames for video
                        for visual in visuals:
                            if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                                frames = self.encode_video(visual, self.max_frames_num)
                                imgs.extend(frames)
                            elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                                img = self.encode_image(visual)
                                imgs.append(img)
                            elif isinstance(visual, Image.Image):
                                img = self.encode_image(visual)
                                imgs.append(img)
                    
                    # Create message content with text and images
                    message_content = [{"type": "text", "text": contexts}]
                    for img in imgs:
                        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                    
                    # Create the body part of the request
                    body = {
                        "model": self.model_version,
                        "messages": [{"role": "user", "content": message_content}]
                    }
                    
                    # Set generation parameters
                    if "max_new_tokens" not in gen_kwargs:
                        gen_kwargs["max_new_tokens"] = 1024
                    if gen_kwargs["max_new_tokens"] > 4096:
                        gen_kwargs["max_new_tokens"] = 4096
                    if "temperature" not in gen_kwargs:
                        gen_kwargs["temperature"] = 0
                    
                    body["max_tokens"] = gen_kwargs["max_new_tokens"]
                    body["temperature"] = gen_kwargs["temperature"]
                    
                    if "o1" in self.model_version or "o3" in self.model_version:
                        body.pop("temperature", None)
                        body["reasoning_effort"] = "medium"
                        body["response_format"] = {"type": "text"}
                        body.pop("max_tokens", None)
                        body["max_completion_tokens"] = gen_kwargs["max_new_tokens"]
                    
                    # Create the complete batch request object with correct format
                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/chat/completions",
                        "body": body
                    }
                    
                    batch_payloads.append(batch_request)
                
                # Create batch file
                batch_file_path = f"batch_requests_{batch_idx}.jsonl"
                with open(batch_file_path, "w") as f:
                    for payload in batch_payloads:
                        f.write(json.dumps(payload) + "\n")
                
                # Submit batch
                try:
                    with open(batch_file_path, "rb") as f:
                        file_response = self.client.files.create(file=f, purpose="batch")
                    
                    batch_response = self.client.batches.create(
                        input_file_id=file_response.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                    )
                    
                    eval_logger.info(f"Batch job {batch_idx+1}/{len(batched_requests)} created with ID: {batch_response.id}")
                    
                    # Poll for completion
                    retry_count = 0
                    max_retries = self.max_retries
                    while batch_response.status not in ["completed", "failed", "cancelled"]:
                        # Sleep for a while before checking the status again
                        base_sleep_time = min(1 + (len(batch_requests) / 10), 5)  # 1-5 seconds based on batch size
                        sleep_time = min(base_sleep_time * (2 ** retry_count), 60)  # Still cap at 60 seconds max
                        time.sleep(sleep_time)

                        eval_logger.info(f"Batch job status: {batch_response.status}...checking again in a moment")
                        batch_response = self.client.batches.retrieve(batch_response.id)
                        retry_count += 1
                        
                        if retry_count > max_retries:
                            eval_logger.error(f"Exceeded maximum retries for batch {batch_idx}")
                            break
                    
                    if batch_response.status == "completed":
                        eval_logger.info(f"Batch job {batch_idx+1}/{len(batched_requests)} completed successfully")
                        
                        # Get results
                        result_file_id = batch_response.output_file_id
                        file_response = self.client.files.content(result_file_id)
                        result_content = file_response.read().decode("utf-8")
                        
                        results = [json.loads(line) for line in result_content.split("\n") if line.strip() != ""]
                        
                        # Process results
                        for result in results:
                            custom_id = result.get("custom_id")
                            if custom_id in request_uuid_map:
                                orig_idx, doc_uuid = request_uuid_map[custom_id]
                                
                                # Extract response text properly from the result based on the correct format
                                response_text = ""
                                if "response" in result and "choices" in result["response"]:
                                    choices = result["response"]["choices"]
                                    if choices and len(choices) > 0 and "message" in choices:
                                        message = choices[0]["message"]
                                        if "content" in message:
                                            response_text = message["content"]
                                
                                # Store result in original position
                                res[orig_idx] = response_text
                                
                                # Update cache if needed
                                if self.continual_mode is True:
                                    self.response_cache[doc_uuid] = response_text
                                
                                # Update progress bar
                                pbar.update(1)
                        
                        # Clean up
                        try:
                            self.client.files.delete(result_file_id)
                            if os.path.exists(batch_file_path):
                                os.remove(batch_file_path)
                        except Exception as e:
                            eval_logger.warning(f"Error cleaning up batch files: {str(e)}")
                    
                    else:
                        eval_logger.error(f"Batch job {batch_idx+1}/{len(batched_requests)} failed with status: {batch_response.status}")
                        if hasattr(batch_response, "errors"):
                            eval_logger.error(f"Errors: {batch_response.errors}")
                        
                        # Handle failure by setting empty responses
                        for i in batch_orig_indices:
                            res[i] = ""
                            pbar.update(1)
                
                except Exception as e:
                    eval_logger.error(f"Error processing batch {batch_idx+1}/{len(batched_requests)}: {str(e)}")
                    # Handle exception by setting empty responses
                    for i in batch_orig_indices:
                        res[i] = ""
                        pbar.update(1)
        
        # Persist cache if needed
        if self.continual_mode is True and self.response_cache:
            with open(self.response_persistent_file, "w") as f:
                json.dump(self.response_cache, f)
        
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for OpenAI compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for OpenAI compatible models")