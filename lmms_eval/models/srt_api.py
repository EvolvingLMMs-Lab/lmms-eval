import asyncio
import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Tuple

import numpy as np
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from openai import AsyncOpenAI, OpenAI
from PIL import Image
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5


@register_model("srt_api")
class SRT_API(lmms):
    def __init__(
        self,
        api_key: str = "sk-123456",
        model_version: str = "lmms-lab/llava-onevision-qwen2-72b-ov",
        modality: str = "video",
        host: str = "127.0.0.1",
        port: int = 30000,
        max_frames_num: int = 32,
        timeout: int = 60,
        chat_template: str = "chatml-llava",
        mem_fraction_static: float = 0.83,
        tp: int = 8,
        chunked_prefill_size: int = 16384,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        num_processes: int = cpu_count() // 2,
        force_sample: bool = False,
        add_time_instruction: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.timeout = timeout
        self.continual_mode = continual_mode
        self.force_sample = force_sample
        self.add_time_instruction = add_time_instruction
        print("force sample:",self.force_sample)
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

        accelerator = Accelerator()
        self.model = model_version
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.chat_template = chat_template
        self.mem_fraction_static = mem_fraction_static
        other_args = []
        other_args.extend(["--chunked-prefill-size", str(chunked_prefill_size)])
        other_args.extend(["--tensor-parallel-size", str(tp)])
        other_args.extend(["--chat-template", self.chat_template])
        other_args.extend(["--mem-fraction-static", str(self.mem_fraction_static)])
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=self.api_key,
            other_args=other_args,
        )
        self.base_url += "/v1"
        if self.modality == "video":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.num_processes = num_processes
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

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        # import pdb; pdb.set_trace()
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > for_get_frames_num or self.force_sample:
            sample_fps = for_get_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in spare_frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames,frame_time,video_time

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    async def generate(self, request):
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        visuals = self.flatten(visuals)
        imgs = []  # multiple images or frames for video
        for visual in visuals:
            if self.modality == "image" or self.modality == "multi-images":
                img = self.encode_image(visual)
                imgs.append(img)
            elif self.modality == "video":
                try:
                    frames,frame_time,video_time = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)
                except Exception as e:
                    eval_logger.error(f"Exception : {e} \n When loading video {visual}")
                    imgs = None
                    break
        
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        if self.add_time_instruction:
            contexts = f"{time_instruciton}\n{contexts}"
        else:
            contexts = f"{contexts}"
        # Handling video decode error
        # If we can't even load using pyav, then we will skip
        if imgs is None:
            resps = ""
            return resps

        messages = []

        # put the images in the first place
        content = []
        for img in imgs:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}, "modalities": self.modality})

        content.append({"type": "text", "text": contexts})
        messages.append({"role": "user", "content": content})

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024

        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0

        for attempt in range(5):
            try:
                response = await self.client.chat.completions.create(model=self.model_version, messages=messages, temperature=gen_kwargs["temperature"], max_tokens=gen_kwargs["max_new_tokens"], timeout=self.timeout)
                response_text = response.choices[0].message.content.strip()
                break  # If successful, break out of the loop

            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.")
                if attempt < 4:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty string
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.")
                    response_text = ""

        return response_text

    def generate_sync(self, request):
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
        if doc_id != 7:
            response = "A"
            return response
        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        visuals = self.flatten(visuals)
        imgs = []  # multiple images or frames for video
        for visual in visuals:
            if self.modality == "image" or self.modality == "multi-images":
                img = self.encode_image(visual)
                imgs.append(img)
            elif self.modality == "video":
                try:
                    frames,frame_time,video_time = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)
                except Exception as e:
                    eval_logger.error(f"Exception : {e} \n When loading video {visual}")
                    imgs = None
                    break


        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        if self.add_time_instruction:
            contexts = f"{time_instruciton}\n{contexts}"
        else:
            contexts = f"{contexts}"
        # Handling video decode error
        # If we can't even load using pyav, then we will skip
        if imgs is None:
            resps = ""
            return resps

        messages = []

        # put the images in the first place
        content = []
        for img in imgs:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}, "modalities": self.modality})

        content.append({"type": "text", "text": contexts})
        messages.append({"role": "user", "content": content})

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024

        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(model=self.model_version, messages=messages, temperature=gen_kwargs["temperature"], max_tokens=gen_kwargs["max_new_tokens"], timeout=self.timeout)
                response_text = response.choices[0].message.content.strip()
                break  # If successful, break out of the loop

            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.")
                if attempt < 4:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty string
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.")
                    response_text = ""

        print("Question:",contexts)
        print("Answer:",response_text)
        import pdb; pdb.set_trace()

        return response_text

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        async def run(requests):
            sem = asyncio.Semaphore(self.num_processes)

            async def _process(request):
                async with sem:
                    return await self.generate(request)

            tasks = [asyncio.create_task(_process(request)) for request in requests]
            for i, task in enumerate(tasks):
                result = await task
                res.append(result)
                pbar.update(1)

        if self.modality == "video":
            for req in requests:
                response = self.generate_sync(req)
                res.append(response)
                pbar.update(1)
        else:
            asyncio.run(run(requests))
        kill_child_process(self.process.pid)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
