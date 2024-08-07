from accelerate import Accelerator, DistributedType
import base64
from io import BytesIO
from copy import deepcopy
from decord import VideoReader, cpu
import numpy as np
from openai import OpenAI
from PIL import Image
import os
import json
from typing import List, Tuple
from tqdm import tqdm
import time

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from loguru import logger as eval_logger

NUM_SECONDS_TO_SLEEP = 5


@register_model("srt_api")
class SRT_API(lmms):
    def __init__(
        self,
        api_key: str = "EMPTY",
        model_version: str = "default",
        modality: str = "video",
        host: str = "127.0.0.1",
        port: int = 30000,
        max_frames_num: int = 32,
        timeout: int = 60,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
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
        self.client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30000/v1")
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
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frame_num = len(vr)
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frames = vr.get_batch(frame_idx).asnumpy()
        except:
            import av

            container = av.open(video_path)

            frames = []
            # https://github.com/PyAV-Org/PyAV/issues/1269
            # https://www.cnblogs.com/beyond-tester/p/17641872.html
            # context = CodecContext.create("libvpx-vp9", "r")
            for packet in container.demux(video=0):
                for frame in packet.decode():
                    frames.append(frame)
            total_frames = len(frames)
            sampled_frm = min(total_frames, for_get_frames_num)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = [frames[i] for i in indices]

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
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
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
            imgs = []  # multiple images or frames for video
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    try:
                        frames = self.encode_video(visual, self.max_frames_num)
                        imgs.extend(frames)
                    except Exception as e:
                        eval_logger.error(f"Exception : {e} \n When loading video {visual}")
                        imgs = None
                        break

            # Handling video decode error
            # If we can't even load using pyav, then we will skip
            if imgs is None:
                resps = ""
                res.append(resps)
                pbar.update(1)
                continue

            messages = []

            # put the images in the first place
            content = []
            for img in imgs:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

            content.append({"type": "text", "text": contexts})
            messages.append({"role": "user", "content": content})
            # if self.image_token not in contexts:  # single image format
            #     content = []
            #     for img in imgs:
            #         content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

            #     content.append({"type": "text", "text": contexts})
            #     messages.append({"role": "user", "content": content})
            # else:  # interleaved format
            #     contexts = contexts.split(self.image_token)
            #     for idx, img in enumerate(imgs):
            #         content = [
            #             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            #             {"type": "text", "text": contexts[idx]},
            #         ]
            #         messages.append({"role": "user", "content": content})
            #     messages.append({"role": "user", "content": [{"type": "text", "text": contexts[-1]}]})

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
        assert False, "GPT4V not support"
