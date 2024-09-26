import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")


from datetime import timedelta

from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs


@register_model("xcomposer2d5")
class XComposer2D5(lmms):
    def __init__(
        self,
        pretrained: str = "internlm/internlm-xcomposer2d5-7b",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        tmp_folder: str = "./temp/xcomposer2d5/",
        **kwargs,
    ):
        super().__init__()

        self.tmp_folder = os.path.abspath(tmp_folder)
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
        eval_logger.info(f"Using temporary folder: {self.tmp_folder}")

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL2, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.path = pretrained
        self._model = AutoModel.from_pretrained(self.path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=self.device_map).half().eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self._model.tokenizer = self._tokenizer

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

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
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image = []

            if self.modality == "image":
                for idx, visual in enumerate(visuals):
                    visual.save(os.path.join(self.tmp_folder, f"tmp_{idx}_{self.rank}_{self.world_size}.jpg"))
                    image.append(os.path.join(self.tmp_folder, f"tmp_{idx}_{self.rank}_{self.world_size}.jpg"))
                if image:
                    image_tokens = [f"Image{i} <ImageHere>; " for i in range(len(visuals))]
                    image_tokens = "".join(image_tokens)
                    contexts = image_tokens + contexts
            elif self.modality == "video":
                image = visuals

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    response, his = self.model.chat(self.tokenizer, contexts, image, do_sample=False, num_beams=1, use_meta=True, max_new_tokens=gen_kwargs["max_new_tokens"])
            except Exception as e:
                eval_logger.error(f"Error : {e}")
                response = ""

            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
