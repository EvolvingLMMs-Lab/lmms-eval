import base64
import logging
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Optional, Tuple, Union

import requests as url_requests
import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("paligemma")
class PaliGemma(lmms):
    def __init__(
        self,
        pretrained: str = "Salesforce/instructblip-vicuna-7b",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=dtype,
            device_map=self._device,
            revision="bfloat16",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
            self.accelerator = accelerator

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:

        res = []
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([reg.args for reg in requests]):
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            model_inputs = self.processor(text=contexts, images=visuals[0][0], return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]

            model_inputs["input_ids"] = model_inputs["input_ids"].to("cuda")
            model_inputs["attention_mask"] = model_inputs["attention_mask"].to("cuda")
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to("cuda")

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                res.append(decoded)
        return res
