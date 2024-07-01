from io import BytesIO
from copy import deepcopy
import os
import base64
from typing import List, Tuple, Union
from tqdm import tqdm
import requests as url_requests
import time
import logging

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils
from typing import List, Optional, Union, Tuple
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from tqdm import tqdm

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
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(pretrained,
                                                                        torch_dtype=dtype,
                                                                        device_map="auto",
                                                                        revision="bfloat16",
                                                                        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)

                
        
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError
    
    def generate_until(self, requests: list[Instance]) -> list[str]:
        
        res = []
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([reg.args for reg in requests]):
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            model_inputs = self.processor(text=contexts, images=visuals[0][0], return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]
            
            model_inputs["input_ids"] = model_inputs["input_ids"].to('cuda')
            model_inputs["attention_mask"] = model_inputs["attention_mask"].to('cuda')
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to('cuda')
            
            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                res.append(decoded)
        return res