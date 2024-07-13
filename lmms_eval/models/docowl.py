from io import BytesIO
from copy import deepcopy
import os
import base64
from typing import List, Tuple, Union
from tqdm import tqdm
import requests as url_requests
import time
import logging
from docowl_infer import DocOwlInfer
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils
from typing import List, Optional, Union, Tuple
import torch
from tqdm import tqdm


@register_model("docowl")
class DocOwl(lmms):
    def __init__(
        self,
        pretrained: str = "Salesforce/instructblip-vicuna-7b",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        model_path = "mPLUG/DocOwl1.5-stage1"
        self.docowl = DocOwlInfer(ckpt_path=model_path, anchors="grid_9", add_global_img=False)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:

        res = []
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([reg.args for reg in requests]):
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            answer = self.docowl.inference(visual, contexts)
            res.append(answer)

        return res
