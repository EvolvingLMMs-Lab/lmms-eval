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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,Conversation 
from tqdm import tqdm
from lmms_eval.models.hfmodel import HFModel

@register_model("sambalingohu")
class SambaLingoHU(HFModel):
    def __init__(
        self,
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:    
        super().__init__(pretrained="sambanovasystems/SambaLingo-Hungarian-Chat-70B", batch_size=batch_size)
        