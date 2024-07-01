from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Tuple, Union
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils
from typing import List, Optional, Union, Tuple
from tqdm import tqdm
import torch


@register_model("hfmodel")
class HFModel(lmms):
    def __init__(
        self,
        pretrained: str = "/import/snvm-sc-podscratch4/jonathanl/generic_checkpoints/llama_3/Meta-Llama-3-8B-Instruct",
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:     
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, 
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2",
                                                          device_map="auto"
                                                          )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, max_length=1024)
        self.batch_size = int(batch_size)
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError
    
    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []
        
        for i in range(0, len(requests), self.batch_size):
            all_prompts = [reg.args[0] for reg in requests[i:i+self.batch_size]]
            all_convos = [{'role': 'user', 'content': prompt} for prompt in all_prompts]
            batch_outputs = self.pipeline(all_convos)
            res.append(batch_outputs[-1]['content'])
        return res
    