import logging
import json

from typing import List, Tuple
from tqdm import tqdm
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from accelerate import Accelerator, DistributedType

eval_logger = logging.getLogger("lmms-eval")


@register_model("from_log")
class FromLog(lmms):
    def __init__(
        self,
        log_file = "",
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.logs = {}

        with open(log_file, "r") as f:
            log_data = json.load(f)
            
        for data in log_data["logs"]:
            id = data["doc_id"]
            response = data["resps"][0]
            self.logs[id] = response

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
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

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            response = self.logs[doc_id]
            res.append(response[0])
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"
