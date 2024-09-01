import json
import os
import re
from datetime import datetime
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("from_log")
class FromLog(lmms):
    def __init__(
        self,
        logs: str = "logs",
        model_name: str = None,
        model_args: str = None,
        have_limits: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.logs = {}

        log_folders = logs.split(",")

        def matched_model(_model_args):
            if model_name and model_name != _model_args["model"]:
                return False

            if model_args:
                _model_args_list = model_args.split(",")

                for _model_arg in _model_args_list:
                    if _model_arg not in _model_args["model_args"]:
                        return False

            if not have_limits and _model_args["limit"] is not None:
                return False

            return True

        for log_folder in log_folders:
            for root, dirs, files in os.walk(log_folder):
                for file in files:
                    if file.endswith(".json"):
                        try:
                            log_file = os.path.join(root, file)

                            with open(log_file, "r") as f:
                                log_data = json.load(f)

                            # check if model is matched
                            _model_args = log_data["args"]
                            if not matched_model(_model_args):
                                raise Exception("Model not matched")

                            # load logs
                            logs = {}
                            for data in log_data["logs"]:
                                id = data["doc_id"]
                                response = data["resps"][0]
                                logs[id] = response

                            task = log_data["model_configs"]["task"]

                            pattern = re.compile(r"\d{4}_\d{4}")

                            if "time" in log_data:
                                log_time = log_data["time"]
                            elif pattern.search(os.path.abspath(log_file)):
                                log_time = pattern.findall(os.path.abspath(log_file))[-1]
                            else:
                                log_time = "unknown"

                            if task not in self.logs or (self.logs[task]["time"] == "unknown" or datetime.strptime(log_time, "%m%d_%H%M") > datetime.strptime(self.logs[task]["time"], "%m%d_%H%M")):
                                self.logs[task] = {"time": log_time, "logs": logs}

                        except Exception as e:
                            pass

        accelerator = Accelerator()
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

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            response = self.logs[task]["logs"][doc_id]
            res.append(response[0])
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "not support"
