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

from PIL import Image

NUM_SECONDS_TO_SLEEP = 5
eval_logger = logging.getLogger("lmms-eval")

try:
    import dashscope
except:
    eval_logger.debug("Can not import Dashscope")

API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY")


@register_model("qwen-vl-api")
class Qwen_VL_API(lmms):
    def __init__(
        self,
        model_version: str = "qwen-vl-max",
        image_token: str = "<image>",  # Use to separate interleaved image and text
        system_prompt: str = "",  # Whether you want some special system prompt here
        tmp_folder: str = "./tmp",  # Due to qwen's api restriction,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_version = model_version
        self.image_token = image_token
        self.system_prompt = system_prompt
        self.tmp_folder = tmp_folder

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        os.makedirs(self.tmp_folder, exist_ok=True)

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []

            for idx, visual in enumerate(visuals):
                visual.save(os.path.join(self.tmp_folder, f"tmp_{idx}_{self.rank}_{self.world_size}.jpg"))
                imgs.append(os.path.join(self.tmp_folder, f"tmp_{idx}_{self.rank}_{self.world_size}.jpg"))

            messages = [{"role": "user", "content": []}]

            if self.image_token not in contexts:
                for img in imgs:
                    messages[0]["content"].append({"image": img})
                messages[0]["content"].append({"text": contexts})
            else:
                contexts = contexts.split(self.image_token)

                for idx, img in enumerate(imgs):
                    messages[0]["content"].append({"text": contexts[idx]})
                    messages[0]["content"].append({"image": img})
                messages[0]["content"].append({"text": contexts[-1]})

            if "max_new_tokens" not in gen_kwargs or gen_kwargs["max_new_tokens"] > 1500:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            for attempt in range(5):
                try:
                    response_data = dashscope.MultiModalConversation.call(model=self.model_version, messages=messages, api_key=API_KEY, max_length=gen_kwargs["max_new_tokens"])
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        res.append("")
                        pbar.update(1)
                        continue
            try:
                res.append(response_data["output"]["choices"][0]["message"]["content"][0]["text"].strip())
            except Exception as e:
                eval_logger.error(f"Error {e} happens when parsing input.")
                eval_logger.error(f"{response_data}")
                res.append("")
            pbar.update(1)

        pbar.close()

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not supported for claude"

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
