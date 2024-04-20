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
    import anthropic
except:
    eval_logger.debug("Can not import anthropic")

API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/complete")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY")


@register_model("claude")
class Claude(lmms):
    def __init__(
        self,
        model_version: str = "claude-3-opus-20240229",
        image_token: str = "<image>", # Use to separate interleaved image and text
        system_prompt: str = "", # Whether you want some special system prompt here
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.image_token = image_token
        self.system_prompt = system_prompt
    
    def encode_image(self, image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
    def get_image_size(self, image):
       # Create a BytesIO object to store the image bytes
        img_byte_array = BytesIO()
        
        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")
        
        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()
        
        return img_size

    # The max file size is 5MB for claude
    def shrink_image_to_file_size(self, img : Image, max_file_size=5242880) -> Image:

        # Get the current size of the image
        original_size = self.get_image_size(img)

        # If the image size is already smaller than the desired size, return
        if original_size <= max_file_size:
            return img

        # Calculate the ratio to shrink the image
        # Somehow I found out sqrt ratio is not enough to shrink the image
        # below threshold, so I guess we do more
        shrink_ratio = (max_file_size / original_size)

        # Resize the image with the calculated ratio
        new_width = int(img.width * shrink_ratio)
        new_height = int(img.height * shrink_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)

        return img

    def generate_until(self, requests) -> List[str]:
        client = anthropic.Anthropic()

        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")


        empty_image_block = {"type" : "image", "source" : {"type": "base64", "media_type": "image/png",}}
        empty_text_block = {"type" : "text"}
        empty_messages = [
            {
                "role" : "user",
                "content" : [],
            }
        ]

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []
            for visual in visuals:
                visual = self.shrink_image_to_file_size(visual)
                img = self.encode_image(visual)
                imgs.append(img)

            messages = deepcopy(empty_messages)
            
            if self.image_token not in contexts:
                for img in imgs:
                    image_block = deepcopy(empty_image_block)
                    image_block["source"]["data"] = img
                    messages[0]["content"].append(image_block)
                text_block = deepcopy(empty_text_block)
                text_block["text"] = contexts
                messages[0]["content"].append(text_block)
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    text_block = deepcopy(empty_text_block)
                    image_block = deepcopy(empty_image_block)
                    text_block["text"] = contexts
                    messages[0]["content"].append(text_block)
                    image_block["source"]["data"] = img
                    messages[0]["content"].append(image_block)
                
                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the messages
                text_block = deepcopy(empty_text_block) 
                text_block["text"] = contexts
                messages["content"].append(text_block)

            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            
            for attempt in range(5):
                try:
                    message = client.messages.create(
                        model=self.model_version,
                        max_tokens= gen_kwargs["max_new_tokens"],
                        system=self.system_prompt,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        messages=messages
                    )
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        res.append("")
                        pbar.update(1)
                        continue

            res.append(message.content[0].text)
            pbar.update(1)
        
        pbar.close()

        return res 

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not supported for claude"