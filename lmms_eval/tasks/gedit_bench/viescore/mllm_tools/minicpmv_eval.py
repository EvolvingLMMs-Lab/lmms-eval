import os
import time
from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


class MiniCPMV:
    def __init__(self) -> None:
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        self.model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", _attn_implementation=attn_implementation).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)

        print(f"Using {attn_implementation} for attention implementation")

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        messages = [{"role": "user", "content": [{"type": "image"}] * len(image_links) + [{"type": "text", "text": text_prompt}]}]
        return messages

    def get_parsed_output(self, inputs):
        res = self.model.chat(
            image=None,
            msgs=inputs,
            tokenizer=self.tokenizer,
            sampling=False,  # if sampling=False, beam_search will be used by default
        )
        return res


if __name__ == "__main__":
    model = MiniCPMV()
    prompt = model.prepare_prompt(
        ["https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg", "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"], "What is difference between two images?"
    )
    # print("prompt : \n", prompt)
    res = model.get_parsed_output(prompt)
    print("result : \n", res)
