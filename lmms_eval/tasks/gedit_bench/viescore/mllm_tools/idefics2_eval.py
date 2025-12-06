import os
import time
from typing import List

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available


class Idefics2:
    def __init__(self, model_path: str = "HuggingFaceM4/idefics2-8b") -> None:
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, _attn_implementation=attn_implementation).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        messages = [{"role": "user", "content": [{"type": "image"}] * len(image_links) + [{"type": "text", "text": text_prompt}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images = [load_image(image_link) for image_link in image_links]  # Support PIL images as well
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def get_parsed_output(self, inputs):
        generate_ids = self.model.generate(**inputs, max_new_tokens=512, num_beams=1)
        generated_text = self.processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return generated_text


if __name__ == "__main__":
    model = Idefics2()
    prompt = model.prepare_prompt(
        ["https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg", "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"], "What is difference between two images?"
    )
    # print("prompt : \n", prompt)
    res = model.get_parsed_output(prompt)
    print("result : \n", res)
