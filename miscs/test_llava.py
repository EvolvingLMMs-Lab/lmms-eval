import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"

prompt_1 = "USER: <image>\nWhat does this image show?\nASSISTANT:"
prompt_2 = "USER: <image> <image> \nWhat is the difference between these two images?\nASSISTANT:"
image_file_1 = "image1.png"
image_file_2 = "image2.png"


def test_llava_generation():
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    raw_image_1 = Image.open(image_file_1)
    raw_image_2 = Image.open(image_file_2)
    inputs = processor(
        [prompt_1, prompt_2],
        [raw_image_1, raw_image_1, raw_image_2],
        padding=True,
        return_tensors="pt",
    ).to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    decoded = processor.batch_decode(output, skip_special_tokens=True)
    print(decoded)
    assert len(decoded) == 2
    assert all(isinstance(s, str) for s in decoded)
