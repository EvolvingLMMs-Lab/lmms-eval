import torch
import torch._dynamo
from transformers import AutoProcessor, Llama4ForConditionalGeneration

torch._dynamo.config.suppress_errors = True

local_path = "/mnt/sfs-common/krhu/.cache/huggingface/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/4bd10c4dc905b4000d76640d07a552344146faec"

processor = AutoProcessor.from_pretrained(local_path)
model = Llama4ForConditionalGeneration.from_pretrained(
    local_path,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


from io import BytesIO

import requests

# Load images from URL (optional: download them beforehand)
from PIL import Image

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "text", "text": "Can you describe the image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

print("inputs")
print(inputs)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=16,
)
print("output")
print(outputs)
# Decode response
response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])[0]
print(response)
