import os

import torch
import torch._dynamo

# 1. Suppress compile-related errors completely
torch._dynamo.config.suppress_errors = True

# 2. Disable torch.compile globally
torch._dynamo.config.disable = True

# 3. Disable tokenizers parallel warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 4. Fix flex attention CUDA compile error (bypass dynamic link)
os.environ["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CFLAGS"] = "-L/usr/lib/x86_64-linux-gnu"

from transformers import AutoProcessor, Llama4ForConditionalGeneration

local_path = "/mnt/sfs-common/krhu/.cache/huggingface/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/4bd10c4dc905b4000d76640d07a552344146faec"

processor = AutoProcessor.from_pretrained(local_path)
model = Llama4ForConditionalGeneration.from_pretrained(
    local_path,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


from io import BytesIO

import requests

# Load images from URL (optional: download them beforehand)
from PIL import Image

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

image1 = Image.open(BytesIO(requests.get(url1).content)).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image1},
            {"type": "text", "text": "Can you describe the image?"},
        ],
    },
]

# DO NOT tokenize here — just get the prompt
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# Process multimodal input with images
inputs = processor(
    images=image1,
    text=prompt,
    return_tensors="pt",
).to(model.device)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=16)
print("output")
print(output)
# Decode response
response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])[0]
print(response)
