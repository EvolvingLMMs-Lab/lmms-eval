import torch
import torch._dynamo
from transformers import AutoProcessor, Llama4ForConditionalGeneration

torch._dynamo.config.suppress_errors = True

from torch.nn.attention.flex_attention import BlockMask

# Adding a custom patch for the flex attention block mask adjustment
from transformers.integrations.flex_attention import make_flex_block_causal_mask

# Monkey patch BlockMask to add auto-adjustment
original_init = BlockMask.__init__


def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._auto_adjust = True


BlockMask.__init__ = patched_init

# Patch the flex attention forward function
from transformers.integrations.flex_attention import (
    flex_attention_forward as original_flex_attention_forward,
)


def patched_flex_attention_forward(module, query, key, value, attention_mask, scaling=None, softcap=None, head_mask=None, **kwargs):
    if isinstance(attention_mask, BlockMask):
        # Check and adjust block mask dimensions if needed
        q_len = query.shape[2]
        kv_len = key.shape[2]
        try:
            attention_mask._adjust(q_len, kv_len)
        except:
            print(f"Auto-adjusting block mask from {attention_mask.shape} to ({q_len}, {kv_len})")
            # Create a new block mask with the correct dimensions if adjustment fails
            # This is a simplified approach and might need refinement
            attention_mask = make_flex_block_causal_mask(torch.ones((query.shape[0], q_len), device=query.device), query_length=q_len, key_length=kv_len)

    return original_flex_attention_forward(module, query, key, value, attention_mask, scaling, softcap, head_mask, **kwargs)


# Replace the original function with our patched version
import transformers.integrations.flex_attention

transformers.integrations.flex_attention.flex_attention_forward = patched_flex_attention_forward

local_path = "/mnt/sfs-common/krhu/.cache/huggingface/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/4bd10c4dc905b4000d76640d07a552344146faec"

processor = AutoProcessor.from_pretrained(local_path)
model = Llama4ForConditionalGeneration.from_pretrained(
    local_path,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

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
