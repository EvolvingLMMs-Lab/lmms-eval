import os

os.environ["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CFLAGS"] = "-L/usr/lib/x86_64-linux-gnu"

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# flex_attention = torch.compile(flex_attention)
attn_mask = torch.ones((4, 1, 2048, 2048), dtype=torch.bool, device="cuda").tril()


def causal(b, h, q_idx, kv_idx):
    h_ = h.new_zeros(h.shape, device="cuda")
    return attn_mask[b][h_][q_idx][kv_idx]


# ✅ FIX: Provide a valid integer for H
block_mask = create_block_mask(causal, B=4, H=1, Q_LEN=2048, KV_LEN=2048)
print(block_mask)

q = torch.randn(4, 1, 2048, 64, device="cuda")
k = torch.randn(4, 1, 2048, 64, device="cuda")
v = torch.randn(4, 1, 2048, 64, device="cuda")

print(flex_attention(q, k, v, block_mask=block_mask))
