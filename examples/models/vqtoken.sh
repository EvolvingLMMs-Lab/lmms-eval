#!/usr/bin/env bash

# From an lmms-eval checkout, install the legacy video decoder and the public
# VQToken runtime first (the runtime supports Python 3.10 and 3.11):
# uv pip install -e ".[video-legacy]"
# uv pip install "llava[runtime] @ git+https://github.com/Hai-chao-Zhang/VQToken.git@a8e3e13e8415b575556dd779e890b77a74ecf52a"
#
# The paper checkpoint uses Hugging Face's access-request flow. Accept its
# terms and run `hf auth login` before evaluation.
accelerate launch --num_processes=1 -m lmms_eval \
    --model vqtoken \
    --model_args pretrained=haichaozhang/VQ-Token-llava-ov-0.5b,vqtoken_selection_method=fixed,vqtoken_max_clusters=32 \
    --tasks videomme \
    --batch_size 1 \
    --limit 1

# vqtoken_selection_method also accepts elbow or silhouette. Configure
# vqtoken_min_clusters and vqtoken_max_clusters to bound adaptive selection.
