#!/usr/bin/env bash

# From an lmms-eval checkout, install the legacy video decoder and the public
# VQToken runtime first (the runtime supports Python 3.10 and 3.11):
# uv pip install -e ".[video-legacy]"
# uv pip install "llava[runtime] @ git+https://github.com/Hai-chao-Zhang/VQToken.git@0314eb9989a7ea843f31bfe0984113529e3f9140"
#
# The paper checkpoint uses Hugging Face's access-request flow. Accept its
# terms and run `hf auth login` before evaluation. This adapter selects the
# checkpoint's released learned VQ-Attention path, not the centroid ablation.
accelerate launch --num_processes=1 -m lmms_eval \
    --model vqtoken \
    --model_args pretrained=haichaozhang/VQ-Token-llava-ov-0.5b,vqtoken_selection_method=fixed,vqtoken_max_clusters=32 \
    --tasks videomme \
    --batch_size 1 \
    --limit 1

# vqtoken_selection_method also accepts elbow or silhouette. Configure
# vqtoken_min_clusters and vqtoken_max_clusters to bound adaptive selection.
# VQ-Attention requires sampled frames <= selected K, so for adaptive methods
# also set max_frames_num <= vqtoken_min_clusters.
