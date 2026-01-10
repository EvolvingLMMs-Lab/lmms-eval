# tested with
# dependenies versions
# NVIDIA-SMI 550.90.07 | Driver Version: 550.90.07 | CUDA Version: 13.1
# "transformers" dont set specific version to follow cambrians requirements
# "accelerate" dont set specific version to follow cambrians requirements
# "torch==2.7.1",
# "torchvision==0.22.1",
# installation
# uv venv -p 3.11
# source .venv/bin/activate
# uv pip install . spacy git+https://github.com/cambrian-mllm/cambrian-s.git
# uv pip install flash-attn --no-build-isolation

NUM_FRAMES=32
MIV_TOKEN_LEN=64
SI_TOKEN_LEN=729

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12345 \
    -m lmms_eval \
    --model cambrians \
    --model_args=pretrained=nyu-visionx/Cambrian-S-3B,conv_template=qwen_2,video_max_frames=${NUM_FRAMES},miv_token_len=${MIV_TOKEN_LEN},si_token_len=${SI_TOKEN_LEN} \
    --tasks vsibench \
    --batch_size 1 \
    --output_path ./logs/
