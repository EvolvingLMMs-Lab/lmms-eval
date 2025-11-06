#!/bin/bash

export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"


accelerate launch --num_processes=1 -m lmms_eval \
    --model gemini_multimodal \
    --model_args model_version=gemini-2.5-flash-image-preview,enable_image_generation=True,response_persistent_folder=/n/fs/vision-mix/bl5652/lmms-eval/logs/gemini_multimodal/response_persistent_folder/ \
    --tasks ueval \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gemini_multimodal \
    --output_path /n/fs/vision-mix/bl5652/lmms-eval/logs/gemini_multimodal/ 

