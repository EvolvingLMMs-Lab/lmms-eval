# Run and exactly reproduce gemma3 results!
# mme as an example

accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
    --model gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks mmmu_val,ai2d,mathvista_testmini \
    --batch_size 1 --output_path ./logs/