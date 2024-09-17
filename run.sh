accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks mme \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks gqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

accelerate launch --num_processes=1 \
-m lmms_eval \
--model gpa_results \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks gqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix gpa_results \
--output_path ./logs/

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-mistral-7b,conv_template=mistral_instruct" \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_gqa \
    --output_path ./cps/logs/
