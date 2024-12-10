###############################################
## Installations after clone the repo##
cd lmms-eval
pip install -e .
## make sure llava is installed ##
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e .
###############################################

# Run llavaone 0.5 for mme
accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks mme \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
# Run llavaone 0.5 for gqa
accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks gqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

# ferret,wildvision
# mmmu,mme,mmbench,seedbench_2_plus,llava_wilder_small,llava_bench_coco,llava_in_the_wild,vibe_eval
accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks llava_wilder_small \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

# gqa,vqav2,ok_vqa,realworldqa
accelerate launch --num_processes=1 \
-m lmms_eval \
--verbosity=INFO \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks mmmu \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

accelerate launch --num_processes=1 \
-m lmms_eval \
--model vlr \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks seedbench_2_plus \
--batch_size 1 \
--log_samples \
--log_samples_suffix vlr \
--output_path ./logs/

accelerate launch --num_processes=1 \
-m lmms_eval \
--model vlr \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks realworldqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix vlr \
--output_path ./logs/
