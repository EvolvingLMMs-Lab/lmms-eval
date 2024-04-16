# install lmms_eval without building dependencies
cd lmms_eval;
pip install --no-deps -U -e .

# install LLaVA without building dependencies
cd LLaVA
pip install --no-deps -U -e .

# install all the requirements that require for reproduce llava results
pip install -r llava_repr_requirements.txt

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,use_flash_attention_2=False"   --tasks mme  --batch_size 1 --log_samples --log_samples_sufix reproduce --output_path ./logs/