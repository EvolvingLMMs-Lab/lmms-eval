# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/flash/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please do not use it unless you are sure about the operation details.

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=True \
#     --tasks mmmu_pro \
#     --batch_size 1

echo "Running Qwen2.5-VL-7B-Instruct"
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks mme \
    --batch_size 1

# uv run python -m lmms_eval --model qwen2_5_vl --model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=602112,interleave_visuals=False,attn_implementation=flash_attention_2,video_sampler=uniform --tasks egoschema --batch_size 1 --output_path results/test.jsonl