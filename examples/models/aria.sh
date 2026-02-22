export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

accelerate launch --num_processes=8 --main_process_port 12348 -m lmms_eval \
    --model aria \
    --model_args pretrained=rhymes-ai/Aria \
    --tasks ai2d,chartqa,docvqa_val,mmmu_pro \
    --batch_size 1