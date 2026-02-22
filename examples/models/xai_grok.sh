export HF_HOME="~/.cache/huggingface"
export OPENAI_API_KEY="xai-xxxxxxxxxx"
export OPENAI_API_BASE="https://api.x.ai/v1"


python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=grok-2-vision-1212 \
    --tasks ai2d,chartqa,docvqa_val,mathvista_testmini,mmmu_pro \
    --batch_size 1