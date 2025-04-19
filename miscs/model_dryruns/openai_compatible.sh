export HF_HOME="~/.cache/huggingface"
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_API_BASE=""
export AZURE_OPENAI_API_VERSION="2023-07-01-preview"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=gpt-4o-2024-11-20,azure_openai=True \
    --tasks mme,mmmu_val \
    --batch_size 1