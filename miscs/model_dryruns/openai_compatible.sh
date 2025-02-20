# cd ~/prod/lmms-eval-public
# pip3 install -e .
# pip3 install openai

python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=grok-2-vision-1212 \
    --tasks mme,mmmu_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix openai_compatible \
    --output_path ./logs \
    --limit=8