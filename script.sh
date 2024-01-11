accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b"   --tasks mme_llava_prompt  --batch_size 1 --log_samples --log_samples_sufix debug --output_path ./logs/


gpu = 8 bs 1:

llava (pretrained=llava-hf/llava-1.5-7b-hf), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
|     Tasks      |Version|Filter|n-shot|  Metric   |Value|   |Stderr |
|----------------|-------|------|-----:|-----------|----:|---|------:|
|mme_llava_prompt|Yaml   |none  |     0|exact_match| 1873|±  |38.4331|

gpu = 8 bs 1 use_flash_attention_2=True:





gpu = 4 bs 1 use_flash_attention_2=True:



accelerate launch --num_processes=8 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b"   --tasks scienceqa  --batch_size 1 --log_samples --log_samples_sufix debug --output_path ./logs/
