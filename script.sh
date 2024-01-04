accelerate launch --num_processes=1 -m lmm_eval --model llava   --model_args pretrained=llava-hf/llava-1.5-7b-hf   --tasks mme_llava_prompt  --batch_size 4 --log_samples --log_samples_sufix debug --output_path ./logs/


gpu = 8:

|     Tasks      |Version|Filter|n-shot|  Metric   |Value|   |Stderr |
|----------------|-------|------|-----:|-----------|----:|---|------:|
|mme_llava_prompt|Yaml   |none  |     0|exact_match| 1862|±  |38.2074|



gpu = 6:

|     Tasks      |Version|Filter|n-shot|  Metric   |Value|   |Stderr |
|----------------|-------|------|-----:|-----------|----:|---|------:|
|mme_llava_prompt|Yaml   |none  |     0|exact_match| 1860|±  |38.1664|

gpu=4: 

|     Tasks      |Version|Filter|n-shot|  Metric   |Value|   |Stderr|
|----------------|-------|------|-----:|-----------|----:|---|-----:|
|mme_llava_prompt|Yaml   |none  |     0|exact_match| 1867|±  | 38.31|


gpu=8, bs=1:

|     Tasks      |Version|Filter|n-shot|  Metric   |Value|   |Stderr |
|----------------|-------|------|-----:|-----------|----:|---|------:|
|mme_llava_prompt|Yaml   |none  |     0|exact_match| 1871|±  |38.3921|



gpu=6, bs=1:
|     Tasks      |Version|Filter|n-shot|  Metric   |Value|   |Stderr|
|----------------|-------|------|-----:|-----------|----:|---|-----:|
|mme_llava_prompt|Yaml   |none  |     0|exact_match| 1863|±  |38.228|


632c632
<         "No"
---
>         "Yes"
636c636
<       "No"
---
>       "Yes"
638c638
<     "exact_match": 1.0
---
>     "exact_match": 0.0
1299c1299
<         "Yes"
---
>         ""
1303c1303
<       "Yes"
---
>       ""
1305c1305
<     "exact_match": 1.0
---
>     "exact_match": 0.0
1908c1908
<         "Yes"
---
>         "No"
1912c1912
<       "Yes"
---
>       "No"
1914c1914
<     "exact_match": 0.0
---
>     "exact_match": 1.0
2488c2488
<         "No"
---
>         "Yes"
2492c2492
<       "No"
---
>       "Yes"
2494c2494
<     "exact_match": 1.0
---
>     "exact_match": 0.0
3822c3822
<         "Yes"