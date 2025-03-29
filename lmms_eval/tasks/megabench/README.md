# MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks [ICLR 2025]

![image](https://github.com/user-attachments/assets/5fd44fa9-0ec2-4298-ad0c-e883cb1edf7f)

MEGA-Bench contains 505 multimodal tasks with diverse data sources, input/output formats, and skill requirements. The taxonomy tree is derived from the application dimension, which guides and calibrates the annotation process. The benchmark is equiped with a suite of 45 evaluation metrics to handle various output formats beyond multiple-choice questions.

Following this doc, the evaluation result contains the final scores and multi-dimensional breakdown, which has a consistent format as [MEGA-Bench Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench). Below is an example from evaluating `llava-ov-7b` on the core set.

```json
{
    "model_summary": {
        "core": {
            "num_eval_tasks": 440,
            "num_eval_samples": 6531,
            "macro_mean_score": 0.21890499112354772
        },
        "open": ...,
        "overall_score": 0.21890499112354772
    },
    "keyword_stats": {
        "skills": {
            "Object Recognition and Classification": {
                "count": 272,
                "num_samples": 4193,
                "tasks": [],
                "average_score": 0.237467941941504
            },
            "Domain-Specific Knowledge and Skills": {
                "count": 66,
                "num_samples": 1153,
                "tasks": [],
                "average_score": 0.2134904060117354
            },
            "Mathematical and Logical Reasoning": {
                "count": 103,
                "num_samples": 1762,
                "tasks": [],
                "average_score": 0.19061988965845886
            },
            "Spatial and Temporal Reasoning": {
                "count": 144,
                "num_samples": 2278,
                "tasks": [],
                "average_score": 0.1953679420247293
            },
            ...
        }
    }
}
```



## Step-1: Get the model response files with lmms-eval

```bash
# Core set (440 tasks)
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava_onevision \
    --tasks megabench_core  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_ov_megabench_core \
    --output_path ./logs/ \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen

# Open-ended set (65 tasks)
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava_onevision \
    --tasks megabench_open  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_ov_megabench_open \
    --output_path ./logs/ \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen
```


## Step-2: Run MEGA-Bench metrics to obtain the evaluation scores and breakdown analysis


Install the dependencies of MEGA-Bench's evaluation metrics.

```bash
pip install -r requirements.txt
```

Evaluate the submission file with stand-alone evaluator adapted from [MEGA-Bench's codebase](https://github.com/TIGER-AI-Lab/MEGA-Bench).

```bash
# Run the metrics for the core set
python lmms_eval/tasks/megabench/evaluator.py --subset_name core --submission_file logs/llava-ov-7b/submissions/megabench_core_all_query_responses.json  --output_file logs/llava-ov-7b/megabench_scores/megabench_core_data_with_scores.json
```

Note: please set up the OpenAI API key in the environment variable `OPENAI_API_KEY` to evaluate the open set.

```bash
# Run the metrics for the open-ended set
python lmms_eval/tasks/megabench/evaluator.py --subset_name open --submission_file logs/llava-ov-7b/submissions/megabench_open_all_query_responses.json  --output_file logs/llava-ov-7b/megabench_scores/megabench_open_data_with_scores.json
```

```bash
# Derive the breakdown results
python lmms_eval/tasks/megabench/breakdown/derive_breakdown_results.py  --input_dir logs/llava-ov-7b/megabench_scores

```

The results in `logs/llava-ov-7b/megabench_scores/analysis` are what used by [MEGA-Bench leaderboard](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench). The leaderboard can be updated by putting the files in the results directory of the leadboard's [HuggingFace space](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench/tree/main/static/eval_results/Default).
