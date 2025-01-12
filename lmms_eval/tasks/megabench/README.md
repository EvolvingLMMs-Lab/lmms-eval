# MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks



Example: evaluate the submission file for core set

```bash
python lmms_eval/tasks/megabench/evaluator.py --subset_name core --submission_file logs/llava-ov-7b/submissions/megabench_core_all_query_responses.json  --output_file logs/llava-ov-7b/submissions/megabench_core_data_with_scores.json

python lmms_eval/tasks/megabench/evaluator.py --subset_name open --submission_file logs/llava-ov-7b/submissions/megabench_open_all_query_responses.json  --output_file logs/llava-ov-7b/submissions/megabench_open_data_with_scores.json
```
