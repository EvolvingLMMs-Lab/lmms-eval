# VideoMathQA: Benchmarking Mathematical Reasoning via Multimodal Understanding in Videos
VideoMathQA is a benchmark designed to evaluate mathematical reasoning in real-world educational videos. It requires models to interpret and integrate information from three modalities, visuals, audio, and text, across time. The benchmark tackles the “needle-in-a-multimodal-haystack” problem, where key information is sparse and spread across different formats and moments in the video.

[![Website](https://img.shields.io/badge/🌐_Project-Website-87CEEB)](https://mbzuai-oryx.github.io/VideoMathQA)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-Access-green)](https://huggingface.co/datasets/MBZUAI/VideoMathQA)
[![🏅 Leaderboard (Reasoning)](https://img.shields.io/badge/🏅_Leaderboard-Reasoning-red)](https://hanoonar.github.io/VideoMathQA/#leaderboard-2)
[![🏅 Leaderboard (Direct)](https://img.shields.io/badge/🏅_Leaderboard-Direct-yellow)](https://hanoonar.github.io/VideoMathQA/#leaderboard)
[![GitHub](https://img.shields.io/badge/📂_GitHub-VideoMathQA-green)](https://github.com/mbzuai-oryx/VideoMathQA)

## Evaluation Strategies

**VideoMathQA** supports the following **evaluation strategies** to comprehensively assess model performance:

1. **MCQ and Multi-Binary (MBin)**  
   - Tasks with `mcq` use a 5-way multiple-choice format.  
   - Tasks with `mbin` use a stricter binary-pairwise evaluation format (correct vs each distractor).  
   - Both formats are available *with* and *without subtitles*, indicated by `_w_subtitles` in the task name.

2. **Direct Answering vs. Chain-of-Thought (CoT)**  
   - Each task can be evaluated under **Direct** or **CoT** prompting.  
   - Tasks containing `_cot` use CoT prompting, where models generate reasoning before the final answer.  
   - Direct answering tasks expect the final answer only, without intermediate reasoning.  
   - CoT tasks require post-processing to extract the final answer (see [Post Processing](#post-processing)).  
   - We maintain **separate leaderboards** for Direct and CoT settings.

3. **Step-wise CoT Evaluation**  
   - For CoT tasks, we additionally evaluate the quality of generated reasoning.  
   - Each response is scored by comparing against annotated solution steps (typically 4–10 steps).  
   - Scoring is done using a small open-source model (Qwen-3-4B in thinking mode), which returns a score (0–10) and rationale.


## Run Evaluation

Please run the following command to start evaluation.

```python
accelerate launch --num_processes=8 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=151200,min_pixels=100352,use_flash_attention_2=True,device_map=auto \
    --tasks videomathqa_mbin \
    --batch_size 1 --log_samples --log_samples_suffix qwen_2_5_vl \
    --output_path output
```

This command starts evaluating the Qwen2.5-VL-3B model on `VideoMathQA` for multi-binary accuracy. The other available `VideoMathQA` tasks are:

1. videomathqa\_mcq
2. videomathqa\_mcq\_w\_subtitles
3. videomathqa\_mcq\_cot
4. videomathqa\_mcq\_cot\_w\_subtitles
5. videomathqa\_mbin
6. videomathqa\_mbin\_w\_subtitles
7. videomathqa\_mbin\_cot
8. videomathqa\_mbin\_cot\_w\_subtitles

`w_subtitles` tasks additionally use subtitles during evaluation. `cot` tasks prompt the model to think step-by-step before answering the question.


## Post Processing
- For tasks with CoT prompting (`_cot`), model outputs typically contain both reasoning and the final answer.
- To enable standardized scoring, we post-process the responses using Qwen-3-4B (in non-thinking mode) to extract only the final answer. This ensures format consistency and removes ambiguity in final answer extraction.

```shell
# Install VLLM
pip install vllm

# Run post-processing
python videomathqa/cot_postprocess.py --input_file <path/to/your/raw_cot_results.jsonl> --output_file <path/to/save/processed_results.jsonl>
```

## CoT Step Evaluation

We provide a [VLLM](https://github.com/vllm-project/vllm)-based script to run CoT step evaluation after inference. The self-contained script is available at [cot\_step\_evaluation.py](cot_step_evaluation.py).

```shell
# Install VLLM
pip install vllm

# Run CoT step evaluation
python videomathqa/cot_step_evaluation.py --gt_file <path/to/the/annotation/parquet_file> --res_file <path/to/the/results/file/generated/after/running/inference/using/lmms_eval>
```