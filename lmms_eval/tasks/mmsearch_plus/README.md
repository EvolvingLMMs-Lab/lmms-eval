# MMSearch-Plus VQA Task

This directory contains the implementation of the MMSearch-Plus VQA task for lmms-eval.

## Overview

MMSearch-Plus is a challenging benchmark designed to test multimodal browsing agents' ability to perform genuine visual reasoning. The dataset contains 311 carefully curated tasks that require extracting and using fine-grained visual cues through iterative image-text retrieval.

**Paper**: [MMSearch-Plus: Benchmarking Provenance-Aware Search for Multimodal Browsing Agents](https://arxiv.org/abs/2508.21475)

**Dataset**: [Cie1/MMSearch-Plus](https://huggingface.co/datasets/Cie1/MMSearch-Plus)

**Project Page**: [https://mmsearch-plus.github.io/](https://mmsearch-plus.github.io/)

## Dataset Structure

Each sample in the dataset contains:
- `question`: The question text (encrypted)
- `answer`: List of valid answer strings (encrypted)
- `num_images`: Number of images in the sample (1-5)
- `img_1` through `img_5`: PIL Image objects
- `arxiv_id`: ArXiv ID if the question is about a paper (optional, encrypted)
- `video_url`: Video URL if the question is about a video (optional, encrypted)
- `category`: Question category (e.g., "Academic Research", "Sports", "Film & TV")
- `difficulty`: "easy" or "difficult"
- `subtask`: Specific subtask type

## Implementation

This is a simple VQA implementation that:

1. **Decrypts** the dataset fields using the provided canary string
2. **Extracts** images from the sample based on `num_images`
3. **Formats** the question as a text prompt
4. **Evaluates** predictions using:
   - **F1 Score**: Token-level overlap between prediction and ground truth
   - **Exact Match**: Binary match after normalization

## Files

- `_mmsearch_plus.yaml`: Group configuration file
- `_default_template_mmsearch_plus.yaml`: Default template with common settings
- `mmsearch_plus_vqa.yaml`: VQA task configuration
- `utils.py`: Helper functions for data loading and evaluation
- `decrypt_utils.py`: Decryption utilities for the encrypted dataset
- `README.md`: This file

## Usage

### Running Evaluation

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks mmsearch_plus_vqa \
    --batch_size 1 \
    --device cuda:0 \
    --log_samples
```

```bash
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=Qwen/Qwen3-VL-32B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks "mmsearch_plus" \
    --batch_size 1 \
    --log_samples \
    --output_path logs
```

### Available Tasks

- `mmsearch_plus_vqa`: Simple VQA evaluation on the full dataset

## Metrics

- **f1_score**: Token-level F1 score (max across all valid answers)
- **exact_match**: Exact match score after normalization (max across all valid answers)

## Notes

### Answer Normalization

Both predictions and ground truth answers are normalized before comparison:
1. Convert to lowercase
2. Remove punctuation
3. Remove extra whitespace

### Multiple Valid Answers

Each sample may have multiple valid answers. The evaluation computes the maximum score across all valid answers to be lenient with answer variations.

## Limitations

This is a **simplified VQA implementation** that:
- Does not include the full browsing agent framework
- Does not perform iterative search and retrieval
- Evaluates based on direct question answering only

For the full agentic evaluation, please refer to the official MMSearch-Plus repository.

## Citation

If you use MMSearch-Plus in your research, please cite:

```bibtex
@article{tao2025mmsearch,
  title={MMSearch-Plus: A Simple Yet Challenging Benchmark for Multimodal Browsing Agents},
  author={Tao, Xijia and Teng, Yihua and Su, Xinxing and Fu, Xinyu and Wu, Jihao and Tao, Chaofan and Liu, Ziru and Bai, Haoli and Liu, Rui and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2508.21475},
  year={2025}
}
```

## Contact

For questions or issues related to this implementation, please open an issue in the lmms-eval repository.

For questions about the MMSearch-Plus dataset or benchmark, please refer to the [project page](https://mmsearch-plus.github.io/).
