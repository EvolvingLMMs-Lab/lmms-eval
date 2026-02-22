# CaptionQA

CaptionQA evaluates how well image captions preserve information for downstream QA tasks.

**Paper**: [CaptionQA: Is Your Caption as Useful as the Image Itself?](https://arxiv.org/abs/2511.21025)  
**Dataset**: [Borise/CaptionQA](https://huggingface.co/datasets/Borise/CaptionQA)  
**Homepage**: [https://captionqa.github.io/website](https://captionqa.github.io/website)

## How It Works

1. **Caption Generation**: The evaluated model generates captions for images
2. **Judge Evaluation**: Qwen2.5-72B-Instruct answers questions based on the captions
3. **Scoring**: Computes accuracy and score based on the judge's answers

## Environment Requirements

### Required Packages

```bash
pip install transformers==4.55.0
pip install sglang[all]>=0.4.0
pip install datasets
pip install requests
```

### Hardware Requirements

The code was tested on 8 AMD MI325 GPU.

- **Caption Model**: 1 GPU with ~8GB VRAM (for Qwen2.5-VL-3B-Instruct)
- **Judge Model**: 2 GPUs with ~80GB VRAM each (for Qwen2.5-72B-Instruct with tp=2)

## Usage

### Basic Command

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks captionqa \
    --batch_size 1 \
    --launcher_args "name=sglang,model=Qwen/Qwen2.5-72B-Instruct,tp=2" \
    --output_path ./logs/captionqa_results
```

### Command Options

- `--model`: The caption model to evaluate (e.g., `qwen2_5_vl`)
- `--model_args`: Model arguments (e.g., `pretrained=Qwen/Qwen2.5-VL-3B-Instruct`)
- `--tasks`: Task name (e.g., `captionqa_natural`)
- `--batch_size`: Batch size for caption generation (recommend `1`)
- `--launcher_args`: Judge server configuration
  - `name=sglang`: Use SGLang as the judge server
  - `model=Qwen/Qwen2.5-72B-Instruct`: Judge model
  - `tp=2`: Tensor parallelism (number of GPUs for judge)
- `--output_path`: Directory to save results

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_URL` | `http://localhost:8000/v1` | Judge server URL (set by launcher) |
| `CAPTIONQA_JUDGE_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | Override judge model name |

## Metrics

| Metric | Description | Higher is Better |
|--------|-------------|------------------|
| `captionqa_score` | Weighted score (1.0 for correct, partial for "cannot answer") | ✅ |
| `captionqa_accuracy` | Percentage of correctly answered questions | ✅ |
| `captionqa_cannot_answer_rate` | Percentage of "cannot answer" responses | ❌ |

## Citation

```bibtex
@misc{yang2025captionqacaptionusefulimage,
      title={CaptionQA: Is Your Caption as Useful as the Image Itself?}, 
      author={Shijia Yang and Yunong Liu and Bohan Zhai and Ximeng Sun and Zicheng Liu and Emad Barsoum and Manling Li and Chenfeng Xu},
      year={2025},
      eprint={2511.21025},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.21025}, 
}