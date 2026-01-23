# Uni-MMMU

Uni-MMMU evaluates unified multimodal models on bidirectional synergy between generation and understanding.

- Paper: https://arxiv.org/abs/2510.13759
- Dataset: https://huggingface.co/datasets/Vchitect/Uni-MMMU-Eval

## Data Setup

1. Request access to the dataset on HuggingFace
2. Download and extract the data:

```bash
git clone https://huggingface.co/datasets/Vchitect/Uni-MMMU-Eval
cd Uni-MMMU-Eval
tar -xvf data.tar -C /path/to/uni_mmmu_data
```

3. Set the environment variable:

```bash
export UNI_MMMU_DATA_DIR=/path/to/uni_mmmu_data
```

## Tasks

- **jigsaw**: Complete a 2x2 image puzzle by selecting the correct patch
- **maze**: Find path through a maze
- **sliding**: Solve sliding puzzle
- **geometry**: Solve geometry problems with visual diagrams

## Usage

```bash
python -m lmms_eval --model <model> --tasks uni_mmmu --limit 10
```

## Metrics

- `exact_match`: Accuracy of exact match
- `frame_accuracy`: For maze/sliding, step-by-step accuracy
