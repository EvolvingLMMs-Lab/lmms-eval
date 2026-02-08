# CoreCognition

## Description

CoreCognition is a large-scale benchmark encompassing 12 core knowledge grounded in developmental cognitive science, designed to evaluate the fundamental core abilities of Multi-modal Large Language Models (MLLMs).

While MLLMs demonstrate impressive abilities over high-level perception and reasoning, their robustness in the wild remains limited, often falling short on tasks that are intuitive and effortless for humans. We examine the hypothesis that these deficiencies stem from the absence of core knowledge—rudimentary core abilities innate to humans.

This dataset contains 1,423 multimodal CoreCognition samples and 80 Concept Hacking questions with images/videos and questions, covering fundamental concepts like object permanence, spatial reasoning, counting, and other core abilities that emerge in human development.

**Paper**: [Core Knowledge Deficits in Multi-Modal Language Models](https://arxiv.org/abs/2410.10855)
**Project Page**: [williamium3000.github.io/core-knowledge](https://williamium3000.github.io/core-knowledge/)
**Dataset**: [williamium/CoreCognition](https://huggingface.co/datasets/williamium/CoreCognition) (subset: `frame-combined`, split: `train`)

## Core Knowledge Concepts (12 Categories)

The benchmark covers these fundamental cognitive concepts grounded in developmental science:

- **Boundary**: The transition from one object to another
- **Continuity**: Objects persist as unified, cohesive entities across space and time
- **Permanence**: Objects do not cease to exist when they are no longer perceived
- **Spatiality**: The a priori understanding of the Euclidean properties of the world
- **Perceptual Constancy**: Changes in appearances don't mean changes in physical properties
- **Intuitive Physics**: Intuitions about the laws of how things interact in the physical world
- **Perspective**: To see what others see
- **Hierarchy**: Understanding of inclusion and exclusion of objects and categories
- **Conservation**: Invariances of properties despite transformations
- **Tool Use**: The capacity to manipulate specific objects to achieve goals
- **Intentionality**: To see what others want
- **Mechanical Reasoning**: Inferring actions from system states and vice versa

## Metrics

- **accuracy**: Overall accuracy - how often the model produces the correct answer
- **accuracy_by_concept**: Per-concept accuracy breakdown across all 12 core knowledge categories

## Answer Matching

Evaluation uses **hybrid matching** as described in the original paper (reference: `lmms_eval/tasks/stare/utils.py`):

### Hybrid Matching (Default when API is provided)

When an LLM judge API is configured, the evaluation uses hybrid matching:
1. **Template matching**: Extract MCQ (A–F) or YORN (YES/NO) from the model output via regex.
2. **LLM judge**: When template matching fails to extract a valid answer, an LLM judge determines correctness based on semantic similarity.

This hybrid approach combines the efficiency of template matching with the robustness of LLM-based semantic evaluation.

### Template Matching Only (when API is not provided)

If no LLM judge API is configured, evaluation uses template matching only, falling back to direct string comparison when template matching fails.

### Configuring Hybrid Matching

To enable hybrid matching with LLM judge:

1. **Enable in configuration**: Set `metadata.use_lmms_judge: true` in `corecognition.yaml`:
   ```yaml
   metadata:
     use_lmms_judge: true  # Enable hybrid matching with LLM judge
   ```

2. **Set environment variables**:
   ```bash
   export API_TYPE=openai  # or "azure", "anthropic", etc., see lmms_eval/llm_judge/providers
   export OPENAI_API_URL="https://api.openai.com/v1"
   export OPENAI_API_KEY=your_api_key_here
   export DEPLOYMENT_NAME=gpt-4o  # or OPENAI_API_MODEL=gpt-4o
   ```

If the API is not configured or `use_lmms_judge` is `false`, the evaluation will use template matching only.

## Usage

```bash
# Evaluate all stages
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks corecognition \
  --batch_size 1 \
  --log_samples \
  --output_path output/corecognition

# Evaluate specific stage
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks corecognition_stage_sensorimotor \
  --batch_size 1 \
  --log_samples \
  --output_path output/corecognition
```

## Citation

```bibtex
@inproceedings{
    li2025core,
    title={Core Knowledge Deficits in Multi-Modal Language Models},
    author={Yijiang Li and Qingying Gao and Tianwei Zhao and Bingyang Wang and Haoran Sun and Haiyun Lyu and Robert D. Hawkins and Nuno Vasconcelos and Tal Golan and Dezhi Luo and Hokin Deng},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=EIK6xxIoCB}
}
```
