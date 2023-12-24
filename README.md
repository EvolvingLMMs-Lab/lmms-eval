# lmm-eval

The API, togegher with many code blocks of this project come from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). **Please read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) before contributing to this project**. Please do not commit to this project directly. Instead, push your changes to another branch and create a pull request.

Below are the changes we made to the original API:

- Intance.args (lmm_eval/api/instance.py) now contains a list of images to be inputted to LMM.
- lm-eval-harness supports all HF LM as single model class. Currently this is not possible of LMM because the input/output format of LMM in HF are not yet unified. Thererfore, we have to create a new class for each LMM model. This is not ideal and we will try to unify them in the future.

## Functionality
- support image-text dataset
- support text only dataset (to evaluate if LMM can retain the performance of LM)
- support CoT and multi sampling 


## Current models

- llava

## Current datasets

- MMMU


## How to run

```bash
pip install -e .
lmm_eval --model llava   --model_args pretrained=llava-hf/llava-1.5-7b-hf   --tasks mmmu     --device cuda:0 
```

## Models to be added


## Datasets to be added