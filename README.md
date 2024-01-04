# lmm-eval

The API, togegher with many code blocks of this project come from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). **Please read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) before contributing to this project**. Please do not commit to this project directly. Instead, push your changes to another branch and create a pull request.

Below are the changes we made to the original API:

- Instance.args (lmm_eval/api/instance.py) now contains a list of images to be inputted to LMM.
- lm-eval-harness supports all HF LM as single model class. Currently this is not possible of LMM because the input/output format of LMM in HF are not yet unified. Thererfore, we have to create a new class for each LMM model. This is not ideal and we will try to unify them in the future.

**It is very easy to add new tasks, but adding new models requires holistic understanding of the codebase**
**I recommend you to spend at least one entire days (8 hours) to look through lm evaluation harness first.**

## How to run

```bash
pip install -e .
```

```bash
lmm_eval --model llava   --model_args pretrained=llava-hf/llava-1.5-7b-hf   --tasks mmmu     --device cuda:0 --batch_size 2 --log_samples True # I have not tested this yet.
accelerate launch -m lmm_eval --model llava   --model_args pretrained=llava-hf/llava-1.5-7b-hf   --tasks mme_llava_prompt  --batch_size 2 --log_samples --output_path ./logs/  # This is working. run accelerate config to set number of GPUs.
accelerate launch -m lmm_eval --model llava   --model_args pretrained=llava-hf/llava-1.5-7b-hf   --tasks mme_llava_prompt --device cuda:0 --batch_size 2 --log_samples True # I have not tested this yet.
```
There are still bugs in the code. I will fix them when I come back from vacation.
## Current models

- llava （only generate_until function. Please help add the other two required functions. You can refer to lm-eval-harness for the required functions and how to implement them.)

## Current datasets
- MME (no extra prompts. work as expected)
- MMMU (only two subsets. I have not runned them yet Please help add all the subsets. )




## Models to be added


## Datasets to be added


## Advanced Functionality (to add later)
- support text only dataset (to evaluate if LMM can retain the performance of LM)
- support CoT and multi sampling 
