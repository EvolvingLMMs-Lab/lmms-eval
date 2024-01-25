# lmms-eval

The API, togegher with many code blocks of this project come from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). **Please read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) before contributing to this project**. Please do not commit to this project directly. Instead, push your changes to another branch and create a pull request.

Below are the changes we made to the original API:

- Instance.args (lmms_eval/api/instance.py) now contains a list of images to be inputted to lmms.
- lm-eval-harness supports all HF LMM as single model class. Currently this is not possible of lmms because the input/output format of lmms in HF are not yet unified. Thererfore, we have to create a new class for each lmms model. This is not ideal and we will try to unify them in the future.

## How to run

```bash
pip install -e .
```

```bash
accelerate launch --num_processes=8 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-13b"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix debug --output_path ./logs/ # Eactly reproduce llava results
accelerate launch --num_processes=8 -m lmms_eval --config example_eval.yaml # Eactly reproduce llava results
```
## Current models

- llava （only generate_until function. Please help add the other two required functions. You can refer to lm-eval-harness for the required functions and how to implement them.)

## Current datasets
- GQA
- MMMU
- SQA-IMG
- MME
- MMVet
- LLaVA-Bench
- LLaVA-Bench-CN

## Datasets to be added and tested


## Datasets to be added



