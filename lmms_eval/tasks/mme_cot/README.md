# MME-CoT

[GitHub](https://github.com/CaraJ7/MME-CoT) / [Homepage](https://mmecot.github.io) / [arXiv](https://arxiv.org/abs/2502.09621)

## Usage

lmms-eval provides a convenient way to run the model inference on MME-CoT. 

To obtain the evaluation score, the users needs to use the evaluation script provided in the official repository of [MME-CoT](https://github.com/CaraJ7/MME-CoT).

### Evaluation

**Step 1: Setup**
Clone the repository and install dependencies:

```
git clone https://github.com/CaraJ7/MME-CoT.git
cd MME-CoT
pip install -r requirements.txt
```

**Step 2: Format Conversion**
Convert lmms-eval outputs to the evaluation format:

```
# For CoT prompt
python tools/update_lmmseval_json.py \
--lmms_eval_json_path mmecot_reasoning_test_for_submission.json \
--save_path results/json/YOUR_MODEL_NAME_cot.json
# For direct prompt
python tools/update_lmmseval_json.py \
--lmms_eval_json_path mmecot_direct_test_for_submission.json \
--save_path results/json/YOUR_MODEL_NAME_dir.json
```

**Step 3: Run Evaluation**
Run the evaluation script (From this point forward, all steps are identical to the step 3 and the following in the [official repository](https://github.com/CaraJ7/MME-CoT/tree/main#:~:text=Run%20the%20evaluation%20script.).


## Citation

```bibtex
@article{jiang2025mme,
  title={MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency},
  author={Jiang, Dongzhi and Zhang, Renrui and Guo, Ziyu and Li, Yanwei and Qi, Yu and Chen, Xinyan and Wang, Liuhui and Jin, Jianhan and Guo, Claire and Yan, Shen and others},
  journal={arXiv preprint arXiv:2502.09621},
  year={2025}
}
```