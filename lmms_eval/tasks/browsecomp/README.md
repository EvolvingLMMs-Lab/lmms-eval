# BrowseComp

BrowseComp is a benchmark for evaluating the ability to find hard-to-locate information on the internet, published by OpenAI.

- Paper: https://arxiv.org/abs/2504.12516
- Dataset: https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
- Reference implementation: https://github.com/openai/simple-evals

## Notes

- Data is XOR-encrypted to prevent training contamination; decryption happens at eval time.
- LLM-as-Judge scoring is used when `use_lmms_judge: true` and API credentials are configured.
- Fallback to exact string match when judge is unavailable.

## Usage

```bash
# Basic evaluation (exact match fallback)
python -m lmms_eval --tasks browsecomp --model <model_name>

# With LLM judge (requires OPENAI_API_KEY)
API_TYPE=openai OPENAI_API_KEY=<key> python -m lmms_eval --tasks browsecomp --model <model_name>
```
