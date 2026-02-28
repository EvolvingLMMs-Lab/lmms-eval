# MME-CC

MME-CC integration for lmms-eval.

- Source dataset: `MaxwellWen/MME-CC`
- Task name: `mme_cc`
- Metric: exact match over extracted reference answers

Quick smoke run:

```bash
python -m lmms_eval \
  --model dummy_video_reader \
  --model_args response=yes \
  --tasks mme_cc \
  --limit 8 \
  --batch_size 1 \
  --log_samples \
  --output_path outputs/mme_cc_smoke
```
