# AI2D input variants

`ai2d`, `ai2d_lite`, `ai2d_no_mask`, and `ai2d_reasoning` keep their existing
datasets and behavior. In particular, legacy `ai2d` can supply a letter-masked
diagram to ordinary text questions on that diagram (see issue #1576).

`ai2d_vlmeval` uses VLMEvalKit's **AI2D_TEST**, whose image is selected per
question. It reuses `ai2d`'s prompts, targets, generation settings and scoring;
it does not use VLMEvalKit's model-output judge. Report this task name with your
results. Its scores are not interchangeable with legacy `ai2d` or `ai2d_no_mask`.

Source: [VLMEvalKit dataset definition](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/image_mcq.py).
The published TSV checksum is MD5 `0f593e0d1c7df9a3d69bf1f947e71975` (3,088
questions). The adapter preserves `index`, `abcLabel` and `image_path`, resolves
reused-image row references before evaluation selection, and preserves the
original A–D option order. The TSV's selected image is authoritative: this task
does **not** reconstruct masks or assume `abcLabel` alone reproduces all of
VLMEvalKit's preprocessing.
The CSV loader keeps literal text such as `NA` as an option rather than
converting it to a missing value.

The task pins the `mm-eval/VLMEvalKit` dataset at revision
`772639df87331ba19da76ea9cc5fa77836a15a18`; its TSV matches the published
MD5 and SHA256 `3b16aa83c6809149df6978c7d5a6ee830b75f0010a60b26d33c004c47d575c4a`.
This avoids depending on the availability of the original OpenXLab download
endpoint. There is no runtime model call or mask generation in data preparation.

```bash
transformers serve HuggingFaceTB/SmolVLM-256M-Instruct --device cuda --dtype float16

OPENAI_API_KEY=EMPTY python -m lmms_eval --model async_openai \
  --model_args model_version=HuggingFaceTB/SmolVLM-256M-Instruct,base_url=http://127.0.0.1:8000/v1,max_retries=1,num_cpus=1 \
  --tasks ai2d_vlmeval --limit 12 --batch_size 1 --log_samples \
  --output_path ./logs/ai2d_vlmeval
```

The lightweight model is useful for checking that real images reach a model
and predictions/metrics are emitted. It may answer `Answer: A` rather than
the requested single letter, which the inherited AI2D filter can mark wrong.
Do not use this small run as an accuracy comparison between input variants.

If the upstream download is unavailable, obtain a checksum-matching TSV and
use an `--include_path` YAML that includes this task's YAML and overrides only
`dataset_kwargs.data_files.test` with the local file. Keep the source checksum
and override in the experiment record. Do not silently use `AI2D_TEST_NO_MASK`
or disable certificate verification.

Dataset attribution/license information is provided by the
[AI2D public-data registry](https://registry.opendata.aws/allenai-diagrams/).
This change distributes task code, not dataset images.
