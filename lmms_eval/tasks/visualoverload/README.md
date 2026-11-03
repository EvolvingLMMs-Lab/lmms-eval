# VisualOverload

[VisualOverload](https://github.com/paulgavrikov/visualoverload) (CVPR 2026)
contains 2,720 VQA questions about dense, high-resolution paintings. The public
test split combines multiple-choice, counting, and OCR questions; answers are private.

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks visualoverload --batch_size 1 --output_path ./results
```

The task uses each record's official `default_prompt` without adding another
answer-format instruction. Both simple and chat-template backends are supported.
The adapter keeps the full image resolution; model processors can still resize
or tile images, so report those settings alongside results.

Predictions are saved to `results/submissions/visualoverload_submission.json` as
a list of `{"question_id": "...", "response": "..."}` records. Raw responses,
including free-form counting and OCR answers, are preserved for the official
server's answer extraction. The `submission` metric has no local numeric score.

Upload the file to the [official evaluator](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit)
(Hugging Face sign-in required). No automatic submission is made. Runs using
`--limit` produce partial files for integration checks, not full benchmark scores.
