# vizdoom_agent

Minimal ViZDoom agentic evaluation using `generate_until_agentic`.

- Each JSONL row is one episode seed.
- The model receives a short video clip of recent frames each round.
- The model must emit exactly one action tool call per round and should not emit
  `<submit>`; the task stops the episode automatically.

```text
<tool_call>{"name":"act","arguments":{"action":"ATTACK"}}</tool_call>
```

- The task writes the full episode video to `outputs/vizdoom_agent` by default.
- Override the video output directory with `LMMS_EVAL_VIZDOOM_OUTPUT_DIR`.

Run a small smoke eval with LLaVA-OneVision2:

```bash
python3 -m lmms_eval \
  --model llava_onevision2 \
  --model_args pretrained=lmms-lab-encoder/LLaVA-OneVision2-8B-Instruct,batch_size=1,max_new_tokens=96,max_num_frames=8,fps=4 \
  --tasks vizdoom_agent \
  --limit 3 \
  --batch_size 1 \
  --agentic_trace_mode full \
  --log_samples
```

Install ViZDoom separately if it is not already available in the environment:

```bash
uv pip install vizdoom
```

The final prediction payload includes `video_path` and `observation_video_path`.
