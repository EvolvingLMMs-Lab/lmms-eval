<!-- lmms-eval v0.6 -->
# HTTP Evaluation Server (Eval-as-a-Service)

Decouples evaluation from training. Submit async evaluation jobs from your training loop without blocking GPU resources.

## Start Server

```python
from lmms_eval.entrypoints import ServerArgs, launch_server

args = ServerArgs(
    host="0.0.0.0",
    port=8000,
    max_completed_jobs=200,
    temp_dir_prefix="lmms_eval_"
)
launch_server(args)  # Blocks, serves at http://0.0.0.0:8000
# API docs at http://0.0.0.0:8000/docs
```

## REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/evaluate` | POST | Submit evaluation job |
| `/jobs/{job_id}` | GET | Job status and results |
| `/queue` | GET | Queue status |
| `/tasks` | GET | List available tasks |
| `/models` | GET | List available models |
| `/jobs/{job_id}` | DELETE | Cancel queued job |
| `/merge` | POST | Merge FSDP2 sharded checkpoints |

## Python Client (Sync)

```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")

# Submit (non-blocking, returns immediately)
job = client.evaluate(
    model="qwen2_5_vl",
    tasks=["mmmu_val", "mme"],
    model_args={"pretrained": "Qwen/Qwen2.5-VL-7B-Instruct"},
    batch_size=1,
    device="cuda:0",
)
print(f"Job submitted: {job['job_id']}")

# Poll for completion
result = client.wait_for_job(job["job_id"], poll_interval=5.0, timeout=3600.0)
print(result["result"])
```

## Python Client (Async)

```python
from lmms_eval.entrypoints import AsyncEvalClient

async with AsyncEvalClient("http://eval-server:8000") as client:
    job = await client.evaluate(
        model="qwen3_vl",
        tasks=["mmmu_val"],
        model_args={"pretrained": "Qwen/Qwen3-VL-4B-Instruct"},
    )
    result = await client.wait_for_job(job["job_id"])
```

## Training Loop Integration

```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")
eval_jobs = []

for epoch in range(num_epochs):
    train_one_epoch()

    if epoch % 5 == 0:
        job = client.evaluate(
            model="vllm",
            model_args={"model": f"checkpoints/epoch_{epoch}"},
            tasks=["mmmu_val", "mathvista"],
        )
        eval_jobs.append((epoch, job["job_id"]))

# Collect all results after training
for epoch, job_id in eval_jobs:
    result = client.wait_for_job(job_id)
    print(f"Epoch {epoch}: {result['result']}")
```

## EvaluateRequest Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model backend name |
| `tasks` | list[string] | Yes | Task names to evaluate |
| `model_args` | dict | No | Model-specific arguments |
| `num_fewshot` | int | No | Few-shot examples (default: 0) |
| `batch_size` | int/string | No | Batch size |
| `device` | string | No | Device (e.g., "cuda:0") |
| `limit` | int/float | No | Limit samples (for testing) |
| `log_samples` | bool | No | Log per-sample outputs (default: true) |
| `num_gpus` | int | No | Number of GPUs (default: 1) |

## Job Lifecycle

`queued` -> `running` -> `completed` | `failed` | `cancelled`

## Key Source Files

| File | Purpose |
|------|---------|
| `entrypoints/http_server.py` | FastAPI server, all REST endpoints |
| `entrypoints/client.py` | `EvalClient` (sync), `AsyncEvalClient` |
| `entrypoints/protocol.py` | Pydantic models: `EvaluateRequest`, `JobInfo` |
| `entrypoints/job_scheduler.py` | Sequential GPU-safe job queue |
| `entrypoints/server_args.py` | `ServerArgs` configuration |

## Security

The server is for trusted environments only. Do NOT expose to untrusted networks without authentication, rate limiting, and network isolation.
