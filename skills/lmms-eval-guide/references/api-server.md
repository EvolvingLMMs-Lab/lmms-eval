<!-- lmms-eval v0.7 -->
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

### Non-Blocking Integration Pattern (Recommended)

The key point is to **never block the trainer on eval submission**. Submit jobs during training, then harvest ready results opportunistically.

```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")
pending = {}  # job_id -> epoch

for epoch in range(num_epochs):
    train_one_epoch()

    # 1) Submit eval asynchronously
    if epoch % 5 == 0:
        job = client.evaluate(
            model="vllm",
            model_args={"model": f"checkpoints/epoch_{epoch}"},
            tasks=["mmmu_val", "mathvista"],
            limit=64,  # optional quick signal
        )
        pending[job["job_id"]] = epoch

    # 2) Non-blocking status checks for already submitted jobs
    completed = []
    for job_id, submitted_epoch in pending.items():
        info = client.get_job(job_id)
        status = info["status"]
        if status == "completed":
            print(f"Eval done for epoch {submitted_epoch}: {info['result']}")
            completed.append(job_id)
        elif status in {"failed", "cancelled"}:
            print(f"Eval {job_id} ({submitted_epoch}) ended with status={status}")
            completed.append(job_id)

    for job_id in completed:
        pending.pop(job_id, None)

# Final drain after training (blocking is okay here)
for job_id, submitted_epoch in pending.items():
    result = client.wait_for_job(job_id)
    print(f"Final eval epoch {submitted_epoch}: {result['result']}")
```

### Training-Job Integration Checklist

1. Run eval server on dedicated eval resources (or at least isolated GPU slots).
2. Use periodic async submission (`every N epochs/steps`) instead of synchronous eval.
3. Track `job_id` with training context (for example, `(epoch, checkpoint_path, job_id)`).
4. Use `/queue` and `/jobs/{job_id}` to monitor backlog and job health.
5. Handle terminal error states explicitly (`failed`, `cancelled`) in trainer logs.
6. Use small `limit` for frequent progress checks; run full benchmarks less frequently.

### Evaluation Cadence Suggestions

Use two lanes of evaluation to balance signal quality and training throughput:

- **Fast lane (high frequency)**: every `N` steps/epochs with small `limit` on 1-2 sentinel tasks.
- **Full lane (low frequency)**: less frequent full-benchmark jobs for checkpoint selection and reporting.

Example policy:

1. Every 1-2 epochs: run `limit=64` on a small validation subset.
2. Every 5-10 epochs: run full `mmmu_val`, `mathvista`, or your publication benchmark set.
3. On major checkpoint milestones: run complete benchmark suite and archive outputs.

### Common Operational Flow

1. `GET /health` -> verify server liveness.
2. `GET /models` and `GET /tasks` -> verify model/task availability before submission.
3. `POST /evaluate` -> submit jobs asynchronously.
4. `GET /queue` -> inspect pressure/backlog.
5. `GET /jobs/{job_id}` -> retrieve per-job status/results.
6. `DELETE /jobs/{job_id}` -> cancel stale queued jobs when needed.

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
