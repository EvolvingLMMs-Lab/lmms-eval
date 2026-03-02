#!/bin/bash
# 串行跑所有测试任务（普通版本）
# 用法: bash eval_all.sh --model bagel_visual_cot --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT", save_intermediate=true

set -e

MODEL=""
MODEL_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --model_args) MODEL_ARGS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash eval_all.sh --model <model> --model_args <args>"
    exit 1
fi

OUTPUT_BASE="./logs/${MODEL}"
mkdir -p "$OUTPUT_BASE"

# 覆盖分布式环境变量，避免连接远程节点
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29314

TASKS=(
    auxsolidmath_easy_visual_cot
    chartqa100_visual_cot
    geometry3k_visual_cot
    babyvision_cot
    illusionbench_arshia__visual_cot_split
    mmsi_cot
    phyx_cot
    realunify_cot
    uni_mmmu_cot
    vsp_cot
    VisualPuzzles_visual_cot
)

for TASK in "${TASKS[@]}"; do
    echo "========================================"
    echo "Running: $TASK"
    echo "========================================"
    uv run python -m lmms_eval \
        --model "$MODEL" \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --output_path "${OUTPUT_BASE}/${TASK}"
    echo "Done: $TASK"
    echo
done

echo "All tasks completed. Results in ${OUTPUT_BASE}/"

# 汇总所有结果
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
OUTPUT_BASE="$OUTPUT_BASE" python3 - <<'PYEOF'
import json, os, glob

output_base = os.environ.get("OUTPUT_BASE")
results_files = glob.glob(f"{output_base}/*/*/results*.json") + glob.glob(f"{output_base}/*/*/*results*.json")

all_results = {}
for fpath in sorted(results_files):
    with open(fpath) as f:
        data = json.load(f)
    for task, metrics in data.get("results", {}).items():
        if task not in all_results:
            all_results[task] = {}
        for k, v in metrics.items():
            if k in ("alias", " ") or "stderr" in k:
                continue
            all_results[task][k] = v

# 打印汇总
print(f"{'Task':<60s} {'Metric':<40s} {'Value'}")
print("-" * 110)
for task, metrics in sorted(all_results.items()):
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{task:<60s} {metric:<40s} {value:.4f}")
        else:
            print(f"{task:<60s} {metric:<40s} {value}")

# 保存汇总 JSON
summary_path = os.path.join(output_base, "summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSummary saved to {summary_path}")
PYEOF
