#!/usr/bin/env bash

set -euo pipefail

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}"

MODEL_VERSION="${MODEL_VERSION:-google/gemini-2.5-flash-image}"
TASKS="${TASKS:-ice_bench}"
LIMIT="${LIMIT:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/openrouter_ice_smoke}"
IMAGE_OUTPUT_DIR="${IMAGE_OUTPUT_DIR:-./logs/openrouter_ice_images}"
USE_OFFICIAL_ICE_SAMPLE="${USE_OFFICIAL_ICE_SAMPLE:-1}"

mkdir -p "${OUTPUT_PATH}" "${IMAGE_OUTPUT_DIR}"

if [[ "${USE_OFFICIAL_ICE_SAMPLE}" == "1" ]]; then
uv run python - <<'PY'
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

zip_path = hf_hub_download(
    repo_id="ali-vilab/ICE-Bench",
    repo_type="dataset",
    filename="dataset.zip",
    token=False,
)

target_jsonl = Path("/tmp/ice_bench_smoke.jsonl")
target_dir = Path("/tmp/ice_bench_smoke_data")
target_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path) as zf:
    with zf.open("data/data.jsonl") as fh:
        first = json.loads(next(fh))

    src_rel = first["SourceImage"]
    instruction = first["Instruction"]
    item_id = first["ItemID"]

    src_out = target_dir / f"{item_id}_src.png"
    with zf.open(src_rel) as src_in:
        src_out.write_bytes(src_in.read())

record = {
    "item_id": item_id,
    "instruction": instruction,
    "source_image": str(src_out),
}
target_jsonl.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Prepared smoke data at {target_jsonl}")
print(f"Source image at {src_out}")
PY
fi

echo "[INFO] Running ICE smoke with model=${MODEL_VERSION} tasks=${TASKS}"

uv run python -m lmms_eval \
  --model openrouter_image_gen \
  --model_args "model_version=${MODEL_VERSION},output_dir=${IMAGE_OUTPUT_DIR},max_new_tokens=4096,image_size=1024x1024" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity INFO

echo "[INFO] Done. Generated images in ${IMAGE_OUTPUT_DIR}/ice_bench"
