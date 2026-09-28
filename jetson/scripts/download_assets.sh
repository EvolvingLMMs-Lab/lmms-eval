#!/usr/bin/env bash
# Pre-fetch models and datasets into the shared HF cache (/opt/hf-cache) so evals can run offline.
# All repos here are public; no HF token needed. Datasets are also prepared into $HF_HOME/datasets
# (the Arrow cache that `datasets` requires offline); this also sidesteps task YAMLs with `token: True`.
#
# Usage:
#   jetson/scripts/download_assets.sh <framework> <size>[-<precision>]   # what run_eval.sh needs, plus MME
#   jetson/scripts/download_assets.sh <kind:repo[:file|config]> ...      # explicit, e.g. model:Qwen/Qwen2.5-VL-3B-Instruct
#     (for datasets the third field is a config name, e.g. dataset:lmms-lab-encoder/LMMs-Eval-Lite:coco2017_cap_val)
# Examples:
#   jetson/scripts/download_assets.sh llamacpp 7b-q8_0
#   jetson/scripts/download_assets.sh vllm 3b-awq
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${IMAGE:-lmms-eval-jetson:latest}
source "$REPO/jetson/frameworks/common.sh"

if [ $# -ge 1 ] && [[ "$1" != *:* ]]; then
  resolve_framework "$1" "${2:?usage: $0 <framework> <size>[-precision]}"
  mapfile -t SPECS < <(fw_assets)
  SPECS+=(dataset:lmms-lab-encoder/MME)
else
  SPECS=("$@")
fi
[ ${#SPECS[@]} -gt 0 ] || { echo "nothing to download" >&2; exit 1; }

docker run --rm -i \
  --user "$(id -u):$(id -g)" $(shared_group_args) \
  -e HOME=/tmp -e HF_HOME="$HF_CACHE" -e HF_HUB_CACHE="$HF_CACHE/hub" -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v "$HF_CACHE":"$HF_CACHE" \
  "$IMAGE" bash -c 'umask 002; python - "$@"' _ "${SPECS[@]}" <<'EOF'
import os
import sys

import datasets
from huggingface_hub import hf_hub_download, snapshot_download

for spec in sys.argv[1:]:
    kind, repo, *extra = spec.split(":", 2)
    config = extra[0] if kind == "dataset" and extra else None
    if kind == "dataset":
        # Only this config's files: some repos (e.g. LMMs-Eval-Lite) bundle many datasets.
        path = snapshot_download(repo, repo_type=kind, token=False, allow_patterns=[f"{config}/*", "*.md", "*.json"] if config else None)
    elif extra:
        path = hf_hub_download(repo, extra[0], repo_type=kind, token=False)
    else:
        path = snapshot_download(repo, repo_type=kind, token=False)
    if kind == "dataset":
        ds = datasets.load_dataset(repo, config, token=False, cache_dir=os.path.join(os.environ["HF_HOME"], "datasets"))
        print(f"    {ds}", flush=True)
    print(f"ok  {spec} -> {path}", flush=True)
EOF
