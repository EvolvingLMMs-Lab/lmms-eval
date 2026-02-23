#!/usr/bin/env bash

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ -z "${FAL_KEY:-}" ]]; then
  FAL_KEY="${FAL_API_KEY:-}"
fi
export FAL_KEY="${FAL_KEY:?Error: FAL_KEY not set in environment}"

FAL_MODEL="${FAL_MODEL:-wan/v2.6/text-to-video}"
PROMPT="${PROMPT:-A cinematic drone shot over a futuristic city at sunset}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
DURATION="${DURATION:-5}"
RESOLUTION="${RESOLUTION:-720p}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
SEED="${SEED:-}"
EXTRA_JSON="${EXTRA_JSON:-{}}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/fal_videogen}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-3}"
MAX_POLLS="${MAX_POLLS:-120}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-fal_video_${timestamp}}"

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] fal.ai video generation"
echo "[INFO] model=${FAL_MODEL}"
echo "[INFO] prompt=${PROMPT}"
echo "[INFO] output_dir=${OUTPUT_DIR}"

PAYLOAD_JSON="$(python3 - "${PROMPT}" "${NEGATIVE_PROMPT}" "${DURATION}" "${RESOLUTION}" "${ASPECT_RATIO}" "${SEED}" "${EXTRA_JSON}" <<'PY'
import json
import sys

prompt = sys.argv[1]
negative_prompt = sys.argv[2]
duration = sys.argv[3]
resolution = sys.argv[4]
aspect_ratio = sys.argv[5]
seed = sys.argv[6]
extra_json = sys.argv[7]

payload = {
    "prompt": prompt,
    "duration": str(duration),
    "resolution": str(resolution),
    "aspect_ratio": aspect_ratio,
}
if negative_prompt:
    payload["negative_prompt"] = negative_prompt
if seed:
    payload["seed"] = int(seed)

extra = json.loads(extra_json)
if not isinstance(extra, dict):
    raise ValueError("EXTRA_JSON must decode to a JSON object")
payload.update(extra)

print(json.dumps(payload, ensure_ascii=True))
PY
)"

submit_url="https://queue.fal.run/${FAL_MODEL}"

submit_json="$(curl -fsS -X POST "${submit_url}" \
  -H "Authorization: Key ${FAL_KEY}" \
  -H "Content-Type: application/json" \
  --data "${PAYLOAD_JSON}")"

REQUEST_ID="$(printf '%s' "${submit_json}" | python3 - <<'PY'
import json
import sys

data = json.load(sys.stdin)
rid = data.get("request_id")
if not rid:
    raise SystemExit("submit response missing request_id")
print(rid)
PY
)"

STATUS_URL="$(printf '%s' "${submit_json}" | python3 - "${FAL_MODEL}" "${REQUEST_ID}" <<'PY'
import json
import sys

data = json.load(sys.stdin)
status_url = data.get("status_url")
if status_url:
    print(status_url)
else:
    model = sys.argv[1]
    rid = sys.argv[2]
    print(f"https://queue.fal.run/{model}/requests/{rid}/status")
PY
)"

RESULT_URL="$(printf '%s' "${submit_json}" | python3 - "${FAL_MODEL}" "${REQUEST_ID}" <<'PY'
import json
import sys

data = json.load(sys.stdin)
response_url = data.get("response_url")
if response_url:
    print(response_url)
else:
    model = sys.argv[1]
    rid = sys.argv[2]
    print(f"https://queue.fal.run/{model}/requests/{rid}")
PY
)"

echo "[INFO] request_id=${REQUEST_ID}"

state=""
status_json=""
for ((i = 1; i <= MAX_POLLS; i++)); do
  status_json="$(curl -fsS "${STATUS_URL}" -H "Authorization: Key ${FAL_KEY}")"
  state="$(printf '%s' "${status_json}" | python3 - <<'PY'
import json
import sys

data = json.load(sys.stdin)
state = data.get("status") or data.get("state") or ""
if not state and isinstance(data.get("response"), dict):
    state = data["response"].get("status", "")
print(str(state).upper())
PY
)"
  echo "[INFO] poll=${i}/${MAX_POLLS} state=${state:-UNKNOWN}"

  if [[ "${state}" == "COMPLETED" ]]; then
    break
  fi
  if [[ "${state}" == "FAILED" || "${state}" == "CANCELED" ]]; then
    echo "[ERROR] fal request ended with state=${state}"
    echo "${status_json}" > "${OUTPUT_DIR}/${OUTPUT_BASENAME}.status.json"
    exit 1
  fi
  sleep "${POLL_INTERVAL_S}"
done

if [[ "${state}" != "COMPLETED" ]]; then
  echo "[ERROR] timed out waiting for completion after ${MAX_POLLS} polls"
  echo "${status_json}" > "${OUTPUT_DIR}/${OUTPUT_BASENAME}.status.json"
  exit 1
fi

result_json="$(curl -fsS "${RESULT_URL}" -H "Authorization: Key ${FAL_KEY}")"
echo "${result_json}" > "${OUTPUT_DIR}/${OUTPUT_BASENAME}.result.json"

VIDEO_URL="$(printf '%s' "${result_json}" | python3 - <<'PY'
import json
import sys

video_exts = (".mp4", ".mov", ".webm", ".mkv", ".avi")

def pick_url(node):
    if isinstance(node, dict):
        if "url" in node and isinstance(node["url"], str):
            u = node["url"]
            if u.startswith("http") and any(ext in u.lower() for ext in video_exts):
                return u
        for key in ("video", "videos", "output", "result", "data"):
            if key in node:
                found = pick_url(node[key])
                if found:
                    return found
        for v in node.values():
            found = pick_url(v)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = pick_url(item)
            if found:
                return found
    elif isinstance(node, str):
        u = node
        if u.startswith("http") and any(ext in u.lower() for ext in video_exts):
            return u
    return None

data = json.load(sys.stdin)
url = pick_url(data)
if not url:
    raise SystemExit("no video URL found in fal response")
print(url)
PY
)"

video_ext="$(python3 - "${VIDEO_URL}" <<'PY'
import sys
import urllib.parse

url = sys.argv[1]
path = urllib.parse.urlparse(url).path.lower()
for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
    if path.endswith(ext):
        print(ext)
        break
else:
    print(".mp4")
PY
)"

output_video_path="${OUTPUT_DIR}/${OUTPUT_BASENAME}${video_ext}"
curl -LfsS "${VIDEO_URL}" -o "${output_video_path}"

echo "[OK] video saved: ${output_video_path}"
echo "[OK] request metadata: ${OUTPUT_DIR}/${OUTPUT_BASENAME}.result.json"
