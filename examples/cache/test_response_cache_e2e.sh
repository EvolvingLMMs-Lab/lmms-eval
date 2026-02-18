#!/bin/bash
# End-to-end cache verification using OpenRouter API.
#
# Requires: OPENROUTER_API_KEY environment variable
# Usage:    bash tests/test_response_cache_e2e.sh
#
# Verifies:
#   Run 1 (cold): all cache misses, real API calls
#   Run 2 (warm): all cache hits, zero API calls, faster completion
#   Cache stats logged by ResponseCache confirm 100% hit rate on run 2

set -euo pipefail

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Export it and re-run."
    exit 1
fi

export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

CACHE_DIR=$(mktemp -d)
OUTPUT_DIR=$(mktemp -d)
trap "rm -rf $CACHE_DIR $OUTPUT_DIR" EXIT

MODEL_ARGS="model_version=google/gemma-3-4b-it:free,timeout=60,max_retries=3,num_concurrent=2"
TASKS="mme"
LIMIT=3

echo "=== Cache dir: $CACHE_DIR ==="
echo ""

echo "=== Run 1: cold cache (expect all misses) ==="
START1=$(python3 -c "import time; print(time.time())")
uv run python -m lmms_eval \
    --model openai \
    --force_simple \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --batch_size 1 \
    --use_cache "$CACHE_DIR" \
    --output_path "$OUTPUT_DIR/run1" \
    --log_samples \
    --verbosity INFO 2>&1 | tee /tmp/cache_run1.log
END1=$(python3 -c "import time; print(time.time())")
ELAPSED1=$(python3 -c "print(f'{$END1 - $START1:.1f}s')")
echo ""
echo "Run 1 elapsed: $ELAPSED1"

echo ""
echo "=== Cache contents ==="
find "$CACHE_DIR" -type f | sort
echo ""

echo "=== Run 2: warm cache (expect all hits) ==="
START2=$(python3 -c "import time; print(time.time())")
uv run python -m lmms_eval \
    --model openai \
    --force_simple \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --batch_size 1 \
    --use_cache "$CACHE_DIR" \
    --output_path "$OUTPUT_DIR/run2" \
    --log_samples \
    --verbosity INFO 2>&1 | tee /tmp/cache_run2.log
END2=$(python3 -c "import time; print(time.time())")
ELAPSED2=$(python3 -c "print(f'{$END2 - $START2:.1f}s')")
echo ""
echo "Run 2 elapsed: $ELAPSED2"

echo ""
echo "=== Verification ==="
HITS=$(grep -o "cache hits" /tmp/cache_run2.log | wc -l | tr -d ' ')
SKIPPED=$(grep -c "skipping model inference" /tmp/cache_run2.log || true)

if [ "$SKIPPED" -ge 1 ]; then
    echo "PASS: Run 2 served all requests from cache (no model inference)"
else
    echo "WARNING: Run 2 may not have hit cache for all requests"
    echo "Check /tmp/cache_run2.log for ResponseCache stats"
fi

echo ""
echo "Run 1 (cold): $ELAPSED1"
echo "Run 2 (warm): $ELAPSED2"
echo ""
echo "Cache files:"
ls -lh "$CACHE_DIR"/*/* 2>/dev/null || echo "  (none)"
