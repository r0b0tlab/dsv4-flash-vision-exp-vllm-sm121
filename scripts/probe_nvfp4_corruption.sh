#!/bin/bash
# NVFP4 KV Cache Corruption Probe
# Sends N identical requests and checks for garbage output patterns.
# Usage: ./probe_nvfp4_corruption.sh <PORT> <MODEL_NAME> <NUM_REQUESTS>
# Default: port 18080, model "Qwen3.6-27B-NVFP4", 20 requests

set -euo pipefail

PORT="${1:-18080}"
MODEL="${2:-Qwen3.6-27B-NVFP4}"
N="${3:-20}"

echo "=== NVFP4 KV Cache Corruption Probe ==="
echo "Server: http://127.0.0.1:${PORT}  Model: ${MODEL}  Requests: ${N}"
echo ""

# Phase 1: Prefill-only test (max_tokens=1)
echo "--- Phase 1: Prefill only (max_tokens=1) ---"
clean=0; corrupt=0
for i in $(seq 1 5); do
  text=$(curl -s "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"What is 2+2?\",\"max_tokens\":1,\"temperature\":0}" \
    2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:30])" 2>/dev/null || echo "ERROR")
  echo "  [$i] $(printf '%-30s' "$text")"
done
echo "  (All should be identical tokens if prefill is correct)"
echo ""

# Phase 2: Decode test (max_tokens=2)  
echo "--- Phase 2: Prefill + 1 decode step (max_tokens=2) ---"
clean=0; corrupt=0
for i in $(seq 1 "$N"); do
  text=$(curl -s "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"What is 2+2?\",\"max_tokens\":2,\"temperature\":0}" \
    2>/dev/null | python3 -c "
import sys, json
t = json.load(sys.stdin)['choices'][0]['text']
has_garbage = any(ord(c) > 0x4000 for c in t) or '!!!!!' in t
print(f'{\"CORRUPT\" if has_garbage else \"CLEAN\"} {repr(t[:40])}')" 2>/dev/null || echo "ERROR")
  
  if echo "$text" | grep -q "CORRUPT"; then
    corrupt=$((corrupt+1))
  else
    clean=$((clean+1))
  fi
  echo "  [$i] $text"
done

echo ""
echo "=== RESULT: ${clean}/${N} clean, ${corrupt}/${N} corrupt ==="
if [ "$corrupt" -gt 0 ]; then
  echo "CORRUPTION DETECTED — NVFP4 KV cache scale factors likely not written."
  echo "Use --kv-cache-dtype fp8 for correct output."
  exit 1
else
  echo "ALL CLEAN — NVFP4 KV cache is working correctly."
  exit 0
fi
