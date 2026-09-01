#!/usr/bin/env bash
# Short decode ladder (512/256 random-ids) writing to /sgl-rw so the host can scp it.
set -u
M=deepseek-v4-flash-vision-exp
OUT="${1:-/sgl-rw/R0-short-ladder.txt}"
: > "$OUT"
bench() {
  local C=$1
  echo "=== c$C ===" | tee -a "$OUT"
  python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
    --model "$M" --dataset-name random-ids --random-input-len 512 --random-output-len 256 \
    --num-prompts $((C*4)) --max-concurrency "$C" --temperature 0 2>&1 | tee -a "$OUT"
}
echo "== warm-up (c2, recorded but not scored) ==" | tee -a "$OUT"
bench 2
for C in 1 2 4 8; do bench "$C"; done
echo SHORT_LADDER_DONE | tee -a "$OUT"
