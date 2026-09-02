#!/usr/bin/env bash
# Final ladder on winning stack (R0 config): random-ids c1/c2/c4/c8 in-container.
set -u
M=deepseek-v4-flash-vision-exp
bench() {
  local C=$1
  python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
    --model "$M" --dataset-name random-ids --random-input-len 512 --random-output-len 256 \
    --num-prompts $((C*4)) --max-concurrency "$C" > "/tmp/final_c${C}.log" 2>&1
}
echo "== warm-up (c2, not recorded) =="
bench 2
for C in 1 2 4 8; do bench "$C"; done
touch /tmp/FINAL_DONE
echo BENCH_SUBMITTED
