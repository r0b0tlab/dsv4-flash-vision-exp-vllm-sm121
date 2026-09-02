#!/usr/bin/env bash
# R2 bench: warm-up + official cells c1/c4/c8 (fixed protocol, matches R0) + gates.
# Runs INSIDE container sglang_dsv4v on node3 (rank0). Invoked via docker exec.
set -u
M=deepseek-v4-flash-vision-exp

bench() { # C tag
  local C=$1 TAG=$2
  python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
    --model "$M" --dataset-name random-ids --random-input-len 512 --random-output-len 256 \
    --num-prompts $((C*4)) --max-concurrency "$C" > "/tmp/r2_${TAG}_c${C}.log" 2>&1
}

echo "== warm-up (c2, not recorded) =="
bench 2 warm
echo "== official cells =="
for C in 1 4 8; do bench "$C" r2; done
touch /tmp/R2_DONE
echo BENCH_SUBMITTED
