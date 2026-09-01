#!/usr/bin/env bash
# Shared-window concurrency ladder: (C, prompt_tokens) cells whose C*prompt ≈ 1.04M. Runs in-container.
# usage: fullctx_ladder.sh OUTFILE
set -u
OUT="$1"; : > "$OUT"
for cell in "1 1040000" "2 520000" "4 260000" "8 130000"; do
  set -- $cell; C=$1; L=$2
  echo "=== c$C prompt=$L ===" | tee -a "$OUT"
  python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
    --model deepseek-v4-flash-vision-exp --dataset-name random-ids \
    --random-input-len "$L" --random-output-len 512 --random-range-ratio 1.0 \
    --num-prompts "$C" --max-concurrency "$C" --temperature 0 2>&1 \
    | grep -E "Successful requests|Failed|Output token throughput|Mean TTFT|Median ITL|Total input tokens" | tee -a "$OUT"
done
echo "FULLCTX_DONE" | tee -a "$OUT"
