#!/usr/bin/env bash
# DSV4V throughput c-ladder via vllm bench serve (client from host venv).
# Usage: ./run-cladder.sh <base-url> <served-model> <tokenizer-path> <out-dir> [concurrency list]
set -euo pipefail
BASE="${1:?base-url}"
MODEL="${2:?served-model}"
TOK="${3:?tokenizer-path}"
OUT="${4:?out-dir}"
LIST="${5:-1 2 4 8 16}"
BENCH_PY="${BENCH_PY:-python3 -m vllm.entrypoints.cli.main bench serve}"

mkdir -p "${OUT}"
for C in ${LIST}; do
  N=$(( C * 2 ))
  echo "== c${C} (${N} prompts) =="
  ${BENCH_PY} \
    --backend vllm \
    --base-url "${BASE}" \
    --model "${MODEL}" \
    --tokenizer "${TOK}" \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --num-prompts "${N}" \
    --ignore-eos \
    --max-concurrency "${C}" \
    --save-result --result-filename "${OUT}/c${C}.json" \
    --disable-tqdm
  python3 - "${OUT}/c${C}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"c{sys.argv[1].split('/')[-1][1:-5]}: output_tok/s={d.get('output_throughput', 0):.2f} "
      f"total_tok/s={d.get('total_token_throughput', 0):.2f} "
      f"ttft_p99={d.get('ttft_p99', 0):.0f}ms success={d.get('completed', '?')}")
PY
done
echo "LADDER_DONE ${OUT}"
