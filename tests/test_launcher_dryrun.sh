#!/usr/bin/env bash
# tests/test_launcher_dryrun.sh — render both rank commands without docker and assert the profile.
set -euo pipefail
cd "$(dirname "$0")/.."
out="$(WORKER_SSH=user@host HEAD_IP=127.0.0.1 SG_DRYRUN=1 SG_STS_TABLE=/sgl-extras/sts.json SG_TOPK_BACKEND=flashinfer SG_TF32=1 SG_MIXED_CHUNK=1 SG_FP4_INDEXER=1 SG_COMPRESS_DTYPE=bf16 bash scripts/run-sglang-dual-gb10.sh)"
for needle in \
  "--context-length 1048576" "--max-running-requests 8" "--cuda-graph-bs 1 2 3 4 6 8" \
  "--reasoning-parser deepseek-v4" "--tool-call-parser deepseekv4" "--watchdog-timeout 1800" \
  "SGLANG_DEFAULT_THINKING=1" "SGLANG_DSV4_REASONING_EFFORT=high" "SGLANG_RAGGED_VERIFY_MODE=compact" \
  "expandable_segments:True" "lmsysorg/sglang:dev-dsv4-flash-vision" "MAX_JOBS=4" \
  "--speculative-dspark-confidence-sts-path /sgl-extras/sts.json" "--dsa-topk-backend flashinfer" \
  "--speculative-dsa-topk-backend flashinfer" "--enable-tf32-matmul" "--enable-mixed-chunk" \
  "--enable-deepseek-v4-fp4-indexer" "SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16" "--node-rank 1" "--node-rank 0"; do
  grep -qF -- "$needle" <<<"$out" || { echo "MISSING: $needle"; echo "$out" | head -c 4000; exit 1; }
done
grep -q "SGLANG_SIMULATE_ACC_LEN" <<<"$out" && { echo "stale SIMULATE env leaked"; exit 1; }
echo "launcher dry-run: PASS"
