#!/usr/bin/env bash
# tests/test_vllm_launcher_dryrun.sh — render both rank commands for the vLLM
# launcher without docker and assert the production profile. Rank1 output is
# printf %q-escaped with doubled backslashes, so spec-config and kwargs keys are
# matched with backslash-tolerant ERE against the plain rank0 line.
set -euo pipefail
cd "$(dirname "$0")/.."
out="$(WORKER_SSH=user@host HEAD_IP=127.0.0.1 SG_DRYRUN=1 \
  MAX_MODEL_LEN=524288 \
  bash scripts/run-vllm-vision-dual-gb10.sh)"
for needle in \
  "--kv-cache-dtype fp8" \
  "--long-prefill-token-threshold 1024" \
  "--tool-call-parser deepseek_v4" "--reasoning-parser deepseek_v4"; do
  grep -qF -- "$needle" <<<"$out" || { echo "MISSING: $needle"; echo "$out" | head -c 4000; exit 1; }
done
for pattern in \
  'node.rank[\\ ]+1' \
  'node.rank[\\ ]+0' \
  'num_speculative_tokens[\\":]*5' \
  'enable_adaptive_verification[\\":]*true' \
  'method[\\":]*dspark' \
  'cudagraph_mode.*FULL_DECODE_ONLY' \
  'cudagraph_capture_sizes.*1.*2.*4.*8.*16.*32.*48' \
  'default-chat-template-kwargs.*thinking.*false'; do
  grep -qE -- "$pattern" <<<"$out" || { echo "MISSING(pattern): $pattern"; echo "$out" | head -c 4000; exit 1; }
done
# The dry-run renders "IMAGE" as a placeholder; assert the default tag in source.
grep -qF 'IMAGE="${DSV4V_IMAGE:-dsv4v-vllm:vision-54631-fi512b-k5adapt}"' scripts/run-vllm-vision-dual-gb10.sh || {
  echo "MISSING: default image tag"; exit 1; }
grep -q "VLLM_VERIFY_PREFIX" <<<"$out" && { echo "stale CPU-prefix diagnostic leaked into default launch"; exit 1; }
grep -q "overlay-sm121-adaptive" <<<"$out" && { echo "overlay mounts leaked into baked-image default launch"; exit 1; }
echo "vllm launcher dry-run: PASS"