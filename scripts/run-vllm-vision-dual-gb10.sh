#!/usr/bin/env bash
# DSV4-Flash-Vision-Exp dual-GB10 TP=2 — official vision image + PR#54631 overlay.
# Run on rank0. Set WORKER_SSH and HEAD_IP for your pair (no baked LAN defaults).
# SG_DRYRUN=1 prints both docker commands and exits 0.
set -euo pipefail

IMAGE="${DSV4V_IMAGE:-dsv4v-vllm:vision-54631-fi512b-k5adapt}"
NAME="${NAME:-dsv4v_vllm}"
MODEL_DIR="${DSV4V_MODEL_DIR:-${HOME}/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
WORKER_MODEL_DIR="${DSV4V_WORKER_MODEL_DIR:-${MODEL_DIR}}"
SERVED="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
WORKER_SSH="${WORKER_SSH:?set WORKER_SSH to rank1 user@host}"
HEAD_IP="${HEAD_IP:?set HEAD_IP to rank0 fabric address}"
MASTER_PORT="${MASTER_PORT:-25000}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.875}"
# DSpark draft n_predict follows dspark_block_size=5 on this image, so k=6 is
# illegal here (MiaAI's k=6 is their MTP n_predict=3 rule). k=5 + baked SM121
# adaptive verification is the production depth after SHORT/PROSE WIN vs k=3.
SPEC_TOKENS="${SPEC_TOKENS:-5}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_EP="${ENABLE_EP:-0}"
FP4_INDEXER="${FP4_INDEXER:-0}"
FI_AUTOTUNE="${FI_AUTOTUNE:-1}"
RANK0_ETH_IF="${RANK0_ETH_IF:-enP2p1s0f1np1}"
RANK0_HCA="${RANK0_HCA:-roceP2p1s0f1}"
RANK1_ETH_IF="${RANK1_ETH_IF:-enP2p1s0f0np0}"
RANK1_HCA="${RANK1_HCA:-roceP2p1s0f0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-${HOME}/.cache/dsv4v-flashinfer}"
TRITON_CACHE="${TRITON_CACHE:-${HOME}/.cache/dsv4v-triton}"
TILELANG_CACHE="${TILELANG_CACHE:-${HOME}/.cache/dsv4v-tilelang}"
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-${HOME}/.cache/dsv4v-vllm}"
MEDIA_DIR="${MEDIA_DIR:-${HOME}/sgl-rw/media}"
SPEC_METHOD="${SPEC_METHOD:-probabilistic}"
SPEC_ADAPTIVE="${SPEC_ADAPTIVE:-true}"
ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-1}"
# Throughput default is thinking off. Quality clients send thinking=high per request.
THINKING_DEFAULT="${THINKING_DEFAULT:-off}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-1024}"
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

fail() { echo "ERROR: $*" >&2; exit 1; }

if [[ -z "${COMPILATION_CONFIG_JSON:-}" ]]; then
  COMPILATION_CONFIG_JSON="$(SPEC_TOKENS="${SPEC_TOKENS}" MAX_NUM_SEQS="${MAX_NUM_SEQS}" python3 - <<'PY'
import json, os
k = int(os.environ["SPEC_TOKENS"])
s = int(os.environ["MAX_NUM_SEQS"])
cap = ((s * (k + 1) + 7) // 8) * 8
sizes = [1, 2, 4]
x = 8
while x <= cap:
    sizes.append(x)
    x *= 2
if cap not in sizes:
    sizes.append(cap)
print(json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": sizes}))
PY
)"
fi

if [[ "${THINKING_DEFAULT}" == "off" ]]; then
  CHAT_KWARGS='{"thinking":false}'
else
  CHAT_KWARGS="{\"thinking\":true,\"reasoning_effort\":\"${REASONING_EFFORT}\"}"
fi

serve_cmd() {
  local rank="$1" headless="${2:-}"
  local spec
  spec=$(printf '{"method":"dspark","model":"/model","num_speculative_tokens":%s,"draft_sample_method":"%s","enable_adaptive_verification":%s}' \
    "${SPEC_TOKENS}" "${SPEC_METHOD}" "${SPEC_ADAPTIVE}")
  local -a c=(
    exec vllm serve /model
    --served-model-name "${SERVED}"
    --host 0.0.0.0 --port "${PORT}"
    --trust-remote-code
    --tensor-parallel-size 2 --pipeline-parallel-size 1
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --block-size 256
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}"
  )
  if [[ "${SPEC_TOKENS}" != "0" && "${SPEC_METHOD}" != "none" && "${SPEC_METHOD}" != "off" ]]; then
    c+=(--speculative-config "${spec}")
  fi
  c+=(
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --default-chat-template-kwargs "${CHAT_KWARGS}"
    --compilation-config "${COMPILATION_CONFIG_JSON}"
    --distributed-executor-backend mp
    --nnodes 2 --node-rank "${rank}"
    --master-addr "${HEAD_IP}" --master-port "${MASTER_PORT}"
    --allowed-local-media-path /media
  )
  [[ "${FI_AUTOTUNE}" == "0" ]] && c+=(--no-enable-flashinfer-autotune)
  [[ "${ENFORCE_EAGER}" == "1" ]] && c+=(--enforce-eager)
  [[ "${ENABLE_EP}" == "1" ]] && c+=(--enable-expert-parallel)
  [[ "${FP4_INDEXER}" == "1" ]] && c+=(--attention_config.use_fp4_indexer_cache True)
  [[ "${ENABLE_PREFIX_CACHE}" == "0" ]] && c+=(--no-enable-prefix-caching)
  [[ -n "${headless}" ]] && c+=(--headless)
  printf '%q ' "${c[@]}"
}

if [[ "${SG_DRYRUN:-0}" == "1" ]]; then
  echo "RANK1: docker run IMAGE -lc $(printf '%q' "$(serve_cmd 1 x)")"
  echo "RANK0: docker run IMAGE -lc $(serve_cmd 0 '')"
  exit 0
fi

[[ -d "${MODEL_DIR}" ]] || fail "model dir missing: ${MODEL_DIR}"
[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || fail "model index missing"
ssh -o BatchMode=yes "${WORKER_SSH}" "test -f '${WORKER_MODEL_DIR}/model.safetensors.index.json'" \
  || fail "worker model dir not ready"
LOCAL_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
REMOTE_ID="$(ssh -o BatchMode=yes "${WORKER_SSH}" "docker image inspect '${IMAGE}' --format '{{.Id}}'")"
[[ "${LOCAL_ID}" == "${REMOTE_ID}" ]] || fail "image parity fail: ${LOCAL_ID} != ${REMOTE_ID}"

common=( --gpus all --ipc=host --network host --entrypoint /bin/bash
  --shm-size=64g --ulimit memlock=-1:-1 --ulimit stack=67108864
  --cap-add=IPC_LOCK --device=/dev/infiniband
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e VLLM_USE_AOT_COMPILE=0
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}"
  -e TILELANG_CLEANUP_TEMP_FILES=1
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e TILELANG_CACHE_DIR=/root/.cache/tilelang
  -e VLLM_CACHE_ROOT=/root/.cache/vllm
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  -e NCCL_NVLS_ENABLE=0
  -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm_fp4_block_scale_moe,flashinfer::trtllm_fp4_block_scale_moe"
  -e MASTER_ADDR="${HEAD_IP}" -e MASTER_PORT="${MASTER_PORT}"
)

_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERIFY_UTILS="${DSV4V_VERIFY_UTILS:-${_ROOT}/docker/overlay-cpu-adaptive/utils.py}"
if [[ -n "${DSV4V_VERIFY_PREFIX:-}" ]]; then
  [[ -f "${VERIFY_UTILS}" ]] || fail "DSV4V_VERIFY_PREFIX set but missing ${VERIFY_UTILS}"
  common+=( -e DSV4V_VERIFY_PREFIX="${DSV4V_VERIFY_PREFIX}" )
  common+=( -v "${VERIFY_UTILS}:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/utils.py:ro" )
  echo "CPU verify prefix=${DSV4V_VERIFY_PREFIX} overlay=${VERIFY_UTILS}"
fi

if [[ "${DSV4V_SM121_ADAPTIVE:-0}" == "1" ]]; then
  IDX_OVERLAY="${_ROOT}/docker/overlay-sm121-adaptive/indexer.py"
  MLA_OVERLAY="${_ROOT}/docker/overlay-sm121-adaptive/sparse_mla.py"
  [[ -f "${IDX_OVERLAY}" && -f "${MLA_OVERLAY}" ]] || fail "DSV4V_SM121_ADAPTIVE=1 missing overlay py"
  common+=( -v "${IDX_OVERLAY}:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py:ro" )
  common+=( -v "${MLA_OVERLAY}:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/sparse_mla.py:ro" )
  SWA_OVERLAY="${_ROOT}/docker/overlay-sm121-adaptive/sparse_swa.py"
  [[ -f "${SWA_OVERLAY}" ]] || fail "missing sparse_swa overlay"
  common+=( -v "${SWA_OVERLAY}:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py:ro" )
  FI_OVERLAY="${_ROOT}/docker/overlay-sm121-adaptive/flashinfer_sparse.py"
  [[ -f "${FI_OVERLAY}" ]] || fail "missing flashinfer_sparse overlay"
  common+=( -v "${FI_OVERLAY}:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:ro" )
  echo "SM121 adaptive overlay on"
fi

mkdir -p "${FLASHINFER_CACHE}" "${TRITON_CACHE}" "${TILELANG_CACHE}" "${VLLM_CACHE_HOST}" "${MEDIA_DIR}"
ssh -o BatchMode=yes "${WORKER_SSH}" "mkdir -p '${FLASHINFER_CACHE}' '${TRITON_CACHE}' '${TILELANG_CACHE}' '${VLLM_CACHE_HOST}' '${MEDIA_DIR}'"
docker rm -f "${NAME}" 2>/dev/null || true
ssh -o BatchMode=yes "${WORKER_SSH}" "docker rm -f '${NAME}' 2>/dev/null || true"

echo "== rank1 (worker) =="
ssh -o BatchMode=yes "${WORKER_SSH}" docker run -d --name "${NAME}" \
  "${common[@]}" \
  -v "${WORKER_MODEL_DIR}:/model:ro" \
  -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
  -v "${TRITON_CACHE}:/root/.cache/triton" \
  -v "${TILELANG_CACHE}:/root/.cache/tilelang" \
  -v "${VLLM_CACHE_HOST}:/root/.cache/vllm" \
  -v "${MEDIA_DIR}:/media" \
  -e NCCL_IB_HCA="${RANK1_HCA}" -e NCCL_SOCKET_IFNAME="${RANK1_ETH_IF}" \
  "${LOCAL_ID}" -lc "$(printf '%q' "$(serve_cmd 1 x)")"

echo "== rank0 (head) =="
docker run -d --name "${NAME}" \
  "${common[@]}" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
  -v "${TRITON_CACHE}:/root/.cache/triton" \
  -v "${TILELANG_CACHE}:/root/.cache/tilelang" \
  -v "${VLLM_CACHE_HOST}:/root/.cache/vllm" \
  -v "${MEDIA_DIR}:/media" \
  -e NCCL_IB_HCA="${RANK0_HCA}" -e NCCL_SOCKET_IFNAME="${RANK0_ETH_IF}" \
  "${LOCAL_ID}" -lc "$(serve_cmd 0 '')"

echo "API: http://${HEAD_IP}:${PORT}/v1/models"
echo "profile: k=${SPEC_TOKENS} len=${MAX_MODEL_LEN} thinking=${THINKING_DEFAULT} breakable=0 long_prefill=${LONG_PREFILL_TOKEN_THRESHOLD}"
echo "graphs: ${COMPILATION_CONFIG_JSON}"
