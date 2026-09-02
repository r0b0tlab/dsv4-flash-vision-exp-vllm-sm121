#!/usr/bin/env bash
# DSV4-Flash-Vision-Exp dual-GB10 TP=2 — official vision image + PR#54631 overlay.
# rank0 = node3 (192.168.5.2), rank1 = node2 (192.168.5.1). Run ON node3.
# SG_DRYRUN=1 prints both docker commands and exits 0.
set -euo pipefail

IMAGE="${DSV4V_IMAGE:-dsv4v-vllm:vision-54631}"
NAME="${NAME:-dsv4v_vllm}"
MODEL_DIR="${DSV4V_MODEL_DIR:-${HOME}/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
WORKER_MODEL_DIR="${DSV4V_WORKER_MODEL_DIR:-${MODEL_DIR}}"
SERVED="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
WORKER_SSH="${WORKER_SSH:-r0b0tdgx@192.168.5.1}"
HEAD_IP="${HEAD_IP:-192.168.5.2}"
MASTER_PORT="${MASTER_PORT:-25000}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.875}"
SPEC_TOKENS="${SPEC_TOKENS:-3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_EP="${ENABLE_EP:-0}"
FP4_INDEXER="${FP4_INDEXER:-0}"
# Do not put `}` inside ${var:-default} — bash ends the expansion at the first `}`.
if [[ -z "${COMPILATION_CONFIG_JSON:-}" ]]; then
  COMPILATION_CONFIG_JSON='{"cudagraph_mode":"FULL"}'
fi
RANK0_ETH_IF="${RANK0_ETH_IF:-enP2p1s0f1np1}"
RANK0_HCA="${RANK0_HCA:-roceP2p1s0f1}"
RANK1_ETH_IF="${RANK1_ETH_IF:-enP2p1s0f0np0}"
RANK1_HCA="${RANK1_HCA:-roceP2p1s0f0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-${HOME}/.cache/dsv4v-flashinfer}"
MEDIA_DIR="${MEDIA_DIR:-${HOME}/sgl-rw/media}"
SPEC_METHOD="${SPEC_METHOD:-probabilistic}"
SPEC_ADAPTIVE="${SPEC_ADAPTIVE:-true}"

fail() { echo "ERROR: $*" >&2; exit 1; }

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
    --speculative-config "${spec}"
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"max"}'
    --compilation-config "${COMPILATION_CONFIG_JSON}"
    --distributed-executor-backend mp
    --nnodes 2 --node-rank "${rank}"
    --master-addr "${HEAD_IP}" --master-port "${MASTER_PORT}"
    --allowed-local-media-path /media
    --no-enable-flashinfer-autotune
  )
  [[ "${ENFORCE_EAGER}" == "1" ]] && c+=(--enforce-eager)
  [[ "${ENABLE_EP}" == "1" ]] && c+=(--enable-expert-parallel)
  [[ "${FP4_INDEXER}" == "1" ]] && c+=(--attention_config.use_fp4_indexer_cache True)
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
  -e TILELANG_CLEANUP_TEMP_FILES=1
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

mkdir -p "${FLASHINFER_CACHE}" "${MEDIA_DIR}"
ssh -o BatchMode=yes "${WORKER_SSH}" "mkdir -p '${FLASHINFER_CACHE}' '${MEDIA_DIR}'"
docker rm -f "${NAME}" 2>/dev/null || true
ssh -o BatchMode=yes "${WORKER_SSH}" "docker rm -f '${NAME}' 2>/dev/null || true"

echo "== rank1 (node2) =="
ssh -o BatchMode=yes "${WORKER_SSH}" docker run -d --name "${NAME}" \
  "${common[@]}" \
  -v "${WORKER_MODEL_DIR}:/model:ro" \
  -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
  -v "${MEDIA_DIR}:/media" \
  -e NCCL_IB_HCA="${RANK1_HCA}" -e NCCL_SOCKET_IFNAME="${RANK1_ETH_IF}" \
  "${LOCAL_ID}" -lc "$(printf '%q' "$(serve_cmd 1 x)")"

echo "== rank0 (node3) =="
docker run -d --name "${NAME}" \
  "${common[@]}" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
  -v "${MEDIA_DIR}:/media" \
  -e NCCL_IB_HCA="${RANK0_HCA}" -e NCCL_SOCKET_IFNAME="${RANK0_ETH_IF}" \
  "${LOCAL_ID}" -lc "$(serve_cmd 0 '')"

echo "API: http://${HEAD_IP}:${PORT}/v1/models (mgmt: http://192.168.3.2:${PORT})"
