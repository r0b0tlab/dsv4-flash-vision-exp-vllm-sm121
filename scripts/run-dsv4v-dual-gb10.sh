#!/usr/bin/env bash
# DSV4-Flash-Vision-Exp dual-GB10 TP=2 launcher — node2 (rank0) + node3 (rank1)
# vLLM v0.28.0 (2cf0a691). Run this ON node2. All flags verified against v0.28.0 tree.
set -euo pipefail

IMAGE="${DSV4V_IMAGE:-dsv4v-vllm:v0.28.0-sm121}"
NAME="${NAME:-dsv4v_vllm}"
MODEL_DIR="${DSV4V_MODEL_DIR:-${HOME}/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
WORKER_MODEL_DIR="${DSV4V_WORKER_MODEL_DIR:-${MODEL_DIR}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
WORKER_SSH="${WORKER_SSH:-r0b0tdgx@192.168.5.1}"  # node2, from node3 over the direct lane
HEAD_IP="${HEAD_IP:-192.168.5.2}"                # node3 lane addr — rank0 rendezvous
MASTER_PORT="${MASTER_PORT:-25000}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-327680}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.835}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
SPEC_TOKENS="${SPEC_TOKENS:-6}"   # must be divisible by num_nextn_predict_layers=3 (engine rule); 6 = v026 production K
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Stock v0.28.0: {"cudagraph_mode":"FULL"} (NONE|PIECEWISE|FULL).
# If the v026 knownfix selector port lands, override with:
#   COMPILATION_CONFIG_JSON='{"cudagraph_implementation":"regular","cudagraph_strict":true}'
COMPILATION_CONFIG_JSON="${COMPILATION_CONFIG_JSON:-{\"cudagraph_mode\":\"FULL\"}}"
# lane interfaces (verified 2026-08-31): per-rank — on node2 the 5.x lane is P2p1s0f0,
# on node3 it is P2p1s0f1. rank0 = the node this script runs on.
RANK0_ETH_IF="${RANK0_ETH_IF:-enP2p1s0f1np1}"
RANK0_HCA="${RANK0_HCA:-roceP2p1s0f1}"
RANK1_ETH_IF="${RANK1_ETH_IF:-enP2p1s0f0np0}"
RANK1_HCA="${RANK1_HCA:-roceP2p1s0f0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-${HOME}/.cache/dsv4v-flashinfer}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ "${KV_CACHE_DTYPE}" =~ ^(nvfp4|nvfp4_4over6|fp8_ds_mla)$ ]] || fail "KV_CACHE_DTYPE must be nvfp4|nvfp4_4over6|fp8_ds_mla"
[[ "${MOE_BACKEND}" == "flashinfer_b12x" ]] || fail "MOE_BACKEND must be flashinfer_b12x"
[[ "${ENFORCE_EAGER}" =~ ^[01]$ ]] || fail "ENFORCE_EAGER must be 0 or 1"
[[ -d "${MODEL_DIR}" ]] || fail "model dir not found: ${MODEL_DIR}"
[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || fail "model index missing — pull incomplete"
ssh -o BatchMode=yes "${WORKER_SSH}" "test -f '${WORKER_MODEL_DIR}/model.safetensors.index.json'" \
  || fail "worker model dir not ready on ${WORKER_SSH}"

KV_BYTES_ARG=()
[[ -n "${KV_CACHE_MEMORY_BYTES}" ]] && KV_BYTES_ARG=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
EAGER_ARG=()
[[ "${ENFORCE_EAGER}" == "1" ]] && EAGER_ARG=(--enforce-eager)

verify_model_py() {
cat <<'PY'
import json, sys, os
d = sys.argv[1]
idx = json.load(open(f"{d}/model.safetensors.index.json"))
need = set(idx["weight_map"].values())
have = {f for f in os.listdir(d) if f.endswith(".safetensors")}
missing = need - have
if missing:
    print(f"MISSING_SHARDS {sorted(missing)[:4]}"); sys.exit(1)
print(f"MODEL_OK shards={len(have)}")
PY
}

verify_model_py | ssh -o BatchMode=yes "${WORKER_SSH}" "python3 - '${WORKER_MODEL_DIR}'" | grep -q '^MODEL_OK' \
  || fail "worker model verify failed"
verify_model_py > /tmp/vm.py.local
python3 /tmp/vm.py.local "${MODEL_DIR}" | grep -q '^MODEL_OK' || fail "rank0 model verify failed"

LOCAL_IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
REMOTE_IMAGE_ID="$(ssh -o BatchMode=yes "${WORKER_SSH}" "docker image inspect '${IMAGE}' --format '{{.Id}}'")"
[[ "${LOCAL_IMAGE_ID}" == "${REMOTE_IMAGE_ID}" ]] \
  || fail "rank image IDs differ: ${LOCAL_IMAGE_ID} != ${REMOTE_IMAGE_ID} — run the transfer gate"
RUNTIME_IMAGE_REF="${LOCAL_IMAGE_ID}"

common_args=(
  --gpus all --ipc=host --network host
  --entrypoint /bin/bash
  --shm-size=64g
  --ulimit memlock=-1:-1 --ulimit stack=67108864:-1
  --cap-add=IPC_LOCK --device=/dev/infiniband
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
  -e VLLM_USE_AOT_COMPILE=0
  -e VLLM_DISABLE_COMPILE_CACHE=1
  -e VLLM_DSV4_ENABLE_MULTI_STREAM=0
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
  -e TILELANG_CLEANUP_TEMP_FILES=1
  -e DG_JIT_USE_NVRTC=0 -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  -e NCCL_NVLS_ENABLE=0 -e NCCL_RAS_ENABLE=1
  -e VLLM_TRITON_MLA_SPARSE=1
  -e MASTER_ADDR="${HEAD_IP}" -e MASTER_PORT="${MASTER_PORT}"
)

serve_cmd() {
  local rank="$1" headless="$2"
  local -a c=(
    exec vllm serve /model
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0 --port "${PORT}"
    --trust-remote-code
    --tensor-parallel-size 2 --pipeline-parallel-size 1
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --block-size 256
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --enable-prefix-caching
    --speculative-config '{"method":"dspark","num_speculative_tokens":'"${SPEC_TOKENS}"',"draft_sample_method":"greedy"}'
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --default-chat-template-kwargs '{"thinking":true}'
    --compilation-config "${COMPILATION_CONFIG_JSON}"
    --moe-backend "${MOE_BACKEND}"
    --distributed-executor-backend mp
    --nnodes 2 --node-rank "${rank}"
    --master-addr "${HEAD_IP}" --master-port "${MASTER_PORT}"
  )
  [[ -n "${KV_CACHE_MEMORY_BYTES}" ]] && c+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
  [[ "${ENFORCE_EAGER}" == "1" ]] && c+=(--enforce-eager)
  [[ -n "${headless}" ]] && c+=(--headless)
  printf '%q ' "${c[@]}"
}

echo "== image identity: ${RUNTIME_IMAGE_REF} (both ranks)"
echo "== stopping previous ${NAME} =="
docker rm -f "${NAME}" 2>/dev/null || true
ssh -o BatchMode=yes "${WORKER_SSH}" "docker rm -f '${NAME}' 2>/dev/null || true"
mkdir -p "${FLASHINFER_CACHE}"
ssh -o BatchMode=yes "${WORKER_SSH}" "mkdir -p '${FLASHINFER_CACHE}'"

echo "== starting rank1 worker (${WORKER_SSH}) =="
ssh -o BatchMode=yes "${WORKER_SSH}" docker run -d --name "${NAME}" \
  "${common_args[@]}" \
  -v "${WORKER_MODEL_DIR}:/model:ro" \
  -v "${FLASHINFER_CACHE}:/cache/flashinfer-workspace" \
  -e NODE_RANK=1 \
  -e NCCL_IB_HCA="${RANK1_HCA}" \
  -e NCCL_SOCKET_IFNAME="${RANK1_ETH_IF}" \
  "${RUNTIME_IMAGE_REF}" -lc "$(serve_cmd 1 x "${RANK1_ETH_IF}" "${RANK1_HCA}")"

echo "== starting rank0 head (this node, ${HEAD_IP}) =="
docker run -d --name "${NAME}" \
  "${common_args[@]}" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${FLASHINFER_CACHE}:/cache/flashinfer-workspace" \
  -e NODE_RANK=0 \
  -e NCCL_IB_HCA="${RANK0_HCA}" \
  -e NCCL_SOCKET_IFNAME="${RANK0_ETH_IF}" \
  "${RUNTIME_IMAGE_REF}" -lc "$(serve_cmd 0 '' "${RANK0_ETH_IF}" "${RANK0_HCA}")"

echo "API: http://${HEAD_IP}:${PORT}/v1/models  (also via mgmt: http://192.168.3.2:${PORT})"
echo "Logs: docker logs -f ${NAME} (node3/rank0) ; ssh ${WORKER_SSH} docker logs -f ${NAME} (node2/rank1)"
