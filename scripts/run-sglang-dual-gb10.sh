#!/usr/bin/env bash
# SGLang DSV4-Flash-Vision-Exp dual-GB10 TP=2 — vision-capable + reasoning=high (v3)
# rank0/rank1 via WORKER_SSH and HEAD_IP (required).
# SG_DRYRUN=1 prints the two docker commands and exits 0 (used by tests).
# Do NOT apply patch_sglang_trackt.py (skip-list) on the vision image.
set -euo pipefail

IMAGE="${SG_IMAGE:-lmsysorg/sglang:dev-dsv4-flash-vision}"
NAME="${NAME:-sglang_dsv4v}"
MODEL_DIR="${SG_MODEL_DIR:-${HOME}/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
WORKER_MODEL_DIR="${SG_WORKER_MODEL_DIR:-${MODEL_DIR}}"
SERVED="${SERVED:-deepseek-v4-flash-vision-exp}"
WORKER_SSH="${WORKER_SSH:?set WORKER_SSH to rank1 user@host}"
HEAD_IP="${HEAD_IP:?set HEAD_IP to rank0 fabric address}"
DIST_PORT="${DIST_PORT:-25001}"
PORT="${PORT:-30000}"
TP="${TP:-2}"
CTX="${CTX:-1048576}"
MEM_STATIC="${MEM_STATIC:-0.85}"
SPEC_ALGO="${SPEC_ALGO:-DSPARK}"
SG_MAX_RUNNING="${SG_MAX_RUNNING:-8}"
SG_GRAPH_BS="${SG_GRAPH_BS:-1 2 3 4 6 8}"
SG_CHUNK="${SG_CHUNK:-8192}"
SG_MAX_PREFILL="${SG_MAX_PREFILL:-16384}"
SG_WATCHDOG="${SG_WATCHDOG:-1800}"
SG_REASONING_PARSER="${SG_REASONING_PARSER:-deepseek-v4}"
SG_TOOL_PARSER="${SG_TOOL_PARSER:-deepseekv4}"
SG_DEFAULT_THINKING="${SG_DEFAULT_THINKING:-1}"
SG_REASONING_EFFORT="${SG_REASONING_EFFORT:-high}"
SG_RAGGED="${SG_RAGGED:-compact}"
SG_FUSED_MARKOV="${SG_FUSED_MARKOV:-1}"
SG_STS_TABLE="${SG_STS_TABLE:-}"
SG_STS_COLLECT="${SG_STS_COLLECT:-}"
SG_COMPRESS_DTYPE="${SG_COMPRESS_DTYPE:-}"
SG_TOPK_BACKEND="${SG_TOPK_BACKEND:-}"
SG_TF32="${SG_TF32:-0}"
SG_MIXED_CHUNK="${SG_MIXED_CHUNK:-0}"
SG_FP4_INDEXER="${SG_FP4_INDEXER:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NCCL_GID="${NCCL_GID:-3}"

fail() { echo "ERROR: $*" >&2; exit 1; }

server_cmd() {
  local rank="$1"
  local -a c=(
    python3 -m sglang.launch_server --model-path /model --trust-remote-code
    --served-model-name "${SERVED}"
    --attention-backend dsv4
    --tp-size "${TP}" --nnodes 2 --node-rank "${rank}"
    --dist-init-addr "${HEAD_IP}:${DIST_PORT}"
    --context-length "${CTX}"
    --mem-fraction-static "${MEM_STATIC}"
    --max-running-requests "${SG_MAX_RUNNING}"
    --chunked-prefill-size "${SG_CHUNK}" --max-prefill-tokens "${SG_MAX_PREFILL}"
    --watchdog-timeout "${SG_WATCHDOG}"
    --reasoning-parser "${SG_REASONING_PARSER}" --tool-call-parser "${SG_TOOL_PARSER}"
    --host 0.0.0.0 --port "${PORT}"
  )
  if [[ -n "${SPEC_ALGO}" && "${SPEC_ALGO}" != "none" && "${SPEC_ALGO}" != "off" ]]; then
    c+=(--speculative-algorithm "${SPEC_ALGO}")
  fi
  if [[ "${SG_DISABLE_GRAPH:-0}" == "1" ]]; then
    c+=(--disable-cuda-graph)
  else
    # shellcheck disable=SC2206
    local -a bs=( ${SG_GRAPH_BS} )
    c+=(--cuda-graph-bs "${bs[@]}")
  fi
  [[ -n "${SG_STS_TABLE}" ]]   && c+=(--speculative-dspark-confidence-sts-path "${SG_STS_TABLE}")
  [[ -n "${SG_TOPK_BACKEND}" ]] && c+=(--dsa-topk-backend "${SG_TOPK_BACKEND}" --speculative-dsa-topk-backend "${SG_TOPK_BACKEND}")
  [[ "${SG_TF32}" == "1" ]]        && c+=(--enable-tf32-matmul)
  [[ "${SG_MIXED_CHUNK}" == "1" ]] && c+=(--enable-mixed-chunk)
  [[ "${SG_FP4_INDEXER}" == "1" ]] && c+=(--enable-deepseek-v4-fp4-indexer)
  if [[ -n "${EXTRA_ARGS}" ]]; then read -r -a extra <<< "${EXTRA_ARGS}"; c+=("${extra[@]}"); fi
  local joined; joined="$(printf '%q ' "${c[@]}")"
  if [[ "${SG_REMOTE_FORM:-0}" == "1" ]]; then printf "'%s'" "${joined}"; else printf '%s' "${joined}"; fi
}

common=( --gpus all --ipc=host --network host --shm-size=64g
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add=IPC_LOCK
  --device=/dev/infiniband
  -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e NCCL_IB_GID_INDEX="${NCCL_GID}"
  -e SGLANG_USE_DEEPGEMM=1
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  -e SGLANG_RAGGED_VERIFY_MODE="${SG_RAGGED}"
  -e SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV="${SG_FUSED_MARKOV}"
  -e SGLANG_DEFAULT_THINKING="${SG_DEFAULT_THINKING}"
  -e SGLANG_DSV4_REASONING_EFFORT="${SG_REASONING_EFFORT}"
  -e MAX_JOBS="${MAX_JOBS:-4}"
  -e FLASHINFER_NVCC_THREADS=2
  -e NINJAFLAGS=-j4 )
[[ -n "${SG_STS_COLLECT}" ]]   && common+=(-e SGLANG_DSPARK_STS_COLLECT_PATH="${SG_STS_COLLECT}")
[[ -n "${SG_COMPRESS_DTYPE}" ]] && common+=(-e SGLANG_DSV4_COMPRESS_STATE_DTYPE="${SG_COMPRESS_DTYPE}")

if [[ "${SG_DRYRUN:-0}" == "1" ]]; then
  echo "RANK1: docker run ${common[*]} ${IMAGE} bash -c $(SG_REMOTE_FORM=1 server_cmd 1)"
  echo "RANK0: docker run ${common[*]} ${IMAGE} bash -c $(server_cmd 0)"
  exit 0
fi

[[ -d "${MODEL_DIR}" ]] || fail "model dir missing: ${MODEL_DIR}"
[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || fail "model index missing"
ssh -o BatchMode=yes "${WORKER_SSH}" "test -f '${WORKER_MODEL_DIR}/model.safetensors.index.json'" || fail "worker model dir not ready"
LOCAL_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
REMOTE_ID="$(ssh -o BatchMode=yes "${WORKER_SSH}" "docker image inspect '${IMAGE}' --format '{{.Id}}'")"
[[ "${LOCAL_ID}" == "${REMOTE_ID}" ]] || fail "image parity fail: ${LOCAL_ID} != ${REMOTE_ID}"
if [[ -n "${SG_STS_TABLE}" ]]; then
  test -f "${HOME}/sgl-extras/$(basename "${SG_STS_TABLE}")" || fail "STS table missing on rank0"
  ssh -o BatchMode=yes "${WORKER_SSH}" "test -f '${HOME}/sgl-extras/$(basename "${SG_STS_TABLE}")'" || fail "STS table missing on rank1"
fi
mkdir -p "${HOME}/sgl-extras" "${HOME}/sgl-rw"
ssh -o BatchMode=yes "${WORKER_SSH}" "mkdir -p ~/sgl-extras ~/sgl-rw"

docker rm -f "${NAME}" 2>/dev/null || true
ssh -o BatchMode=yes "${WORKER_SSH}" "docker rm -f '${NAME}' 2>/dev/null || true"

echo "== rank1 (node2) =="
ssh -o BatchMode=yes "${WORKER_SSH}" docker run -d --name "${NAME}" "${common[@]}" \
  -e NCCL_IB_HCA=roceP2p1s0f0 -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 \
  -v "${WORKER_MODEL_DIR}:/model:ro" -v "${HOME}/sgl-extras:/sgl-extras:ro" -v "${HOME}/sgl-rw:/sgl-rw" -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
  "${IMAGE}" bash -c "$(SG_REMOTE_FORM=1 server_cmd 1)"

echo "== rank0 (node3) =="
docker run -d --name "${NAME}" "${common[@]}" \
  -e NCCL_IB_HCA=roceP2p1s0f1 -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -v "${MODEL_DIR}:/model:ro" -v "${HOME}/sgl-extras:/sgl-extras:ro" -v "${HOME}/sgl-rw:/sgl-rw" -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
  "${IMAGE}" bash -c "$(server_cmd 0)"

{
  echo "CTX=${CTX}"
  echo "MEM_STATIC=${MEM_STATIC}"
  echo "SG_MAX_RUNNING=${SG_MAX_RUNNING}"
  echo "SG_GRAPH_BS=${SG_GRAPH_BS}"
  echo "SG_CHUNK=${SG_CHUNK}"
  echo "SG_MAX_PREFILL=${SG_MAX_PREFILL}"
  echo "SG_WATCHDOG=${SG_WATCHDOG}"
  echo "SG_RAGGED=${SG_RAGGED}"
  echo "SG_FUSED_MARKOV=${SG_FUSED_MARKOV}"
  echo "SG_DEFAULT_THINKING=${SG_DEFAULT_THINKING}"
  echo "SG_REASONING_EFFORT=${SG_REASONING_EFFORT}"
  echo "SG_STS_TABLE=${SG_STS_TABLE}"
  echo "SG_STS_COLLECT=${SG_STS_COLLECT}"
  echo "SG_COMPRESS_DTYPE=${SG_COMPRESS_DTYPE}"
  echo "SG_TOPK_BACKEND=${SG_TOPK_BACKEND}"
  echo "SG_TF32=${SG_TF32}"
  echo "SG_MIXED_CHUNK=${SG_MIXED_CHUNK}"
  echo "SG_FP4_INDEXER=${SG_FP4_INDEXER}"
  echo "SPEC_ALGO=${SPEC_ALGO}"
  echo "SG_DISABLE_GRAPH=${SG_DISABLE_GRAPH:-0}"
  echo "EXTRA_ARGS=${EXTRA_ARGS}"
} > "${HOME}/sgl-rw/last-launch.env"
echo "API: http://${HEAD_IP}:${PORT}/v1/models"
echo "logs: docker logs -f ${NAME} (here) | ssh ${WORKER_SSH} docker logs -f ${NAME}"
