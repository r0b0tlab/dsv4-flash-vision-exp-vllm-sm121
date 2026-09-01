#!/usr/bin/env bash
# SGLang DSV4-Flash-Vision-Exp dual-GB10 TP=2 — Track T (stock fp8 KV, packaged DSpark)
# rank0 = node3 (192.168.5.2), rank1 = node2 (192.168.5.1), run ON node3.
set -euo pipefail

IMAGE="${SG_IMAGE:-sglang-dsv4v:0.5.19-vision}"
NAME="${NAME:-sglang_dsv4v}"
MODEL_DIR="${SG_MODEL_DIR:-${HOME}/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
WORKER_MODEL_DIR="${SG_WORKER_MODEL_DIR:-${MODEL_DIR}}"
SERVED="${SERVED:-deepseek-v4-flash-vision-exp}"
WORKER_SSH="${WORKER_SSH:-r0b0tdgx@192.168.5.1}"
HEAD_IP="${HEAD_IP:-192.168.5.2}"
DIST_PORT="${DIST_PORT:-25001}"
PORT="${PORT:-30000}"
TP="${TP:-2}"
CTX="${CTX:-262144}"
MEM_STATIC="${MEM_STATIC:-0.85}"
# DSpark: bundled in these weights; gamma=5 from dspark_block_size -> draft tokens 6
SPEC_ALGO="${SPEC_ALGO:-DSPARK}"
SPEC_GAMMA="${SPEC_GAMMA:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NCCL_GID="${NCCL_GID:-3}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "${MODEL_DIR}" ]] || fail "model dir missing: ${MODEL_DIR}"
[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || fail "model index missing"
ssh -o BatchMode=yes "${WORKER_SSH}" "test -f '${WORKER_MODEL_DIR}/model.safetensors.index.json'" \
  || fail "worker model dir not ready"
LOCAL_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
REMOTE_ID="$(ssh -o BatchMode=yes "${WORKER_SSH}" "docker image inspect '${IMAGE}' --format '{{.Id}}'")"
[[ "${LOCAL_ID}" == "${REMOTE_ID}" ]] || fail "image parity fail: ${LOCAL_ID} != ${REMOTE_ID}"

server_cmd() {
  local rank="$1" headless="$2"
  local -a c=(
    python3 -m sglang.launch_server --model-path /model --trust-remote-code
    --served-model-name "${SERVED}"
    --attention-backend dsv4
    --tp-size "${TP}" --nnodes 2 --node-rank "${rank}"
    --dist-init-addr "${HEAD_IP}:${DIST_PORT}"
    --context-length "${CTX}"
    --mem-fraction-static "${MEM_STATIC}"
    --host 0.0.0.0 --port "${PORT}"
    --speculative-algorithm "${SPEC_ALGO}"
    --speculative-dspark-block-size "${SPEC_GAMMA}"
  )
  [[ -n "${headless}" ]] && c+=(--headless)
  if [[ -n "${EXTRA_ARGS}" ]]; then read -r -a extra <<< "${EXTRA_ARGS}"; c+=("${extra[@]}"); fi
  printf '%q ' "${c[@]}"
}

docker rm -f "${NAME}" 2>/dev/null || true
ssh -o BatchMode=yes "${WORKER_SSH}" "docker rm -f '${NAME}' 2>/dev/null || true"

common=( --gpus all --ipc=host --network host --shm-size=64g
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add=IPC_LOCK
  --device=/dev/infiniband
  -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e NCCL_IB_GID_INDEX="${NCCL_GID}"
  -e NCCL_IB_HCA=roceP2p1s0f0
  -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0
  -e SGLANG_USE_DEEPGEMM=1 )

echo "== rank1 (node2) =="
ssh -o BatchMode=yes "${WORKER_SSH}" docker run -d --name "${NAME}" \
  "${common[@]}" \
  -e NCCL_IB_HCA=roceP2p1s0f0 -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 \
  -v "${WORKER_MODEL_DIR}:/model:ro" \
  "${IMAGE}" bash -c "$(server_cmd 1 --headless)"

echo "== rank0 (node3) =="
docker run -d --name "${NAME}" \
  "${common[@]}" \
  -e NCCL_IB_HCA=roceP2p1s0f1 -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -v "${MODEL_DIR}:/model:ro" \
  "${IMAGE}" bash -c "$(server_cmd 0 '')"

echo "API: http://${HEAD_IP}:${PORT}/v1/models (mgmt: http://192.168.3.2:${PORT})"
echo "logs: docker logs -f ${NAME} (here) | ssh ${WORKER_SSH} docker logs -f ${NAME}"
