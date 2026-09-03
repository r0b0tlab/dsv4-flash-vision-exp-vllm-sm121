#!/usr/bin/env bash
# Publication evals on the live baked 512k k=5 adaptive serve.
# Do not SIGKILL this client.
set -euo pipefail
ROOT="${ROOT:?set ROOT to the repo root}"
EV="${EV:-${ROOT}/evidence/vision-opt/V0/prod-512k-k5-adapt}"
BASE="${BASE:?set BASE to the serve base URL, e.g. http://<rank0-ip>:8000}"
export Q200V2_RUNNER="${Q200V2_RUNNER:?set Q200V2_RUNNER to the frozen q200v2 runner root}"
IMAGE_ID="${IMAGE_ID:-sha256:f8b73f965834ff8439bf266827229ef71bfdf291689e05b63dc59a74003b517d}"
PROFILE_ID="${PROFILE_ID:-adbdc0a89e3eb48d281d7c5fb6bf0e8adf98e073fa26e6fe8ca986b56be9daab}"
CANDIDATE_ID="${CANDIDATE_ID:-vision-54631-fi512b-k5adapt}"
mkdir -p "${EV}"
cd "${EV}"

echo "==== Q200v2 text180 $(date -Is) ===="
python3 "${ROOT}/scripts/run_q200v2_dsv4.py" \
  --base-url "${BASE}" \
  --run-id "${EV}/q200v2-text180" \
  --model deepseek-v4-flash-vision-exp \
  --image-id "${IMAGE_ID}" \
  --profile-id "${PROFILE_ID}" \
  --candidate-id "${CANDIDATE_ID}" \
  --workers 1 \
  --max-tokens 8192 \
  --timeout 1800 \
  > "${EV}/q200v2-text180.stdout" 2> "${EV}/q200v2-text180.stderr"
echo "Q200_TEXT_RC=$?"

echo "==== BFCL hard20 $(date -Is) ===="
BFCL_ROOT="${EV}/bfcl-hard20-run"
rm -rf "${BFCL_ROOT}"
mkdir -p "${BFCL_ROOT}"
export BFCL_PROJECT_ROOT="${BFCL_ROOT}"
export OPENAI_BASE_URL="${BASE}/v1"
export OPENAI_API_KEY="EMPTY"
export Q200_SERVED_MODEL="deepseek-v4-flash-vision-exp"
export Q200_IMAGE_ID="${IMAGE_ID}"
export Q200_PROFILE_ID="${PROFILE_ID}"
export Q200_CANDIDATE_ID="${CANDIDATE_ID}"
export Q200_BFCL_TIMING_PATH="${EV}/bfcl-hard20-timing.json"
export Q200_BFCL_REGISTRY="${Q200_BFCL_REGISTRY:-qwen38-flash-next-hard20-FC}"
export BFCL_NUM_THREADS=1
export BFCL_HTTP_TIMEOUT=3600
export BFCL_MAX_TOKENS=8192
# Do NOT pre-create the timing sidecar: run mode requires it absent.
rm -f "${Q200_BFCL_TIMING_PATH}"
python3 "${ROOT}/scripts/run_bfcl_hard20_dsv4.py" run \
  > "${EV}/bfcl-hard20.stdout" 2> "${EV}/bfcl-hard20.stderr"
echo "BFCL_RC=$?"

echo "==== NIAH 524032 $(date -Is) ===="
python3 "${ROOT}/scripts/run-niah-advertised.py" \
  --base-url "${BASE}" \
  --output "${EV}/niah-524288.json" \
  --target-tokens 524032 \
  > "${EV}/niah.stdout" 2> "${EV}/niah.stderr"
echo "NIAH_RC=$?"
echo "==== DONE $(date -Is) ===="