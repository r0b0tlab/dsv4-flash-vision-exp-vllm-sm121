vLLM control lane torn down 2026-09-01 ~14:00Z to free both GPUs for SGLang Track T.
Evidence already captured: evidence/VERDICT.md, BRINGUP-VERDICT.md, cladder/c{1,2,4}.json, niah-262k.json, niah-multikey.json.
Relaunch: node3 `~/projects/DeepSeek-V4-Flash-Vision-Exp/scripts/run-dsv4v-dual-gb10.sh` with DSV4V_IMAGE=dsv4v-vllm:v0.28.0-sm121-visionfix KV_CACHE_DTYPE=fp8_ds_mla MAX_MODEL_LEN=262144 GPU_MEMORY_UTILIZATION=0.875.
