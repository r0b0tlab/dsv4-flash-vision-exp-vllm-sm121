# DSV4-Flash-Vision-Exp dual-GB10 run-ready package (WIP)

Do not stop the live serve to use this package. Live: `http://<node3>:8000` max_model_len=524288.

## Identity
- Image: `dsv4v-vllm:vision-54631-fi512b` `sha256:c4c0d7b269b2b0af0d3edaf92cfab1add8644d99eb518f6dbf3e99511ea4e84f` arm64
- Image size: 22,261,278,770 bytes
- Checkpoint (in-container `/model`): 167,831,876,729 bytes (~156.3 GiB)
- KV: 1,207,784 tokens (2.30× at 524288)
- Spec: DSpark K=3 probabilistic, adaptive=false, thinking=high
- Graphs: FULL_DECODE_ONLY capture [1,2,4,8]; FlashInfer autotune ON (skip FP4 MoE op only)

## Launch
From node3 (rank0), after loading the tar and placing weights:

```
DSV4V_IMAGE=dsv4v-vllm:vision-54631-fi512b MAX_MODEL_LEN=524288 \
SPEC_TOKENS=3 SPEC_METHOD=probabilistic REASONING_EFFORT=high \
SPEC_ADAPTIVE=false GPU_MEMORY_UTILIZATION=0.875 \
bash scripts/run-vllm-vision-dual-gb10.sh
```

Weights are NOT in the tar. Mount `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.

## Gates so far
- Vision short 8/8 PASS (synthetic PNG exact-match)
- Vision canary PASS
- SHORT c1 ~31 tok/s at 512k (not the SGLang-text 43.61 floor)
- Q200v2 text180: 180/180 transport; e2e mean 39.39 tok/s agg 41.65; grader INCOMPLETE (humaneval sandbox errors). BFCL-hard20 not run in this cut.
- NIAH advertised 524288 (25/50/90 + multi-key 33/66): **NIAH_PASS** (thinking=false on retrieval requests only)

Live serve left running. Do not docker rm.

