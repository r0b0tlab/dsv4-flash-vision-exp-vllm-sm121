# DeepSeek-V4-Flash-Vision-Exp · vLLM · dual GB10

r0b0tlab (@mr_r0b0t). Vision + 512k context on 2× NVIDIA GB10. Speculative decoding k=3, thinking=high, fp8 KV.

Results: `publication/html/index.html`

## Click-run

Measured image (arm64):

```
ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121:vision-54631-fi512b
```

Image id `sha256:c4c0d7b269b2b0af0d3edaf92cfab1add8644d99eb518f6dbf3e99511ea4e84f`

Weights are not in the image. Place `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on both ranks, then on rank0:

```bash
docker pull ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121:vision-54631-fi512b
docker tag ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121:vision-54631-fi512b dsv4v-vllm:vision-54631-fi512b

WORKER_SSH=user@rank1 HEAD_IP=<rank0-fabric-ip> \
DSV4V_IMAGE=dsv4v-vllm:vision-54631-fi512b \
MAX_MODEL_LEN=524288 SPEC_TOKENS=3 REASONING_EFFORT=high \
GPU_MEMORY_UTILIZATION=0.875 \
bash scripts/run-vllm-vision-dual-gb10.sh
```

Vision gate:

```bash
python3 scripts/vision_short_bench.py http://127.0.0.1:8000 deepseek-v4-flash-vision-exp
```

## This cut

- max_model_len 524288, k=3 probabilistic, thinking=high
- Vision short 8/8
- Q200v2 text180 e2e ~40.4 tok/s aggregate
- HumanEval local 36/40; GSM8K 74/80; IFEval 37/40; hard reasoning 19/20
- BFCL-hard20 13/20 (frozen thinking=low contract)
- NIAH 25/50/90 + multi-key PASS (~480k constructed vs 524k target)
- SHORT c1 28.88 / c8 58.24 tok/s
