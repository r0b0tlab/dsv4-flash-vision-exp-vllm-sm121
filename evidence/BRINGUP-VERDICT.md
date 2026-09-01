# Bring-up verdict — 2026-09-01

## Engine: READY (TP=2, node3 rank0 + node2 rank1)

| Item | Value |
|---|---|
| Image | `dsv4v-vllm:v0.28.0-sm121-visionfix` = `sha256:1bdd5701bf1d…` (both ranks, parity PASS) |
| Base | official `vllm/vllm-openai:v0.28.0` (`2cf0a691`) + 5-file patch + flashinfer 0.6.18 trio (python/cubin/jit-cache cu130, from flashinfer.ai/whl indexes) |
| Served | `deepseek-v4-flash-vision-exp`, max_model_len 262144 |
| KV | `fp8_ds_mla`, **371,677 tokens** (1.42x of max len), util 0.875 |
| Weights | 80.1 GiB/rank, 231s load; DSpark draft loaded: 97 params (packaged 3-layer mtp head) |
| Graphs | CUDA graphs FULL (target) + dspark graphs (13) captured, 5.60 GiB |
| Autotune | flashinfer 24 configs, SM121-aware cache `121a/` |
| Spec decode | DSpark K=6 greedy — **mean acceptance length 4.0–4.5** |
| Gates | semantic arithmetic 4/4; max_tokens=1 vs 2 clean; no corruption signatures |

## Patches carried in image (docker/patch_vision.py — anchors asserted at build)
1. Weight-loader skip: `vision.` `aligner` `image_start/end/newline/pad` (no vision wrapper upstream; text-only serving)
2. Zero-init `e_score_correction_bias` (checkpoint is bias-free noaux_tc)
3. Target-model loader: drop `gate.bias_vl` / `e_score_correction_bias` weights
4. DSpark draft loader: same drop for `.ffn.gate.bias_vl`
5. `sparse_swa.py`: allocate FlashMLA tile_sched on family-120 (defensive; FlashInfer path is active)

## Root causes closed on the way (7 engine admissions)
- K=7→K=6 (divisible by nextn=3) · nvfp4-KV blocked upstream (validator) → fp8_ds_mla
- b12x MoE invalid for MXFP4 → engine default (DeepGEMM E8M0)
- homeless vision tensors → skip · bias-free router → zero-init
- FlashMLA sparse decode unsupported on SM121 → flashinfer 0.6.18 (PR #4380, topk192)
- flashinfer cubin/jit-cache live at flashinfer.ai/whl, not PyPI (PyPI cubin stops at 0.6.13)

## Capacity note
util 0.90 exceeds per-rank free (107.8/121.69 GiB) → cap ≈ 0.886. Production profile = 262144 @ 0.875.
Full 1M window NOT reachable with fp8 KV on 2×128G (needs ~57 GiB KV/rank vs ~21 available) — NIAH claims capped at 262k unless/until nvfp4 KV lands upstream.
