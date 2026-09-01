# Stage-1 admission probe — RESULT (2026-09-01)

## Attempt
`sglang-dsv4v:0.5.19-vision` (`33486ccefb32`, upstream main `4c2c169e` + patched env) with
`--attention-backend dsv4 --kv-cache-dtype fp4_mx_block16 --tp 1`, on node2 (probe;
node3 GPUs fully occupied by the live vLLM control lane).

Result: **config-time rejection, exactly as source analysis predicted.** Probe died at
arg-resolution (no GPU needed to know); runtime OOM in the probe log is incidental
(the control lane holds both GPUs).

## Verdict (from source, lines verified in worktree `~/worktrees/sglang-dsv4v` @ `4c2c169e`)

| # | Gate | File:line | Content |
|---|---|---|---|
| 1 | dtype allowlist | `arg_groups/overrides.py:907-930` `_deepseek_v4_kv_cache_dtype` | `assert kv_cache_dtype in ["fp8_e4m3", "bfloat16"]` for `DeepseekV4ForCausalLM`; fp4 rejected before anything runs |
| 2 | pool routing | `mem_cache/kv_cache_configurator.py:1146` → `_build_dsv4_kv_pool`; FP4 branch (line 1044) unreachable for DSA/DSV4 | DSV4 always builds `DeepSeekV4TokenToKVPool` |
| 3 | layout lock | `mem_cache/deepseek_v4_memory_pool.py:121` | `assert bytes_per_token == 448 + 64*2 + 8` (584 B/token FP8-nope+BF16-rope, block-64 UE8M0 scales) |
| 4 | write kernel | `kernels/ops/attention/dsv4/quant_k_cache.py:72` | `quant_to_nope_fp8_rope_bf16_pack_triton` — FP8-only fused quant+pack (hidden 512 → 448 FP8 + 64 BF16 + 7×u8 scales) |
| 5 | read path | `deepseek_v4_backend.py` (`fused_k_norm_rope_flashmla`, `is_fp8_kvcache=True` at :1787) + `dequant_k_cache.py` (prefill gathers) | decode kernel consumes the FP8 layout in-graph |

## What upstream DOES support (adjacent, not ours)
- `nvfp4` / `fp4_mx_block16` KV: real (MLA non-DSA pool `MLATokenToKVPoolFP4`, trtllm-MLA fp4 decode, ~1.78× tokens vs FP8) — but only for non-DSA MLA models; GLM-DSA gets fp8-only `flashinfer_sparse_mla` on SM120/121; DSV4 is excluded by gate #1.
- **MoE good news**: DSV4 + MXFP4 experts auto-selects `--moe-runner-backend flashinfer_mxfp4` on SM90/100/120 (`arg_groups/model_overrides/deepseek_v4.py:44-60`) — the modern analogue of the v026 B12X fast path, no patch needed.
- SM120/121 sparse-decode template exists: `kernels/ops/attention/flash_mla_sm120.py` (`_gather_and_dequant`, triton Tiled-V2 variant) — the pattern a FP4 DSA decode path would follow.

## Port surface for true NVFP4 DSA-KV (Stage 2), estimated
1. overrides.py: extend dtype allowlist for DSV4 (trivial, flag-guarded)
2. New `DeepSeekV4SingleKVPoolFP4` (~352-380 B/token incl. FP8-E4M3 block-16 scales; FP4 nope 224B + BF16 rope 128B + scales) + configurator route
3. New `quant_to_nope_fp4_rope_bf16_pack_triton` (write kernel, CUDA-graph-safe)
4. Decode read path: FP4 gather+dequant fused into `flash_mla_sm120` triton kernel OR trtllm-style fp4 decode if layout-compatible — **risk center**
5. DSpark draft pool + graph paths share the layout
Estimated: 2-4 days focused engineering + validation, with a real risk the sparse-DSA read path needs kernel-level work beyond Triton adaptation.

## Recommendation
Split the campaign: run Track T (throughput) on stock SGLang FP8-KV first (moE `flashinfer_mxfp4` is already auto-selected; spec/graph knobs per skill) — that is where 80-100 tok/s lives. Treat FP4-KV-DSA as a separate follow-on port decision after Track T results, since FP4 KV buys capacity/context, not decode speed.
