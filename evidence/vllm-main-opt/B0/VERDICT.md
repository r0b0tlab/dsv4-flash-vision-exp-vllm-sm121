# B0 — vLLM vision image + #54631 overlay + FlashInfer dual-topk512 — 2026-09-02

## Identity

- Image: `dsv4v-vllm:vision-54631-fi512b` `sha256:c4c0d7b269b2b0af0d3edaf92cfab1add8644d99eb518f6dbf3e99511ea4e84f`
- Base: `vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130` (`0.28.1rc1.dev137+g5ab628dd1`)
- Overlay: surgical PR #54631 (streaming VL loader + DSpark n_predict from dspark_block_size=5)
- FlashInfer: dual-cache SM120 prefill TOPK=512 instantiation (stock 0.6.18 only had dual TOPK=128) + deleted prebuilt `flashinfer_jit_cache/.../sparse_mla_sm120.so` so JIT rebuilt
- TP=2 node3 rank0 + node2 rank1, API http://192.168.3.2:8000
- KV fp8_ds_mla, max_model_len 262144, util 0.86, K=3, adaptive_verification=false, no flashinfer autotune
- Architecture: DeepseekV4ForConditionalGeneration (vision ON)
- KV tokens: 627,397 (2.39× at 262k)

## Canaries

| Probe | Result |
|---|---|
| 17×23 thinking-off | **391** in 4.2s |
| SM12X canary 2+2 | PASS text `4` |
| Vision 96×64 red/blue PNG | **LEFT=red; RIGHT=blue**, 135 prompt tokens (~107 image), 29.5s |

## Screen (thinking off, not publish-grade)

| cell | agg tok/s | per-stream p50 |
|---|---|---|
| latency C1 (76 tok prose) | — | **29.8** |
| c1 count-up 256 | **56.6** | 56.6 |
| c2 | 99.3 | 49.7 |
| c4 | 38.7 (stall outlier) | 9.7 |
| c8 | 169.8 | 21.2 |

Below 85 c1 / 300 c8-at-200k targets. v028 stock c1 was 11.3 — this is faster, not yet in-class record.

## Rejected / required on SM121

- `enable_adaptive_verification=true`: DeepseekV4IndexerBackend cannot trim on device
- FlashInfer sparse-MLA **prefill autotune** at 8192 tokens: unsupported DSV4 dual config
- Stock FlashInfer 0.6.18 dual-cache prefill only TOPK=128; Vision-Exp is index_topk=512 — needs the fi512 JIT rebuild
- NVFP4 KV: still `nvfp4 KV cache is not supported with MLA` in this tree
- 1M window not admitted on fp8 (need NVFP4 port)

## Not done

S1 lever sweep, NVFP4 KV, 1M NIAH, SM12X systems full, publication.
