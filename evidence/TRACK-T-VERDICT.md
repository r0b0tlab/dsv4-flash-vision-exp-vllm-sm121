# Track T result — SGLang dual-GB10 TP=2, packaged DSpark — 2026-09-01

## Verdict: 80–100 tok/s TARGET MET at c8 — and every rung beats the vLLM v0.28 lane 2.8–3.4×

Stack: upstream SGLang `4c2c169e` (image `sglang-dsv4v:0.5.19-vision` = `86466ee96176…`, both ranks parity PASS),
`--attention-backend dsv4`, `flashinfer_mxfp4` MoE runner (auto-selected), fp8_e4m3 KV,
**bundled DSpark γ=5 / verify 6** (in-weight head, zero extra artifacts), full verify CUDA graphs.
Image patches: vision-tensor skip list only (weights preserved for Track V); flashinfer trio aligned 0.6.18.

## Throughput ladder (512-in / 256-out, greedy, ignore_eos)

| Concurrency | Output tok/s | vLLM v0.28 lane | Speedup |
|---|---|---|---|
| c1 | **34.82** | 11.29 | 3.1× |
| c2 | **47.34** | 14.08 | 3.4× |
| c4 | **58.06** | 21.09 | 2.8× |
| c8 | **88.53** | (KV-capped) | — |

- Per-stream decode: median ITL ≈ 16.6 ms (~60 tok/s/stream) at c1.
- c16 NOT run (user constraint): fp8-KV 262k window leaves ~1.42× concurrency headroom; c8 is the honest ceiling cell here.

## Spec attribution (attributing the win)
- DSpark γ=5 active on every rung (bundled head, `DeepseekV4ForCausalLMDSpark`, gamma=5 → verify 6).
- Acceptance readout: `/get_internal_state` `avg_spec_accept_length` returned null on this build
  (skill-noted failure mode) — per-stream ITL 16.6 ms and 34.8 tok/s at c1 vs vLLM AR-lane 11.29 tok/s
  attribute the c1 gain to (DSpark + flashinfer_mxfp4 MoE + dsv4 backend) jointly; a K-ladder (γ∈{3,5}, AR) isolates
  the DSpark share as follow-up.

## Vision tower status (per user directive: do not lose it)
- Vision/aligner/image tensors remain in the checkpoint; load path skips them with a warning
  (`patch_sglang_trackt.py`), text-serving only in Track T.
- Track V (next): port `inference/{vision.py,image_processor.py}` reference as a SGLang multimodal wrapper + wire
  `bias_vl` router bias — the skip list gets deleted then, replaced by real modules.

## Artifacts
- `evidence/t-ladder/tl{1,2,4,8}.log` — raw bench_serving outputs
- `scripts/run-sglang-dual-gb10.sh` — reproducible dual-node launcher (image parity + model gates)
- `docker/{Dockerfile.sglang-dsv4v,patch_sglang_trackt.py}` — image definition
- Stage-1 FP4-KV finding: `notes/stage1-admission.md` (port = separate go/no-go)
