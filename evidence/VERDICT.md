# CAMPAIGN VERDICT — DeepSeek-V4-Flash-Vision-Exp DSpark SM121 dual-GB10 — 2026-09-01

## Outcome: ENGINE QUALIFIED + BENCHMARKED (local; not published)

| Dimension | Result |
|---|---|
| Weights | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` @ `e46e16bf…` (167.8GB, 48 shards, both ranks sha-verified) |
| Engine | vLLM **v0.28.0** (`2cf0a691`) + 5-file patch + **flashinfer 0.6.18** trio (python/cubin/jit-cache cu130) |
| Image | `dsv4v-vllm:v0.28.0-sm121-visionfix` = `sha256:1bdd5701bf1d…` (both ranks parity PASS) |
| Serving | TP=2, node3 rank0 + node2 rank1, RoCE lane, **CUDA graphs FULL + dspark graphs** |
| Spec decode | DSpark K=6 greedy (packaged 3-layer head), **mean acceptance length 4.0–4.5** |
| KV | `fp8_ds_mla` — 371,677 tokens; util 0.875; max_model_len 262,144 |
| Correctness gates | semantic arithmetic 4/4 · prefill/decode isolation clean · no corruption |
| **NIAH @ advertised window (262,080)** | single-needle 25/50/90% **PASS** · multikey LAST-needle@90% w/ distractors@33/66 **PASS** |
| Throughput (512in/256out, ignore-eos) | c1 **11.29** out tok/s · c2 **14.08** · c4 **21.09** (total 33.9/42.3/63.3 tok/s) |

## Divergence from plan target (disclosed)
- **NVFP4 KV cache NOT served**: v0.28.0 upstream hard-blocks nvfp4 KV on MLA (config validator);
  `nvfp4_ds_mla` (our v0.26 lane) never upstreamed. Serving on `fp8_ds_mla`. The nvfp4-KV
  goal needs either an upstream unblock or a fork port (7 knownfix-class items remain).
- **Full 1M context NOT reachable** on 2×128G with fp8 KV (needs ~57 GiB KV/rank vs ~21 free).
  Advertised-window NIAH capped at 262,144.
- **Vision**: text-only (vLLM has no deepseek_v4 vision wrapper at v0.28.0; 267 vision tensors
  skipped at load; router bias_vl dropped — selection-only, exact for text per reference impl).

## Files
- Patches: `docker/patch_vision.py` + `docker/Dockerfile.visionfix` (anchors asserted at build)
- Launcher: `scripts/run-dsv4v-dual-gb10.sh` (rank0=node3; env knobs documented inline)
- NIAH: `scripts/run-niah-advertised.py` (single 25/50/90 + multikey last-needle@90 33/66)
- Evidence: `evidence/BRINGUP-VERDICT.md`, `evidence/cladder/c{1,2,4}.json`, node3 `~/dsv4v-niah-262k.json`, `~/dsv4v-niah-mk.json`
- Knownfix audit: `notes/v0280-knownfix-port-audit.md` (6 upstream / 7 obsolete / 1 ported / 7 deferred)

## Next (not executed — requires go)
- K-ladder (K=3/6 AR-vs-spec comparison), c8+, GSM8K quality canary
- nvfp4_ds_mla port (7-item nontrivial list) for the original KV target + 1M window
- Publication pack (r0b0bench ledger/GH/HF) — not asked
