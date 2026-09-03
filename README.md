# DeepSeek-V4-Flash-Vision-Exp · vLLM · dual GB10

r0b0tlab (@mr_r0b0t). Vision + 512k context on 2× NVIDIA GB10 (SM121). Speculative decoding k=5 with adaptive verification, fp8 KV.

Results: `publication/html/index.html`

## Click-run

Measured image (arm64):

```
ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121:vision-54631-fi512b-k5adapt
```

Image id `sha256:f8b73f965834ff8439bf266827229ef71bfdf291689e05b63dc59a74003b517d`
(manifest digest `sha256:78eb0a105a3b4cfc9e28da477d1c5285ffc76e3fb1e621d26799906976c8fbcc`)

Weights are not in the image. Place `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on both ranks, then on rank0:

```bash
docker pull ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121:vision-54631-fi512b-k5adapt
docker tag ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121:vision-54631-fi512b-k5adapt dsv4v-vllm:vision-54631-fi512b-k5adapt

WORKER_SSH=user@rank1 HEAD_IP=<rank0-fabric-ip> \
MAX_MODEL_LEN=524288 \
bash scripts/run-vllm-vision-dual-gb10.sh
```

The launcher defaults are the production profile: k=5, `enable_adaptive_verification=true`, thinking off by default (quality clients send `thinking=true, reasoning_effort=high` per request), fp8 KV, FDO CUDA graphs, FlashInfer autotune with the fp4-block-scale MoE ops skipped.

Vision gate:

```bash
python3 scripts/vision_short_bench.py http://127.0.0.1:8000 deepseek-v4-flash-vision-exp
```

## Patches and modifications (official image → this runtime)

Start from the official per-model image `vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130`. Everything below is cumulative; each step is a thin overlay with an in-build assertion, and the full chain reproduces the published image.

1. **PR #54631 surgical overlay** (`docker/Dockerfile.overlay-54631`, base = official image): streaming weights loader (`nvidia/vl_model.py`, `nvidia/model.py`) and DSpark speculative config that derives `n_predict` from `dspark_block_size` (`config/speculative.py`). Without this the image dies on first decode.

2. **FlashInfer dual-cache sparse-MLA TOPK=512 prefill kernel** (`docker/Dockerfile.fi512`, base = step 1): replaces `flashinfer/data/csrc/sparse_mla_sm120_prefill.cu` with the TOPK=512 build. The stock kernel's `page_block_size=64` prefill path is incompatible with this dual-cache layout on SM120/SM121 — engine dead on first prefill. Build asserts the kernel source contains `topk == 512`.

3. **SM121 adaptive verification** (`docker/Dockerfile.sm121-adaptive`, base = step 2 = `vision-54631-fi512b`): four source overlays that make vLLM's DSpark adaptive verification — which upstream gates to SM90/SM100 — run on SM121, and fix the one kernel-side incompatibility that surfaced once it ran. Files live in `docker/overlay-sm121-adaptive/`:

   - `indexer.py` — DSA indexer decode path. Upstream allows device-decided query lengths (flattened `q_lens`) only on Hopper with DeepGEMM (`is_device_capability_family(90) and has_deep_gemm()`), and varlen paged MQA logits only on SM100. SM121 already runs the same per-token flattened decode (uniform `decode_len=1`); we extend the flatten gate to `is_device_capability_family(120)`, and the indexer logs `use_flattening=True supports_varlen=False`. Hopper's DeepGEMM path is untouched — no SM90/SM100 predicate was modified.
   - `sparse_mla.py` — sparse-MLA attention metadata builder. CUDA-graph support is a class-level `UNIFORM_BATCH`; adaptive verification needs per-step non-uniform verify widths, so we add a `get_cudagraph_support` override returning `ALWAYS` when `current_platform.get_device_capability().major == 12` (SM120/SM121 only). Also makes the C128A `extra_indices` slice a first-dim-only contiguous view (`[:num_decode_tokens]` of the full-width buffer instead of a `[:N, :W]` slice).
   - `sparse_swa.py` — same `ALWAYS`-on-SM12x graph-support override for the SWA metadata builder.
   - `flashinfer_sparse.py` — before the SM120 TVM paged-attention call, squeeze 4-D `[tokens, 1, topk]` index tensors to the 2-D/3-D contiguous forms the kernel checks (`eidx must be contiguous`), including the compressed-cache (C4A/C128A) extra indices. This was the boot-time crash once adaptive verification was live: the kernel rejects non-contiguous index views.

   Result: k=5 drafts 5 tokens every step (DSpark `n_predict=5`), and the target verifies only what the confidence head says it needs instead of all 5+1.

4. **Launch profile** (`scripts/run-vllm-vision-dual-gb10.sh`): `enable_adaptive_verification=true` with k=5 (k=6 is illegal here — `num_speculative_tokens` must divide `dspark_block_size=5`), `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, FDO graph capture sized to `max_num_seqs*(k+1)` rounded up to 8 (`[1,2,4,8,16,32,48]`), `--long-prefill-token-threshold 1024`, persistent Triton/TileLang/vLLM/FlashInfer JIT caches, `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` limited to the fp4-block-scale MoE ops (autotune stays on), `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`, and request-level `thinking_token_budget` on quality runs so reasoning cannot consume the whole `max_tokens`.

## What was NOT taken

- **B12X** (`flashinfer_b12x` / `VLLM_USE_B12X_MOE`): an NVFP4 expert path. This checkpoint is FP8 (`quant_method=fp8`); Spark `b12x` images also drop vision tensors.
- **`nvfp4_ds_mla` KV**: not in this vLLM enum; the fp8 KV path is what admits 512k on 2×128 GB.
- **Community DSpark recipes** (Anemll vision graft, k=6 MTP, NVFP4-KV patch series): measured on their own stacks; their k=6 rule is `n_predict=3` on a different image and does not transfer.

## This cut

- max_model_len 524288 (KV 1,281,052 tokens, 2.44× concurrency at full window), k=5 adaptive, fp8 KV
- Vision short 8/8
- SHORT c1 think-off 43.23 tok/s · PROSE c1 med 34.87 · TD2W300 51.9 (complete)
- Q200v2 quality + NIAH: see `publication/html/index.html` and `evidence/vision-opt/V0/prod-512k-k5-adapt/`