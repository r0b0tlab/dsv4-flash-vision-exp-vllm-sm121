# HANDOFF — full-ctx-opt paused for cluster reboot — 2026-09-01 17:18 CDT

Operator is rebooting the 3-node GB10 cluster. Recovery owner killed. Do **not** resume NIAH until throughput levers (STS → R3 compute → R4 numerics) are measured.

## Resume in one command (after both workers SSH)

On **node3** (rank0), after `ssh gb10-node3` and `ssh r0b0tdgx@192.168.5.1` both work, image `sglang-dsv4v:0.5.19-vision` present on both ranks:

```bash
ssh gb10-node3 'cd ~/dsv4v-scripts && SG_STS_COLLECT=/sgl-rw/sts/shard bash run-sglang-dual-gb10.sh'
```

That is **R1** (same 1M + max-thinking profile as R0, plus STS logit collection). Then follow the plan: think traffic → `python3 -m sglang.benchmark.dspark_sts_fit` → R2 relaunch with `--speculative-dspark-confidence-sts-path /sgl-extras/sts.json`.

Plan: `~/.hermes/plans/2026-09-01_155500-dsv4v-sglang-1m-max-thinking-throughput.md`
Project: `~/projects/DeepSeek-V4-Flash-Vision-Exp` git `5c653ad` (plus this handoff).
Launcher v2: `scripts/run-sglang-dual-gb10.sh` (also `~/dsv4v-scripts/` on node3).

## Why we stopped

Orphaned 1M NIAH prefill (client killed; scheduler kept the request) starved unified memory. ICMP + `/v1/models` lived; sshd banner and decode timed out on **both** node3 and node2. Prefill ran well past the 12–15 min budget → stuck, not finishing. Reboot is the control path.

## Banked (do not re-measure)

R0 1M + max-thinking **ADMITTED**. Evidence: `evidence/full-ctx-opt/R0/`

| Item | Value |
|---|---|
| image | `sglang-dsv4v:0.5.19-vision` `sha256:86466ee96176…` |
| ctx / max_model_len | **1048576** |
| max_running_requests | **8** (not the DSV4 auto-256) |
| cuda-graph-bs | `1 2 3 4 6 8` |
| max_total_num_tokens | **1,690,880** (was 1,142,784 @ 262k) |
| graph capture | target 82s/2.79GB, draft 4.3s (was 165s/6.07GB + 26s) |
| avail GPU after graphs | 11.14 GB (was 6.11) |
| thinking | `SGLANG_DEFAULT_THINKING=1` `SGLANG_DSV4_REASONING_EFFORT=max`; prompt_tokens low=5 high=84 max=97=default |
| gates | 5/5 |
| short c1/c4/c8 | 41.53 / 60.73 / 89.94 tok/s (c1 +7.8% vs t-opt 38.51) |
| think c1 greedy | **44.3** tok/s, mean 1171 tok, reasoning_present=1.0 |
| think c4 greedy | 68.55 agg; c4 t=1.0/0.95 = 56.2 |
| accept len | still ~2.4–3.8 / 6 (log `accept len:`) — STS is the next lever |
| NIAH 1M | **cancelled** (client killed at START 25%; leftover prefill wedged the hosts) |

t-opt FINAL still the 262k speed floor: compact + fused-greedy-Markov, SPS/align-tier/γ=4 **rejected**.

## Vision — maintained

Image was **not** rebuilt. `docker/patch_sglang_trackt.py` still skip-lists `vision.*` `aligner*` `image_*` `gate.bias_vl` at load; checkpoint tensors stay on disk (`inference/`, 263 `vision.*`). Text path does not address them. Track V = upstream PR **#37253** (open, +2156/−157, 25 files, head `61f962cf`) — **not** a throughput patch. Rebuild cost ~4–6 h; DSpark+images verified on 4×B200 cookbook, **not** on 2×GB10/SM121.

## Next after reboot (order)

1. Confirm node3 + node2 SSH, `docker image inspect sglang-dsv4v:0.5.19-vision` parity, ports 30000/25001 free, `free -g` avail ≳ 40 GB after containers gone.
2. **R1** STS collect (command above). Drive `think_bench.py` c4 greedy until `~/sgl-rw/sts/shard.*.pt` ≥ 8. Fit with `dspark_sts_fit --gamma 5`. Copy `sts.json` to both `~/sgl-extras/`. Collection numbers are **not** evidence.
3. **R2** relaunch with `SG_STS_TABLE=/sgl-extras/sts.json`. Short + think ladders vs R0. Adopt ≥5% think c1 or c4, no short c1/c2 regression >3%.
4. **R3** one restart: `SG_TOPK_BACKEND=flashinfer SG_TF32=1 SG_MIXED_CHUNK=1` (+ STS table if adopted).
5. **R4** FP4 indexer + `SG_COMPRESS_DTYPE=bf16` only after speed work; **then** 1M NIAH.
6. Do **not** register Hermes endpoint without explicit OK.

## Topology (rank0 must be node3)

- rank0 node3 `gn100-2eea` ring `192.168.5.2` API `http://192.168.3.2:30000` ssh `gb10-node3`
- rank1 node2 `r0b0tdgx1` ring `192.168.5.1` user `r0b0tdgx` (node3→node2 keys work; reverse does not)
- image + weights already on both nodes (`~/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`)

## Do not retry

- γ via `--speculative-dspark-block-size 4` (shape crash)
- SPS cost table (regressed −11..−19%)
- align-verify-to-graph-tier (regressed)
- Q8 prefill (SM90), DP-attention (incompatible with DSpark)
- mem-fraction > 0.85
- NIAH until R3 (or R4) is done
