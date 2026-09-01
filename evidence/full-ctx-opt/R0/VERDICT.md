# R0 — 1M + max-thinking baseline — 2026-09-01

## Admission (PASS)

| Fact | t-opt 262k | R0 1M |
|---|---|---|
| context_len / max_model_len | 262144 | **1048576** |
| max_running_requests | 256 (auto) | **8 (pinned)** |
| decode graph buckets | 51 (bs≤256) | **6 (1 2 3 4 6 8)** |
| target_verify capture | 164.92 s / 6.07 GB | **82.05 s / 2.79 GB** |
| draft_decode capture | 26.47 s / 1.08 GB | **4.34 s / ~0 GB** |
| max_total_num_tokens | 1,142,784 | **1,690,880** |
| available_gpu_mem after graphs | 6.11 GB | **11.14 GB** |
| host MemAvailable | ~4 GB | **~9 GB** |
| engine e2e | 974 s | **852 s** |

`max_total_num_tokens=1690880` ≥ 1.06M target. 1M c1 is admitted with ~1.61× token headroom.

## Thinking defaults (PASS)

`prompt_tokens` at max_tokens=1: default=97, low=5, high=84, max=97. default == max, low < high < max.
`--reasoning-parser deepseek-v4 --tool-call-parser deepseekv4` + `SGLANG_DEFAULT_THINKING=1` + `SGLANG_DSV4_REASONING_EFFORT=max`.
think_bench `reasoning_present_frac=1.0`. Arithmetic gates 5/5 (numbers appear after `</think>`).

## Short ladder (random-ids 512/256 greedy)

| C | t-opt FINAL | R0 | note |
|---|---|---|---|
| c1 | 38.51 | **41.53** | +7.8% |
| c2 warmup | — | 48.21 | |
| c2 | 56.04 | 37.85 | one-shot; treat as variance, c1/c4/c8 in band |
| c4 | 60.25–65.40 | **60.73** | in band |
| c8 | 83–98 | **89.94** | in band |

Zero failed requests. Median ITL c1 13.28 ms.

## Think ladder (effort=max)

| cell | agg tok/s | per-stream med | mean completion | finish |
|---|---|---|---|---|
| c1 greedy | **44.30** | 44.73 | 1171 | stop |
| c4 greedy | **68.55** | 20.28 | 3268 | mix length/stop (6144 cap) |
| c4 t=1.0 top_p=0.95 | **56.20** | 22.00 | 3732 | mix length/stop |

vs t-opt prose: c1 32.5 → 44.3 (+36%); c4 63.5 → 68.55. Thinking-max is not slower than prose on this head.

Accept len from rank-0 decode log during think traffic: ~2.4–3.8 (typical ~2.6–2.8), cap len 6.00 — same shallow ceiling as t-opt. `/get_internal_state` still returns no avg_spec_accept_length.

## NIAH 1M

In flight: `python3 scripts/run-niah-advertised.py --base-url http://192.168.3.2:30000 --target-tokens 1048512` → `evidence/full-ctx-opt/R0/niah-1m.json`. Full-context concurrency ladder deferred until NIAH PASS (same 1M prefill cost).

## Next

R1 STS collect on this admitted 1M server (needs restart with collect env). Do not start R1 until NIAH 1M single+multikey complete.
