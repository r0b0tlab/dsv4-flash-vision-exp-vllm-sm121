# t-opt R2 — full stack (compact + fused-markov + SPS table + max-running 16) — 2026-09-01

## Config
- R0 stack + `--speculative-dspark-sps-table-path /sgl-extras/sps.json` + `--max-running-requests 16`
- SPS table (profiled R1 on the live R0 server, 5 bs × 3 M cells): bias=73.77ms,
  alpha=[0, 23.5, 61.5, 120.6, 185.4]ms, theta=[0, 2.36, -8.98]ms.

## Result vs R0 (R0: c1 38.20 / c4 65.40 / c8 98.03)

| C | R0 | R2 | delta |
|---|---|---|---|
| c1 | 38.20 | **31.01** | **−18.8%** |
| c4 | 65.40 | **58.40** | **−10.7%** |
| c8 | 98.03 | **84.13** | **−14.2%** |

Median ITL: c1 18.51 ms, c4 36.75, c8 49.73.

## Gates: PASS (5/5 — arithmetic 4/4, max_tokens prefix, health)

## Analysis
Server logs during the R2 bench show `cap len: 5.5–5.9` throughout — the SPS table did NOT
reduce verify width; cap stays pinned near the γ=5 full width. Meanwhile per-step throughput
dropped vs R0 at every concurrency. Attribution: the profiled table's bias term (73.8ms) is
far above this hardware's actual verify cost, so the cost model overestimates verify and
the scheduler under-schedules speculation; the table adds per-step scheduling overhead
without width reduction benefit. The profile was taken under SGLANG_SIMULATE_ACC_LEN=1.0
(deterministic KV conditioning), which inflates measured step times.

## Verdict: REVERT SPS table (do not adopt). R0 remains the best stack.
Accept-len stats from the bench (bs1: 1.6–4.4, bs4/8: 2.0–3.6, mean ≈ 2.9) show acceptance
is shallow relative to γ=5 full width → next lever: γ sweep (γ=4 then γ=3) on the R0 stack.
