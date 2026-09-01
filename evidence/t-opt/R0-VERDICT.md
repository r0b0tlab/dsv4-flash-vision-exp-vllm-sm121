# t-opt R0 — compact + fused-greedy-markov (γ5, no SPS yet) — 2026-09-01

## Result vs base (base: c1 34.82 / c4 58.06 / c8 88.53)

| C | base | R0 | delta |
|---|---|---|---|
| c1 | 34.82 | **38.20** | **+9.7%** |
| c4 | 58.06 | **65.40** | **+12.6%** |
| c8 | 88.53 | 83.57 → 92.38 → **98.03** (3 runs) | first run was cold-cache variance; warm = **+10.7%** |

Median ITL: c1 14.42 ms (was 13.9-16.6), c4 27.85, c8 49.05.

## Gates: PASS
Arithmetic 4/4 (391/144/12/73); max_tokens 1→2 clean ('AL'→'ALP'); server healthy.

## Verdict: ADOPT R0 knobs (compact + fused greedy markov)
Warm-run c8 = 98.03 (+10.7%). All cells ≥ +9.7% on warm measurement. c1 target (60) not yet
reached — remaining gap is per-stream draft floor, next lever is the SPS table (R1→R2).
Note for protocol: warm the endpoint (1 short bench) before recording official cells.
