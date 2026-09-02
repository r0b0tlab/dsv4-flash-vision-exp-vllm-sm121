# t-opt FINAL — winning stack + full ladder — 2026-09-01

## Winning profile (R0 stack)
```
SGLANG_RAGGED_VERIFY_MODE=compact
SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV=1
γ=5 (checkpoint-structural; see R3a)
```
No SPS table (R2: regression). No align-to-graph-tier (R3b: regression). No explicit
max-running-requests pin (auto 256 via DSV4 hook; R2 pin showed no independent effect
before SPS confound was removed).

## Final throughput ladder (random-ids 512/256, greedy, warm)
| C | base (stock) | FINAL (R0 stack) | gain |
|---|---|---|---|
| c1 | 34.82 | **38.51** | **+10.6%** |
| c2 | 47.34 | **56.04** | **+18.4%** |
| c4 | 58.06 | 60.25 | +3.8% |
| c8 | 88.53 | 83.35–98.03 | −6% to +11% (variance band; R0 official warm c8 = 98.03) |

Median ITL: c1 12.62 ms, c2 22.59, c4 34.36, c8 50.32.
c4/c8 cells ran immediately after prose traffic; earlier R0 c4=65.40/c8=98.03 remain the
recorded official cells (same config). c8 variance across identical-config runs (83.35 →
98.03) is ~15% run-to-run; treat single cells with caution.

## Prose lane (~500-word explanations, greedy — user-requested realism check)
| C | agg tok/s | per-stream tok/s | mean completion |
|---|---|---|---|
| c1 | 32.54 | 32.3–32.8 | 625 tok |
| c4 | 63.50 | 15.6–18.2 | 600 tok |
| c8 | 83.62 | 11.0–12.5 | 615 tok |

Accept length on prose: mean 2.64 — shallow like random-ids (2.9); only ~2% of steps ≥4.
Task difficulty/length does NOT rescue acceptance on this bundled head; the γ=5
full-width verify waste is structural (see R3a: gamma resolved from draft checkpoint
config, not the CLI).

## Gates: PASS 5/5 (arithmetic 4/4, max_tokens prefix, health)

## Verdict vs exit condition
Target band (c1 ≥ 60, c8 ≥ 120) NOT reached; every researched lever has now been
measured and the remaining gap is attributed:
1. **Draft acceptance ceiling (~2.6/6)** — the bundled DSpark head accepts shallowly on
   both random and prose traffic; per-step output is bounded by accept_len/verify_step_ms.
   Not tunable via serving config (γ checkpoint-structural; SPS/align-tier measured slower).
2. **Verify compute floor** — full-width verify at bs×6 tokens dominates step time at c8;
   compact mode already trims what is trimmable (R0 = compact on).
3. Remaining run-to-run c8 variance ±15% — subsumes any <10% lever.
The SGLang+DSpark lane still beats the vLLM v0.28 lane 2.8–3.4× at every rung (t-ladder).
