# t-opt R3 — ablations — 2026-09-01

## R3a — γ=4 attempt: STRUCTURALLY BLOCKED (crash)
`--speculative-dspark-block-size 4` crashed at graph capture:
`deepseek_v4_dspark.py:898 compute_confidence: x_post_hc.view(bs, self.gamma, -1)` with
`self.gamma=5` — `self.gamma` is resolved from the DRAFT CHECKPOINT config
(`dspark_config.resolve_gamma()`), not the CLI flag; block-size 4 changed token width but
not gamma → shape mismatch `[256,5,-1] invalid for 4194304`. The bundled head is
structurally γ=5; no γ sweep is possible without retraining/patching the draft head.
Traceback: `crash-traceback.txt`.

## R3b — align-verify-tokens-to-graph-tier: REGRESSION (rejected)
R0 stack + `--speculative-dspark-align-verify-tokens-to-graph-tier`:
| C | R0 | R3b | delta |
|---|---|---|---|
| c1 | 38.20 | 31.79 | −16.8% |
| c4 | 65.40 | 56.37 | −13.8% |
| c8 | 98.03 | 86.15 | −12.1% |

Gates 5/5 PASS (correct, just slower). Source read confirms the flag CEILS the verify
budget up to the padded graph tier (fills padding with extra draft tokens) — with mean
accept 2.6/6 the extra drafted tokens are mostly rejected, so the deeper speculation costs
more than it returns. c8 median ITL did improve (42.6 vs 49.1 ms) but aggregate still lost.

## Accept-length reality check (prose + random-ids)
- random-ids bench: mean accept ≈ 2.9 (R2 logs)
- prose lane (~500-word explanations, greedy): mean accept 2.56; only 2% of steps ≥ 4
- Conclusion: acceptance is shallow for BOTH traffic shapes on this bundled head;
  full-width γ=5 verify wastes ~55–60% of verify compute, and it cannot be lowered
  (checkpoint-structural). This is the dominant remaining inefficiency.

## Verdict: R0 stack (compact verify + fused greedy markov, γ5) is the final profile.
