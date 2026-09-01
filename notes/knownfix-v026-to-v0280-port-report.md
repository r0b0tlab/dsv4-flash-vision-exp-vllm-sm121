# v0.26 knownfix (21 commits) -> vLLM v0.28.0 classification & port report

- Branch: `local/v0280-dsv4v-sm121` @ `2cf0a6915ce544dc493a0990f2ea38d81601128a` (v0.28.0)
- Worktree: `/home/r0b0tdgx/worktrees/vllm-v0280-dsv4v`
- Source stack: branch `local/v026-knownfix-integrated` (21 commits over v0.26 base `568afb3a1`) in `/home/r0b0tdgx/.cache/dspark-vllm-v026-build/source`
- Date: 2026-08-31
- Status: IN PROGRESS — verdicts appended per commit below.

| # | Commit | Subject | Verdict | Notes |
|---|--------|---------|---------|-------|
| 15 | `dc4660582` | fix(dspark): fix DFlash prepare BLOCK_SIZE to stop gen-length JIT | **PORTED** | Same dynamic `BLOCK_SIZE = min(256, next_power_of_2(...))` still present in v0.28 `dflash/speculator.py:698`; applied identical fixed `BLOCK_SIZE = 256` fix. |
