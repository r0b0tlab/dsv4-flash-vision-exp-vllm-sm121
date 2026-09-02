# S1 PIECEWISE — REJECTED

`cudagraph_mode=PIECEWISE` parsed (launcher JSON-brace bug fixed) then died at capture:

`tvm.error.InternalError: eidx must be contiguous`

17-bucket piecewise capture vs 2-bucket FULL_DECODE_ONLY. Do not retry without a FlashInfer eidx-contiguous fix.

Adopted remains FULL → engine FULL_DECODE_ONLY, K=5 probabilistic.
