# S1 FP4 indexer — REJECTED (SM121)

`--attention_config.use_fp4_indexer_cache True` maps to `indexer_kv_dtype=mxfp4`.

Worker error:
`indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs (sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and earlier architectures are not supported.`

Do not retry on GB10/SM121. Adopted remains FP8 indexer cache.
