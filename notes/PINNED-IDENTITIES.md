# Pinned 2026-08-31
weights   deepseek-ai/DeepSeek-V4-Flash-Vision-Exp @ e46e16bf6035c6f317eb2ac7458eb0362926d402 (167.8GB, 48 shards, FP8 blockwise, packaged DSpark mtp.* x3 layers, 263 vision.* tensors)
engine    vLLM v0.28.0 = 2cf0a6915ce544dc493a0990f2ea38d81601128a
rank0     node2 r0b0tdgx1  ring 192.168.1.2   (user r0b0tdgx)
rank1     node3 gn100-2eea ring 192.168.3.2
node2->node3 lane: 192.168.5.2 (verified)
kv target nvfp4 (v0.28.0 enum: nvfp4 | nvfp4_4over6; nvfp4_ds_mla REMOVED)
spec      {"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}  # no "model" key - packaged head
deletion authorizations 2026-08-31: Ornith-1.0-397B (all nodes); qwen38-flash-next BF16 SOURCE only (candidate/builds/qwen38-live PROTECTED)
