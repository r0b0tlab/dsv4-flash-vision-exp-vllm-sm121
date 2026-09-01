"""Patch stock v0.28.0 deepseek_v4 for the Vision-Exp checkpoint (text-only serving).

Checkpoint facts (verified against index + reference inference/model.py):
- Carries homeless vision tensors: vision.*, aligner.*, image_start, image_end,
  image_newline, image_pad. Stock vLLM has no deepseek_v4 vision wrapper.
- Router: topk_method=noaux_tc + scoring_func=sqrtsoftplus, has bias_vl
  (vision-token routing bias) but NO e_score_correction_bias tensors for the
  39 non-hash MoE layers. Bias shifts expert SELECTION only, never the
  routing weights (weights = f(original_scores) in the reference), so with
  text-only input the bias has no effect. Zero-init is mathematically exact.

Patches (all in vllm/models/deepseek_v4/nvidia/model.py):
1. AutoWeightsLoader skip list += vision/aligner/image_* prefixes.
2. Gate param creation: zero-init e_score_correction_bias (bias-free recipe).
3. Layer.load_weights: early-continue for gate.bias_vl / e_score_correction_bias
   names (text serving: selection bias unused; AutoWeightsLoader has already
   returned by the time the generator reaches these, so the skip list cannot
   handle them).
"""
import pathlib

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py"
)
text = P.read_text()

# 1) loader skip list
OLD_LOADER = 'AutoWeightsLoader(self, skip_substrs=["mtp."])'
NEW_LOADER = (
    'AutoWeightsLoader(self, skip_substrs=['
    '"mtp.", "vision.", "aligner", '
    '"image_start", "image_end", "image_newline", "image_pad", '
    '"e_score_correction_bias", "gate.bias_vl"])'
)
assert OLD_LOADER in text, "loader anchor not found"
text = text.replace(OLD_LOADER, NEW_LOADER)

# 2) zero-init bias param (unique to the noaux_tc branch)
OLD_BIAS = (
    "self.gate.e_score_correction_bias = nn.Parameter(\n"
    "                torch.empty(config.n_routed_experts, dtype=torch.float32),"
)
NEW_BIAS = (
    "self.gate.e_score_correction_bias = nn.Parameter(\n"
    "                torch.zeros(config.n_routed_experts, dtype=torch.float32),"
)
assert OLD_BIAS in text, "bias anchor not found"
text = text.replace(OLD_BIAS, NEW_BIAS)

# 3) router-bias drop in the layer-level load_weights loop.
#    Inserted right before the attention-sink branch so it precedes the final
#    else that would otherwise KeyError into params_dict.
OLD_LOOP = (
    "                elif \"attn_sink\" in name:\n"
    "                    if is_pp_missing_parameter(name, self):\n"
    "                        continue\n"
    "                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]\n"
)
NEW_LOOP = (
    "                elif (\"gate.bias_vl\" in name\n"
    "                        or \"e_score_correction_bias\" in name):\n"
    "                    # Vision-Exp router bias: selection-only, unused for\n"
    "                    # text-only serving (see reference inference/model.py).\n"
    "                    continue\n"
    "                elif \"attn_sink\" in name:\n"
    "                    if is_pp_missing_parameter(name, self):\n"
    "                        continue\n"
    "                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]\n"
)
assert OLD_LOOP in text, "loop anchor not found"
text = text.replace(OLD_LOOP, NEW_LOOP)

# 4) same drop in the DSpark draft loader (dspark.py): it remaps
#    .ffn.gate.bias -> e_score_correction_bias and would KeyError on bias_vl.
D = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py"
)
dtext = D.read_text()
OLD_D = (
    "                if name.endswith(\".ffn.gate.bias\"):\n"
    "                    name = name.replace(\n"
    "                        \".ffn.gate.bias\", \".ffn.gate.e_score_correction_bias\"\n"
    "                    )\n"
)
NEW_D = (
    "                if name.endswith(\".ffn.gate.bias_vl\"):\n"
    "                    # Vision-Exp selection-only router bias: skip (text-only).\n"
    "                    continue\n"
    "                if name.endswith(\".ffn.gate.bias\"):\n"
    "                    name = name.replace(\n"
    "                        \".ffn.gate.bias\", \".ffn.gate.e_score_correction_bias\"\n"
    "                    )\n"
)
assert OLD_D in dtext, "dspark anchor not found"
D.write_text(dtext.replace(OLD_D, NEW_D))

# 5) sparse_swa.py: build_tile_scheduler early-returns all-None on
#    device-capability family 120 (SM121), assuming the FlashInfer DSV4
#    backend. With FLASHMLA_SPARSE_DSV4 forced (stock flashinfer wheel lacks
#    the SM120 sparse decode specialization, PR #4380), flashmla.py asserts
#    tile_sched entries are non-None. The empty SchedMeta allocation is
#    harmless for the FlashInfer path (it ignores tile_sched_*), so allocate
#    on all CUDA platforms.
S = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/"
    "mla/sparse_swa.py"
)
stext = S.read_text()
OLD_S = (
    "        if (\n"
    "            num_decode_tokens == 0\n"
    "            or current_platform.is_rocm()\n"
    "            or current_platform.is_xpu()\n"
    "            or current_platform.is_device_capability_family(120)\n"
    "        ):\n"
    "            return out\n"
)
NEW_S = (
    "        if (\n"
    "            num_decode_tokens == 0\n"
    "            or current_platform.is_rocm()\n"
    "            or current_platform.is_xpu()\n"
    "        ):\n"
    "            return out\n"
)
assert OLD_S in stext, "sparse_swa guard anchor not found"
S.write_text(stext.replace(OLD_S, NEW_S))

P.write_text(text)
print("PATCHED v6 (target+draft loader + tile_sched family-120 guard)", P)
