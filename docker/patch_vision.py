"""Patch stock v0.28.0 deepseek_v4 for the Vision-Exp checkpoint (text-only serving).

1. Loader: skip homeless vision tensors (vision.*, aligner, image_*).
2. Router bias: this checkpoint uses noaux_tc + sqrtsoftplus with NO
   e_score_correction_bias tensors (trained bias-free). Stock code eagerly
   creates the bias param when topk_method == noaux_tc and then the loader
   KeyErrors on the missing weight. Fix: init the param to zeros and skip its
   weight — zero bias is mathematically identical to no bias.
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
    '"e_score_correction_bias"])'
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

P.write_text(text)
print("PATCHED (vision-skip + zero-bias)", P)
