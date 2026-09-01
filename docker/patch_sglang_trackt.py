"""Track T image patch: DSV4-Flash-Vision-Exp on SGLang.

The Vision-Exp checkpoint carries vision-tower + aligner + image-* tensors that
stock SGLang (no deepseek_v4 vision wrapper yet) has no parameters for. The
target loader's only unknown-weight skip is `startswith("mtp")`, so
`aligner.gate_up_proj.weight` KeyErrors. We extend that guard to the full
vision-family prefix set (same list proven on the vLLM v0.28 lane). Weights
are NOT removed from the checkpoint — Track V will consume them when the
vision wrapper is ported; text serving simply does not address them.

Applied at image build to:
  python/sglang/srt/models/deepseek_v4.py
"""
import pathlib

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/deepseek_v4.py"
)
text = P.read_text()

OLD = (
    'if name not in params_dict and name.startswith("mtp"):\n'
    "                            break"
)
NEW = (
    'if name not in params_dict and name.startswith(\n'
    '                                ("mtp", "vision.", "aligner", "image_start",\n'
    '                                 "image_end", "image_newline", "image_pad",\n'
    '                                 "e_score_correction_bias", "gate.bias_vl")):\n'
    "                            break"
)
count = text.count(OLD)
assert count >= 1, f"vision-skip anchor not found (count={count})"
text = text.replace(OLD, NEW)
P.write_text(text)
print(f"PATCHED track-t: {count} site(s) in {P}")
