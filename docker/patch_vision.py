"""Patch stock v0.28.0 DeepseekV4ForCausalLM weight loader to ignore vision/aligner tensors.

The DeepSeek-V4-Flash-Vision-Exp checkpoint carries vision.* + aligner.* tensors,
but stock vLLM v0.28.0 has no deepseek_v4 vision wrapper, so AutoWeightsLoader
raises on the homeless prefixes. Text-only serving: skip them explicitly.
"""
import pathlib

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py"
)
OLD = 'AutoWeightsLoader(self, skip_substrs=["mtp."])'
NEW = (
    'AutoWeightsLoader(self, skip_substrs=['
    '"mtp.", "vision.", "aligner", '
    '"image_start", "image_end", "image_newline", "image_pad"])'
)
text = P.read_text()
assert OLD in text, "anchor not found — stock file changed?"
P.write_text(text.replace(OLD, NEW))
print("PATCHED", P)
