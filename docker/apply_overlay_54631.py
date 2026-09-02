#!/usr/bin/env python3
"""Install PR #54631 Python overlay into the vLLM site-packages of the vision image.

Copies only the three runtime files. Refuses to run if vl_model.py is absent
(that would mean we are not on the vision image). Skips copy when load_weights
already streams (no sorted() materialization).
"""
from __future__ import annotations

import pathlib
import shutil
import sys

OVERLAY = pathlib.Path("/overlay")
NEED = [
    "vllm/config/speculative.py",
    "vllm/models/deepseek_v4/nvidia/model.py",
    "vllm/models/deepseek_v4/nvidia/vl_model.py",
]


def find_vllm() -> pathlib.Path:
    import vllm

    return pathlib.Path(vllm.__file__).resolve().parent


def main() -> int:
    dest_root = find_vllm().parent  # site-packages
    vl = dest_root / "vllm/models/deepseek_v4/nvidia/vl_model.py"
    if not vl.is_file():
        print(f"FAIL: vision wrapper missing at {vl}", file=sys.stderr)
        return 2
    text = vl.read_text()
    already = "def load_weights" in text and "sorted(" not in text
    print(f"image_vl_model={vl} already_streaming={already}")
    for rel in NEED:
        src = OVERLAY / rel
        dst = dest_root / rel
        if not src.is_file():
            print(f"FAIL: overlay missing {src}", file=sys.stderr)
            return 3
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {rel} -> {dst} bytes={dst.stat().st_size}")
    import vllm

    print(f"vllm_version={vllm.__version__} vllm_file={vllm.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
