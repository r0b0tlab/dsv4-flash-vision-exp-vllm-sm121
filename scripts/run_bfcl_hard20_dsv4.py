#!/usr/bin/env python3
"""Run frozen BFCL-hard20 against DSV4. Must be executed with cwd = q200v2 runner root
or PYTHONPATH=that/scripts. Clears registrar .file_locks so run mode can start."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/r0b0tdgx/qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner/scripts")))
import run_bfcl_hard20 as b  # noqa: E402

_orig = b.start_fresh_run


def start_fresh_run(root: Path, binding_sha256: str):
    for path in list(root.rglob("*")):
        if path.is_file() and ".file_locks" in path.parts:
            path.unlink()
    for path in sorted((p for p in root.rglob(".file_locks") if p.is_dir()), reverse=True):
        path.rmdir()
    return _orig(root, binding_sha256)


b.start_fresh_run = start_fresh_run
raise SystemExit(b.main())
