#!/usr/bin/env python3
"""Run BFCL-hard20 against DSV4. Thinking=high like the rest of the quality set.

Must be executed with Q200V2_RUNNER set. Clears registrar .file_locks so run mode can start.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__import__("os").environ["Q200V2_RUNNER"]) / "scripts"))
import run_bfcl_hard20 as b  # noqa: E402

HIGH = {
    "enable_thinking": True,
    "thinking": True,
    "reasoning_effort": "high",
}
b.REQUIRED_CHAT_KWARGS = HIGH

_orig = b.start_fresh_run


def start_fresh_run(root: Path, binding_sha256: str):
    for path in list(root.rglob("*")):
        if path.is_file() and ".file_locks" in path.parts:
            path.unlink()
    for path in sorted((p for p in root.rglob(".file_locks") if p.is_dir()), reverse=True):
        path.rmdir()
    return _orig(root, binding_sha256)


b.start_fresh_run = start_fresh_run

_orig_query = b.Q200OpenAICompletionsHandler._query_FC


def _query_FC(self, inference_data):
    orig_gen = self.generate_with_backoff

    def gen(**kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        extra["chat_template_kwargs"] = dict(HIGH)
        extra["thinking_token_budget"] = 2048
        kwargs["extra_body"] = extra
        return orig_gen(**kwargs)

    self.generate_with_backoff = gen
    try:
        return _orig_query(self, inference_data)
    finally:
        self.generate_with_backoff = orig_gen


b.Q200OpenAICompletionsHandler._query_FC = _query_FC
raise SystemExit(b.main())
