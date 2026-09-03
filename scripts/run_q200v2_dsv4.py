#!/usr/bin/env python3
"""Run frozen Q200v2 text180 against DSV4 Vision-Exp (thinking=high).

Reuses qwen38 Q200v2 dataset SHA + graders. Dummy admission (this is not the
qwen38 two-rank lease). Empty content is filled from reasoning_content.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

RUNNER = Path(__import__("os").environ["Q200V2_RUNNER"])
sys.path.insert(0, str(RUNNER / "scripts"))
spec = importlib.util.spec_from_file_location("run_quality_set", RUNNER / "scripts" / "run_quality_set.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class DummyAdmission:
    @contextlib.contextmanager
    def request(self, row_id: str):
        yield {"bypass": True, "row_id": row_id}


def validate_high(chat_kwargs):
    values = dict(chat_kwargs)
    if values.get("enable_thinking") is not True or values.get("thinking") is not True:
        raise ValueError("Q200 DSV4 requires thinking=true")
    if values.get("reasoning_effort") not in ("high", "low", "max"):
        raise ValueError("reasoning_effort must be high|low|max")
    return values


_orig_chat = q.chat


def chat_dsv4(base, prompt, max_tokens, timeout, *, model=q.MODEL, chat_kwargs=None):
    result = _orig_chat(base, prompt, max_tokens, timeout, model=model, chat_kwargs=chat_kwargs)
    if result.get("error") and "content must be non-empty" in str(result.get("error")):
        # retry parse: original already returned error; call again is wasteful.
        pass
    if not (result.get("content") or "").strip() and (result.get("reasoning_content") or "").strip():
        result["content"] = result["reasoning_content"]
        result["text"] = result["reasoning_content"]
        if result.get("error") and "content must be non-empty" in str(result.get("error")):
            result["error"] = None
    ct = ((result.get("usage") or {}).get("completion_tokens") or 0)
    el = result.get("elapsed_seconds") or 0
    result["e2e_tok_s"] = (ct / el) if el and ct else None
    return result


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--profile-id", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    q.validate_native_thinking = validate_high
    q.chat = chat_dsv4
    q.DEFAULT_CHAT_KWARGS = {"enable_thinking": True, "thinking": True, "reasoning_effort": "high"}
    dataset = str(RUNNER / "artifacts" / "quality-text-180-v2.jsonl")
    t0 = time.perf_counter()
    summary = q.run_quality(
        base_url=args.base_url,
        run_id=args.run_id,
        dataset_path=dataset,
        admission=DummyAdmission(),
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        workers=args.workers,
        human_eval_timeout=12.0,
        image_id=args.image_id,
        profile_id=args.profile_id,
        candidate_id=args.candidate_id,
        chat_kwargs=q.DEFAULT_CHAT_KWARGS,
    )
    wall = time.perf_counter() - t0
    rows_path = Path(f"{args.run_id}.rows.jsonl")
    e2e = []
    sum_ct = 0.0
    sum_el = 0.0
    if rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            u = row.get("usage") or {}
            el = float(row.get("elapsed_seconds") or row.get("elapsed") or 0)
            ct = float(u.get("completion_tokens") or 0)
            sum_ct += ct
            sum_el += el
            if el and ct:
                e2e.append(ct / el)
    summary["campaign_wall_seconds"] = wall
    summary["e2e_tok_s"] = {
        "n": len(e2e),
        "mean": (sum(e2e) / len(e2e)) if e2e else None,
        "min": min(e2e) if e2e else None,
        "max": max(e2e) if e2e else None,
        "sum_completion_tokens": int(sum_ct),
        "sum_elapsed_s": round(sum_el, 3),
        "aggregate_completion_over_wall": (sum_ct / sum_el) if sum_el else None,
    }
    Path(f"{args.run_id}.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "rows", "correct_count", "incorrect_count", "ungraded_count", "grader_error_count", "e2e_tok_s", "campaign_wall_seconds") if k in summary}, indent=2))
    return 0 if summary.get("status") == "SCORED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
