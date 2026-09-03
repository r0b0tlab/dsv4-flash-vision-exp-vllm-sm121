#!/usr/bin/env python3
"""TD2W300: count 1-300 spelled out. Throughput only.

Usage: td2w300.py BASE_URL MODEL [--thinking high|off] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPT = "one two three four five. Continue through three hundred in English words, spaces only."


def metrics(base: str) -> dict:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, val = parts[0].split("{", 1)[0], parts[-1]
        if name in (
            "vllm:generation_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:prompt_tokens_total",
        ):
            try:
                out[name] = float(val)
            except ValueError:
                pass
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--thinking", choices=("high", "off"), default="off")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()
    http_timeout = a.timeout or max(600, int(a.max_tokens / 8) + 180)
    kwargs = (
        {"thinking": False}
        if a.thinking == "off"
        else {"thinking": True, "enable_thinking": True, "reasoning_effort": "high"}
    )
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": a.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": kwargs,
    }
    before = metrics(a.base)
    t0 = time.time()
    req = urllib.request.Request(
        a.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=http_timeout) as r:
        d = json.load(r)
    elapsed = time.time() - t0
    after = metrics(a.base)
    msg = d["choices"][0]["message"]
    usage = d.get("usage") or {}
    ct = int(usage.get("completion_tokens") or 0)
    pt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    rsn = int(details.get("reasoning_tokens") or 0)
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    listed = "three hundred" in content.lower()
    row = {
        "name": "TD2W300",
        "prompt": PROMPT,
        "thinking": a.thinking,
        "max_tokens": a.max_tokens,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rsn,
        "output_tok_s": round(ct / elapsed, 3) if elapsed else None,
        "finish_reason": d["choices"][0].get("finish_reason"),
        "content_chars": len(content),
        "content_head": content[:180],
        "content_tail": content[-180:],
        "reasoning_chars": len(reasoning),
        "reasoning_head": reasoning[:180],
        "reasoning_tail": reasoning[-180:],
        "listed_three_hundred": listed,
        "complete": bool(listed and d["choices"][0].get("finish_reason") == "stop"),
        "metrics_delta": {
            k: (after.get(k, 0) - before.get(k, 0)) for k in sorted(set(before) | set(after))
        },
    }
    acc = row["metrics_delta"].get("vllm:spec_decode_num_accepted_tokens_total")
    draft = row["metrics_delta"].get("vllm:spec_decode_num_draft_tokens_total")
    if acc is not None and draft:
        row["accept_rate"] = round(acc / draft, 4)
        row["accept_len_est"] = round(5 * acc / draft, 3) if draft else None
    print(json.dumps(row, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(row, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
