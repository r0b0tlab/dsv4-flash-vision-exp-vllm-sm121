#!/usr/bin/env python3
"""NVFP4 KV semantic gate for reasoning evals.

Run this after low-level corruption probes pass but before GSM8K@100/full GSM8K.
It checks that the endpoint both avoids known decode-corruption signatures and
answers a few deterministic arithmetic prompts semantically correctly.

Usage:
  python3 scripts/probe_nvfp4_semantic_gate.py \
    --base-url http://127.0.0.1:18080 \
    --model Qwen3.6-27B-NVFP4 \
    --out /tmp/nvfp4-semantic-gate.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Probe:
    name: str
    prompt: str
    must_match: str


PROBES = [
    Probe("add_2_2", "What is 2+2? Answer briefly.", r"\b4\b"),
    Probe(
        "multiply_17_23",
        "What is 17 times 23? Answer with the number and one short sentence.",
        r"\b391\b",
    ),
    Probe(
        "robe_fiber",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts total? Answer briefly.",
        r"\b3\b",
    ),
    Probe(
        "download_rate",
        "Carla is downloading a 200 GB file at 25 GB per hour. "
        "How many hours will it take? Answer briefly.",
        r"\b8\b",
    ),
]

BAD_PATTERNS = ["!!!!", "icicic", "liblib", "WindowSizeResolver", "另另", "砚砚"]


def request_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        obj = json.loads(resp.read().decode())
    return obj["choices"][0].get("text") or ""


def corruption_reason(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if any(pattern in text for pattern in BAD_PATTERNS):
        return "known_bad_pattern"
    alnum = sum(ch.isalnum() for ch in text)
    punct = sum(ch in string.punctuation for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)
    if len(text) >= 20 and alnum == 0 and punct > 10:
        return "punctuation_only"
    if len(text) >= 30 and non_ascii / max(len(text), 1) > 0.35:
        return "high_non_ascii_ratio"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Server base URL without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", default="nvfp4-semantic-gate.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with out.open("w") as f:
        idx = 0
        for _ in range(args.repeat):
            for probe in PROBES:
                idx += 1
                t0 = time.time()
                try:
                    text = request_completion(
                        args.base_url, args.model, probe.prompt, args.max_tokens
                    )
                    corr = corruption_reason(text)
                    semantic_ok = re.search(probe.must_match, text) is not None
                    ok = corr is None and semantic_ok
                    reason = corr or ("clean_semantic" if ok else "wrong_answer")
                    rec = {
                        "i": idx,
                        "name": probe.name,
                        "ok": ok,
                        "reason": reason,
                        "semantic_ok": semantic_ok,
                        "must_match": probe.must_match,
                        "prompt": probe.prompt,
                        "text": text,
                        "latency_s": round(time.time() - t0, 3),
                    }
                except Exception as exc:  # noqa: BLE001 - CLI probe should log all failures
                    rec = {
                        "i": idx,
                        "name": probe.name,
                        "ok": False,
                        "reason": "exception",
                        "prompt": probe.prompt,
                        "error": repr(exc),
                        "latency_s": round(time.time() - t0, 3),
                    }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                preview = rec.get("text", rec.get("error", "")).replace("\n", " ")[:120]
                print(
                    f"{idx:02d} {rec['name']} ok={rec['ok']} "
                    f"reason={rec['reason']} text={preview!r}",
                    flush=True,
                )

    summary = {
        "total": len(records),
        "passed": all(r["ok"] for r in records),
        "clean_semantic": sum(r["ok"] for r in records),
        "bad": sum(not r["ok"] for r in records),
        "out": str(out),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
