#!/usr/bin/env python3
"""DSV4-Flash-Vision-Exp max-context NIAH: advertised window at 25/50/90% depth + multi-key 33/66.

Adapted from the proven v026 fixed-300k chat NIAH runner. Uses the server
/tokenize endpoint (no host AutoTokenizer). Depth = fraction of target tokens.
Multi-key variant embeds 3 needles at 33/66/probe depths; the answer must be
the probe needle (third).
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NONCE = "R0B0BENCH_NIAH_SLOT"
FILLER = "The archive records ordinary weather observations for a routine day. "
PREFIX = "BEGIN DOCUMENT\n"
SUFFIX = (
    "\nEND DOCUMENT\n"
    "Question: What is the exact value after NEEDLE:? Answer with that value only."
)
SYSTEM = (
    "You are a precise retrieval system. The supplied document contains exactly one "
    "line beginning with NEEDLE:. Ignore the surrounding filler and answer the final "
    "question with only the value from that line."
)
MULTIKEY_NONCES = ["ni4h7q2x9", "k7d2m8v4", "t3r9w5y1"]


def request_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail[:1000]}") from error


def get_served_model(base):
    data = request_json(base + "/v1/models", {}, 600)
    return data["data"][0]["id"]


def tokenize(base, model, text):
    data = request_json(
        base + "/tokenize",
        {"model": model, "prompt": text},
        timeout=600,
    )
    tokens = data.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response keys: {sorted(data)}")
    return tokens


def atomic_write(path, document):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def build_document(base, model, target_tokens, depth):
    filler_tokens = len(tokenize(base, model, FILLER))
    needle_line = f"\nNEEDLE: {NONCE}\n"
    fixed_tokens = len(tokenize(base, model, PREFIX + needle_line + SUFFIX))
    total_repeats = max(1, (target_tokens - fixed_tokens) // filler_tokens)
    pre_repeats = int(total_repeats * depth)
    post_repeats = total_repeats - pre_repeats
    doc = PREFIX + FILLER * pre_repeats + needle_line + FILLER * post_repeats + SUFFIX
    prompt_tokens = len(tokenize(base, model, doc))
    needle_start = len(tokenize(base, model, PREFIX + FILLER * pre_repeats))
    meta = {
        "target_prompt_tokens": target_tokens,
        "raw_prompt_tokens": prompt_tokens,
        "needle_start_tokens": needle_start,
        "actual_depth": needle_start / prompt_tokens if prompt_tokens else None,
        "filler_token_width": filler_tokens,
        "pre_repeats": pre_repeats,
        "post_repeats": post_repeats,
    }
    return doc, meta


def build_multikey(base, model, target_tokens, probe_depth):
    filler_tokens = len(tokenize(base, model, FILLER))
    line1 = f"\nNEEDLE: {MULTIKEY_NONCES[0]}\n"
    line2 = f"\nNEEDLE: {MULTIKEY_NONCES[1]}\n"
    line3 = f"\nNEEDLE: {MULTIKEY_NONCES[2]}\n"
    fixed = len(tokenize(base, model, PREFIX + line1 + line2 + line3 + SUFFIX))
    total_repeats = max(3, (target_tokens - fixed) // filler_tokens)
    c1 = int(total_repeats * 0.33)
    c2 = int(total_repeats * 0.66) - c1
    c3 = int(total_repeats * probe_depth) - c1 - c2
    c4 = total_repeats - c1 - c2 - c3
    if min(c1, c2, c3, c4) < 1:
        c1 = 1
        c2 = 1
        c3 = max(1, int(total_repeats * probe_depth) - 2)
        c4 = max(1, total_repeats - c1 - c2 - c3)
    doc = (
        PREFIX
        + FILLER * c1 + line1
        + FILLER * c2 + line2
        + FILLER * c3 + line3
        + FILLER * c4 + SUFFIX
    )
    meta = {
        "target_prompt_tokens": target_tokens,
        "raw_prompt_tokens": len(tokenize(base, model, doc)),
        "distractor_nonces": MULTIKEY_NONCES[:2],
        "probe_nonce": MULTIKEY_NONCES[2],
        "needle_depths": [0.33, 0.66, probe_depth],
    }
    return doc, meta


def run_case(base, model, label, depth, target_tokens, multikey=False):
    if multikey:
        document, construction = build_multikey(base, model, target_tokens, depth)
        expected = MULTIKEY_NONCES[2]
    else:
        document, construction = build_document(base, model, target_tokens, depth)
        expected = NONCE
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": document},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.perf_counter()
    data = request_json(base + "/v1/chat/completions", body, timeout=43200)
    elapsed = time.perf_counter() - started
    message = data["choices"][0]["message"]
    content = (message.get("content") or "")[:500]
    usage = data.get("usage") or {}
    return {
        "label": label,
        "requested_depth": depth,
        "multikey": multikey,
        **construction,
        "api_prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_s": round(elapsed, 3),
        "response": content,
        "needle_retrieved": expected in content,
        "transport_ok": True,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.5.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-tokens", type=int, default=1048512,
                        help="default: 1048576 - 64 (advertised window minus headroom)")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    model = get_served_model(base)
    output = Path(args.output)
    target = args.target_tokens

    if output.exists():
        document = json.loads(output.read_text())
    else:
        document = {
            "schema_version": 3,
            "method": "advertised_window_niah_plus_multikey",
            "model": model,
            "nonce": NONCE,
            "target_tokens": target,
            "results": {},
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
    cases = [
        ("25%", 0.25, False),
        ("50%", 0.50, False),
        ("90%", 0.90, False),
        ("mk33", 0.33, True),
        ("mk66", 0.66, True),
    ]
    for label, depth, mk in cases:
        if label in document["results"]:
            print(f"SKIP existing {label}", flush=True)
            continue
        print(f"START {label}", flush=True)
        try:
            result = run_case(base, model, label, depth, target, multikey=mk)
        except Exception as error:
            result = {
                "label": label,
                "requested_depth": depth,
                "transport_ok": False,
                "needle_retrieved": False,
                "error": str(error),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
        document["results"][label] = result
        atomic_write(output, document)
        print(json.dumps(result, sort_keys=True, default=str), flush=True)
    results = document["results"]
    complete = {"25%", "50%", "90%"}.issubset(set(results))
    passed = complete and all(
        bool(row.get("needle_retrieved")) for row in results.values()
    )
    document["verdict"] = "NIAH_PASS" if passed else "NIAH_FAIL"
    document["eligible_count"] = sum(bool(r.get("transport_ok")) for r in results.values())
    document["semantic_pass_count"] = sum(
        bool(r.get("needle_retrieved")) for r in results.values()
    )
    document["infra_error_count"] = sum(
        not bool(r.get("transport_ok")) for r in results.values()
    )
    document["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write(output, document)
    print(document["verdict"], flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
