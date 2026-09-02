#!/usr/bin/env python3
"""Vision + text canaries against the OpenAI-compatible endpoint.

Usage: vision_canary.py BASE_URL MODEL [--out PATH]
Exit 0 only if every check PASSes.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
import zlib

# Minimal 96x64 PNG: left half red, right half blue. No Pillow required.
def _png_red_blue(w: int = 96, h: int = 64) -> bytes:
    raw = bytearray()
    for _y in range(h):
        raw.append(0)  # filter none
        for x in range(w):
            if x < w // 2:
                raw.extend(b"\xff\x00\x00")
            else:
                raw.extend(b"\x00\x00\xff")

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + tag + data + crc.to_bytes(4, "big")

    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def chat(base: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    d["_elapsed_s"] = time.time() - t0
    return d


def content_of(d: dict) -> str:
    msg = d["choices"][0]["message"]
    return (msg.get("content") or "") + " " + (msg.get("reasoning_content") or msg.get("reasoning") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    results = []
    fail = 0

    # 1 text arithmetic
    d = chat(a.base, {
        "model": a.model,
        "messages": [{"role": "user", "content": "What is 17*23? Answer with the number."}],
        "max_tokens": 2048, "temperature": 0,
    })
    text = content_of(d)
    ok = "391" in text
    results.append({"id": "arith_17x23", "ok": ok, "elapsed_s": d["_elapsed_s"],
                    "usage": d.get("usage"), "snippet": text[-200:]})
    fail += not ok

    # 2 vision color
    b64 = base64.b64encode(_png_red_blue()).decode()
    d = chat(a.base, {
        "model": a.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "The image is split vertically. Reply exactly: LEFT=<color>; RIGHT=<color> using red or blue."},
        ]}],
        "max_tokens": 256, "temperature": 0,
    })
    text = content_of(d).lower()
    usage = d.get("usage") or {}
    ok = "left=red" in text.replace(" ", "") and "right=blue" in text.replace(" ", "")
    results.append({"id": "vision_red_blue", "ok": ok, "elapsed_s": d["_elapsed_s"],
                    "usage": usage, "prompt_tokens": usage.get("prompt_tokens"),
                    "snippet": content_of(d)[:400]})
    fail += not ok

    # 3 text path still works after vision
    d = chat(a.base, {
        "model": a.model,
        "messages": [{"role": "user", "content": "Reply with exactly 4 and nothing else. What is 2+2?"}],
        "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    })
    text = (d["choices"][0]["message"].get("content") or "").strip()
    ok = "4" in text
    results.append({"id": "text_after_vision", "ok": ok, "elapsed_s": d["_elapsed_s"],
                    "snippet": text[:80]})
    fail += not ok

    summary = {"pass": len(results) - fail, "fail": fail, "results": results}
    print(json.dumps(summary, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
    print("VISION_CANARY:", "PASS" if fail == 0 else "FAIL")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
