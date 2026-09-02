#!/usr/bin/env python3
"""Short deterministic vision gate (8 items). Synthetic PNGs, exact-match answers.

Usage: vision_short_bench.py BASE_URL MODEL [--out PATH]
Exit 0 iff all 8 PASS.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
import zlib


def _png(w: int, h: int, rgb_at) -> bytes:
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw.extend(rgb_at(x, y))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + tag + data + crc.to_bytes(4, "big")

    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def red_blue(w=96, h=64):
    return _png(w, h, lambda x, y: b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff")


def solid(rgb, w=64, h=64):
    return _png(w, h, lambda x, y: rgb)


def three_dots():
    def rgb(x, y):
        for cx, cy in ((16, 32), (48, 32), (80, 32)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 36:
                return b"\x00\x00\x00"
        return b"\xff\xff\xff"

    return _png(96, 64, rgb)


def digit_seven():
    def rgb(x, y):
        # crude 7: top bar + diagonal
        if 8 <= y <= 14 and 16 <= x <= 48:
            return b"\x00\x00\x00"
        if 14 < y <= 56 and abs((x - 48) + (y - 14) * 0.4) < 4:
            return b"\x00\x00\x00"
        return b"\xff\xff\xff"

    return _png(64, 64, rgb)


ITEMS = [
    {
        "id": "v-red-blue",
        "png": red_blue,
        "prompt": 'Identify left and right half colors. Reply exactly: LEFT=<color>; RIGHT=<color> using red or blue.',
        "ok": lambda t: "left=red" in t and "right=blue" in t,
    },
    {
        "id": "v-solid-green",
        "png": lambda: solid(b"\x00\x80\x00"),
        "prompt": "What is the dominant color of this image? Reply with one word: red, green, blue, black, or white.",
        "ok": lambda t: "green" in t,
    },
    {
        "id": "v-solid-white",
        "png": lambda: solid(b"\xff\xff\xff"),
        "prompt": "Is this image mostly white or mostly black? Reply with one word.",
        "ok": lambda t: "white" in t,
    },
    {
        "id": "v-count-3",
        "png": three_dots,
        "prompt": "How many black filled circles are in this image? Reply with a single integer.",
        "ok": lambda t: any(tok == "3" for tok in t.replace(",", " ").split()),
    },
    {
        "id": "v-digit-7",
        "png": digit_seven,
        "prompt": "What digit is drawn in black on white? Reply with a single digit 0-9.",
        "ok": lambda t: "7" in t,
    },
    {
        "id": "v-not-photo",
        "png": lambda: solid(b"\xff\x00\x00"),
        "prompt": "Is this a photograph of a real-world scene? Reply yes or no.",
        "ok": lambda t: t.strip().startswith("no") or " no " in f" {t} ",
    },
    {
        "id": "v-blue-field",
        "png": lambda: solid(b"\x00\x00\xff"),
        "prompt": "What is the dominant color? Reply with one word: red, green, or blue.",
        "ok": lambda t: "blue" in t,
    },
    {
        "id": "v-split-again",
        "png": red_blue,
        "prompt": "Which side is blue, left or right? Reply with one word: left or right.",
        "ok": lambda t: "right" in t.split()[0] if t.split() else "right" in t,
    },
]


def chat(base: str, model: str, prompt: str, png: bytes, timeout: int = 600) -> dict:
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(png)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True, "reasoning_effort": "high"},
    }
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


def text_of(d: dict) -> str:
    msg = d["choices"][0]["message"]
    return ((msg.get("content") or "") + " " + (msg.get("reasoning_content") or msg.get("reasoning") or "")).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    results = []
    fail = 0
    for item in ITEMS:
        try:
            d = chat(a.base, a.model, item["prompt"], item["png"]())
            blob = text_of(d)
            ok = bool(item["ok"](blob))
            usage = d.get("usage") or {}
            results.append({
                "id": item["id"],
                "ok": ok,
                "elapsed_s": d["_elapsed_s"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "e2e_tok_s": (usage.get("completion_tokens") or 0) / d["_elapsed_s"] if d["_elapsed_s"] else None,
                "snippet": blob[:240],
            })
            fail += not ok
        except Exception as exc:
            results.append({"id": item["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            fail += 1
    out = {"pass": len(ITEMS) - fail, "fail": fail, "n": len(ITEMS), "results": results}
    print(json.dumps(out, indent=2))
    if a.out:
        open(a.out, "w").write(json.dumps(out, indent=2) + "\n")
    print("VISION_SHORT: PASS" if fail == 0 else "VISION_SHORT: FAIL")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
