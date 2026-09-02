#!/usr/bin/env python3
"""Prose bench: N concurrent ~500-word explanation requests, greedy.
Measures per-stream output tok/s and aggregate; accept-len read from server log after.
Usage: prose_bench.py <base_url> <model> <concurrency> <reps_per_stream>"""
import json, sys, time, urllib.request, concurrent.futures as cf

BASE, MODEL, C, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4] or "1")

TOPICS = [
    "how speculative decoding works in large language models, covering the draft model, verification, and acceptance",
    "why the sky is blue, including Rayleigh scattering and how it affects different wavelengths of light",
    "how vaccines train the immune system, describing antigens, B cells, antibodies, and memory cells",
    "how a four-stroke car engine works, covering intake, compression, power, and exhaust strokes",
    "how HTTPS keeps web browsing secure, including certificates, TLS handshake, and encryption",
    "why airplanes stay in the air, explaining lift, the wing shape, and pressure differences",
    "how coffee gets from a plant to a cup, covering growing, harvesting, roasting, and brewing",
    "how the internet routes a message across the world, mentioning DNS, packets, and routers",
]

def one(i):
    topic = TOPICS[i % len(TOPICS)]
    prompt = (f"Write a clear, well-organized explanation of {topic}. "
              "Aim for about 500 words. Use plain prose with short paragraphs.")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.time() - t0
    comp = d["usage"]["completion_tokens"]
    return dt, comp

jobs = [(i, r) for r in range(REPS) for i in range(C)]
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=C) as ex:
    results = list(ex.map(lambda j: one(j[0]), jobs))
wall = time.time() - t0
tot = sum(c for _, c in results)
rates = sorted(c / dt for dt, c in results)
print(f"prose c{C} x{REPS}: {len(results)} reqs, {tot} out tokens, wall {wall:.1f}s")
print(f"aggregate out tok/s = {tot / wall:.2f}")
print(f"per-stream tok/s: min {rates[0]:.2f} med {rates[len(rates)//2]:.2f} max {rates[-1]:.2f}")
mean_dt = sum(dt for dt, _ in results) / len(results)
mean_ct = sum(c for _, c in results) / len(results)
print(f"mean completion tokens = {mean_ct:.0f} (target ~500+)")
