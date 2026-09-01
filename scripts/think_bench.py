#!/usr/bin/env python3
"""Thinking-mode throughput bench against the SGLang OpenAI endpoint.

Usage: think_bench.py BASE_URL MODEL CONCURRENCY REPS --effort max --temperature 0 [--top-p 1.0]
       [--max-tokens 6144] [--label R0-greedy] [--out evidence/.../think-c1-greedy.json]
Measures aggregate output tok/s (reasoning + content), per-stream tok/s, mean completion tokens,
and avg_spec_accept_length delta from /get_internal_state (None if the build 404s it)."""
import argparse, json, sys, time, urllib.request, concurrent.futures as cf

PROMPTS = [
    "A tank is filled by pipe A in 6 h and by pipe B in 9 h; a drain empties it in 12 h. All three open at t=0; after 2 h the drain is closed. When is the tank full? Show every step.",
    "Prove that for every integer n >= 1, 7 divides 8^n - 1. Then find the smallest n such that 7^2 divides 8^n - 1.",
    "Write a Python function that returns the longest palindromic substring in O(n^2) time, with unit tests and a complexity argument.",
    "Three boxes contain two gold coins, two silver coins, and one of each. You draw a gold coin from a random box. Probability the other coin in that box is gold? Explain two ways.",
    "Design a rate limiter for 10k QPS across 4 API servers with a shared Redis. Compare token bucket vs sliding log; give pseudocode and failure modes.",
    "Solve: x^2 + y^2 = 25 and x*y = 12 for real x,y. List all solutions and verify each.",
    "Explain why the sky is blue and sunsets are red using Rayleigh scattering; include the wavelength dependence formula and one numerical comparison.",
    "A 2 kg block slides down a 30 degree frictionless incline of length 4 m, then across a floor with mu=0.3. How far does it travel on the floor? Show units.",
    "Implement LRU cache in Python with O(1) get/put using a dict and doubly linked list; include tests for eviction order.",
    "How many trailing zeros does 125! have? Generalize to n! and prove the formula.",
    "Compare Raft and Paxos leader election; describe a network partition scenario and what each protocol guarantees.",
    "Find all primes p such that p^2 + 2 is also prime. Prove your answer.",
    "Write a SQL query to find the second-highest salary per department, handling ties, and explain the window function choice.",
    "A ladder 10 m long leans against a wall; its foot slides away at 1 m/s. How fast is the top falling when the foot is 6 m from the wall?",
    "Explain speculative decoding with a draft model: acceptance rule for greedy decoding, expected speedup formula in terms of acceptance rate alpha and draft length k.",
    "Determine whether the series sum_{n=1}^inf (n!)/(n^n) converges; justify with a named test.",
]

def one(base, model, i, effort, temperature, top_p, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort},
    }
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    return {"dt": dt, "completion_tokens": d["usage"]["completion_tokens"],
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "has_reasoning": bool(msg.get("reasoning_content")),
            "finish": d["choices"][0].get("finish_reason")}

def accept_len(base):
    try:
        with urllib.request.urlopen(f"{base}/get_internal_state", timeout=10) as r:
            s = json.load(r)
        s = s[0] if isinstance(s, list) else s
        return s.get("avg_spec_accept_length")
    except Exception:
        return None

def summarize(results, wall):
    tot = sum(r["completion_tokens"] for r in results)
    rates = sorted(r["completion_tokens"] / r["dt"] for r in results)
    return {"n": len(results), "total_completion_tokens": tot, "wall_s": round(wall, 2),
            "aggregate_out_tok_s": round(tot / wall, 2),
            "per_stream_tok_s": {"min": round(rates[0], 2), "med": round(rates[len(rates)//2], 2), "max": round(rates[-1], 2)},
            "mean_completion_tokens": round(tot / len(results), 1),
            "reasoning_present_frac": round(sum(r["has_reasoning"] for r in results) / len(results), 3),
            "finish_reasons": sorted({r["finish"] for r in results})}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("concurrency", type=int); ap.add_argument("reps", type=int)
    ap.add_argument("--effort", default="max"); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0); ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--label", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    a0 = accept_len(a.base)
    jobs = [i for _ in range(a.reps) for i in range(a.concurrency)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        results = list(ex.map(lambda i: one(a.base, a.model, i, a.effort, a.temperature, a.top_p, a.max_tokens), jobs))
    wall = time.time() - t0
    s = summarize(results, wall)
    s.update({"label": a.label, "effort": a.effort, "temperature": a.temperature, "top_p": a.top_p,
              "concurrency": a.concurrency, "accept_len_before": a0, "accept_len_after": accept_len(a.base)})
    print(json.dumps(s, indent=2))
    if a.out:
        with open(a.out, "w") as f: json.dump({"summary": s, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
