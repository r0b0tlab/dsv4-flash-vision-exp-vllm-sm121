#!/usr/bin/env python3
"""1 Hz nvidia-smi + optional /metrics scrape. No hostnames in output.

Usage: telemetry_sampler.py --out PATH [--metrics-url URL] [--interval 1]
Stop on SIGINT/SIGTERM. Writes JSONL.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
import urllib.request

STOP = False


def _stop(*_):
    global STOP
    STOP = True


def smi() -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,power.draw,temperature.gpu,utilization.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "i": int(parts[0]),
                "w": float(parts[1]) if parts[1] not in ("", "[N/A]") else None,
                "c": float(parts[2]) if parts[2] not in ("", "[N/A]") else None,
                "u": float(parts[3]) if parts[3] not in ("", "[N/A]") else None,
                "mhz": float(parts[4]) if parts[4] not in ("", "[N/A]") else None,
            }
        )
    return {"gpus": gpus}


def metrics(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            text = r.read().decode()
    except Exception:
        return {}
    want = (
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
    )
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in want:
            try:
                out[name] = float(line.split()[-1])
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-url", default="")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    with open(args.out, "a") as f:
        while not STOP:
            row = {"t": time.time(), "smi": smi()}
            if args.metrics_url:
                row["prom"] = metrics(args.metrics_url)
            f.write(json.dumps(row) + "\n")
            f.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
